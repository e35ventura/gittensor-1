"""Composition root for verification, scaling, placement, routing, and rewards."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

from gittensor.compute.autoscaler import AutoscaleDecision, FleetAutoscaler
from gittensor.compute.config import ComputeConfig
from gittensor.compute.models import GPURecord, GPURegistration, GPUState, Release
from gittensor.compute.placement import GlobalGepetto, PlacementPlan, ReleaseDemand
from gittensor.compute.routing import CapacityUnavailable, FastestFinishRouter, RouteDecision, RoutingGPU
from gittensor.compute.settlement import (
    FundingPlan,
    SettlementResult,
    aggregate_miner_rewards,
    funding_plan,
    settle_ready_seconds,
)
from gittensor.compute.verification import SparkComputeClient, SparkVerifier, index_snapshots


@dataclass(frozen=True)
class ControlTick:
    autoscaling: AutoscaleDecision
    funding: FundingPlan
    placement: PlacementPlan
    ready_gpus: int
    registered_gpus: int


class ComputeControlPlane:
    """Single-process reference control plane.

    Production deployments should place registrations and assignment epochs in
    durable storage and replace the in-process reservation lock with a regional
    atomic store. The algorithms and state transitions remain the same.
    """

    def __init__(self, config: ComputeConfig, *, clock=time.time) -> None:
        self.config = config
        self.clock = clock
        self.releases: dict[str, Release] = {}
        self.gpus: dict[str, GPURecord] = {}
        self.verifier = SparkVerifier(config.verification)
        self.verification_client = SparkComputeClient(config.verification)
        self.autoscaler = FleetAutoscaler(
            config.autoscaling,
            floor=config.fleet.floor,
            certified_slots_per_gpu=config.fleet.certified_slots_per_gpu,
            initial_target=config.fleet.initial_target,
        )
        self.router = FastestFinishRouter(config.router.reservation_ttl_seconds)
        self.gepetto = GlobalGepetto(config.placement.minimum_residency_seconds)
        self.max_budget_per_hour = config.fleet.max_budget_per_hour
        self.funding = self._funding_plan()
        self.placement = PlacementPlan({}, {}, ())
        now = self.clock()
        self._last_tick_at = now
        self._last_ready_account_at = now
        self._settlement_started_at = now
        self._funded_target_seconds = 0.0
        self._ready_seconds: dict[str, float] = {}
        self._accepted: dict[str, int] = {}
        self._rejected: dict[str, int] = {}
        self._service_seconds_total: dict[str, float] = {}
        self._lock = threading.RLock()

    def register_release(self, release: Release) -> None:
        with self._lock:
            self.releases[release.release_digest] = release

    def register_gpu(self, registration: GPURegistration, now: float | None = None) -> None:
        with self._lock:
            if registration.release_digest not in self.releases:
                raise ValueError('GPU release must be approved before registration')
            if registration.certified_slots != self.config.fleet.certified_slots_per_gpu:
                raise ValueError('GPU certified_slots must match the operator-certified concurrency')
            timestamp = self.clock() if now is None else now
            existing = self.gpus.get(registration.gpu_id)
            if existing and existing.registration.miner_uid != registration.miner_uid:
                raise ValueError('a GPU cannot move between miner UIDs without deregistration')
            self.gpus[registration.gpu_id] = GPURecord(
                registration=registration,
                state=GPUState.REGISTERED,
                assignment_started_at=timestamp,
            )

    def begin_assignment(self, gpu_id: str, release_digest: str, now: float | None = None) -> None:
        """Begin the global Gepetto transition for one GPU.

        The old canary binding remains in place, which prevents READY until the
        new runtime is loaded and ``bind_runtime_canary`` is called.
        """
        with self._lock:
            if release_digest not in self.releases:
                raise ValueError('release is not approved')
            record = self.gpus[gpu_id]
            self._account_ready_seconds(self.clock() if now is None else now)
            record.registration = replace(record.registration, release_digest=release_digest)
            record.state = GPUState.DRAINING
            record.lease = None
            record.revocation_reason = 'release transition in progress'
            record.assignment_started_at = self.clock() if now is None else now

    def bind_runtime_canary(self, gpu_id: str, release_digest: str) -> None:
        """Record the trusted SparkCompute canary bootstrap for a loaded runtime."""
        with self._lock:
            record = self.gpus[gpu_id]
            if record.registration.release_digest != release_digest:
                raise ValueError('canary binding must match the current Gepetto assignment')
            record.registration = replace(record.registration, canary_release_digest=release_digest)
            record.state = GPUState.RUNTIME_VERIFY
            record.revocation_reason = None

    def refresh_verification(
        self,
        snapshots: Sequence[Mapping[str, Any]] | None = None,
        now: float | None = None,
    ) -> dict[str, str]:
        timestamp = self.clock() if now is None else now
        source = self.verification_client.fetch_status() if snapshots is None else snapshots
        indexed = index_snapshots(source)
        outcomes: dict[str, str] = {}
        with self._lock:
            self._account_ready_seconds(timestamp)
            evaluated = {}
            for gpu_id, record in self.gpus.items():
                release = self.releases.get(record.registration.release_digest)
                if release is None:
                    evaluated[gpu_id] = None, 'assigned release is no longer approved'
                    continue
                outcome = self.verifier.evaluate(
                    record.registration,
                    release,
                    indexed.get(record.registration.spark_node_id),
                    timestamp,
                )
                evaluated[gpu_id] = outcome.lease, outcome.reason

            uuid_to_gpus: dict[str, list[str]] = {}
            for gpu_id, (lease, _) in evaluated.items():
                if lease is not None:
                    uuid_to_gpus.setdefault(lease.hardware_uuid, []).append(gpu_id)
            duplicate_gpus = {gpu_id for gpu_ids in uuid_to_gpus.values() if len(gpu_ids) > 1 for gpu_id in gpu_ids}

            for gpu_id, record in self.gpus.items():
                lease, reason = evaluated[gpu_id]
                if gpu_id in duplicate_gpus:
                    lease = None
                    reason = 'duplicate SparkCompute GPU UUID is registered more than once'
                if lease is not None:
                    record.lease = lease
                    record.state = GPUState.READY
                    record.revocation_reason = None
                    outcomes[gpu_id] = GPUState.READY.value
                else:
                    record.lease = None
                    record.state = GPUState.QUARANTINED
                    record.revocation_reason = reason
                    outcomes[gpu_id] = reason or GPUState.QUARANTINED.value
        return outcomes

    def route(
        self,
        release_digest: str,
        requester_region: str,
        expected_service_seconds: float,
        now: float | None = None,
    ) -> RouteDecision:
        timestamp = self.clock() if now is None else now
        if release_digest not in self.releases:
            raise CapacityUnavailable('requested release is not approved')
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_leases(timestamp)
            candidates = [
                RoutingGPU(
                    gpu_id=record.registration.gpu_id,
                    endpoint=record.registration.endpoint,
                    release_digest=record.registration.release_digest,
                    performance_class=record.registration.performance_class,
                    certified_slots=record.registration.certified_slots,
                    # The reference router owns all active reservations, so
                    # its internal atomic count is the source of truth.
                    reported_active_slots=0,
                    remaining_work_seconds=record.reported_remaining_work_seconds,
                    service_seconds=max(0.0, expected_service_seconds),
                    rtt_ms=float(
                        record.registration.latency_by_region_ms.get(
                            requester_region,
                            self.config.router.default_rtt_ms,
                        )
                    ),
                )
                for record in self.gpus.values()
                if record.is_ready(timestamp) and record.registration.release_digest == release_digest
            ]
            try:
                decision = self.router.route(candidates, release_digest, timestamp)
            except CapacityUnavailable:
                self._record_demand(release_digest, expected_service_seconds, rejected=True)
                raise
            self._record_demand(release_digest, expected_service_seconds, rejected=False)
            return decision

    def complete_reservation(self, reservation_id: str) -> bool:
        return self.router.complete(reservation_id)

    def update_budget(self, max_budget_per_hour: float, now: float | None = None) -> FundingPlan:
        """Apply the current compute share of subnet emissions as an hourly budget."""
        minimum_budget = self.config.fleet.floor * self.config.fleet.target_price_per_gpu_hour
        if max_budget_per_hour < minimum_budget:
            raise ValueError('max_budget_per_hour must continue to fund the configured GPU floor')
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self.max_budget_per_hour = max_budget_per_hour
            self.funding = self._funding_plan()
            return self.funding

    def tick(self, now: float | None = None) -> ControlTick:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_leases(timestamp)
            elapsed = max(timestamp - self._last_tick_at, 1e-9)
            active_counts = self.router.active_counts(timestamp)
            active_slots = sum(active_counts.values())
            rejected_concurrent = sum(
                self._rejected.get(release_digest, 0) / elapsed * self._average_service_seconds(release_digest)
                for release_digest in self.releases
            )
            autoscaling = self.autoscaler.update(
                active_slots=active_slots,
                rejected_concurrent_demand=rejected_concurrent,
                funded_target=self.funding.funded_target,
                now=timestamp,
            )
            self.funding = self._funding_plan()
            release_demand = [
                ReleaseDemand(
                    release_digest=release_digest,
                    concurrent_demand=(self._accepted.get(release_digest, 0) + self._rejected.get(release_digest, 0))
                    / elapsed
                    * self._average_service_seconds(release_digest),
                )
                for release_digest in self.releases
            ]
            ready = [record for record in self.gpus.values() if record.is_ready(timestamp)]
            self.placement = self.gepetto.plan(
                ready,
                self.releases.values(),
                release_demand,
                timestamp,
            )
            self._accepted.clear()
            self._rejected.clear()
            self._service_seconds_total.clear()
            self._last_tick_at = timestamp
            return ControlTick(
                autoscaling=autoscaling,
                funding=self.funding,
                placement=self.placement,
                ready_gpus=len(ready),
                registered_gpus=len(self.gpus),
            )

    def settle(self, now: float | None = None) -> tuple[SettlementResult, dict[int, Decimal]]:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            window_seconds = timestamp - self._settlement_started_at
            window_budget = (
                self.funding.target_price_per_gpu_hour * Decimal(str(self._funded_target_seconds)) / Decimal(3600)
            )
            result = settle_ready_seconds(
                self.funding,
                window_seconds,
                self._ready_seconds,
                window_budget_override=window_budget,
            )
            miner_rewards = aggregate_miner_rewards(
                result,
                {gpu_id: record.registration.miner_uid for gpu_id, record in self.gpus.items()},
            )
            self._ready_seconds = {}
            self._funded_target_seconds = 0.0
            self._settlement_started_at = timestamp
            return result, miner_rewards

    def status(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_leases(timestamp)
            active = self.router.active_counts(timestamp)
            return {
                'desired_target': self.autoscaler.desired_target,
                'funded_target': self.funding.funded_target,
                'funding_shortfall': self.funding.funding_shortfall,
                'target_price_per_gpu_hour': str(self.funding.target_price_per_gpu_hour),
                'max_budget_per_hour': str(self.funding.max_budget_per_hour),
                'verification_source': {
                    'repository': self.config.verification.source_repository,
                    'commit': self.config.verification.source_commit,
                },
                'registered_gpus': len(self.gpus),
                'ready_gpus': sum(record.is_ready(timestamp) for record in self.gpus.values()),
                'placements': dict(self.placement.assignments),
                'gpus': {
                    gpu_id: {
                        'miner_uid': record.registration.miner_uid,
                        'state': record.state.value,
                        'release_digest': record.registration.release_digest,
                        'active_reservations': active.get(gpu_id, 0),
                        'lease_expires_at': record.lease.expires_at if record.lease else None,
                        'revocation_reason': record.revocation_reason,
                    }
                    for gpu_id, record in sorted(self.gpus.items())
                },
            }

    def _funding_plan(self) -> FundingPlan:
        return funding_plan(
            self.autoscaler.desired_target,
            self.config.fleet.target_price_per_gpu_hour,
            self.max_budget_per_hour,
        )

    def _record_demand(self, release_digest: str, service_seconds: float, *, rejected: bool) -> None:
        target = self._rejected if rejected else self._accepted
        target[release_digest] = target.get(release_digest, 0) + 1
        self._service_seconds_total[release_digest] = self._service_seconds_total.get(release_digest, 0.0) + max(
            0.0, service_seconds
        )

    def _average_service_seconds(self, release_digest: str) -> float:
        count = self._accepted.get(release_digest, 0) + self._rejected.get(release_digest, 0)
        return self._service_seconds_total.get(release_digest, 0.0) / count if count else 1.0

    def _account_ready_seconds(self, now: float) -> None:
        if now <= self._last_ready_account_at:
            return
        elapsed = now - self._last_ready_account_at
        self._funded_target_seconds += self.funding.funded_target * elapsed
        for gpu_id, record in self.gpus.items():
            if record.state != GPUState.READY or record.lease is None:
                continue
            eligible_until = min(now, record.lease.expires_at)
            seconds = max(0.0, eligible_until - self._last_ready_account_at)
            self._ready_seconds[gpu_id] = self._ready_seconds.get(gpu_id, 0.0) + seconds
        self._last_ready_account_at = now

    def _expire_leases(self, now: float) -> None:
        for record in self.gpus.values():
            if record.state == GPUState.READY and (record.lease is None or not record.lease.is_live(now)):
                record.state = GPUState.QUARANTINED
                record.lease = None
                record.revocation_reason = 'verification lease expired'
