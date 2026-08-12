"""Composition root for verification, scaling, placement, routing, and rewards."""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

from gittensor.compute.artifacts import ReleaseArtifactVerifier
from gittensor.compute.assignment import AssignmentExecutor
from gittensor.compute.autoscaler import AutoscaleDecision, FleetAutoscaler
from gittensor.compute.config import ComputeConfig
from gittensor.compute.models import (
    AssignmentCommand,
    GPURecord,
    GPURegistration,
    GPUState,
    Release,
    RoutingObservation,
    RuntimeEvidence,
    VerificationLease,
)
from gittensor.compute.placement import GlobalGepetto, PlacementPlan, ReleaseDemand
from gittensor.compute.routing import CapacityUnavailable, FastestFinishRouter, RouteDecision, RoutingGPU
from gittensor.compute.settlement import (
    FundingPlan,
    SettlementResult,
    aggregate_miner_rewards,
    funding_plan,
    settle_ready_seconds,
)
from gittensor.compute.storage import SQLiteStateStore
from gittensor.compute.verification import SparkComputeClient, SparkVerifier, index_snapshots
from gittensor.compute.weight_challenges import WeightChallengeTransport, WeightChallengeVerifier


@dataclass(frozen=True)
class ControlTick:
    autoscaling: AutoscaleDecision
    funding: FundingPlan
    placement: PlacementPlan
    ready_gpus: int
    registered_gpus: int


class ComputeControlPlane:
    """Durable regional control plane for one globally coordinated GPU fleet."""

    def __init__(
        self,
        config: ComputeConfig,
        *,
        clock=time.time,
        store: SQLiteStateStore | None = None,
        assignment_executor: AssignmentExecutor | None = None,
        weight_verifier: WeightChallengeVerifier | None = None,
        weight_transport: WeightChallengeTransport | None = None,
        spark_node_owners: Mapping[str, str] | None = None,
        release_artifact_verifier: ReleaseArtifactVerifier | None = None,
    ) -> None:
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
        self.router = FastestFinishRouter(
            config.router.reservation_ttl_seconds,
            config.router.equivalent_finish_epsilon_seconds,
        )
        self.gepetto = GlobalGepetto(config.placement.minimum_residency_seconds)
        self.store = store
        self.assignment_executor = assignment_executor
        self.weight_verifier = weight_verifier
        self.weight_transport = weight_transport
        self.spark_node_owners = dict(spark_node_owners or {})
        self.release_artifact_verifier = release_artifact_verifier
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
        if self.store and (saved := self.store.load_state()):
            self._restore_state(saved, now)

    def register_release(self, release: Release) -> None:
        with self._lock:
            existing = self.releases.get(release.release_digest)
            if existing is not None and existing != release:
                raise ValueError('an approved release digest is immutable')
            if existing is None and self.release_artifact_verifier is not None:
                self.release_artifact_verifier.verify(release)
            self.releases[release.release_digest] = release
            self._persist()

    def register_gpu(self, registration: GPURegistration, now: float | None = None) -> None:
        """Register a trusted, already-resolved claim.

        Public API callers use ``register_miner_gpu`` so UID ownership always
        comes from the live metagraph and assignments always come from Gepetto.
        """
        with self._lock:
            if registration.release_digest and registration.release_digest not in self.releases:
                raise ValueError('GPU release must be approved before registration')
            if registration.certified_slots != self.config.fleet.certified_slots_per_gpu:
                raise ValueError('GPU certified_slots must match the operator-certified concurrency')
            timestamp = self.clock() if now is None else now
            existing = self.gpus.get(registration.gpu_id)
            if existing and existing.registration.miner_uid != registration.miner_uid:
                raise ValueError('a GPU cannot move between miner UIDs without deregistration')
            if existing and existing.registration.miner_hotkey != registration.miner_hotkey:
                raise ValueError('a GPU cannot move between hotkeys without deregistration')
            if existing and (
                existing.registration.release_digest != registration.release_digest
                or existing.registration.canary_release_digest != registration.canary_release_digest
            ):
                raise ValueError('registration cannot change a Gepetto assignment or runtime binding')
            self.gpus[registration.gpu_id] = GPURecord(
                registration=registration,
                state=GPUState.REGISTERED,
                assignment_started_at=timestamp,
            )
            self._persist()

    def register_miner_gpu(
        self,
        *,
        miner_uid: int,
        miner_hotkey: str,
        gpu_id: str,
        spark_node_id: str,
        endpoint: str,
        region: str,
        now: float | None = None,
    ) -> None:
        """Register hardware ownership without accepting miner-selected placement."""
        if self.spark_node_owners and self.spark_node_owners.get(spark_node_id) != miner_hotkey:
            raise ValueError('SparkCompute node is not enrolled to this miner hotkey')
        self.register_gpu(
            GPURegistration(
                gpu_id=gpu_id,
                spark_node_id=spark_node_id,
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                endpoint=endpoint,
                region=region,
                release_digest='',
                canary_release_digest='',
                certified_slots=self.config.fleet.certified_slots_per_gpu,
            ),
            now=now,
        )

    def begin_assignment(self, gpu_id: str, release_digest: str, now: float | None = None) -> None:
        """Begin the global Gepetto transition for one GPU.

        The old canary binding remains in place, which prevents READY until the
        miner acknowledges the exact runtime evidence for this assignment epoch.
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
            record.assignment_epoch += 1
            record.weight_verified_at = 0.0
            record.runtime_verified_at = 0.0
            self._persist()

    def acknowledge_assignment(
        self,
        gpu_id: str,
        epoch: int,
        state: GPUState,
        evidence: RuntimeEvidence | None = None,
        now: float | None = None,
    ) -> GPUState:
        """Advance an assignment only through the epoch-bound lifecycle."""
        timestamp = self.clock() if now is None else now
        with self._lock:
            record = self.gpus[gpu_id]
            if epoch != record.assignment_epoch:
                raise ValueError('assignment acknowledgement has a stale epoch')
            allowed = {
                GPUState.DRAINING: GPUState.LOADING,
                GPUState.LOADING: GPUState.RUNTIME_VERIFY,
            }
            if allowed.get(record.state) != state:
                raise ValueError(f'invalid assignment transition: {record.state.value} -> {state.value}')
            if state == GPUState.RUNTIME_VERIFY:
                if evidence is None:
                    raise ValueError('runtime evidence is required before RUNTIME_VERIFY')
                release = self.releases[record.registration.release_digest]
                expected = RuntimeEvidence(
                    release_digest=release.release_digest,
                    model_repository=release.model_repository,
                    model_revision=release.model_revision,
                    runtime_digest=release.runtime_digest,
                    runtime_commit=release.runtime_commit,
                    container_image=release.container_image,
                    container_digest=release.container_digest,
                    filesystem_digest=release.filesystem_digest,
                )
                if evidence != expected:
                    raise ValueError('runtime evidence does not match the approved release')
                record.registration = replace(
                    record.registration,
                    canary_release_digest=record.registration.release_digest,
                )
                record.runtime_verified_at = timestamp
            record.state = state
            record.revocation_reason = None
            self._persist()
            return record.state

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
                snapshot = indexed.get(record.registration.spark_node_id)
                if release is None and record.registration.release_digest:
                    evaluated[gpu_id] = None, 'assigned release is no longer approved', True
                    continue
                fully_bound = (
                    release is not None
                    and record.registration.canary_release_digest == record.registration.release_digest
                )
                if fully_bound:
                    assert release is not None
                    outcome = self.verifier.evaluate(record.registration, release, snapshot, timestamp)
                    if (
                        outcome.accepted
                        and self.config.verification.require_weight_challenges
                        and timestamp - record.weight_verified_at
                        >= self.config.verification.weight_verification_ttl_seconds
                    ):
                        outcome = type(outcome)(None, 'model weight challenge is missing or stale')
                    evaluated[gpu_id] = outcome.lease, outcome.reason, True
                else:
                    outcome = self.verifier.evaluate_hardware(record.registration, snapshot, timestamp)
                    evaluated[gpu_id] = outcome.lease, outcome.reason, False

            uuid_to_gpus: dict[str, list[str]] = {}
            for gpu_id, (lease, _, _) in evaluated.items():
                if lease is not None:
                    uuid_to_gpus.setdefault(lease.hardware_uuid, []).append(gpu_id)
            duplicate_gpus = {gpu_id for gpu_ids in uuid_to_gpus.values() if len(gpu_ids) > 1 for gpu_id in gpu_ids}

            for gpu_id, record in self.gpus.items():
                lease, reason, full_verification = evaluated[gpu_id]
                if gpu_id in duplicate_gpus:
                    lease = None
                    reason = 'duplicate SparkCompute GPU UUID is registered more than once'
                if lease is not None:
                    record.lease = lease
                    record.revocation_reason = None
                    if full_verification:
                        record.state = GPUState.READY
                        outcomes[gpu_id] = GPUState.READY.value
                    else:
                        if record.state == GPUState.QUARANTINED:
                            record.state = GPUState.REGISTERED
                        outcomes[gpu_id] = record.state.value
                else:
                    record.lease = None
                    record.revocation_reason = reason
                    if record.state not in {GPUState.DRAINING, GPUState.LOADING, GPUState.RUNTIME_VERIFY}:
                        record.state = GPUState.QUARANTINED
                    outcomes[gpu_id] = reason or GPUState.QUARANTINED.value
            self._persist()
        return outcomes

    def run_weight_challenges(self, now: float | None = None) -> dict[str, bool]:
        """Challenge a random fleet sample, three at a time by default."""
        if not self.config.verification.require_weight_challenges:
            return {}
        verifier = self.weight_verifier
        transport = self.weight_transport
        if verifier is None or transport is None:
            raise RuntimeError('weight challenge verifier and transport are required')
        timestamp = self.clock() if now is None else now
        with self._lock:
            candidates = [
                record
                for record in self.gpus.values()
                if record.state in {GPUState.RUNTIME_VERIFY, GPUState.READY, GPUState.QUARANTINED}
                and record.registration.release_digest in self.releases
                and record.registration.canary_release_digest == record.registration.release_digest
            ]
        # Scale the batch so the whole fleet can be refreshed twice within the
        # verification TTL. ``challenge_sample_size`` remains the minimum batch.
        required_for_fleet = math.ceil(
            len(candidates)
            * self.config.verification.weight_challenge_interval_seconds
            / (self.config.verification.weight_verification_ttl_seconds / 2)
        )
        count = min(
            len(candidates),
            max(self.config.verification.challenge_sample_size, required_for_fleet),
        )
        randomizer = random.SystemRandom()
        randomizer.shuffle(candidates)
        candidates.sort(key=lambda item: item.weight_verified_at)
        selected = candidates[:count]
        outcomes: dict[str, bool] = {}

        def challenge_one(record: GPURecord) -> tuple[str, str, int, bool]:
            gpu_id = record.registration.gpu_id
            release_digest = record.registration.release_digest
            assignment_epoch = record.assignment_epoch
            release = self.releases[release_digest]
            challenge = None
            try:
                challenge = verifier.issue(gpu_id, release, timestamp)
                digest = transport.answer(record, challenge)
                accepted = verifier.verify(challenge.challenge_id, gpu_id, digest, self.clock())
            except Exception:
                if challenge is not None:
                    verifier.cancel(challenge.challenge_id)
                accepted = False
            return gpu_id, release_digest, assignment_epoch, accepted

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, count), thread_name_prefix='weight-challenge') as executor:
            futures = [executor.submit(challenge_one, record) for record in selected]
            results = [future.result() for future in as_completed(futures)]
        for gpu_id, release_digest, assignment_epoch, accepted in results:
            with self._lock:
                current = self.gpus.get(gpu_id)
                if current is None:
                    continue
                if (
                    current.registration.release_digest != release_digest
                    or current.assignment_epoch != assignment_epoch
                ):
                    continue
                if accepted:
                    current.weight_verified_at = timestamp
                    if current.state == GPUState.QUARANTINED:
                        current.state = GPUState.RUNTIME_VERIFY
                    current.revocation_reason = None
                else:
                    current.weight_verified_at = 0.0
                    current.lease = None
                    current.state = GPUState.QUARANTINED
                    current.revocation_reason = 'model weight challenge failed'
                outcomes[gpu_id] = accepted
                self._persist()
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
                    reported_active_slots=(
                        record.gateway_active_slots
                        if timestamp - record.gateway_telemetry_updated_at < self.config.router.telemetry_ttl_seconds
                        else 0
                    ),
                    remaining_work_seconds=(
                        record.gateway_remaining_work_seconds
                        if timestamp - record.gateway_telemetry_updated_at < self.config.router.telemetry_ttl_seconds
                        else 0.0
                    ),
                    service_seconds=max(
                        0.0,
                        record.service_seconds_ewma or expected_service_seconds,
                    ),
                    rtt_ms=float(
                        record.measured_rtt_by_region_ms.get(
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
                self._persist()
                raise
            self._record_demand(release_digest, expected_service_seconds, rejected=False)
            self._persist()
            return decision

    def complete_reservation(self, reservation_id: str) -> bool:
        completed = self.router.complete(reservation_id)
        if completed:
            with self._lock:
                self._persist()
        return completed

    def record_routing_observation(self, observation: RoutingObservation, now: float | None = None) -> None:
        """Update gateway-measured RTT and service performance, never miner claims."""
        timestamp = self.clock() if now is None else now
        if observation.measured_rtt_ms < 0 or observation.service_seconds < 0:
            raise ValueError('routing observations cannot be negative')
        if observation.remaining_work_seconds < 0:
            raise ValueError('remaining_work_seconds cannot be negative')
        with self._lock:
            record = self.gpus[observation.gpu_id]
            if not 0 <= observation.observed_active_slots <= record.registration.certified_slots:
                raise ValueError('observed_active_slots exceeds the certified concurrency')
            alpha = 0.2
            previous_rtt = record.measured_rtt_by_region_ms.get(observation.requester_region)
            record.measured_rtt_by_region_ms[observation.requester_region] = (
                observation.measured_rtt_ms
                if previous_rtt is None
                else alpha * observation.measured_rtt_ms + (1 - alpha) * previous_rtt
            )
            if observation.success:
                record.service_seconds_ewma = (
                    observation.service_seconds
                    if record.service_seconds_ewma <= 0
                    else alpha * observation.service_seconds + (1 - alpha) * record.service_seconds_ewma
                )
            record.gateway_active_slots = observation.observed_active_slots
            record.gateway_remaining_work_seconds = observation.remaining_work_seconds
            record.gateway_telemetry_updated_at = timestamp
            self._persist()

    def update_gpu_telemetry(
        self,
        gpu_id: str,
        miner_hotkey: str,
        *,
        active_slots: int,
        remaining_work_seconds: float,
        now: float | None = None,
    ) -> None:
        timestamp = self.clock() if now is None else now
        with self._lock:
            record = self.gpus[gpu_id]
            if record.registration.miner_hotkey != miner_hotkey:
                raise ValueError('hotkey does not own this GPU')
            if not 0 <= active_slots <= record.registration.certified_slots:
                raise ValueError('active_slots exceeds the certified concurrency')
            if remaining_work_seconds < 0:
                raise ValueError('remaining_work_seconds cannot be negative')
            record.reported_active_slots = active_slots
            record.reported_remaining_work_seconds = remaining_work_seconds
            record.telemetry_updated_at = timestamp
            self._persist()

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
            self._persist()
            return self.funding

    def tick(self, now: float | None = None, *, execute: bool = False) -> ControlTick:
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
            eligible_hardware = [
                record
                for record in self.gpus.values()
                if (
                    record.state in {GPUState.DRAINING, GPUState.LOADING, GPUState.RUNTIME_VERIFY}
                    and record.registration.release_digest in self.releases
                )
                or (
                    record.lease is not None
                    and record.lease.is_live(timestamp)
                    and record.state != GPUState.QUARANTINED
                )
            ]
            self.placement = self.gepetto.plan(
                eligible_hardware,
                self.releases.values(),
                release_demand,
                timestamp,
            )
            ready = [record for record in self.gpus.values() if record.is_ready(timestamp)]
            if execute:
                self._execute_placement(timestamp)
            self._accepted.clear()
            self._rejected.clear()
            self._service_seconds_total.clear()
            self._last_tick_at = timestamp
            self._persist()
            return ControlTick(
                autoscaling=autoscaling,
                funding=self.funding,
                placement=self.placement,
                ready_gpus=len(ready),
                registered_gpus=len(self.gpus),
            )

    def _execute_placement(self, now: float) -> None:
        if self.assignment_executor is None:
            return
        for transition in self.placement.transitions:
            record = self.gpus.get(transition.gpu_id)
            release = self.releases.get(transition.to_release)
            if record is None or release is None:
                continue
            if record.state not in {GPUState.REGISTERED, GPUState.READY}:
                continue
            epoch = record.assignment_epoch + 1
            command = AssignmentCommand(
                gpu_id=record.registration.gpu_id,
                miner_hotkey=record.registration.miner_hotkey,
                epoch=epoch,
                release_digest=release.release_digest,
                model_id=release.model_id,
                model_repository=release.model_repository,
                model_revision=release.model_revision,
                runtime_digest=release.runtime_digest,
                runtime_commit=release.runtime_commit,
                container_image=release.container_image,
                container_digest=release.container_digest,
                filesystem_digest=release.filesystem_digest,
            )
            try:
                self.assignment_executor.dispatch(record, command)
            except Exception as exc:
                record.revocation_reason = f'assignment dispatch failed: {exc}'
                continue
            self.begin_assignment(record.registration.gpu_id, release.release_digest, now=now)

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
            hotkey_rewards: dict[str, Decimal] = {}
            for gpu_id, amount in result.gpu_rewards.items():
                record = self.gpus[gpu_id]
                hotkey = record.registration.miner_hotkey or f'uid:{record.registration.miner_uid}'
                hotkey_rewards[hotkey] = hotkey_rewards.get(hotkey, Decimal(0)) + amount
            window_id = f'{self._settlement_started_at:.6f}:{timestamp:.6f}'
            if self.store:
                self.store.record_settlement(
                    window_id,
                    self._settlement_started_at,
                    timestamp,
                    {hotkey: str(amount) for hotkey, amount in hotkey_rewards.items()},
                    {
                        'window_budget': str(result.window_budget),
                        'total_ready_seconds': result.total_ready_seconds,
                        'effective_ready_gpus': result.effective_ready_gpus,
                    },
                )
            self._ready_seconds = {}
            self._funded_target_seconds = 0.0
            self._settlement_started_at = timestamp
            self._persist()
            return result, miner_rewards

    def status(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_leases(timestamp)
            active = self.router.active_counts(timestamp)
            status = {
                'desired_target': self.autoscaler.desired_target,
                'funded_target': self.funding.funded_target,
                'funding_shortfall': self.funding.funding_shortfall,
                'target_price_per_gpu_hour': str(self.funding.target_price_per_gpu_hour),
                'max_budget_per_hour': str(self.funding.max_budget_per_hour),
                'verification_source': {
                    'repository': self.config.verification.source_repository,
                    'commit': self.config.verification.source_commit,
                    'protocol': self.config.verification.expected_verifier_protocol,
                    'measurement': self.config.verification.expected_verifier_measurement,
                },
                'registered_gpus': len(self.gpus),
                'ready_gpus': sum(record.is_ready(timestamp) for record in self.gpus.values()),
                'placements': dict(self.placement.assignments),
                'gpus': {
                    gpu_id: {
                        'miner_uid': record.registration.miner_uid,
                        'miner_hotkey': record.registration.miner_hotkey,
                        'state': record.state.value,
                        'assignment_epoch': record.assignment_epoch,
                        'release_digest': record.registration.release_digest,
                        'active_reservations': active.get(gpu_id, 0),
                        'lease_expires_at': record.lease.expires_at if record.lease else None,
                        'weight_verified_at': record.weight_verified_at or None,
                        'telemetry_updated_at': record.telemetry_updated_at or None,
                        'gateway_telemetry_updated_at': record.gateway_telemetry_updated_at or None,
                        'revocation_reason': record.revocation_reason,
                    }
                    for gpu_id, record in sorted(self.gpus.items())
                },
            }
            self._persist()
            return status

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

    def _persist(self) -> None:
        if self.store is not None:
            self.store.save_state(self._export_state())

    def _export_state(self) -> dict[str, Any]:
        now = self.clock()
        return {
            'version': 1,
            'releases': [asdict(value) for value in self.releases.values()],
            'gpus': [
                {
                    'registration': asdict(record.registration),
                    'state': record.state.value,
                    'lease': asdict(record.lease) if record.lease else None,
                    'revocation_reason': record.revocation_reason,
                    'assignment_started_at': record.assignment_started_at,
                    'assignment_epoch': record.assignment_epoch,
                    'reported_remaining_work_seconds': record.reported_remaining_work_seconds,
                    'reported_active_slots': record.reported_active_slots,
                    'telemetry_updated_at': record.telemetry_updated_at,
                    'measured_rtt_by_region_ms': record.measured_rtt_by_region_ms,
                    'service_seconds_ewma': record.service_seconds_ewma,
                    'gateway_active_slots': record.gateway_active_slots,
                    'gateway_remaining_work_seconds': record.gateway_remaining_work_seconds,
                    'gateway_telemetry_updated_at': record.gateway_telemetry_updated_at,
                    'weight_verified_at': record.weight_verified_at,
                    'runtime_verified_at': record.runtime_verified_at,
                }
                for record in self.gpus.values()
            ],
            'autoscaler': {
                'desired_target': self.autoscaler.desired_target,
                'rejection_demand_ewma': self.autoscaler.rejection_demand_ewma,
                'high_since': self.autoscaler.high_since,
                'low_since': self.autoscaler.low_since,
                'last_scaled_at': self.autoscaler.last_scaled_at,
            },
            'max_budget_per_hour': self.max_budget_per_hour,
            'placement': {
                'assignments': dict(self.placement.assignments),
                'replica_counts': dict(self.placement.replica_counts),
            },
            'last_tick_at': self._last_tick_at,
            'last_ready_account_at': self._last_ready_account_at,
            'settlement_started_at': self._settlement_started_at,
            'funded_target_seconds': self._funded_target_seconds,
            'ready_seconds': self._ready_seconds,
            'accepted': self._accepted,
            'rejected': self._rejected,
            'service_seconds_total': self._service_seconds_total,
            'reservations': self.router.export_state(now),
        }

    def _restore_state(self, state: Mapping[str, Any], now: float) -> None:
        if int(state.get('version', 0)) != 1:
            raise ValueError('unsupported compute state version')
        self.releases = {value['release_digest']: Release(**value) for value in state.get('releases', [])}
        self.gpus = {}
        for value in state.get('gpus', []):
            registration = GPURegistration(**value['registration'])
            lease = VerificationLease(**value['lease']) if value.get('lease') else None
            record = GPURecord(
                registration=registration,
                state=GPUState(value['state']),
                lease=lease,
                revocation_reason=value.get('revocation_reason'),
                assignment_started_at=float(value.get('assignment_started_at', 0)),
                assignment_epoch=int(value.get('assignment_epoch', 0)),
                reported_remaining_work_seconds=float(value.get('reported_remaining_work_seconds', 0)),
                reported_active_slots=int(value.get('reported_active_slots', 0)),
                telemetry_updated_at=float(value.get('telemetry_updated_at', 0)),
                measured_rtt_by_region_ms={
                    key: float(item) for key, item in value.get('measured_rtt_by_region_ms', {}).items()
                },
                service_seconds_ewma=float(value.get('service_seconds_ewma', 0)),
                gateway_active_slots=int(value.get('gateway_active_slots', 0)),
                gateway_remaining_work_seconds=float(value.get('gateway_remaining_work_seconds', 0)),
                gateway_telemetry_updated_at=float(value.get('gateway_telemetry_updated_at', 0)),
                weight_verified_at=float(value.get('weight_verified_at', 0)),
                runtime_verified_at=float(value.get('runtime_verified_at', 0)),
            )
            self.gpus[registration.gpu_id] = record
        autoscaler = state.get('autoscaler', {})
        self.autoscaler.desired_target = int(autoscaler.get('desired_target', self.autoscaler.desired_target))
        self.autoscaler.rejection_demand_ewma = float(autoscaler.get('rejection_demand_ewma', 0))
        self.autoscaler.high_since = autoscaler.get('high_since')
        self.autoscaler.low_since = autoscaler.get('low_since')
        self.autoscaler.last_scaled_at = autoscaler.get('last_scaled_at')
        self.max_budget_per_hour = float(state.get('max_budget_per_hour', self.max_budget_per_hour))
        self.funding = self._funding_plan()
        placement = state.get('placement', {})
        self.placement = PlacementPlan(
            placement.get('assignments', {}),
            placement.get('replica_counts', {}),
            (),
        )
        self._last_tick_at = float(state.get('last_tick_at', now))
        self._last_ready_account_at = float(state.get('last_ready_account_at', now))
        self._settlement_started_at = float(state.get('settlement_started_at', now))
        self._funded_target_seconds = float(state.get('funded_target_seconds', 0))
        self._ready_seconds = {key: float(value) for key, value in state.get('ready_seconds', {}).items()}
        self._accepted = {key: int(value) for key, value in state.get('accepted', {}).items()}
        self._rejected = {key: int(value) for key, value in state.get('rejected', {}).items()}
        self._service_seconds_total = {
            key: float(value) for key, value in state.get('service_seconds_total', {}).items()
        }
        self.router.restore_state(state.get('reservations', []), now)
        self._expire_leases(now)
