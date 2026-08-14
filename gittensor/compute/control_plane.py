"""Composition root for verification, scaling, placement, routing, and rewards."""

from __future__ import annotations

import math
import random
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import Any, Iterator, Mapping, Sequence

from gittensor.compute.artifacts import ReleaseArtifactVerifier
from gittensor.compute.assignment import AssignmentExecutor
from gittensor.compute.autoscaler import AutoscaleDecision, FleetAutoscaler
from gittensor.compute.config import ComputeConfig
from gittensor.compute.emission_oracle import EmissionObservation
from gittensor.compute.inference_tokens import issue_inference_capability
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
from gittensor.compute.settlement_auth import SettlementSigner
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
        settlement_signer: SettlementSigner | None = None,
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
            initial_target=config.fleet.initial_target,
        )
        self.router = FastestFinishRouter(
            config.router.reservation_ttl_seconds,
            config.router.equivalent_finish_epsilon_seconds,
        )
        self.gepetto = GlobalGepetto(
            config.placement.minimum_residency_seconds,
            switch_sustain_seconds=config.placement.switch_sustain_seconds,
            planning_horizon_seconds=config.placement.planning_horizon_seconds,
            minimum_switch_gain_gpu=config.placement.minimum_switch_gain_gpu,
            target_utilization=config.autoscaling.utilization_up,
        )
        self.store = store
        self.router.attach_store(store)
        self.assignment_executor = assignment_executor
        self.weight_verifier = weight_verifier
        self.weight_transport = weight_transport
        self.spark_node_owners = dict(spark_node_owners or {})
        self.release_artifact_verifier = release_artifact_verifier
        self.settlement_signer = settlement_signer
        self.max_budget_per_hour = config.fleet.max_budget_per_hour
        self.subnet_miner_emission_value_per_hour = config.fleet.subnet_miner_emission_value_per_hour
        self.emission_oracle_observation: EmissionObservation | None = None
        self.emission_oracle_error: str | None = None
        self.funding = self._funding_plan()
        self.last_autoscale: AutoscaleDecision | None = None
        self.placement = PlacementPlan({}, {}, ())
        now = self.clock()
        self._last_tick_at = now
        self._last_ready_account_at = now
        self._settlement_started_at = now
        self._funded_target_seconds = 0.0
        self._funded_budget_seconds = 0.0
        self._compute_emission_share_seconds = 0.0
        self._ready_seconds: dict[str, float] = {}
        self._accepted: dict[str, int] = {}
        self._rejected: dict[str, int] = {}
        self._rejected_capacity_seconds: dict[str, float] = {}
        self._completed_capacity_seconds: dict[str, float] = {}
        self._pending_reservation_deletions: set[str] = set()
        self._pending_expired_reservation_deletions: set[str] = set()
        self._lock = threading.RLock()
        if self.store and (saved := self.store.load_state()):
            self._restore_state(saved, now)

    def register_release(self, release: Release) -> None:
        with self._lock:
            if release.max_concurrency > self.config.fleet.certified_slots_per_gpu:
                raise ValueError('release max_concurrency exceeds the operator-certified hardware ceiling')
            existing = self.releases.get(release.release_digest)
            if existing is not None and existing != release:
                raise ValueError('an approved release digest is immutable')
            if existing is None and self.release_artifact_verifier is not None:
                self.release_artifact_verifier.verify(release)
            self.releases[release.release_digest] = release
            self._persist()

    def catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    'release_digest': release.release_digest,
                    'model_id': release.model_id,
                    'model_revision': release.model_revision,
                    'token_proof_scheme': release.token_proof_scheme,
                    'max_concurrency': release.max_concurrency,
                    'max_context_tokens': release.max_context_tokens,
                    'kv_bytes_per_token': release.kv_bytes_per_token,
                    'kv_cache_capacity_bytes': release.kv_cache_capacity_bytes,
                    'request_overhead_tokens': release.request_overhead_tokens,
                }
                for release in sorted(self.releases.values(), key=lambda item: (item.model_id, item.release_digest))
            ]

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
            if existing and existing.administratively_disabled:
                raise ValueError('GPU is administratively disabled')
            if existing and existing.registration.miner_uid != registration.miner_uid:
                raise ValueError('a GPU cannot move between miner UIDs without deregistration')
            if existing and existing.registration.miner_hotkey != registration.miner_hotkey:
                raise ValueError('a GPU cannot move between hotkeys without deregistration')
            if existing and (
                existing.registration.release_digest != registration.release_digest
                or existing.registration.canary_release_digest != registration.canary_release_digest
            ):
                raise ValueError('registration cannot change a Gepetto assignment or runtime binding')
            existing_node = next(
                (
                    record
                    for gpu_id, record in self.gpus.items()
                    if gpu_id != registration.gpu_id and record.registration.spark_node_id == registration.spark_node_id
                ),
                None,
            )
            if existing_node is not None:
                raise ValueError('a SparkCompute node can register only one GPU identity')
            if existing is not None:
                if existing.registration != registration:
                    raise ValueError('GPU registration is immutable after admission')
                return
            self.gpus[registration.gpu_id] = GPURecord(
                registration=registration,
                state=GPUState.REGISTERED,
                assignment_started_at=timestamp,
                assignment_token=secrets.token_hex(32),
            )
            self._persist()

    def disable_gpu(self, gpu_id: str, reason: str, now: float | None = None) -> None:
        """Immediately remove one GPU from routing until an operator re-enables it."""
        reason = reason.strip()
        if not reason or len(reason) > 500:
            raise ValueError('disable reason must contain 1 to 500 characters')
        timestamp = self.clock() if now is None else now
        revocation: tuple[GPURecord, int, str] | None = None
        with self._lock:
            record = self.gpus[gpu_id]
            self._account_ready_seconds(timestamp)
            self._complete_gpu_reservations(gpu_id, timestamp)
            record.administratively_disabled = True
            record.state = GPUState.QUARANTINED
            record.lease = None
            record.assignment_token = secrets.token_hex(32)
            record.revocation_reason = f'administratively disabled: {reason}'
            record.revocation_pending = record.assignment_epoch > 0
            self._remove_from_placement(gpu_id)
            self._persist()
            if record.assignment_epoch > 0:
                revocation = (
                    GPURecord(registration=record.registration),
                    record.assignment_epoch,
                    record.revocation_reason,
                )
        if revocation is not None:
            self._dispatch_assignment_revocation(*revocation)

    def enable_gpu(self, gpu_id: str, now: float | None = None) -> None:
        """Re-admit hardware only through a new assignment and verification chain."""
        timestamp = self.clock() if now is None else now
        with self._lock:
            record = self.gpus[gpu_id]
            if not record.administratively_disabled:
                return
            if record.revocation_pending or record.assignment_dispatch_in_flight:
                raise ValueError('assignment revocation has not been acknowledged by the miner agent')
            record.administratively_disabled = False
            record.registration = replace(record.registration, release_digest='', canary_release_digest='')
            record.state = GPUState.REGISTERED
            record.lease = None
            record.assignment_started_at = timestamp
            record.assignment_dispatched = False
            record.assignment_token = secrets.token_hex(32)
            record.runtime_stream_public_key = ''
            record.runtime_verified_at = 0.0
            record.weight_verified_at = 0.0
            record.consecutive_inference_failures = 0
            record.inference_quarantined_at = 0.0
            record.revocation_reason = 're-enabled; fresh assignment and verification required'
            self._remove_from_placement(gpu_id)
            self._persist()

    def revoke_release(self, release_digest: str, reason: str, now: float | None = None) -> None:
        """Stop a compromised release and force every affected GPU through Gepetto again."""
        reason = reason.strip()
        if not reason or len(reason) > 500:
            raise ValueError('revocation reason must contain 1 to 500 characters')
        timestamp = self.clock() if now is None else now
        revocations: list[tuple[GPURecord, int, str]] = []
        with self._lock:
            if release_digest not in self.releases:
                raise KeyError(release_digest)
            self._account_ready_seconds(timestamp)
            del self.releases[release_digest]
            for gpu_id, record in self.gpus.items():
                if record.registration.release_digest != release_digest:
                    continue
                self._complete_gpu_reservations(gpu_id, timestamp)
                record.registration = replace(record.registration, release_digest='', canary_release_digest='')
                record.state = GPUState.QUARANTINED
                record.administratively_disabled = True
                record.lease = None
                record.assignment_started_at = timestamp
                record.assignment_token = secrets.token_hex(32)
                record.runtime_stream_public_key = ''
                record.runtime_verified_at = 0.0
                record.weight_verified_at = 0.0
                record.revocation_reason = f'release revoked: {reason}'
                record.revocation_pending = record.assignment_epoch > 0
                if record.assignment_epoch > 0:
                    revocations.append(
                        (GPURecord(registration=record.registration), record.assignment_epoch, record.revocation_reason)
                    )
            self.placement = PlacementPlan(
                {
                    gpu_id: assigned
                    for gpu_id, assigned in self.placement.assignments.items()
                    if assigned != release_digest
                },
                {digest: count for digest, count in self.placement.replica_counts.items() if digest != release_digest},
                (),
                {digest: value for digest, value in self.placement.shortages.items() if digest != release_digest},
                {digest: value for digest, value in self.placement.deferred.items() if digest != release_digest},
            )
            self._persist()
        errors = []
        for revocation in revocations:
            try:
                self._dispatch_assignment_revocation(*revocation)
            except Exception as exc:
                errors.append(f'{revocation[0].registration.gpu_id}: {exc}')
        if errors:
            raise RuntimeError(f'release was revoked, but miner revocation delivery failed: {"; ".join(errors)}')

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
        revocation: tuple[GPURecord, int, str] | None = None
        with self._lock:
            record = self.gpus[gpu_id]
            if record.administratively_disabled:
                raise ValueError('disabled GPU cannot acknowledge assignments')
            if epoch != record.assignment_epoch:
                raise ValueError('assignment acknowledgement has a stale epoch')
            if not record.assignment_dispatched:
                raise ValueError('assignment has not been accepted by the miner agent')
            expected_evidence = None
            if state == GPUState.RUNTIME_VERIFY:
                if evidence is None:
                    raise ValueError('runtime evidence is required before RUNTIME_VERIFY')
                release = self.releases[record.registration.release_digest]
                expected_evidence = RuntimeEvidence(
                    release_digest=release.release_digest,
                    model_repository=release.model_repository,
                    model_revision=release.model_revision,
                    tokenizer_repository=release.tokenizer_repository,
                    tokenizer_revision=release.tokenizer_revision,
                    runtime_digest=release.runtime_digest,
                    runtime_commit=release.runtime_commit,
                    container_image=release.container_image,
                    container_digest=release.container_digest,
                    filesystem_digest=release.filesystem_digest,
                    stream_public_key=evidence.stream_public_key,
                )
                if evidence != expected_evidence or not evidence.stream_public_key:
                    raise ValueError('runtime evidence does not match the approved release')
                if record.state in {GPUState.RUNTIME_VERIFY, GPUState.READY}:
                    if evidence.stream_public_key == record.runtime_stream_public_key:
                        return record.state
                    self._account_ready_seconds(timestamp)
                    reason = 'runtime signing key rotated; new assignment is required'
                    assigned_epoch = record.assignment_epoch
                    self._quarantine_gpu(gpu_id, reason, timestamp)
                    self._persist()
                    revocation = (GPURecord(registration=record.registration), assigned_epoch, reason)
            if revocation is None:
                if record.state == state:
                    return record.state
                if state == GPUState.LOADING and record.state in {GPUState.RUNTIME_VERIFY, GPUState.READY}:
                    return record.state
                allowed = {
                    GPUState.DRAINING: GPUState.LOADING,
                    GPUState.LOADING: GPUState.RUNTIME_VERIFY,
                }
                if allowed.get(record.state) != state:
                    raise ValueError(f'invalid assignment transition: {record.state.value} -> {state.value}')
                if state == GPUState.RUNTIME_VERIFY:
                    assert evidence is not None
                    record.registration = replace(
                        record.registration,
                        canary_release_digest=record.registration.release_digest,
                    )
                    record.runtime_verified_at = timestamp
                    record.runtime_stream_public_key = evidence.stream_public_key
                record.state = state
                record.revocation_reason = None
                self._persist()
                return record.state
        assert revocation is not None
        self._dispatch_assignment_revocation(*revocation)
        return GPUState.QUARANTINED

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
                if record.administratively_disabled:
                    evaluated[gpu_id] = None, record.revocation_reason, False, True
                    continue
                release = self.releases.get(record.registration.release_digest)
                snapshot = indexed.get(record.registration.spark_node_id)
                if release is None and record.registration.release_digest:
                    evaluated[gpu_id] = None, 'assigned release is no longer approved', True, True
                    continue
                fully_bound = (
                    release is not None
                    and record.registration.canary_release_digest == record.registration.release_digest
                )
                if fully_bound:
                    assert release is not None
                    outcome = self.verifier.evaluate(record.registration, release, snapshot, timestamp)
                    hard_failure = False
                    if (
                        outcome.accepted
                        and outcome.lease is not None
                        and self.config.verification.require_stream_proof
                        and outcome.lease.stream_public_key != record.runtime_stream_public_key
                    ):
                        outcome = type(outcome)(None, 'attested stream key does not match the active runtime')
                        hard_failure = True
                    if (
                        outcome.accepted
                        and self.config.verification.require_weight_challenges
                        and timestamp - record.weight_verified_at
                        >= self.config.verification.weight_verification_ttl_seconds
                    ):
                        outcome = type(outcome)(None, 'model weight challenge is missing or stale')
                    if (
                        outcome.accepted
                        and record.inference_quarantined_at > 0
                        and outcome.lease is not None
                        and outcome.lease.verified_at <= record.inference_quarantined_at
                    ):
                        outcome = type(outcome)(None, 'new verification is required after inference quarantine')
                    evaluated[gpu_id] = outcome.lease, outcome.reason, True, hard_failure
                else:
                    outcome = self.verifier.evaluate_hardware(record.registration, snapshot, timestamp)
                    evaluated[gpu_id] = outcome.lease, outcome.reason, False, False

            uuid_to_gpus: dict[str, list[str]] = {}
            for gpu_id, (lease, _, _, _) in evaluated.items():
                if lease is not None:
                    uuid_to_gpus.setdefault(lease.hardware_uuid, []).append(gpu_id)
            duplicate_gpus = {gpu_id for gpu_ids in uuid_to_gpus.values() if len(gpu_ids) > 1 for gpu_id in gpu_ids}

            for gpu_id, record in self.gpus.items():
                lease, reason, full_verification, hard_failure = evaluated[gpu_id]
                if record.administratively_disabled:
                    record.lease = None
                    outcomes[gpu_id] = record.revocation_reason or GPUState.QUARANTINED.value
                    continue
                if gpu_id in duplicate_gpus:
                    lease = None
                    reason = 'duplicate SparkCompute GPU UUID is registered more than once'
                    hard_failure = True
                if lease is not None:
                    record.lease = lease
                    if full_verification:
                        record.revocation_reason = None
                        record.consecutive_inference_failures = 0
                        record.inference_quarantined_at = 0.0
                        record.state = GPUState.READY
                        outcomes[gpu_id] = GPUState.READY.value
                    else:
                        if record.state == GPUState.QUARANTINED and not record.revocation_pending:
                            record.state = GPUState.REGISTERED
                            record.revocation_reason = None
                        outcomes[gpu_id] = record.state.value
                else:
                    if hard_failure or record.state not in {
                        GPUState.DRAINING,
                        GPUState.LOADING,
                        GPUState.RUNTIME_VERIFY,
                    }:
                        self._quarantine_gpu(
                            gpu_id,
                            reason or 'verification failed',
                            timestamp,
                        )
                    else:
                        record.lease = None
                        record.revocation_reason = reason
                    outcomes[gpu_id] = reason or GPUState.QUARANTINED.value
            self._persist()
        self.retry_pending_revocations()
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
                if not record.administratively_disabled
                and record.state in {GPUState.RUNTIME_VERIFY, GPUState.READY, GPUState.QUARANTINED}
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
                challenge = verifier.issue(gpu_id, release, self.clock())
                digest = transport.answer(record, challenge)
                accepted = verifier.verify(challenge.challenge_id, gpu_id, digest, self.clock())
            except Exception:
                if challenge is not None:
                    verifier.cancel(challenge.challenge_id)
                accepted = False
            return gpu_id, release_digest, assignment_epoch, accepted

        futures = []
        with ThreadPoolExecutor(max_workers=min(32, max(1, count)), thread_name_prefix='weight-challenge') as executor:
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
                    self._quarantine_gpu(
                        gpu_id,
                        'model weight challenge failed',
                        timestamp,
                    )
                outcomes[gpu_id] = accepted
                self._persist()
        self.retry_pending_revocations()
        return outcomes

    def route(
        self,
        release_digest: str,
        requester_region: str,
        expected_service_seconds: float,
        estimated_input_tokens: int = 1,
        max_output_tokens: int = 1,
        now: float | None = None,
    ) -> RouteDecision:
        timestamp = self.clock() if now is None else now
        if (
            not math.isfinite(expected_service_seconds)
            or expected_service_seconds <= 0
            or expected_service_seconds > self.config.router.maximum_service_seconds
        ):
            raise ValueError('expected_service_seconds must be positive and within the configured maximum')
        for field_name, value in {
            'estimated_input_tokens': estimated_input_tokens,
            'max_output_tokens': max_output_tokens,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{field_name} must be a positive integer')
        with self._lock:
            release = self.releases.get(release_digest)
            if release is None:
                raise CapacityUnavailable('requested release is not approved')
            context_tokens = estimated_input_tokens + max_output_tokens
            if context_tokens > release.max_context_tokens:
                raise ValueError('request exceeds the approved release context limit')
            request_kv_bytes = context_tokens * release.kv_bytes_per_token
            if request_kv_bytes > release.kv_cache_capacity_bytes:
                raise ValueError('request exceeds the approved release KV-cache capacity')
            request_capacity_units = max(
                1.0 / release.max_concurrency,
                request_kv_bytes / release.kv_cache_capacity_bytes,
            )
            self._account_ready_seconds(timestamp)
            self._expire_leases(timestamp)
            candidates = [
                RoutingGPU(
                    gpu_id=record.registration.gpu_id,
                    endpoint=record.registration.endpoint,
                    release_digest=record.registration.release_digest,
                    performance_class=record.registration.performance_class,
                    certified_slots=release.max_concurrency,
                    observed_active_slots=(
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
                        expected_service_seconds
                        * record.request_estimate_ratio_ewma_by_release.get(release_digest, 1.0),
                    ),
                    rtt_ms=float(
                        record.measured_rtt_by_region_ms.get(
                            requester_region,
                            self.config.router.default_rtt_ms,
                        )
                    ),
                    stream_public_key=record.lease.stream_public_key if record.lease else '',
                    eligible_until=record.lease.expires_at if record.lease else timestamp,
                    kv_cache_capacity_bytes=release.kv_cache_capacity_bytes,
                )
                for record in self.gpus.values()
                if record.is_ready(timestamp) and record.registration.release_digest == release_digest
            ]
            try:
                decision = self.router.route(
                    candidates,
                    release_digest,
                    timestamp,
                    request_kv_bytes=request_kv_bytes,
                    request_capacity_units=request_capacity_units,
                )
            except CapacityUnavailable:
                self._record_demand(
                    release_digest,
                    expected_service_seconds,
                    request_capacity_units,
                    rejected=True,
                )
                raise
            self._record_demand(
                release_digest,
                expected_service_seconds,
                request_capacity_units,
                rejected=False,
            )
            selected = self.gpus[decision.gpu_id]
            assert selected.lease is not None
            capability = issue_inference_capability(
                selected.assignment_token,
                reservation_id=decision.reservation_id,
                gpu_id=decision.gpu_id,
                release_digest=release_digest,
                expires_at=min(decision.expires_at, selected.lease.expires_at),
                reserved_kv_bytes=decision.reserved_kv_bytes,
            )
            return replace(decision, inference_token=capability)

    def complete_reservation(self, reservation_id: str, now: float | None = None) -> bool:
        timestamp = self.clock() if now is None else now
        with self._lock:
            reservation = self.router.detach(reservation_id)
            previous_completed_seconds: float | None = None
            if reservation is not None:
                occupied_seconds = max(
                    0.0,
                    min(timestamp, reservation.expires_at) - max(reservation.created_at, self._last_tick_at),
                )
                previous_completed_seconds = self._completed_capacity_seconds.get(reservation.release_digest)
                self._completed_capacity_seconds[reservation.release_digest] = (
                    previous_completed_seconds or 0.0
                ) + occupied_seconds * reservation.capacity_units
            if self.store is not None:
                pending = set(self._pending_reservation_deletions)
                expired = set(self._pending_expired_reservation_deletions)
                try:
                    completed = self.store.save_state_and_complete_reservation(
                        self._export_state(),
                        reservation_id,
                        pending,
                        expired,
                    )
                except Exception:
                    if reservation is not None:
                        self.router.reattach(reservation)
                        if previous_completed_seconds is None:
                            self._completed_capacity_seconds.pop(reservation.release_digest, None)
                        else:
                            self._completed_capacity_seconds[reservation.release_digest] = previous_completed_seconds
                    raise
                self._pending_reservation_deletions.difference_update(pending)
                self._pending_expired_reservation_deletions.difference_update(expired)
                return completed or reservation is not None
            return reservation is not None

    def _complete_gpu_reservations(self, gpu_id: str, now: float) -> None:
        self._pending_reservation_deletions.add(gpu_id)
        for reservation in self.router.detach_gpu(gpu_id):
            occupied_seconds = max(
                0.0,
                min(now, reservation.expires_at) - max(reservation.created_at, self._last_tick_at),
            )
            self._completed_capacity_seconds[reservation.release_digest] = (
                self._completed_capacity_seconds.get(reservation.release_digest, 0.0)
                + occupied_seconds * reservation.capacity_units
            )

    def _remove_from_placement(self, gpu_id: str) -> None:
        assignments = dict(self.placement.assignments)
        release_digest = assignments.pop(gpu_id, None)
        replica_counts = dict(self.placement.replica_counts)
        if release_digest is not None:
            replica_counts[release_digest] = max(0, replica_counts.get(release_digest, 1) - 1)
        self.placement = PlacementPlan(
            assignments,
            replica_counts,
            (),
            dict(self.placement.shortages),
            dict(self.placement.deferred),
        )

    def _dispatch_assignment_revocation(self, record: GPURecord, epoch: int, reason: str) -> bool:
        if self.assignment_executor is None:
            return False
        revoke = getattr(self.assignment_executor, 'revoke', None)
        if not callable(revoke):
            return False
        revoke(record, epoch, reason)
        with self._lock:
            current = self.gpus.get(record.registration.gpu_id)
            if current is not None and current.assignment_epoch == epoch:
                current.assignment_dispatched = False
                current.revocation_pending = False
                self._persist()
        return True

    def retry_pending_revocations(self) -> dict[str, str]:
        """Retry durable assignment tombstones after network failure or restart."""
        with self._lock:
            pending = [
                (
                    GPURecord(registration=record.registration),
                    record.assignment_epoch,
                    record.revocation_reason or 'assignment was revoked',
                )
                for record in self.gpus.values()
                if record.revocation_pending and record.assignment_epoch > 0
            ]
        outcomes: dict[str, str] = {}
        for record, epoch, reason in pending:
            gpu_id = record.registration.gpu_id
            try:
                delivered = self._dispatch_assignment_revocation(record, epoch, reason)
            except Exception as exc:
                outcomes[gpu_id] = str(exc)
            else:
                if not delivered:
                    outcomes[gpu_id] = 'revocation transport is unavailable'
                    continue
                with self._lock:
                    current = self.gpus.get(gpu_id)
                    if current is not None and current.assignment_epoch == epoch:
                        current.assignment_dispatch_in_flight = False
                        self._persist()
                outcomes[gpu_id] = 'revoked'
        return outcomes

    def renew_reservation(self, reservation_id: str, now: float | None = None) -> float | None:
        """Extend a reservation only while its exact assignment remains eligible."""
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._expire_leases(timestamp)
            reservation = self.router.reservation(reservation_id)
            if reservation is None:
                return None
            record = self.gpus.get(reservation.gpu_id)
            if (
                record is None
                or not record.is_ready(timestamp)
                or record.registration.release_digest != reservation.release_digest
            ):
                self.complete_reservation(reservation_id, timestamp)
                return None
            assert record.lease is not None
            return self.router.renew(
                reservation_id,
                timestamp,
                eligible_until=record.lease.expires_at,
            )

    def record_routing_observation(self, observation: RoutingObservation, now: float | None = None) -> None:
        """Update gateway-measured RTT and service performance, never miner claims."""
        timestamp = self.clock() if now is None else now
        numeric_observations = (
            observation.measured_rtt_ms,
            observation.service_seconds,
            observation.remaining_work_seconds,
            observation.expected_service_seconds,
        )
        if not all(math.isfinite(value) for value in numeric_observations):
            raise ValueError('routing observations must be finite')
        if observation.measured_rtt_ms < 0 or observation.service_seconds < 0:
            raise ValueError('routing observations cannot be negative')
        if observation.remaining_work_seconds < 0:
            raise ValueError('remaining_work_seconds cannot be negative')
        if observation.expected_service_seconds < 0:
            raise ValueError('expected_service_seconds cannot be negative')
        if not isinstance(observation.success, bool):
            raise ValueError('routing observation success must be a boolean')
        with self._lock:
            reservation = self.router.reservation(observation.reservation_id)
            if reservation is None:
                raise ValueError('routing observation reservation is missing, expired, or already consumed')
            if reservation.gpu_id != observation.gpu_id:
                raise ValueError('routing observation GPU does not match its reservation')
            record = self.gpus[observation.gpu_id]
            release_digest = observation.release_digest or record.registration.release_digest
            if release_digest != record.registration.release_digest:
                raise ValueError('routing observation release does not match the GPU assignment')
            if reservation.release_digest != release_digest:
                raise ValueError('routing observation release does not match its reservation')
            if not isinstance(observation.observed_active_slots, int) or isinstance(
                observation.observed_active_slots, bool
            ):
                raise ValueError('observed_active_slots must be an integer')
            release = self.releases[release_digest]
            if not 0 <= observation.observed_active_slots <= release.max_concurrency:
                raise ValueError('observed_active_slots exceeds the certified concurrency')
            alpha = 0.2
            if observation.measured_rtt_ms > 0:
                previous_rtt = record.measured_rtt_by_region_ms.get(observation.requester_region)
                record.measured_rtt_by_region_ms[observation.requester_region] = (
                    observation.measured_rtt_ms
                    if previous_rtt is None
                    else alpha * observation.measured_rtt_ms + (1 - alpha) * previous_rtt
                )
            if observation.success:
                record.consecutive_inference_failures = 0
                previous_service = record.service_seconds_ewma_by_release.get(release_digest, 0.0)
                record.service_seconds_ewma_by_release[release_digest] = (
                    observation.service_seconds
                    if previous_service <= 0
                    else alpha * observation.service_seconds + (1 - alpha) * previous_service
                )
                if observation.expected_service_seconds > 0:
                    ratio = observation.service_seconds / observation.expected_service_seconds
                    previous_ratio = record.request_estimate_ratio_ewma_by_release.get(release_digest, 1.0)
                    record.request_estimate_ratio_ewma_by_release[release_digest] = max(
                        0.05,
                        min(20.0, alpha * ratio + (1 - alpha) * previous_ratio),
                    )
            else:
                record.consecutive_inference_failures += 1
                if record.consecutive_inference_failures >= self.config.router.failure_quarantine_threshold:
                    self._account_ready_seconds(timestamp)
                    record.inference_quarantined_at = timestamp
                    self._quarantine_gpu(
                        observation.gpu_id,
                        'repeated real-inference failures require fresh verification',
                        timestamp,
                    )
            if record.state != GPUState.QUARANTINED:
                record.gateway_active_slots = observation.observed_active_slots
                record.gateway_remaining_work_seconds = observation.remaining_work_seconds
                record.gateway_telemetry_updated_at = timestamp
            if record.state != GPUState.QUARANTINED:
                self.complete_reservation(observation.reservation_id, timestamp)
            self._persist()
        self.retry_pending_revocations()

    def update_budget(
        self,
        max_budget_per_hour: float | None,
        now: float | None = None,
        *,
        subnet_miner_emission_value_per_hour: float | None = None,
    ) -> FundingPlan:
        """Update the price oracle value and optional cap in the target-price unit."""
        minimum_budget = self.config.fleet.floor * self.config.fleet.target_price_per_gpu_hour
        if max_budget_per_hour is not None and not math.isfinite(max_budget_per_hour):
            raise ValueError('max_budget_per_hour must be finite')
        if max_budget_per_hour is not None and max_budget_per_hour < minimum_budget:
            raise ValueError('max_budget_per_hour must continue to fund the configured GPU floor')
        if subnet_miner_emission_value_per_hour is not None and (
            not math.isfinite(subnet_miner_emission_value_per_hour) or subnet_miner_emission_value_per_hour <= 0
        ):
            raise ValueError('subnet_miner_emission_value_per_hour must be finite and positive')
        if self.config.emission_oracle.enabled and subnet_miner_emission_value_per_hour is not None:
            raise ValueError('manual emission values are disabled while the automatic oracle is enabled')
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_emission_oracle(timestamp)
            self.max_budget_per_hour = max_budget_per_hour
            if subnet_miner_emission_value_per_hour is not None:
                self.subnet_miner_emission_value_per_hour = subnet_miner_emission_value_per_hour
            self.funding = self._funding_plan(timestamp)
            self._persist()
            return self.funding

    def apply_emission_observation(
        self,
        observation: EmissionObservation,
        now: float | None = None,
    ) -> FundingPlan:
        if observation.currency != self.config.fleet.target_price_currency:
            raise ValueError('emission observation currency does not match target price currency')
        if observation.observed_at > (self.clock() if now is None else now) + 120:
            raise ValueError('emission observation is future-dated')
        timestamp = self.clock() if now is None else now
        if not math.isfinite(observation.value_per_hour) or observation.value_per_hour <= 0:
            raise ValueError('emission observation value must be finite and positive')
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_emission_oracle(timestamp)
            self.subnet_miner_emission_value_per_hour = observation.value_per_hour
            self.emission_oracle_observation = observation
            self.emission_oracle_error = None
            self.funding = self._funding_plan(timestamp)
            self._persist()
            return self.funding

    def record_emission_oracle_failure(self, error: Exception | str) -> None:
        with self._lock:
            self.emission_oracle_error = str(error)
            self._persist()

    def emission_oracle_is_fresh(self, now: float | None = None) -> bool:
        if not self.config.emission_oracle.enabled:
            return True
        timestamp = self.clock() if now is None else now
        observation = self.emission_oracle_observation
        return bool(
            observation is not None
            and 0 <= timestamp - observation.observed_at <= self.config.emission_oracle.max_refresh_staleness_seconds
        )

    def tick(self, now: float | None = None, *, execute: bool = False) -> ControlTick:
        timestamp = self.clock() if now is None else now
        pending: list[tuple[GPURecord, AssignmentCommand]] = []
        with self._lock, self._rollback_tick_on_error():
            self._account_ready_seconds(timestamp)
            self._expire_emission_oracle(timestamp)
            self._expire_leases(timestamp, persist=False)
            elapsed = max(timestamp - self._last_tick_at, 1e-9)
            expired_reservations = self.router.take_expired_reservations(timestamp)
            for reservation in expired_reservations:
                self._pending_expired_reservation_deletions.add(reservation.reservation_id)
                occupied_until = min(timestamp, reservation.expires_at)
                occupied_seconds = max(
                    0.0,
                    occupied_until - max(reservation.created_at, self._last_tick_at),
                )
                self._completed_capacity_seconds[reservation.release_digest] = (
                    self._completed_capacity_seconds.get(reservation.release_digest, 0.0)
                    + occupied_seconds * reservation.capacity_units
                )
            active_capacity = self.router.active_capacity_by_release(timestamp)
            active_reservations = self.router.active_reservations(timestamp)
            live_gpu_equivalents = sum(active_capacity.values())
            active_capacity_seconds: dict[str, float] = {}
            for reservation in active_reservations:
                occupied_seconds = max(0.0, timestamp - max(reservation.created_at, self._last_tick_at))
                active_capacity_seconds[reservation.release_digest] = (
                    active_capacity_seconds.get(reservation.release_digest, 0.0)
                    + occupied_seconds * reservation.capacity_units
                )
            window_gpu_equivalents = (
                sum(self._completed_capacity_seconds.values()) + sum(active_capacity_seconds.values())
            ) / elapsed
            observed_gpu_equivalents = max(live_gpu_equivalents, window_gpu_equivalents)
            rejected_gpu_equivalents = sum(self._rejected_capacity_seconds.values()) / elapsed
            autoscaling = self.autoscaler.update(
                active_gpu_equivalents=observed_gpu_equivalents,
                rejected_gpu_equivalents=rejected_gpu_equivalents,
                funded_target=self.funding.funded_target,
                now=timestamp,
            )
            self.last_autoscale = autoscaling
            self.funding = self._funding_plan(timestamp)
            release_demand = [
                ReleaseDemand(
                    release_digest=release_digest,
                    gpu_equivalent_demand=max(
                        active_capacity.get(release_digest, 0.0),
                        (
                            self._completed_capacity_seconds.get(release_digest, 0.0)
                            + active_capacity_seconds.get(release_digest, 0.0)
                        )
                        / elapsed,
                    )
                    + self._rejected_capacity_seconds.get(release_digest, 0.0) / elapsed,
                )
                for release_digest in self.releases
            ]
            eligible_hardware = [
                record
                for record in self.gpus.values()
                if record.lease is not None
                and record.lease.is_live(timestamp)
                and record.state != GPUState.QUARANTINED
                and not record.administratively_disabled
                and not record.revocation_pending
            ]
            self.placement = self.gepetto.plan(
                eligible_hardware,
                self.releases.values(),
                release_demand,
                timestamp,
            )
            if execute and self.assignment_executor is None and self.placement.transitions:
                raise RuntimeError('placement transitions require an assignment executor')
            if execute:
                pending = self._stage_placement(timestamp)
            else:
                self.placement = PlacementPlan(
                    dict(self.placement.assignments),
                    dict(self.placement.replica_counts),
                    (),
                    dict(self.placement.shortages),
                    dict(self.placement.deferred),
                )
            ready = [record for record in self.gpus.values() if record.is_ready(timestamp)]
            self._accepted.clear()
            self._rejected.clear()
            self._rejected_capacity_seconds.clear()
            self._completed_capacity_seconds.clear()
            self._last_tick_at = timestamp
            self._persist()
            result = ControlTick(
                autoscaling=autoscaling,
                funding=self.funding,
                placement=self.placement,
                ready_gpus=len(ready),
                registered_gpus=len(self.gpus),
            )
        if pending:
            self._dispatch_assignments(pending)
        return result

    @contextmanager
    def _rollback_tick_on_error(self) -> Iterator[None]:
        """Restore volatile control state when the tick checkpoint fails."""
        checkpoint = (
            deepcopy(self.gpus),
            deepcopy(self.autoscaler),
            deepcopy(self.gepetto),
            self.funding,
            self.last_autoscale,
            self.placement,
            self._last_tick_at,
            self._last_ready_account_at,
            self._funded_target_seconds,
            self._funded_budget_seconds,
            self._compute_emission_share_seconds,
            dict(self._ready_seconds),
            dict(self._accepted),
            dict(self._rejected),
            dict(self._rejected_capacity_seconds),
            dict(self._completed_capacity_seconds),
            set(self._pending_reservation_deletions),
            set(self._pending_expired_reservation_deletions),
            self.router.checkpoint(),
        )
        try:
            yield
        except Exception:
            (
                self.gpus,
                self.autoscaler,
                self.gepetto,
                self.funding,
                self.last_autoscale,
                self.placement,
                self._last_tick_at,
                self._last_ready_account_at,
                self._funded_target_seconds,
                self._funded_budget_seconds,
                self._compute_emission_share_seconds,
                self._ready_seconds,
                self._accepted,
                self._rejected,
                self._rejected_capacity_seconds,
                self._completed_capacity_seconds,
                self._pending_reservation_deletions,
                self._pending_expired_reservation_deletions,
                routing_checkpoint,
            ) = checkpoint
            self.router.restore_checkpoint(routing_checkpoint)
            raise

    def _stage_placement(self, now: float) -> list[tuple[GPURecord, AssignmentCommand]]:
        for transition in self.placement.transitions:
            record = self.gpus.get(transition.gpu_id)
            release = self.releases.get(transition.to_release)
            if record is None or release is None:
                continue
            if record.state not in {GPUState.REGISTERED, GPUState.READY}:
                continue
            self._account_ready_seconds(now)
            record.registration = replace(record.registration, release_digest=release.release_digest)
            record.state = GPUState.DRAINING
            record.lease = None
            record.revocation_reason = 'release transition awaiting miner-agent acceptance'
            record.assignment_started_at = now
            record.assignment_epoch += 1
            record.assignment_dispatched = False
            record.assignment_token = secrets.token_hex(32)
            record.weight_verified_at = 0.0
            record.runtime_verified_at = 0.0
            record.runtime_stream_public_key = ''

        pending: list[tuple[GPURecord, AssignmentCommand]] = []
        for record in self.gpus.values():
            if (
                record.state not in {GPUState.DRAINING, GPUState.LOADING}
                or (
                    (record.assignment_dispatched or record.assignment_dispatch_in_flight)
                    and now - record.assignment_last_dispatched_at < self.config.assignment.redispatch_interval_seconds
                )
                or record.registration.release_digest not in self.releases
            ):
                continue
            release = self.releases[record.registration.release_digest]
            command = AssignmentCommand(
                gpu_id=record.registration.gpu_id,
                miner_hotkey=record.registration.miner_hotkey,
                epoch=record.assignment_epoch,
                release_digest=release.release_digest,
                model_id=release.model_id,
                model_repository=release.model_repository,
                model_revision=release.model_revision,
                tokenizer_repository=release.tokenizer_repository,
                tokenizer_revision=release.tokenizer_revision,
                runtime_digest=release.runtime_digest,
                runtime_commit=release.runtime_commit,
                container_image=release.container_image,
                container_digest=release.container_digest,
                filesystem_digest=release.filesystem_digest,
                weight_files=release.weight_files,
                token_proof_scheme=release.token_proof_scheme,
                certified_slots=release.max_concurrency,
                max_context_tokens=release.max_context_tokens,
                kv_cache_capacity_bytes=release.kv_cache_capacity_bytes,
                kv_bytes_per_token=release.kv_bytes_per_token,
                request_overhead_tokens=release.request_overhead_tokens,
                assignment_token=record.assignment_token,
            )
            record.assignment_last_dispatched_at = now
            record.assignment_dispatch_in_flight = True
            pending.append((GPURecord(registration=record.registration), command))
        return pending

    def _dispatch_assignments(self, pending: Sequence[tuple[GPURecord, AssignmentCommand]]) -> None:
        assignment_executor = self.assignment_executor
        assert assignment_executor is not None

        def dispatch(item: tuple[GPURecord, AssignmentCommand]) -> tuple[GPURecord, int, str | None]:
            record, command = item
            try:
                assignment_executor.dispatch(record, command)
            except Exception as exc:
                return record, command.epoch, str(exc)
            return record, command.epoch, None

        worker_count = min(32, max(1, len(pending)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='assignment-dispatch') as executor:
            outcomes = list(executor.map(dispatch, pending))
        with self._lock:
            late_revocations: list[tuple[GPURecord, int, str]] = []
            for dispatched_record, epoch, error in outcomes:
                gpu_id = dispatched_record.registration.gpu_id
                record = self.gpus.get(gpu_id)
                if record is None or record.assignment_epoch != epoch:
                    continue
                record.assignment_dispatch_in_flight = False
                if record.administratively_disabled:
                    if error is None:
                        record.revocation_pending = True
                        late_revocations.append(
                            (dispatched_record, epoch, record.revocation_reason or 'assignment was revoked')
                        )
                    continue
                if error is None:
                    record.assignment_dispatched = True
                    record.revocation_reason = 'release transition in progress'
                else:
                    record.assignment_dispatched = False
                    record.revocation_reason = f'assignment dispatch failed: {error}'
            self._persist()
        for revocation in late_revocations:
            self._dispatch_assignment_revocation(*revocation)

    def settle(self, now: float | None = None) -> tuple[SettlementResult, dict[int, Decimal]]:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_emission_oracle(timestamp)
            window_seconds = timestamp - self._settlement_started_at
            if window_seconds <= 0:
                raise ValueError('settlement timestamp must advance the current window')
            window_budget = Decimal(str(self._funded_budget_seconds)) / Decimal(3600)
            reserved_compute_emission_share = self._compute_emission_share_seconds / window_seconds
            result = settle_ready_seconds(
                self.funding,
                window_seconds,
                self._ready_seconds,
                window_budget_override=window_budget,
                scarcity_reward_exponent=self.config.fleet.scarcity_reward_exponent,
                scarcity_multiplier_cap=self.config.fleet.scarcity_multiplier_cap,
            )
            paid_ratio = float(result.distributed_budget / result.window_budget) if result.window_budget > 0 else 0.0
            paid_compute_emission_share = reserved_compute_emission_share * paid_ratio
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
            started_at = self._settlement_started_at
            ready_seconds_checkpoint = self._ready_seconds
            funded_target_checkpoint = self._funded_target_seconds
            funded_budget_checkpoint = self._funded_budget_seconds
            emission_share_checkpoint = self._compute_emission_share_seconds
            self._ready_seconds = {}
            self._funded_target_seconds = 0.0
            self._funded_budget_seconds = 0.0
            self._compute_emission_share_seconds = 0.0
            self._settlement_started_at = timestamp
            if self.store:
                try:
                    hotkey_reward_payload = {hotkey: str(amount) for hotkey, amount in hotkey_rewards.items()}
                    settlement_metadata: dict[str, Any] = {
                        'window_budget': str(result.window_budget),
                        'distributed_budget': str(result.distributed_budget),
                        'unspent_budget': str(result.unspent_budget),
                        'total_ready_seconds': result.total_ready_seconds,
                        'effective_ready_gpus': result.effective_ready_gpus,
                        'effective_funded_gpus': result.effective_funded_gpus,
                        'scarcity_multiplier': result.scarcity_multiplier,
                        'compute_emission_share': paid_compute_emission_share,
                        'compute_reserved_emission_share': reserved_compute_emission_share,
                        'target_price_currency': self.config.fleet.target_price_currency,
                        'subnet_miner_emission_value_per_hour': str(self.subnet_miner_emission_value_per_hour),
                        'emission_epoch_block': (
                            self.emission_oracle_observation.epoch_block if self.emission_oracle_observation else None
                        ),
                    }
                    if self.settlement_signer is not None:
                        settlement_metadata.update(
                            self.settlement_signer.sign(
                                window_id,
                                started_at,
                                timestamp,
                                hotkey_reward_payload,
                                settlement_metadata,
                            )
                        )
                    finalized = self.store.finalize_settlement(
                        window_id,
                        started_at,
                        timestamp,
                        hotkey_reward_payload,
                        settlement_metadata,
                        self._export_state(),
                    )
                    if not finalized:
                        raise RuntimeError('settlement window was already finalized')
                except Exception:
                    self._ready_seconds = ready_seconds_checkpoint
                    self._funded_target_seconds = funded_target_checkpoint
                    self._funded_budget_seconds = funded_budget_checkpoint
                    self._compute_emission_share_seconds = emission_share_checkpoint
                    self._settlement_started_at = started_at
                    raise
            return result, miner_rewards

    def latest_settlement(self, max_age_seconds: float | None = None) -> dict[str, Any] | None:
        if self.store is None:
            return None
        max_age = max_age_seconds or self.config.control_loop.settlement_interval_seconds * 2
        return self.store.latest_settlement(max_age)

    def status(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else now
        with self._lock:
            self._account_ready_seconds(timestamp)
            self._expire_emission_oracle(timestamp)
            self._expire_leases(timestamp)
            active = self.router.active_counts(timestamp)
            status = {
                'desired_target': self.autoscaler.desired_target,
                'funded_target': self.funding.funded_target,
                'funding_shortfall': self.funding.funding_shortfall,
                'target_price_per_gpu_hour': str(self.funding.target_price_per_gpu_hour),
                'target_price_currency': self.config.fleet.target_price_currency,
                'subnet_miner_emission_value_per_hour': str(self.subnet_miner_emission_value_per_hour),
                'target_budget_per_hour': str(
                    Decimal(self.autoscaler.desired_target) * Decimal(str(self.config.fleet.target_price_per_gpu_hour))
                ),
                'external_budget_cap_per_hour': (
                    str(self.max_budget_per_hour) if self.max_budget_per_hour is not None else None
                ),
                'funded_pool_per_hour': str(self.funding.max_budget_per_hour),
                'compute_emission_share': self._current_compute_emission_share(),
                'emission_oracle': {
                    'enabled': self.config.emission_oracle.enabled,
                    'fresh': self.emission_oracle_is_fresh(timestamp),
                    'currency': self.config.fleet.target_price_currency,
                    'observation': asdict(self.emission_oracle_observation)
                    if self.emission_oracle_observation
                    else None,
                    'error': self.emission_oracle_error,
                },
                'verification_source': {
                    'repository': self.config.verification.source_repository,
                    'commit': self.config.verification.source_commit,
                    'protocol': self.config.verification.expected_verifier_protocol,
                    'measurement': self.config.verification.expected_verifier_measurement,
                    'signed_status_required': self.config.verification.require_status_signature,
                    'trusted_signer_count': len(self.config.verification.trusted_verifier_public_keys),
                },
                'registered_gpus': len(self.gpus),
                'ready_gpus': sum(record.is_ready(timestamp) for record in self.gpus.values()),
                'supply': {
                    'gpu_equivalent_demand': self.last_autoscale.concurrent_demand
                    if self.last_autoscale
                    else 0.0,
                    'required_target': self.last_autoscale.required_target if self.last_autoscale else self.config.fleet.floor,
                    'shortage_gpus': self.last_autoscale.supply_shortage if self.last_autoscale else 0.0,
                },
                'placements': dict(self.placement.assignments),
                'placement_shortages': dict(self.placement.shortages),
                'placement_deferred': dict(self.placement.deferred),
                'gpus': {
                    gpu_id: {
                        'miner_uid': record.registration.miner_uid,
                        'miner_hotkey': record.registration.miner_hotkey,
                        'state': record.state.value,
                        'assignment_epoch': record.assignment_epoch,
                        'assignment_dispatch_in_flight': record.assignment_dispatch_in_flight,
                        'release_digest': record.registration.release_digest,
                        'active_reservations': active.get(gpu_id, 0),
                        'lease_expires_at': record.lease.expires_at if record.lease else None,
                        'weight_verified_at': record.weight_verified_at or None,
                        'gateway_telemetry_updated_at': record.gateway_telemetry_updated_at or None,
                        'revocation_reason': record.revocation_reason,
                        'administratively_disabled': record.administratively_disabled,
                        'revocation_pending': record.revocation_pending,
                    }
                    for gpu_id, record in sorted(self.gpus.items())
                },
            }
            return status

    def _funding_plan(self, now: float | None = None) -> FundingPlan:
        return self._funding_plan_for_freshness(self.emission_oracle_is_fresh(now))

    def _funding_plan_for_freshness(self, oracle_fresh: bool) -> FundingPlan:
        funding_target = self.autoscaler.desired_target
        if self.config.emission_oracle.enabled and not oracle_fresh:
            funding_target = self.config.fleet.floor
        target_budget = Decimal(funding_target) * Decimal(str(self.config.fleet.target_price_per_gpu_hour))
        emission_budget_limit = Decimal(str(self.subnet_miner_emission_value_per_hour)) * Decimal(
            str(self.config.fleet.max_compute_emission_share)
        )
        available_budget = min(target_budget, emission_budget_limit)
        if self.max_budget_per_hour is not None:
            available_budget = min(available_budget, Decimal(str(self.max_budget_per_hour)))
        return funding_plan(
            self.autoscaler.desired_target,
            self.config.fleet.target_price_per_gpu_hour,
            available_budget,
        )

    def _expire_emission_oracle(self, now: float) -> None:
        if not self.config.emission_oracle.enabled or self.emission_oracle_is_fresh(now):
            return
        stale_funding = self._funding_plan_for_freshness(False)
        if self.funding == stale_funding:
            return
        self.funding = stale_funding

    def _record_demand(
        self,
        release_digest: str,
        service_seconds: float,
        capacity_units: float,
        *,
        rejected: bool,
    ) -> None:
        target = self._rejected if rejected else self._accepted
        target[release_digest] = target.get(release_digest, 0) + 1
        if rejected:
            self._rejected_capacity_seconds[release_digest] = self._rejected_capacity_seconds.get(
                release_digest, 0.0
            ) + max(0.0, service_seconds) * max(0.0, capacity_units)
            self._persist()

    def _account_ready_seconds(self, now: float) -> None:
        if now <= self._last_ready_account_at:
            return
        observation = self.emission_oracle_observation
        stale_funding = self._funding_plan_for_freshness(False)
        if self.config.emission_oracle.enabled and observation is not None and self.funding != stale_funding:
            stale_at = observation.observed_at + self.config.emission_oracle.max_refresh_staleness_seconds
            if self._last_ready_account_at < stale_at < now:
                self._account_ready_interval(stale_at)
                self.funding = stale_funding
        self._account_ready_interval(now)

    def _account_ready_interval(self, now: float) -> None:
        if now <= self._last_ready_account_at:
            return
        elapsed = now - self._last_ready_account_at
        self._funded_target_seconds += self.funding.funded_target * elapsed
        self._funded_budget_seconds += float(self.funding.max_budget_per_hour) * elapsed
        self._compute_emission_share_seconds += self._current_compute_emission_share() * elapsed
        for gpu_id, record in self.gpus.items():
            if record.state != GPUState.READY or record.lease is None:
                continue
            eligible_until = min(now, record.lease.expires_at)
            seconds = max(0.0, eligible_until - self._last_ready_account_at)
            self._ready_seconds[gpu_id] = self._ready_seconds.get(gpu_id, 0.0) + seconds
        self._last_ready_account_at = now

    def _current_compute_emission_share(self) -> float:
        funded_pool_value = float(self.funding.max_budget_per_hour)
        return min(
            self.config.fleet.max_compute_emission_share,
            funded_pool_value / self.subnet_miner_emission_value_per_hour,
        )

    def _expire_leases(self, now: float, *, persist: bool = True) -> None:
        changed = False
        for gpu_id, record in self.gpus.items():
            if record.state == GPUState.READY and (record.lease is None or not record.lease.is_live(now)):
                self._quarantine_gpu(gpu_id, 'verification lease expired', now)
                changed = True
        if changed and persist:
            self._persist()

    def _quarantine_gpu(self, gpu_id: str, reason: str, now: float) -> None:
        """Remove contradictory or stale capacity from every serving surface."""
        record = self.gpus[gpu_id]
        had_assignment = bool(record.registration.release_digest) and record.assignment_epoch > 0
        self._complete_gpu_reservations(gpu_id, now)
        record.registration = replace(record.registration, release_digest='', canary_release_digest='')
        record.state = GPUState.QUARANTINED
        record.lease = None
        record.assignment_token = secrets.token_hex(32)
        record.assignment_dispatched = False
        record.assignment_dispatch_in_flight = False
        record.revocation_pending = record.revocation_pending or had_assignment
        record.gateway_active_slots = 0
        record.gateway_remaining_work_seconds = 0.0
        record.gateway_telemetry_updated_at = 0.0
        record.runtime_verified_at = 0.0
        record.runtime_stream_public_key = ''
        record.weight_verified_at = 0.0
        record.revocation_reason = reason
        self._remove_from_placement(gpu_id)

    def _persist(self) -> None:
        if self.store is None:
            self._pending_reservation_deletions.clear()
            self._pending_expired_reservation_deletions.clear()
            return
        pending = set(self._pending_reservation_deletions)
        expired = set(self._pending_expired_reservation_deletions)
        self.store.save_state_and_delete_gpu_reservations(self._export_state(), pending, expired)
        self._pending_reservation_deletions.difference_update(pending)
        self._pending_expired_reservation_deletions.difference_update(expired)

    def _export_state(self) -> dict[str, Any]:
        return {
            'version': 2,
            'releases': [asdict(value) for value in self.releases.values()],
            'gpus': [
                {
                    'registration': asdict(record.registration),
                    'state': record.state.value,
                    'lease': asdict(record.lease) if record.lease else None,
                    'revocation_reason': record.revocation_reason,
                    'assignment_started_at': record.assignment_started_at,
                    'assignment_epoch': record.assignment_epoch,
                    'measured_rtt_by_region_ms': record.measured_rtt_by_region_ms,
                    'service_seconds_ewma_by_release': record.service_seconds_ewma_by_release,
                    'request_estimate_ratio_ewma_by_release': record.request_estimate_ratio_ewma_by_release,
                    'gateway_active_slots': record.gateway_active_slots,
                    'gateway_remaining_work_seconds': record.gateway_remaining_work_seconds,
                    'gateway_telemetry_updated_at': record.gateway_telemetry_updated_at,
                    'weight_verified_at': record.weight_verified_at,
                    'runtime_verified_at': record.runtime_verified_at,
                    'runtime_stream_public_key': record.runtime_stream_public_key,
                    'assignment_dispatched': record.assignment_dispatched,
                    'assignment_dispatch_in_flight': record.assignment_dispatch_in_flight,
                    'assignment_last_dispatched_at': record.assignment_last_dispatched_at,
                    'assignment_token': record.assignment_token,
                    'consecutive_inference_failures': record.consecutive_inference_failures,
                    'inference_quarantined_at': record.inference_quarantined_at,
                    'administratively_disabled': record.administratively_disabled,
                    'revocation_pending': record.revocation_pending,
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
            'last_autoscale': asdict(self.last_autoscale) if self.last_autoscale else None,
            'max_budget_per_hour': self.max_budget_per_hour,
            'subnet_miner_emission_value_per_hour': self.subnet_miner_emission_value_per_hour,
            'emission_oracle_observation': asdict(self.emission_oracle_observation)
            if self.emission_oracle_observation
            else None,
            'emission_oracle_error': self.emission_oracle_error,
            'placement': {
                'assignments': dict(self.placement.assignments),
                'replica_counts': dict(self.placement.replica_counts),
                'shortages': dict(self.placement.shortages),
                'deferred': dict(self.placement.deferred),
                'shortage_since': self.gepetto.export_state(),
            },
            'last_tick_at': self._last_tick_at,
            'last_ready_account_at': self._last_ready_account_at,
            'settlement_started_at': self._settlement_started_at,
            'funded_target_seconds': self._funded_target_seconds,
            'funded_budget_seconds': self._funded_budget_seconds,
            'compute_emission_share_seconds': self._compute_emission_share_seconds,
            'ready_seconds': self._ready_seconds,
            'accepted': self._accepted,
            'rejected': self._rejected,
            'rejected_capacity_seconds': self._rejected_capacity_seconds,
            'completed_capacity_seconds': self._completed_capacity_seconds,
        }

    def _restore_state(self, state: Mapping[str, Any], now: float) -> None:
        state_version = int(state.get('version', 0))
        if state_version not in {1, 2}:
            raise ValueError('unsupported compute state version')
        self.releases = {value['release_digest']: Release(**value) for value in state.get('releases', [])}
        self.gpus = {}
        for value in state.get('gpus', []):
            registration = GPURegistration(**value['registration'])
            lease = VerificationLease(**value['lease']) if value.get('lease') else None
            administratively_disabled = bool(value.get('administratively_disabled', False))
            dispatch_was_in_flight = bool(value.get('assignment_dispatch_in_flight', False))
            record = GPURecord(
                registration=registration,
                state=GPUState(value['state']),
                lease=lease,
                revocation_reason=value.get('revocation_reason'),
                assignment_started_at=float(value.get('assignment_started_at', 0)),
                assignment_epoch=int(value.get('assignment_epoch', 0)),
                measured_rtt_by_region_ms={
                    key: float(item) for key, item in value.get('measured_rtt_by_region_ms', {}).items()
                },
                service_seconds_ewma_by_release={
                    key: float(item) for key, item in value.get('service_seconds_ewma_by_release', {}).items()
                },
                request_estimate_ratio_ewma_by_release={
                    key: float(item) for key, item in value.get('request_estimate_ratio_ewma_by_release', {}).items()
                },
                gateway_active_slots=int(value.get('gateway_active_slots', 0)),
                gateway_remaining_work_seconds=float(value.get('gateway_remaining_work_seconds', 0)),
                gateway_telemetry_updated_at=float(value.get('gateway_telemetry_updated_at', 0)),
                weight_verified_at=float(value.get('weight_verified_at', 0)),
                runtime_verified_at=float(value.get('runtime_verified_at', 0)),
                runtime_stream_public_key=str(value.get('runtime_stream_public_key', '')),
                assignment_dispatched=bool(value.get('assignment_dispatched', False)),
                # A process restart ends local delivery but leaves its remote
                # result ambiguous. Enabled assignments retry the same epoch;
                # disabled assignments retry their tombstone first.
                assignment_dispatch_in_flight=False,
                assignment_last_dispatched_at=float(value.get('assignment_last_dispatched_at', 0)),
                assignment_token=str(value.get('assignment_token', '')),
                consecutive_inference_failures=int(value.get('consecutive_inference_failures', 0)),
                inference_quarantined_at=float(value.get('inference_quarantined_at', 0)),
                administratively_disabled=administratively_disabled,
                revocation_pending=(
                    bool(value.get('revocation_pending', False))
                    or (administratively_disabled and dispatch_was_in_flight)
                ),
            )
            self.gpus[registration.gpu_id] = record
        autoscaler = state.get('autoscaler', {})
        self.autoscaler.desired_target = int(autoscaler.get('desired_target', self.autoscaler.desired_target))
        self.autoscaler.rejection_demand_ewma = float(autoscaler.get('rejection_demand_ewma', 0))
        self.autoscaler.high_since = autoscaler.get('high_since')
        self.autoscaler.low_since = autoscaler.get('low_since')
        self.autoscaler.last_scaled_at = autoscaler.get('last_scaled_at')
        saved_last_autoscale = state.get('last_autoscale')
        self.last_autoscale = (
            AutoscaleDecision(**saved_last_autoscale) if isinstance(saved_last_autoscale, Mapping) else None
        )
        saved_budget = state.get('max_budget_per_hour', self.max_budget_per_hour)
        self.max_budget_per_hour = float(saved_budget) if saved_budget is not None else None
        saved_emission_value = state.get(
            'subnet_miner_emission_value_per_hour',
            self.subnet_miner_emission_value_per_hour,
        )
        self.subnet_miner_emission_value_per_hour = float(saved_emission_value)
        if (
            not math.isfinite(self.subnet_miner_emission_value_per_hour)
            or self.subnet_miner_emission_value_per_hour <= 0
        ):
            raise ValueError('persisted subnet miner emission value must be finite and positive')
        saved_observation = state.get('emission_oracle_observation')
        self.emission_oracle_observation = (
            EmissionObservation(**saved_observation) if isinstance(saved_observation, Mapping) else None
        )
        self.emission_oracle_error = (
            str(state['emission_oracle_error']) if state.get('emission_oracle_error') is not None else None
        )
        saved_account_at = float(state.get('last_ready_account_at', now))
        observation = self.emission_oracle_observation
        historical_oracle_fresh = bool(
            not self.config.emission_oracle.enabled
            or (
                observation is not None
                and 0
                <= saved_account_at - observation.observed_at
                <= self.config.emission_oracle.max_refresh_staleness_seconds
            )
        )
        self.funding = self._funding_plan_for_freshness(historical_oracle_fresh)
        placement = state.get('placement', {})
        self.placement = PlacementPlan(
            placement.get('assignments', {}),
            placement.get('replica_counts', {}),
            (),
            placement.get('shortages', {}),
            placement.get('deferred', {}),
        )
        self.gepetto.restore_state(placement.get('shortage_since', {}))
        self._last_tick_at = float(state.get('last_tick_at', now))
        self._last_ready_account_at = saved_account_at
        self._settlement_started_at = float(state.get('settlement_started_at', now))
        self._funded_target_seconds = float(state.get('funded_target_seconds', 0))
        self._funded_budget_seconds = float(
            state.get(
                'funded_budget_seconds',
                self._funded_target_seconds * float(self.config.fleet.target_price_per_gpu_hour),
            )
        )
        self._compute_emission_share_seconds = float(state.get('compute_emission_share_seconds', 0))
        self._ready_seconds = {key: float(value) for key, value in state.get('ready_seconds', {}).items()}
        self._accepted = {key: int(value) for key, value in state.get('accepted', {}).items()}
        self._rejected = {key: int(value) for key, value in state.get('rejected', {}).items()}
        if state_version == 2:
            self._rejected_capacity_seconds = {
                key: float(value) for key, value in state.get('rejected_capacity_seconds', {}).items()
            }
            self._completed_capacity_seconds = {
                key: float(value) for key, value in state.get('completed_capacity_seconds', {}).items()
            }
        else:
            self._rejected_capacity_seconds = {
                key: float(value) / self.releases[key].max_concurrency
                for key, value in state.get('rejected_service_seconds', {}).items()
                if key in self.releases
            }
            self._completed_capacity_seconds = {
                key: float(value) / self.releases[key].max_concurrency
                for key, value in state.get('completed_slot_seconds', {}).items()
                if key in self.releases
            }
        self._account_ready_seconds(now)
        self._expire_emission_oracle(now)
        legacy_service_seconds = {key: float(value) for key, value in state.get('service_seconds_total', {}).items()}
        for release_digest, total_seconds in legacy_service_seconds.items():
            if release_digest not in self.releases:
                continue
            accepted = self._accepted.get(release_digest, 0)
            rejected = self._rejected.get(release_digest, 0)
            total_requests = accepted + rejected
            if total_requests <= 0:
                continue
            self._rejected_capacity_seconds.setdefault(
                release_digest,
                total_seconds * rejected / total_requests / self.releases[release_digest].max_concurrency,
            )
        if self.store is None:
            self.router.restore_state(state.get('reservations', []), now)
        self._expire_leases(now)
