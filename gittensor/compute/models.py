"""Shared domain models for the compute control plane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping
from urllib.parse import urlparse


class GPUState(str, Enum):
    REGISTERED = 'REGISTERED'
    DRAINING = 'DRAINING'
    LOADING = 'LOADING'
    RUNTIME_VERIFY = 'RUNTIME_VERIFY'
    READY = 'READY'
    QUARANTINED = 'QUARANTINED'


@dataclass(frozen=True)
class Release:
    release_digest: str
    model_id: str
    runtime_digest: str
    model_repository: str = ''
    model_revision: str = ''
    tokenizer_revision: str = ''
    container_image: str = ''
    container_digest: str = ''
    filesystem_digest: str = ''
    runtime_commit: str = ''
    weight_files: Mapping[str, int] = field(default_factory=dict)
    token_proof_scheme: str = 'hmac-sha256-v1'
    minimum_replicas: int = 0
    placement_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.release_digest or not self.model_id or not self.runtime_digest:
            raise ValueError('release digest, model id, and runtime digest are required')
        if self.minimum_replicas < 0 or self.placement_weight <= 0:
            raise ValueError('release placement values are invalid')
        if self.model_revision and len(self.model_revision) != 40:
            raise ValueError('model_revision must be an immutable 40-character commit')
        if self.tokenizer_revision and len(self.tokenizer_revision) != 40:
            raise ValueError('tokenizer_revision must be an immutable 40-character commit')
        if any(not path or size < 1 for path, size in self.weight_files.items()):
            raise ValueError('weight_files must map non-empty paths to positive byte sizes')

    def validate_production_manifest(self) -> None:
        """Reject a release that cannot be independently reproduced and checked."""
        required = {
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'tokenizer_revision': self.tokenizer_revision,
            'container_image': self.container_image,
            'container_digest': self.container_digest,
            'filesystem_digest': self.filesystem_digest,
            'runtime_commit': self.runtime_commit,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError(f'release manifest is missing: {", ".join(missing)}')
        if not self.weight_files:
            raise ValueError('release manifest must list at least one model weight file')
        expected_digest = self.computed_release_digest()
        if self.release_digest != expected_digest:
            raise ValueError(f'release_digest must equal the canonical manifest digest: {expected_digest}')

    def computed_release_digest(self) -> str:
        manifest = {
            'model_id': self.model_id,
            'runtime_digest': self.runtime_digest,
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'tokenizer_revision': self.tokenizer_revision,
            'container_image': self.container_image,
            'container_digest': self.container_digest,
            'filesystem_digest': self.filesystem_digest,
            'runtime_commit': self.runtime_commit,
            'weight_files': dict(sorted(self.weight_files.items())),
            'token_proof_scheme': self.token_proof_scheme,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


@dataclass(frozen=True)
class GPURegistration:
    gpu_id: str
    spark_node_id: str
    miner_uid: int
    endpoint: str
    region: str
    release_digest: str
    canary_release_digest: str
    miner_hotkey: str = ''
    performance_class: str = 'rtx-5090'
    certified_slots: int = 1

    def __post_init__(self) -> None:
        if not all(
            (
                self.gpu_id,
                self.spark_node_id,
                self.endpoint,
                self.region,
            )
        ):
            raise ValueError('GPU registration fields cannot be empty')
        if self.miner_uid < 0 or self.certified_slots < 1:
            raise ValueError('miner_uid and certified_slots must be non-negative')
        parsed_endpoint = urlparse(self.endpoint)
        if parsed_endpoint.scheme != 'https' or not parsed_endpoint.netloc:
            raise ValueError('GPU endpoint must be an absolute HTTPS URL')


@dataclass(frozen=True)
class VerificationLease:
    gpu_id: str
    hardware_type: str
    hardware_uuid: str
    driver_version: str
    release_digest: str
    verified_at: float
    last_heartbeat: float
    expires_at: float
    verifier_protocol: str = ''
    verifier_measurement: str = ''
    evidence_digest: str = ''

    def is_live(self, now: float) -> bool:
        return now < self.expires_at


@dataclass
class GPURecord:
    registration: GPURegistration
    state: GPUState = GPUState.REGISTERED
    lease: VerificationLease | None = None
    revocation_reason: str | None = None
    assignment_started_at: float = 0.0
    assignment_epoch: int = 0
    reported_remaining_work_seconds: float = 0.0
    reported_active_slots: int = 0
    telemetry_updated_at: float = 0.0
    measured_rtt_by_region_ms: dict[str, float] = field(default_factory=dict)
    service_seconds_ewma: float = 0.0
    gateway_active_slots: int = 0
    gateway_remaining_work_seconds: float = 0.0
    gateway_telemetry_updated_at: float = 0.0
    weight_verified_at: float = 0.0
    runtime_verified_at: float = 0.0

    def is_ready(self, now: float) -> bool:
        return self.state == GPUState.READY and self.lease is not None and self.lease.is_live(now)


@dataclass(frozen=True)
class PlacementTransition:
    gpu_id: str
    from_release: str | None
    to_release: str
    states: tuple[GPUState, ...] = (
        GPUState.DRAINING,
        GPUState.LOADING,
        GPUState.RUNTIME_VERIFY,
        GPUState.READY,
    )


@dataclass(frozen=True)
class AssignmentCommand:
    gpu_id: str
    miner_hotkey: str
    epoch: int
    release_digest: str
    model_id: str
    model_repository: str
    model_revision: str
    runtime_digest: str
    runtime_commit: str
    container_image: str
    container_digest: str
    filesystem_digest: str


@dataclass(frozen=True)
class RuntimeEvidence:
    release_digest: str
    model_repository: str
    model_revision: str
    runtime_digest: str
    runtime_commit: str
    container_image: str
    container_digest: str
    filesystem_digest: str


@dataclass(frozen=True)
class RoutingObservation:
    gpu_id: str
    requester_region: str
    measured_rtt_ms: float
    service_seconds: float
    success: bool
    observed_active_slots: int
    remaining_work_seconds: float
