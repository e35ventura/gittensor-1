"""Shared domain models for the compute control plane."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Mapping
from urllib.parse import urlparse

_COMMIT_PATTERN = re.compile(r'[0-9a-f]{40}')
_SHA256_PATTERN = re.compile(r'sha256:[0-9a-f]{64}')
_REPOSITORY_PATTERN = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')


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
    tokenizer_repository: str = ''
    tokenizer_revision: str = ''
    container_image: str = ''
    container_digest: str = ''
    filesystem_digest: str = ''
    runtime_commit: str = ''
    weight_files: Mapping[str, int] = field(default_factory=dict)
    token_proof_scheme: str = 'sr25519-response-v1'
    minimum_replicas: int = 0
    placement_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.release_digest or not self.model_id or not self.runtime_digest:
            raise ValueError('release digest, model id, and runtime digest are required')
        if self.minimum_replicas < 0 or self.placement_weight <= 0:
            raise ValueError('release placement values are invalid')
        if self.model_revision and not _COMMIT_PATTERN.fullmatch(self.model_revision):
            raise ValueError('model_revision must be an immutable 40-character commit')
        if self.tokenizer_revision and not _COMMIT_PATTERN.fullmatch(self.tokenizer_revision):
            raise ValueError('tokenizer_revision must be an immutable 40-character commit')
        if any(not path or size < 1 for path, size in self.weight_files.items()):
            raise ValueError('weight_files must map non-empty paths to positive byte sizes')

    def validate_production_manifest(self) -> None:
        """Reject a release that cannot be independently reproduced and checked."""
        required = {
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'tokenizer_repository': self.tokenizer_repository,
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
        if not _REPOSITORY_PATTERN.fullmatch(self.model_repository):
            raise ValueError('model_repository must be an owner/repository identifier')
        if not _REPOSITORY_PATTERN.fullmatch(self.tokenizer_repository):
            raise ValueError('tokenizer_repository must be an owner/repository identifier')
        if not _COMMIT_PATTERN.fullmatch(self.runtime_commit):
            raise ValueError('runtime_commit must be an immutable 40-character commit')
        for field_name in ('runtime_digest', 'container_digest', 'filesystem_digest'):
            if not _SHA256_PATTERN.fullmatch(str(getattr(self, field_name))):
                raise ValueError(f'{field_name} must be a lowercase sha256 digest')
        if '@' in self.container_image or any(character.isspace() for character in self.container_image):
            raise ValueError('container_image must not include a digest or whitespace')
        if self.token_proof_scheme != 'sr25519-response-v1':
            raise ValueError('token_proof_scheme must be sr25519-response-v1')
        for path in self.weight_files:
            parsed_path = PurePosixPath(path)
            if parsed_path.is_absolute() or '..' in parsed_path.parts or '.' in parsed_path.parts:
                raise ValueError('weight file paths must stay inside the pinned repository')
        expected_digest = self.computed_release_digest()
        if self.release_digest != expected_digest:
            raise ValueError(f'release_digest must equal the canonical manifest digest: {expected_digest}')

    def computed_release_digest(self) -> str:
        manifest = {
            'model_id': self.model_id,
            'runtime_digest': self.runtime_digest,
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'tokenizer_repository': self.tokenizer_repository,
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
        if parsed_endpoint.path not in {'', '/'} or parsed_endpoint.query or parsed_endpoint.fragment:
            raise ValueError('GPU endpoint must be an HTTPS origin without a path, query, or fragment')
        try:
            endpoint_ip = ipaddress.ip_address(parsed_endpoint.hostname or '')
        except ValueError:
            endpoint_ip = None
        if endpoint_ip is not None and not endpoint_ip.is_global:
            raise ValueError('GPU endpoint IP must be globally routable')


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
    stream_public_key: str = ''

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
    measured_rtt_by_region_ms: dict[str, float] = field(default_factory=dict)
    service_seconds_ewma_by_release: dict[str, float] = field(default_factory=dict)
    request_estimate_ratio_ewma_by_release: dict[str, float] = field(default_factory=dict)
    gateway_active_slots: int = 0
    gateway_remaining_work_seconds: float = 0.0
    gateway_telemetry_updated_at: float = 0.0
    weight_verified_at: float = 0.0
    runtime_verified_at: float = 0.0
    runtime_stream_public_key: str = ''
    assignment_dispatched: bool = False
    assignment_dispatch_in_flight: bool = False
    assignment_last_dispatched_at: float = 0.0
    assignment_token: str = ''
    consecutive_inference_failures: int = 0
    inference_quarantined_at: float = 0.0
    administratively_disabled: bool = False
    revocation_pending: bool = False

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
    tokenizer_repository: str
    tokenizer_revision: str
    runtime_digest: str
    runtime_commit: str
    container_image: str
    container_digest: str
    filesystem_digest: str
    weight_files: Mapping[str, int]
    token_proof_scheme: str
    certified_slots: int
    assignment_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.certified_slots, int) or isinstance(self.certified_slots, bool):
            raise ValueError('certified_slots must be an integer')
        if self.certified_slots < 1:
            raise ValueError('certified_slots must be positive')


@dataclass(frozen=True)
class RuntimeEvidence:
    release_digest: str
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    runtime_digest: str
    runtime_commit: str
    container_image: str
    container_digest: str
    filesystem_digest: str
    stream_public_key: str = ''


@dataclass(frozen=True)
class RoutingObservation:
    reservation_id: str
    gpu_id: str
    requester_region: str
    measured_rtt_ms: float
    service_seconds: float
    success: bool
    observed_active_slots: int
    remaining_work_seconds: float
    expected_service_seconds: float = 0.0
    release_digest: str = ''
