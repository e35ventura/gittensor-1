"""Shared domain models for the compute control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


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
    minimum_replicas: int = 0
    placement_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.release_digest or not self.model_id or not self.runtime_digest:
            raise ValueError('release digest, model id, and runtime digest are required')
        if self.minimum_replicas < 0 or self.placement_weight <= 0:
            raise ValueError('release placement values are invalid')


@dataclass(frozen=True)
class GPURegistration:
    gpu_id: str
    spark_node_id: str
    miner_uid: int
    endpoint: str
    region: str
    release_digest: str
    canary_release_digest: str
    performance_class: str = 'rtx-5090'
    certified_slots: int = 1
    latency_by_region_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            (
                self.gpu_id,
                self.spark_node_id,
                self.endpoint,
                self.region,
                self.release_digest,
                self.canary_release_digest,
            )
        ):
            raise ValueError('GPU registration fields cannot be empty')
        if self.miner_uid < 0 or self.certified_slots < 1:
            raise ValueError('miner_uid and certified_slots must be non-negative')


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

    def is_live(self, now: float) -> bool:
        return now < self.expires_at


@dataclass
class GPURecord:
    registration: GPURegistration
    state: GPUState = GPUState.REGISTERED
    lease: VerificationLease | None = None
    revocation_reason: str | None = None
    assignment_started_at: float = 0.0
    reported_remaining_work_seconds: float = 0.0

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
