"""Configuration and validation for the compute sub-subnet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FleetConfig:
    floor: int
    initial_target: int
    certified_slots_per_gpu: int
    target_price_per_gpu_hour: float
    max_budget_per_hour: float


@dataclass(frozen=True)
class AutoscalingConfig:
    ewma_alpha: float
    utilization_up: float
    utilization_down: float
    sustain_up_seconds: float
    sustain_down_seconds: float
    cooldown_seconds: float


@dataclass(frozen=True)
class VerificationConfig:
    status_url: str
    timeout_seconds: float
    lease_ttl_seconds: float
    expected_hardware: str
    require_model_canary: bool
    bearer_token_env: str | None
    source_repository: str
    source_commit: str
    expected_verifier_protocol: str = 'sparkcompute-v1'
    expected_verifier_measurement: str = ''
    require_verifier_measurement: bool = False
    require_runtime_attestation: bool = False
    require_weight_challenges: bool = False
    weight_challenge_interval_seconds: float = 30.0
    weight_challenge_ttl_seconds: float = 30.0
    weight_verification_ttl_seconds: float = 900.0
    challenge_sample_size: int = 3
    require_signed_containers: bool = False
    cosign_binary: str = 'cosign'
    cosign_public_key_path: str = ''


@dataclass(frozen=True)
class RouterConfig:
    reservation_ttl_seconds: float
    default_rtt_ms: float
    equivalent_finish_epsilon_seconds: float = 0.025
    telemetry_ttl_seconds: float = 30.0


@dataclass(frozen=True)
class PlacementConfig:
    minimum_residency_seconds: float
    control_interval_seconds: float


@dataclass(frozen=True)
class StateConfig:
    database_path: str = 'gittensor-compute.sqlite3'


@dataclass(frozen=True)
class IdentityConfig:
    netuid: int = 74
    network: str = 'finney'
    signature_ttl_seconds: float = 60.0
    metagraph_refresh_seconds: float = 60.0
    spark_node_owners_path: str = ''


@dataclass(frozen=True)
class AssignmentConfig:
    request_timeout_seconds: float = 15.0
    bearer_token_env: str | None = 'GITTENSOR_ASSIGNMENT_TOKEN'


@dataclass(frozen=True)
class ControlLoopConfig:
    verification_interval_seconds: float = 30.0
    settlement_interval_seconds: float = 3600.0


@dataclass(frozen=True)
class ComputeConfig:
    fleet: FleetConfig
    autoscaling: AutoscalingConfig
    verification: VerificationConfig
    router: RouterConfig
    placement: PlacementConfig
    state: StateConfig = StateConfig()
    identity: IdentityConfig = IdentityConfig()
    assignment: AssignmentConfig = AssignmentConfig()
    control_loop: ControlLoopConfig = ControlLoopConfig()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> 'ComputeConfig':
        config = cls(
            fleet=FleetConfig(**raw['fleet']),
            autoscaling=AutoscalingConfig(**raw['autoscaling']),
            verification=VerificationConfig(**raw['verification']),
            router=RouterConfig(**raw['router']),
            placement=PlacementConfig(**raw['placement']),
            state=StateConfig(**raw.get('state', {})),
            identity=IdentityConfig(**raw.get('identity', {})),
            assignment=AssignmentConfig(**raw.get('assignment', {})),
            control_loop=ControlLoopConfig(**raw.get('control_loop', {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        fleet = self.fleet
        scaling = self.autoscaling
        if fleet.floor < 1:
            raise ValueError('fleet.floor must be at least 1')
        if fleet.initial_target < fleet.floor:
            raise ValueError('fleet.initial_target cannot be below fleet.floor')
        if fleet.certified_slots_per_gpu < 1:
            raise ValueError('fleet.certified_slots_per_gpu must be at least 1')
        if fleet.target_price_per_gpu_hour <= 0:
            raise ValueError('fleet.target_price_per_gpu_hour must be positive')
        minimum_budget = fleet.floor * fleet.target_price_per_gpu_hour
        if fleet.max_budget_per_hour < minimum_budget:
            raise ValueError('fleet.max_budget_per_hour must fund the configured floor')
        if not 0 < scaling.ewma_alpha <= 1:
            raise ValueError('autoscaling.ewma_alpha must be in (0, 1]')
        if not 0 <= scaling.utilization_down < scaling.utilization_up <= 1:
            raise ValueError('autoscaling thresholds must satisfy 0 <= down < up <= 1')
        if (
            min(
                scaling.sustain_up_seconds,
                scaling.sustain_down_seconds,
                scaling.cooldown_seconds,
                self.verification.lease_ttl_seconds,
                self.router.reservation_ttl_seconds,
                self.placement.control_interval_seconds,
                self.verification.weight_challenge_interval_seconds,
                self.verification.weight_challenge_ttl_seconds,
                self.verification.weight_verification_ttl_seconds,
                self.router.telemetry_ttl_seconds,
                self.identity.signature_ttl_seconds,
                self.identity.metagraph_refresh_seconds,
                self.assignment.request_timeout_seconds,
                self.control_loop.verification_interval_seconds,
                self.control_loop.settlement_interval_seconds,
            )
            <= 0
        ):
            raise ValueError('all intervals and TTLs must be positive')
        if self.router.equivalent_finish_epsilon_seconds < 0:
            raise ValueError('router.equivalent_finish_epsilon_seconds cannot be negative')
        if self.verification.challenge_sample_size < 1:
            raise ValueError('verification.challenge_sample_size must be positive')
        if (
            self.verification.require_weight_challenges
            and self.verification.weight_challenge_interval_seconds
            > self.verification.weight_verification_ttl_seconds / 2
        ):
            raise ValueError('weight challenge interval must be at most half the weight verification TTL')
        if self.verification.require_verifier_measurement and not self.verification.expected_verifier_measurement:
            raise ValueError('strict verifier mode requires expected_verifier_measurement')
        if self.verification.require_signed_containers and not self.verification.cosign_public_key_path:
            raise ValueError('signed container mode requires cosign_public_key_path')
        if self.identity.netuid < 0:
            raise ValueError('identity.netuid cannot be negative')


def load_compute_config(path: str | Path) -> ComputeConfig:
    with Path(path).open() as handle:
        return ComputeConfig.from_mapping(json.load(handle))
