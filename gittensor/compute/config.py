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


@dataclass(frozen=True)
class RouterConfig:
    reservation_ttl_seconds: float
    default_rtt_ms: float


@dataclass(frozen=True)
class PlacementConfig:
    minimum_residency_seconds: float
    control_interval_seconds: float


@dataclass(frozen=True)
class ComputeConfig:
    fleet: FleetConfig
    autoscaling: AutoscalingConfig
    verification: VerificationConfig
    router: RouterConfig
    placement: PlacementConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> 'ComputeConfig':
        config = cls(
            fleet=FleetConfig(**raw['fleet']),
            autoscaling=AutoscalingConfig(**raw['autoscaling']),
            verification=VerificationConfig(**raw['verification']),
            router=RouterConfig(**raw['router']),
            placement=PlacementConfig(**raw['placement']),
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
            )
            <= 0
        ):
            raise ValueError('all intervals and TTLs must be positive')


def load_compute_config(path: str | Path) -> ComputeConfig:
    with Path(path).open() as handle:
        return ComputeConfig.from_mapping(json.load(handle))
