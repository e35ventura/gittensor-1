"""Configuration and validation for the compute sub-subnet."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gittensor.compute.http_json import load_json_object

_COMMIT_PATTERN = re.compile(r'[0-9a-f]{40}')
_SHA256_PATTERN = re.compile(r'sha256:[0-9a-f]{64}')
_PUBLIC_KEY_PATTERN = re.compile(r'[0-9a-f]{64}')


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'{field_name} must be a list of strings')
    return tuple(value)


@dataclass(frozen=True)
class FleetConfig:
    floor: int
    initial_target: int
    certified_slots_per_gpu: int
    target_price_per_gpu_hour: float
    subnet_miner_emission_value_per_hour: float
    max_budget_per_hour: float | None = None
    max_compute_emission_share: float = 0.50
    target_price_currency: str = 'USD'
    scarcity_reward_exponent: float = 0.5
    scarcity_multiplier_cap: float = 2.0


@dataclass(frozen=True)
class EmissionOracleConfig:
    enabled: bool = False
    refresh_interval_seconds: float = 60.0
    max_refresh_staleness_seconds: float = 300.0
    max_epoch_age_seconds: float = 9_000.0
    request_timeout_seconds: float = 5.0
    tao_usd_price_sources: tuple[str, ...] = ('coinbase', 'coingecko')
    minimum_price_sources: int = 2
    maximum_price_divergence: float = 0.05


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
    require_status_signature: bool = False
    trusted_verifier_public_keys: tuple[str, ...] = ()
    require_runtime_attestation: bool = False
    require_stream_proof: bool = False
    require_confidential_compute: bool = False
    require_uniqueness_challenge: bool = False
    uniqueness_challenge_ttl_seconds: float = 900.0
    uniqueness_batch_size: int = 3
    require_weight_challenges: bool = False
    weight_challenge_interval_seconds: float = 30.0
    weight_challenge_ttl_seconds: float = 30.0
    weight_verification_ttl_seconds: float = 900.0
    challenge_sample_size: int = 3
    require_signed_containers: bool = False
    cosign_binary: str = 'cosign'
    cosign_public_key_path: str = ''
    status_urls: tuple[str, ...] = ()
    refresh_workers: int = 32


@dataclass(frozen=True)
class RouterConfig:
    reservation_ttl_seconds: float
    default_rtt_ms: float
    equivalent_finish_epsilon_seconds: float = 0.025
    telemetry_ttl_seconds: float = 30.0
    maximum_service_seconds: float = 900.0
    failure_quarantine_threshold: int = 3


@dataclass(frozen=True)
class PlacementConfig:
    minimum_residency_seconds: float
    control_interval_seconds: float
    switch_sustain_seconds: float = 120.0
    planning_horizon_seconds: float = 900.0
    minimum_switch_gain_gpu: float = 0.10


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
    redispatch_interval_seconds: float = 60.0
    validator_wallet_name: str = 'default'
    validator_wallet_hotkey: str = 'default'
    validator_wallet_path: str | None = None


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
    emission_oracle: EmissionOracleConfig = EmissionOracleConfig()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> 'ComputeConfig':
        verification = dict(raw['verification'])
        verification['status_urls'] = _string_tuple(
            verification.get('status_urls', ()),
            'verification.status_urls',
        )
        verification['trusted_verifier_public_keys'] = _string_tuple(
            verification.get('trusted_verifier_public_keys', ()),
            'verification.trusted_verifier_public_keys',
        )
        emission_oracle = dict(raw.get('emission_oracle', {}))
        if 'tao_usd_price_sources' in emission_oracle:
            emission_oracle['tao_usd_price_sources'] = _string_tuple(
                emission_oracle['tao_usd_price_sources'],
                'emission_oracle.tao_usd_price_sources',
            )
        config = cls(
            fleet=FleetConfig(**raw['fleet']),
            autoscaling=AutoscalingConfig(**raw['autoscaling']),
            verification=VerificationConfig(**verification),
            router=RouterConfig(**raw['router']),
            placement=PlacementConfig(**raw['placement']),
            state=StateConfig(**raw.get('state', {})),
            identity=IdentityConfig(**raw.get('identity', {})),
            assignment=AssignmentConfig(**raw.get('assignment', {})),
            control_loop=ControlLoopConfig(**raw.get('control_loop', {})),
            emission_oracle=EmissionOracleConfig(**emission_oracle),
        )
        config.validate()
        return config

    def validate(self) -> None:
        fleet = self.fleet
        scaling = self.autoscaling
        finite_values = [
            fleet.target_price_per_gpu_hour,
            fleet.subnet_miner_emission_value_per_hour,
            fleet.max_compute_emission_share,
            fleet.scarcity_reward_exponent,
            fleet.scarcity_multiplier_cap,
            scaling.ewma_alpha,
            scaling.utilization_up,
            scaling.utilization_down,
            scaling.sustain_up_seconds,
            scaling.sustain_down_seconds,
            scaling.cooldown_seconds,
            self.verification.timeout_seconds,
            self.verification.lease_ttl_seconds,
            self.verification.weight_challenge_interval_seconds,
            self.verification.weight_challenge_ttl_seconds,
            self.verification.weight_verification_ttl_seconds,
            self.verification.uniqueness_challenge_ttl_seconds,
            self.router.reservation_ttl_seconds,
            self.router.default_rtt_ms,
            self.router.equivalent_finish_epsilon_seconds,
            self.router.telemetry_ttl_seconds,
            self.router.maximum_service_seconds,
            self.placement.minimum_residency_seconds,
            self.placement.control_interval_seconds,
            self.placement.switch_sustain_seconds,
            self.placement.planning_horizon_seconds,
            self.placement.minimum_switch_gain_gpu,
            self.identity.signature_ttl_seconds,
            self.identity.metagraph_refresh_seconds,
            self.assignment.request_timeout_seconds,
            self.assignment.redispatch_interval_seconds,
            self.control_loop.verification_interval_seconds,
            self.control_loop.settlement_interval_seconds,
            self.emission_oracle.refresh_interval_seconds,
            self.emission_oracle.max_refresh_staleness_seconds,
            self.emission_oracle.max_epoch_age_seconds,
            self.emission_oracle.request_timeout_seconds,
            self.emission_oracle.maximum_price_divergence,
        ]
        if fleet.max_budget_per_hour is not None:
            finite_values.append(fleet.max_budget_per_hour)
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError('all numeric compute configuration values must be finite')
        integer_values = {
            'fleet.floor': fleet.floor,
            'fleet.initial_target': fleet.initial_target,
            'fleet.certified_slots_per_gpu': fleet.certified_slots_per_gpu,
            'verification.challenge_sample_size': self.verification.challenge_sample_size,
            'verification.refresh_workers': self.verification.refresh_workers,
            'verification.uniqueness_batch_size': self.verification.uniqueness_batch_size,
            'router.failure_quarantine_threshold': self.router.failure_quarantine_threshold,
            'identity.netuid': self.identity.netuid,
            'emission_oracle.minimum_price_sources': self.emission_oracle.minimum_price_sources,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values.values()):
            raise ValueError('GPU counts, sample size, and netuid must be integers')
        if fleet.floor < 1:
            raise ValueError('fleet.floor must be at least 1')
        if fleet.initial_target < fleet.floor:
            raise ValueError('fleet.initial_target cannot be below fleet.floor')
        if fleet.certified_slots_per_gpu < 1:
            raise ValueError('fleet.certified_slots_per_gpu must be at least 1')
        if fleet.target_price_per_gpu_hour <= 0:
            raise ValueError('fleet.target_price_per_gpu_hour must be positive')
        if fleet.target_price_currency not in {'USD', 'TAO'}:
            raise ValueError('fleet.target_price_currency must be USD or TAO')
        if fleet.subnet_miner_emission_value_per_hour <= 0:
            raise ValueError('fleet.subnet_miner_emission_value_per_hour must be positive')
        if not 0 < fleet.max_compute_emission_share <= 0.90:
            raise ValueError('fleet.max_compute_emission_share must be in (0, 0.90]')
        if not 0 < fleet.scarcity_reward_exponent < 1:
            raise ValueError('fleet.scarcity_reward_exponent must be in (0, 1)')
        if fleet.scarcity_multiplier_cap < 1:
            raise ValueError('fleet.scarcity_multiplier_cap must be at least 1')
        minimum_budget = fleet.floor * fleet.target_price_per_gpu_hour
        if fleet.max_budget_per_hour is not None and fleet.max_budget_per_hour < minimum_budget:
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
                self.placement.planning_horizon_seconds,
                self.verification.weight_challenge_interval_seconds,
                self.verification.weight_challenge_ttl_seconds,
                self.verification.weight_verification_ttl_seconds,
                self.verification.uniqueness_challenge_ttl_seconds,
                self.router.telemetry_ttl_seconds,
                self.router.maximum_service_seconds,
                self.identity.signature_ttl_seconds,
                self.identity.metagraph_refresh_seconds,
                self.assignment.request_timeout_seconds,
                self.assignment.redispatch_interval_seconds,
                self.control_loop.verification_interval_seconds,
                self.control_loop.settlement_interval_seconds,
                self.emission_oracle.refresh_interval_seconds,
                self.emission_oracle.max_refresh_staleness_seconds,
                self.emission_oracle.max_epoch_age_seconds,
                self.emission_oracle.request_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError('all intervals and TTLs must be positive')
        if self.placement.minimum_residency_seconds < 0 or self.placement.switch_sustain_seconds < 0:
            raise ValueError('placement residency and sustain periods cannot be negative')
        if self.placement.minimum_switch_gain_gpu < 0:
            raise ValueError('placement.minimum_switch_gain_gpu cannot be negative')
        if self.router.equivalent_finish_epsilon_seconds < 0:
            raise ValueError('router.equivalent_finish_epsilon_seconds cannot be negative')
        if self.router.maximum_service_seconds < self.router.reservation_ttl_seconds:
            raise ValueError('router.maximum_service_seconds cannot be below reservation_ttl_seconds')
        if self.router.failure_quarantine_threshold < 1:
            raise ValueError('router.failure_quarantine_threshold must be positive')
        if self.verification.challenge_sample_size < 1:
            raise ValueError('verification.challenge_sample_size must be positive')
        if self.verification.refresh_workers < 1:
            raise ValueError('verification.refresh_workers must be positive')
        if self.verification.uniqueness_batch_size < 2:
            raise ValueError('verification.uniqueness_batch_size must be at least two')
        oracle = self.emission_oracle
        known_price_sources = {'coinbase', 'coingecko'}
        if not 0 <= oracle.maximum_price_divergence <= 1:
            raise ValueError('emission_oracle.maximum_price_divergence must be in [0, 1]')
        if oracle.minimum_price_sources < 1:
            raise ValueError('emission_oracle.minimum_price_sources must be positive')
        if len(set(oracle.tao_usd_price_sources)) != len(oracle.tao_usd_price_sources):
            raise ValueError('emission oracle price sources must be unique')
        if not set(oracle.tao_usd_price_sources) <= known_price_sources:
            raise ValueError('emission oracle contains an unknown TAO/USD price source')
        if fleet.target_price_currency == 'USD' and oracle.enabled:
            if oracle.minimum_price_sources > len(oracle.tao_usd_price_sources):
                raise ValueError('emission oracle minimum price sources exceeds configured sources')
        if not self.verification.status_url and not self.verification.status_urls:
            raise ValueError('verification requires status_url or status_urls')
        verifier_urls = tuple(self.verification.status_urls) or (self.verification.status_url,)
        if len(set(verifier_urls)) != len(verifier_urls):
            raise ValueError('verification status URLs must be unique')
        if (
            self.verification.require_weight_challenges
            and self.verification.weight_challenge_interval_seconds
            > self.verification.weight_verification_ttl_seconds / 2
        ):
            raise ValueError('weight challenge interval must be at most half the weight verification TTL')
        if self.verification.require_verifier_measurement and not self.verification.expected_verifier_measurement:
            raise ValueError('strict verifier mode requires expected_verifier_measurement')
        if self.verification.require_verifier_measurement:
            if not _COMMIT_PATTERN.fullmatch(self.verification.source_commit):
                raise ValueError('strict verifier source_commit must be an immutable 40-character commit')
            if not _SHA256_PATTERN.fullmatch(self.verification.expected_verifier_measurement):
                raise ValueError('strict expected_verifier_measurement must be a lowercase sha256 digest')
            if self.verification.expected_verifier_measurement == f'sha256:{"0" * 64}':
                raise ValueError('strict expected_verifier_measurement must be replaced with the deployed digest')
            if not self.verification.require_status_signature:
                raise ValueError('strict verifier measurement requires signed verifier status responses')
        if self.verification.require_status_signature:
            if not isinstance(self.verification.trusted_verifier_public_keys, (tuple, list)):
                raise ValueError('trusted verifier public keys must be a list')
            verifier_keys = tuple(self.verification.trusted_verifier_public_keys)
            if not all(isinstance(key, str) for key in verifier_keys):
                raise ValueError('trusted verifier public keys must be strings')
            if not verifier_keys:
                raise ValueError('signed verifier status requires at least one trusted public key')
            if len(set(verifier_keys)) != len(verifier_keys):
                raise ValueError('trusted verifier public keys must be unique')
            if any(not _PUBLIC_KEY_PATTERN.fullmatch(key) or key == '0' * 64 for key in verifier_keys):
                raise ValueError('trusted verifier public keys must be nonzero lowercase 32-byte hex keys')
        if self.verification.require_stream_proof and not self.verification.require_runtime_attestation:
            raise ValueError('stream proof mode requires runtime attestation')
        if self.verification.require_runtime_attestation and not self.verification.require_verifier_measurement:
            raise ValueError('runtime attestation requires a pinned verifier measurement')
        if self.verification.require_uniqueness_challenge and not self.verification.require_verifier_measurement:
            raise ValueError('uniqueness challenge requires a pinned verifier measurement')
        if self.verification.require_confidential_compute and not self.verification.require_runtime_attestation:
            raise ValueError('confidential compute mode requires runtime attestation')
        if self.verification.require_signed_containers and not self.verification.cosign_public_key_path:
            raise ValueError('signed container mode requires cosign_public_key_path')
        if self.identity.netuid < 0:
            raise ValueError('identity.netuid cannot be negative')


def load_compute_config(path: str | Path) -> ComputeConfig:
    return ComputeConfig.from_mapping(load_json_object(Path(path).read_bytes()))
