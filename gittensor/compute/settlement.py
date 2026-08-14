"""Lium-style target-price funding and READY-second settlement."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True)
class FundingPlan:
    desired_target: int
    funded_target: int
    target_price_per_gpu_hour: Decimal
    max_budget_per_hour: Decimal
    funding_shortfall: int


@dataclass(frozen=True)
class SettlementResult:
    funding: FundingPlan
    window_seconds: float
    window_budget: Decimal
    total_ready_seconds: float
    effective_ready_gpus: float
    gpu_rewards: Mapping[str, Decimal]
    unspent_budget: Decimal
    distributed_budget: Decimal
    effective_funded_gpus: float
    scarcity_multiplier: float


def funding_plan(
    desired_target: int,
    target_price_per_gpu_hour: float | Decimal,
    max_budget_per_hour: float | Decimal,
) -> FundingPlan:
    price = Decimal(str(target_price_per_gpu_hour))
    budget = Decimal(str(max_budget_per_hour))
    if desired_target < 0 or price <= 0 or budget < 0:
        raise ValueError('funding inputs must be non-negative and price must be positive')
    funded = min(desired_target, int(budget // price))
    return FundingPlan(
        desired_target=desired_target,
        funded_target=funded,
        target_price_per_gpu_hour=price,
        max_budget_per_hour=budget,
        funding_shortfall=max(0, desired_target - funded),
    )


def settle_ready_seconds(
    funding: FundingPlan,
    window_seconds: float,
    ready_seconds_by_gpu: Mapping[str, float],
    *,
    window_budget_override: Decimal | None = None,
    scarcity_reward_exponent: float = 0.5,
    scarcity_multiplier_cap: float = 2.0,
) -> SettlementResult:
    if window_seconds <= 0:
        raise ValueError('window_seconds must be positive')
    if not 0 < scarcity_reward_exponent < 1:
        raise ValueError('scarcity_reward_exponent must be in (0, 1)')
    if scarcity_multiplier_cap < 1:
        raise ValueError('scarcity_multiplier_cap must be at least 1')
    clean_seconds = {
        gpu_id: min(window_seconds, max(0.0, float(seconds))) for gpu_id, seconds in ready_seconds_by_gpu.items()
    }
    total_ready_seconds = sum(clean_seconds.values())
    window_budget = window_budget_override
    if window_budget is None:
        window_budget = (
            Decimal(funding.funded_target)
            * funding.target_price_per_gpu_hour
            * Decimal(str(window_seconds))
            / Decimal(3600)
        )
    elif window_budget < 0:
        raise ValueError('window_budget_override must be non-negative')
    window_hours = Decimal(str(window_seconds)) / Decimal(3600)
    funded_denominator = funding.target_price_per_gpu_hour * window_hours
    effective_funded_gpus = float(window_budget / funded_denominator) if funded_denominator > 0 else 0.0
    effective_ready_gpus = total_ready_seconds / window_seconds
    scarcity_multiplier = 1.0
    if total_ready_seconds == 0:
        distributed_budget = Decimal(0)
        rewards = {gpu_id: Decimal(0) for gpu_id in clean_seconds}
        unspent = window_budget
    else:
        if 0 < effective_ready_gpus < effective_funded_gpus:
            scarcity_multiplier = min(
                scarcity_multiplier_cap,
                (effective_funded_gpus / effective_ready_gpus) ** scarcity_reward_exponent,
            )
        supply_value = (
            funding.target_price_per_gpu_hour
            * Decimal(str(effective_ready_gpus))
            * window_hours
            * Decimal(str(scarcity_multiplier))
        )
        distributed_budget = min(window_budget, supply_value)
        denominator = Decimal(str(total_ready_seconds))
        rewards = {
            gpu_id: distributed_budget * Decimal(str(seconds)) / denominator
            for gpu_id, seconds in clean_seconds.items()
        }
        unspent = window_budget - distributed_budget
    return SettlementResult(
        funding=funding,
        window_seconds=window_seconds,
        window_budget=window_budget,
        total_ready_seconds=total_ready_seconds,
        effective_ready_gpus=effective_ready_gpus,
        gpu_rewards=rewards,
        unspent_budget=unspent,
        distributed_budget=distributed_budget,
        effective_funded_gpus=effective_funded_gpus,
        scarcity_multiplier=scarcity_multiplier,
    )


def aggregate_miner_rewards(
    result: SettlementResult,
    miner_uid_by_gpu: Mapping[str, int],
) -> dict[int, Decimal]:
    rewards: dict[int, Decimal] = {}
    for gpu_id, amount in result.gpu_rewards.items():
        miner_uid = miner_uid_by_gpu[gpu_id]
        rewards[miner_uid] = rewards.get(miner_uid, Decimal(0)) + amount
    return rewards
