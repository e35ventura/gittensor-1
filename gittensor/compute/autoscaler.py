"""Chutes-style target scaling over the global GPU fleet."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gittensor.compute.config import AutoscalingConfig


@dataclass(frozen=True)
class AutoscaleDecision:
    desired_target: int
    utilization: float
    concurrent_demand: float
    rejection_demand_ewma: float
    required_target: int
    supply_shortage: float
    changed: bool
    reason: str


class FleetAutoscaler:
    """Adjust the desired target after sustained high or low utilization.

    Accepted and rejected work arrive as time-weighted GPU equivalents. A
    request consumes the larger of its concurrency share and KV-cache share,
    so releases with different safe capacities remain comparable. The target
    has a floor and intentionally has no product-level maximum.
    """

    def __init__(
        self,
        config: AutoscalingConfig,
        floor: int,
        initial_target: int,
    ) -> None:
        self.config = config
        self.floor = floor
        self.desired_target = max(floor, initial_target)
        self.rejection_demand_ewma = 0.0
        self.high_since: float | None = None
        self.low_since: float | None = None
        self.last_scaled_at: float | None = None

    def update(
        self,
        *,
        active_gpu_equivalents: float,
        rejected_gpu_equivalents: float,
        funded_target: int,
        now: float,
    ) -> AutoscaleDecision:
        alpha = self.config.ewma_alpha
        self.rejection_demand_ewma = (
            alpha * max(0.0, rejected_gpu_equivalents) + (1.0 - alpha) * self.rejection_demand_ewma
        )
        demand = max(0.0, float(active_gpu_equivalents)) + self.rejection_demand_ewma
        capacity = max(1, funded_target)
        utilization = demand / capacity
        desired_capacity = max(1, self.desired_target)
        desired_utilization = demand / desired_capacity
        required_capacity = demand / self.config.utilization_up if demand > 0 else 0.0
        required_target = max(self.floor, math.floor(required_capacity) + 1) if demand > 0 else self.floor
        supply_shortage = max(0.0, required_capacity - funded_target)
        changed = False
        reason = 'inside hysteresis band'

        if utilization >= self.config.utilization_up and required_target > self.desired_target:
            self.low_since = None
            self.high_since = self.high_since if self.high_since is not None else now
            if now - self.high_since >= self.config.sustain_up_seconds:
                new_target = required_target
                changed = new_target != self.desired_target
                self.desired_target = new_target
                self.last_scaled_at = now
                self.high_since = None
                reason = 'sustained high utilization'
            else:
                reason = 'high utilization is not yet sustained'
        elif desired_utilization <= self.config.utilization_down:
            self.high_since = None
            self.low_since = self.low_since if self.low_since is not None else now
            cooldown_ready = self.last_scaled_at is None or now - self.last_scaled_at >= self.config.cooldown_seconds
            if now - self.low_since >= self.config.sustain_down_seconds and cooldown_ready:
                new_target = max(self.floor, self.desired_target - 1)
                changed = new_target != self.desired_target
                self.desired_target = new_target
                if changed:
                    self.last_scaled_at = now
                self.low_since = None
                reason = 'sustained low utilization'
            elif not cooldown_ready:
                reason = 'low utilization is inside scale-down cooldown'
            else:
                reason = 'low utilization is not yet sustained'
        elif utilization >= self.config.utilization_up:
            self.high_since = None
            self.low_since = None
            reason = 'desired target already covers measured demand; funding is constrained'
        else:
            self.high_since = None
            self.low_since = None

        return AutoscaleDecision(
            desired_target=self.desired_target,
            utilization=utilization,
            concurrent_demand=demand,
            rejection_demand_ewma=self.rejection_demand_ewma,
            required_target=required_target,
            supply_shortage=supply_shortage,
            changed=changed,
            reason=reason,
        )
