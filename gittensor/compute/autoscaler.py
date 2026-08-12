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
    changed: bool
    reason: str


class FleetAutoscaler:
    """Adjust the desired target after sustained high or low utilization.

    Rejections are converted to concurrent-equivalent demand using
    ``rejected_requests_per_second * expected_service_seconds``. The target has
    a floor and intentionally has no product-level maximum.
    """

    def __init__(
        self,
        config: AutoscalingConfig,
        floor: int,
        certified_slots_per_gpu: int,
        initial_target: int,
    ) -> None:
        self.config = config
        self.floor = floor
        self.certified_slots_per_gpu = certified_slots_per_gpu
        self.desired_target = max(floor, initial_target)
        self.rejection_demand_ewma = 0.0
        self.high_since: float | None = None
        self.low_since: float | None = None
        self.last_scaled_at: float | None = None

    def update(
        self,
        *,
        active_slots: int,
        rejected_concurrent_demand: float,
        funded_target: int,
        now: float,
    ) -> AutoscaleDecision:
        alpha = self.config.ewma_alpha
        self.rejection_demand_ewma = (
            alpha * max(0.0, rejected_concurrent_demand) + (1.0 - alpha) * self.rejection_demand_ewma
        )
        demand = max(0.0, float(active_slots)) + self.rejection_demand_ewma
        capacity = max(1, funded_target * self.certified_slots_per_gpu)
        utilization = demand / capacity
        changed = False
        reason = 'inside hysteresis band'

        if utilization >= self.config.utilization_up:
            self.low_since = None
            self.high_since = self.high_since if self.high_since is not None else now
            if now - self.high_since >= self.config.sustain_up_seconds:
                required = math.ceil(demand / (self.certified_slots_per_gpu * self.config.utilization_up))
                new_target = max(self.desired_target + 1, required, self.floor)
                changed = new_target != self.desired_target
                self.desired_target = new_target
                self.last_scaled_at = now
                self.high_since = None
                reason = 'sustained high utilization'
            else:
                reason = 'high utilization is not yet sustained'
        elif utilization <= self.config.utilization_down:
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
        else:
            self.high_since = None
            self.low_since = None

        return AutoscaleDecision(
            desired_target=self.desired_target,
            utilization=utilization,
            concurrent_demand=demand,
            rejection_demand_ewma=self.rejection_demand_ewma,
            changed=changed,
            reason=reason,
        )
