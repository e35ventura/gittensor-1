"""Global Gepetto release placement over every verified GPU."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from gittensor.compute.models import GPURecord, GPUState, PlacementTransition, Release


@dataclass(frozen=True)
class ReleaseDemand:
    release_digest: str
    gpu_equivalent_demand: float


@dataclass(frozen=True)
class PlacementPlan:
    assignments: Mapping[str, str]
    replica_counts: Mapping[str, int]
    transitions: Sequence[PlacementTransition]
    shortages: Mapping[str, float] = field(default_factory=dict)
    deferred: Mapping[str, str] = field(default_factory=dict)


class GlobalGepetto:
    """Produce one subnet-level ``gpu_id -> release_digest`` assignment map.

    Supply scaling is deliberately outside this class. Gepetto sees a fixed
    verified fleet and repairs only release placement shortages. A demand-driven
    switch must survive the configured sustain period and recover more
    capacity-seconds over the planning horizon than it costs to drain and load.
    """

    def __init__(
        self,
        minimum_residency_seconds: float,
        *,
        switch_sustain_seconds: float = 0.0,
        planning_horizon_seconds: float = 900.0,
        minimum_switch_gain_gpu: float = 0.0,
        target_utilization: float = 1.0,
    ) -> None:
        self.minimum_residency_seconds = minimum_residency_seconds
        self.switch_sustain_seconds = switch_sustain_seconds
        self.planning_horizon_seconds = planning_horizon_seconds
        self.minimum_switch_gain_gpu = minimum_switch_gain_gpu
        self.target_utilization = target_utilization
        self._shortage_since: dict[str, float] = {}

    def plan(
        self,
        gpus: Iterable[GPURecord],
        releases: Iterable[Release],
        demand: Iterable[ReleaseDemand],
        now: float,
    ) -> PlacementPlan:
        gpu_list = sorted(gpus, key=lambda gpu: gpu.registration.gpu_id)
        release_list = sorted(releases, key=lambda release: release.release_digest)
        if not gpu_list or not release_list:
            self._shortage_since.clear()
            return PlacementPlan({}, {}, ())

        release_by_digest = {release.release_digest: release for release in release_list}
        demand_by_release = {item.release_digest: max(0.0, item.gpu_equivalent_demand) for item in demand}
        ready_counts = {
            release.release_digest: sum(
                1
                for gpu in gpu_list
                if gpu.state == GPUState.READY and gpu.registration.release_digest == release.release_digest
            )
            for release in release_list
        }
        required_capacity = {
            release.release_digest: max(
                float(release.minimum_replicas),
                demand_by_release.get(release.release_digest, 0.0) / self.target_utilization,
            )
            for release in release_list
        }
        shortages = {
            digest: max(0.0, required - ready_counts.get(digest, 0)) for digest, required in required_capacity.items()
        }
        self._update_shortage_timers(shortages, release_by_digest, ready_counts, now)

        quotas = self._replica_quotas(len(gpu_list), release_list, required_capacity)
        assignments: dict[str, str] = {}
        remaining = dict(quotas)

        # Locked assignments remain in place until minimum residency expires.
        for gpu in gpu_list:
            current = gpu.registration.release_digest
            locked = now - gpu.assignment_started_at < self.minimum_residency_seconds
            if locked and current in remaining:
                assignments[gpu.registration.gpu_id] = current
                remaining[current] = max(0, remaining[current] - 1)

        # Keep existing assignments where they still fit the desired map.
        for gpu in gpu_list:
            if gpu.registration.gpu_id in assignments:
                continue
            current = gpu.registration.release_digest
            if remaining.get(current, 0) > 0:
                assignments[gpu.registration.gpu_id] = current
                remaining[current] -= 1

        unfilled = [
            digest
            for digest, count in sorted(
                remaining.items(),
                key=lambda item: (-shortages.get(item[0], 0.0), item[0]),
            )
            for _ in range(count)
        ]
        available = [gpu for gpu in gpu_list if gpu.registration.gpu_id not in assignments]
        transitions: list[PlacementTransition] = []
        deferred: dict[str, str] = {}
        planned_additions: dict[str, int] = {}
        for target_digest in unfilled:
            if not available:
                break
            target = release_by_digest[target_digest]
            candidate = min(
                available,
                key=lambda gpu: (
                    bool(gpu.registration.release_digest),
                    max(0.0, gpu.gateway_remaining_work_seconds) + target.estimated_load_seconds,
                    gpu.registration.gpu_id,
                ),
            )
            available.remove(candidate)
            current = candidate.registration.release_digest
            addition_index = planned_additions.get(target_digest, 0)
            marginal_shortage = max(0.0, shortages[target_digest] - addition_index)
            minimum_gap = max(0, target.minimum_replicas - ready_counts.get(target_digest, 0))
            allowed, reason = self._switch_allowed(
                candidate,
                target,
                marginal_shortage,
                addition_index < minimum_gap,
                now,
            )
            if current and current != target_digest and not allowed:
                assignments[candidate.registration.gpu_id] = current
                deferred.setdefault(target_digest, reason)
                continue
            assignments[candidate.registration.gpu_id] = target_digest
            if current != target_digest:
                planned_additions[target_digest] = addition_index + 1
                transitions.append(
                    PlacementTransition(
                        gpu_id=candidate.registration.gpu_id,
                        from_release=current or None,
                        to_release=target_digest,
                    )
                )

        # A deferred switch never makes hardware disappear from the global map.
        fallback = max(release_list, key=lambda release: (release.placement_weight, release.release_digest))
        for gpu in available:
            current = gpu.registration.release_digest
            assignments[gpu.registration.gpu_id] = current if current in release_by_digest else fallback.release_digest

        replica_counts = {
            release.release_digest: sum(1 for value in assignments.values() if value == release.release_digest)
            for release in release_list
        }
        return PlacementPlan(assignments, replica_counts, tuple(transitions), shortages, deferred)

    def export_state(self) -> dict[str, float]:
        return dict(self._shortage_since)

    def restore_state(self, shortage_since: Mapping[str, float]) -> None:
        self._shortage_since = {
            str(digest): float(started_at)
            for digest, started_at in shortage_since.items()
            if math.isfinite(float(started_at))
        }

    def _update_shortage_timers(
        self,
        shortages: Mapping[str, float],
        releases: Mapping[str, Release],
        ready_counts: Mapping[str, int],
        now: float,
    ) -> None:
        active: set[str] = set()
        for digest, shortage in shortages.items():
            minimum_missing = ready_counts.get(digest, 0) < releases[digest].minimum_replicas
            if shortage > self.minimum_switch_gain_gpu or minimum_missing:
                active.add(digest)
                self._shortage_since.setdefault(digest, now)
        for digest in tuple(self._shortage_since):
            if digest not in active:
                self._shortage_since.pop(digest, None)

    def _switch_allowed(
        self,
        gpu: GPURecord,
        target: Release,
        shortage: float,
        minimum_missing: bool,
        now: float,
    ) -> tuple[bool, str]:
        if not gpu.registration.release_digest:
            return True, ''
        if minimum_missing:
            return True, ''
        if shortage <= self.minimum_switch_gain_gpu:
            return False, 'placement gain is inside the hysteresis band'
        shortage_since = self._shortage_since.get(target.release_digest, now)
        if now - shortage_since < self.switch_sustain_seconds:
            return False, 'placement shortage is not yet sustained'
        benefit_capacity_seconds = min(1.0, shortage) * self.planning_horizon_seconds
        switching_cost_seconds = max(0.0, gpu.gateway_remaining_work_seconds) + target.estimated_load_seconds
        if benefit_capacity_seconds <= switching_cost_seconds:
            return False, 'switching cost exceeds the planning-horizon benefit'
        return True, ''

    @staticmethod
    def _replica_quotas(
        total_gpus: int,
        releases: Sequence[Release],
        demand: Mapping[str, float],
    ) -> dict[str, int]:
        quotas = {release.release_digest: 0 for release in releases}
        remaining = total_gpus

        priority = sorted(releases, key=lambda item: (-item.placement_weight, item.release_digest))
        while remaining > 0 and any(quotas[item.release_digest] < item.minimum_replicas for item in priority):
            for release in priority:
                if remaining <= 0:
                    break
                if quotas[release.release_digest] < release.minimum_replicas:
                    quotas[release.release_digest] += 1
                    remaining -= 1

        if remaining <= 0:
            return quotas

        weights = {
            release.release_digest: max(0.0, demand.get(release.release_digest, 0.0)) * release.placement_weight
            for release in releases
        }
        if sum(weights.values()) == 0:
            weights[priority[0].release_digest] = 1.0
        weight_total = sum(weights.values())
        raw = {digest: remaining * weight / weight_total for digest, weight in weights.items()}
        floors = {digest: math.floor(value) for digest, value in raw.items()}
        for digest, count in floors.items():
            quotas[digest] += count
        leftovers = remaining - sum(floors.values())
        remainder_order = sorted(
            releases,
            key=lambda release: (
                -(raw[release.release_digest] - floors[release.release_digest]),
                -release.placement_weight,
                release.release_digest,
            ),
        )
        for release in remainder_order[:leftovers]:
            quotas[release.release_digest] += 1
        return quotas
