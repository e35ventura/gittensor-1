"""Global Gepetto release placement over every verified GPU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from gittensor.compute.models import GPURecord, PlacementTransition, Release


@dataclass(frozen=True)
class ReleaseDemand:
    release_digest: str
    concurrent_demand: float


@dataclass(frozen=True)
class PlacementPlan:
    assignments: Mapping[str, str]
    replica_counts: Mapping[str, int]
    transitions: Sequence[PlacementTransition]


class GlobalGepetto:
    """Produce one subnet-level ``gpu_id -> release_digest`` assignment map."""

    def __init__(self, minimum_residency_seconds: float) -> None:
        self.minimum_residency_seconds = minimum_residency_seconds

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
            return PlacementPlan({}, {}, ())

        demand_by_release = {item.release_digest: max(0.0, item.concurrent_demand) for item in demand}
        quotas = self._replica_quotas(len(gpu_list), release_list, demand_by_release)
        assignments: dict[str, str] = {}
        remaining = dict(quotas)

        # Locked assignments remain in place until minimum residency expires.
        for gpu in gpu_list:
            current = gpu.registration.release_digest
            locked = now - gpu.assignment_started_at < self.minimum_residency_seconds
            if locked and current in remaining:
                assignments[gpu.registration.gpu_id] = current
                remaining[current] = max(0, remaining[current] - 1)

        # Keep existing assignments where they still fit the target map.
        for gpu in gpu_list:
            if gpu.registration.gpu_id in assignments:
                continue
            current = gpu.registration.release_digest
            if remaining.get(current, 0) > 0:
                assignments[gpu.registration.gpu_id] = current
                remaining[current] -= 1

        unfilled = [release_digest for release_digest, count in sorted(remaining.items()) for _ in range(count)]
        for gpu, release_digest in zip(
            (gpu for gpu in gpu_list if gpu.registration.gpu_id not in assignments),
            unfilled,
        ):
            assignments[gpu.registration.gpu_id] = release_digest

        transitions = tuple(
            PlacementTransition(
                gpu_id=gpu.registration.gpu_id,
                from_release=gpu.registration.release_digest,
                to_release=assignments[gpu.registration.gpu_id],
            )
            for gpu in gpu_list
            if assignments.get(gpu.registration.gpu_id) != gpu.registration.release_digest
        )
        replica_counts = {
            release.release_digest: sum(1 for value in assignments.values() if value == release.release_digest)
            for release in release_list
        }
        return PlacementPlan(assignments, replica_counts, transitions)

    @staticmethod
    def _replica_quotas(
        total_gpus: int,
        releases: Sequence[Release],
        demand: Mapping[str, float],
    ) -> dict[str, int]:
        quotas = {release.release_digest: 0 for release in releases}
        remaining = total_gpus

        # Minimums are filled in stable priority order when supply is scarce.
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
        raw = {release_digest: remaining * weight / weight_total for release_digest, weight in weights.items()}
        floors = {release_digest: math.floor(value) for release_digest, value in raw.items()}
        for release_digest, count in floors.items():
            quotas[release_digest] += count
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
