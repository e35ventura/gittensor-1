"""Atomic, idle-first, fastest-finish request routing."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Iterable


class CapacityUnavailable(RuntimeError):
    """Raised immediately when no compatible GPU has a free certified slot."""


@dataclass(frozen=True)
class RoutingGPU:
    gpu_id: str
    endpoint: str
    release_digest: str
    performance_class: str
    certified_slots: int
    reported_active_slots: int
    remaining_work_seconds: float
    service_seconds: float
    rtt_ms: float


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    gpu_id: str
    release_digest: str
    service_seconds: float
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class RouteDecision:
    reservation_id: str
    gpu_id: str
    endpoint: str
    expected_completion_seconds: float
    expires_at: float


class FastestFinishRouter:
    """Reserve one GPU without creating an internal request queue."""

    def __init__(self, reservation_ttl_seconds: float) -> None:
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self._reservations: dict[str, Reservation] = {}
        self._lock = threading.Lock()

    def route(
        self,
        candidates: Iterable[RoutingGPU],
        release_digest: str,
        now: float,
    ) -> RouteDecision:
        with self._lock:
            self._expire(now)
            local = self._active_by_gpu()
            ranked: list[tuple[tuple[float, ...], RoutingGPU, int, float]] = []
            for gpu in candidates:
                if gpu.release_digest != release_digest:
                    continue
                active = gpu.reported_active_slots + local.get(gpu.gpu_id, 0)
                if active >= gpu.certified_slots:
                    continue
                locally_reserved_work = sum(
                    reservation.service_seconds
                    for reservation in self._reservations.values()
                    if reservation.gpu_id == gpu.gpu_id
                )
                expected_completion = (
                    gpu.rtt_ms / 1000.0
                    + max(0.0, gpu.remaining_work_seconds)
                    + locally_reserved_work
                    + max(0.0, gpu.service_seconds)
                )
                # For equivalent RTX 5090s, spread one request to every idle GPU
                # before stacking requests. ECT decides within each load tier.
                rank = (
                    0.0 if active == 0 else 1.0,
                    expected_completion,
                    float(active),
                )
                ranked.append((rank, gpu, active, expected_completion))
            if not ranked:
                raise CapacityUnavailable('no compatible READY GPU has a free certified slot')
            _, selected, _, expected_completion = min(ranked, key=lambda item: (item[0], item[1].gpu_id))
            reservation_id = uuid.uuid4().hex
            expires_at = now + self.reservation_ttl_seconds
            self._reservations[reservation_id] = Reservation(
                reservation_id=reservation_id,
                gpu_id=selected.gpu_id,
                release_digest=release_digest,
                service_seconds=selected.service_seconds,
                created_at=now,
                expires_at=expires_at,
            )
            return RouteDecision(
                reservation_id=reservation_id,
                gpu_id=selected.gpu_id,
                endpoint=selected.endpoint,
                expected_completion_seconds=expected_completion,
                expires_at=expires_at,
            )

    def complete(self, reservation_id: str) -> bool:
        with self._lock:
            return self._reservations.pop(reservation_id, None) is not None

    def active_counts(self, now: float) -> dict[str, int]:
        with self._lock:
            self._expire(now)
            return self._active_by_gpu()

    def _expire(self, now: float) -> None:
        expired = [key for key, value in self._reservations.items() if value.expires_at <= now]
        for key in expired:
            self._reservations.pop(key, None)

    def _active_by_gpu(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reservation in self._reservations.values():
            counts[reservation.gpu_id] = counts.get(reservation.gpu_id, 0) + 1
        return counts
