"""Atomic expected-completion-time request routing."""

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

    def __init__(self, reservation_ttl_seconds: float, equivalent_finish_epsilon_seconds: float = 0.025) -> None:
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.equivalent_finish_epsilon_seconds = equivalent_finish_epsilon_seconds
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
            ranked: list[tuple[RoutingGPU, int, float]] = []
            for gpu in candidates:
                if gpu.release_digest != release_digest:
                    continue
                gpu_reservations = [
                    reservation for reservation in self._reservations.values() if reservation.gpu_id == gpu.gpu_id
                ]
                locally_active = local.get(gpu.gpu_id, 0)
                active = max(gpu.reported_active_slots, locally_active)
                if active >= gpu.certified_slots:
                    continue
                # Fresh gateway telemetry accounts for requests already running.
                # Only reservations not yet visible to that observation are added.
                unobserved_count = max(0, locally_active - gpu.reported_active_slots)
                unobserved = (
                    sorted(gpu_reservations, key=lambda item: item.created_at)[-unobserved_count:]
                    if unobserved_count
                    else []
                )
                locally_reserved_work = sum(reservation.service_seconds for reservation in unobserved)
                expected_completion = (
                    gpu.rtt_ms / 1000.0
                    + max(0.0, gpu.remaining_work_seconds)
                    + locally_reserved_work
                    + max(0.0, gpu.service_seconds)
                )
                ranked.append((gpu, active, expected_completion))
            if not ranked:
                raise CapacityUnavailable('no compatible READY GPU has a free certified slot')
            fastest = min(item[2] for item in ranked)
            equivalent = [item for item in ranked if item[2] <= fastest + self.equivalent_finish_epsilon_seconds]
            # Completion time is the primary rule. Within a genuinely equivalent
            # finish-time band, prefer fewer active requests so equivalent local
            # GPUs each receive one request before any is stacked.
            selected, _, expected_completion = min(
                equivalent,
                key=lambda item: (item[1], item[2], item[0].gpu_id),
            )
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

    def export_state(self, now: float) -> list[dict[str, str | float]]:
        with self._lock:
            self._expire(now)
            return [
                {
                    'reservation_id': value.reservation_id,
                    'gpu_id': value.gpu_id,
                    'release_digest': value.release_digest,
                    'service_seconds': value.service_seconds,
                    'created_at': value.created_at,
                    'expires_at': value.expires_at,
                }
                for value in sorted(self._reservations.values(), key=lambda item: item.reservation_id)
            ]

    def restore_state(self, reservations: Iterable[dict[str, str | float]], now: float) -> None:
        with self._lock:
            self._reservations = {
                str(value['reservation_id']): Reservation(
                    reservation_id=str(value['reservation_id']),
                    gpu_id=str(value['gpu_id']),
                    release_digest=str(value['release_digest']),
                    service_seconds=float(value['service_seconds']),
                    created_at=float(value['created_at']),
                    expires_at=float(value['expires_at']),
                )
                for value in reservations
                if float(value['expires_at']) > now
            }

    def _expire(self, now: float) -> None:
        expired = [key for key, value in self._reservations.items() if value.expires_at <= now]
        for key in expired:
            self._reservations.pop(key, None)

    def _active_by_gpu(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reservation in self._reservations.values():
            counts[reservation.gpu_id] = counts.get(reservation.gpu_id, 0) + 1
        return counts
