"""Atomic expected-completion-time request routing."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Iterable

from gittensor.compute.storage import SQLiteStateStore


class CapacityUnavailable(RuntimeError):
    """Raised immediately when no compatible GPU has certified request capacity."""


@dataclass(frozen=True)
class RoutingGPU:
    gpu_id: str
    endpoint: str
    release_digest: str
    performance_class: str
    certified_slots: int
    observed_active_slots: int
    remaining_work_seconds: float
    service_seconds: float
    rtt_ms: float
    stream_public_key: str = ''
    eligible_until: float = float('inf')
    kv_cache_capacity_bytes: int = 131_072


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    gpu_id: str
    release_digest: str
    service_seconds: float
    created_at: float
    expires_at: float
    kv_bytes: int = 0
    capacity_units: float = 0.0


@dataclass(frozen=True)
class RouteDecision:
    reservation_id: str
    gpu_id: str
    endpoint: str
    expected_completion_seconds: float
    expires_at: float
    stream_public_key: str = ''
    inference_token: str = ''
    reserved_kv_bytes: int = 0
    capacity_units: float = 0.0


class FastestFinishRouter:
    """Reserve one GPU without creating an internal request queue."""

    def __init__(self, reservation_ttl_seconds: float, equivalent_finish_epsilon_seconds: float = 0.025) -> None:
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.equivalent_finish_epsilon_seconds = equivalent_finish_epsilon_seconds
        self._reservations: dict[str, Reservation] = {}
        self._expired_reservations: list[Reservation] = []
        self._lock = threading.Lock()
        self._store: SQLiteStateStore | None = None

    def attach_store(self, store: SQLiteStateStore | None) -> None:
        """Use transactional reservation rows instead of whole-state checkpoints."""
        self._store = store
        if store is not None:
            self.restore_state(store.load_reservations(), float('-inf'))

    def route(
        self,
        candidates: Iterable[RoutingGPU],
        release_digest: str,
        now: float,
        *,
        request_kv_bytes: int = 1,
        request_capacity_units: float = 0.25,
    ) -> RouteDecision:
        if not isinstance(request_kv_bytes, int) or isinstance(request_kv_bytes, bool) or request_kv_bytes < 1:
            raise ValueError('request_kv_bytes must be a positive integer')
        if not math.isfinite(request_capacity_units) or not 0 < request_capacity_units <= 1:
            raise ValueError('request_capacity_units must be in (0, 1]')
        with self._lock:
            self._expire(now)
            local = self._active_by_gpu()
            local_kv = self._active_kv_by_gpu()
            ranked: list[tuple[RoutingGPU, int, float]] = []
            for gpu in candidates:
                if gpu.release_digest != release_digest:
                    continue
                locally_active = local.get(gpu.gpu_id, 0)
                active = max(gpu.observed_active_slots, locally_active)
                if active >= gpu.certified_slots:
                    continue
                if local_kv.get(gpu.gpu_id, 0) + request_kv_bytes > gpu.kv_cache_capacity_bytes:
                    continue
                expected_completion = (
                    gpu.rtt_ms / 1000.0 + max(0.0, gpu.remaining_work_seconds) + max(0.0, gpu.service_seconds)
                )
                ranked.append((gpu, active, expected_completion))
            if not ranked:
                raise CapacityUnavailable('no compatible READY GPU has free concurrency and KV capacity')
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
            expires_at = min(
                now + max(self.reservation_ttl_seconds, selected.service_seconds + 30.0),
                selected.eligible_until,
            )
            reservation = Reservation(
                reservation_id=reservation_id,
                gpu_id=selected.gpu_id,
                release_digest=release_digest,
                service_seconds=selected.service_seconds,
                created_at=now,
                expires_at=expires_at,
                kv_bytes=request_kv_bytes,
                capacity_units=request_capacity_units,
            )
            self._reservations[reservation_id] = reservation
            if self._store is not None:
                try:
                    self._store.create_reservation(
                        reservation.reservation_id,
                        reservation.gpu_id,
                        reservation.release_digest,
                        reservation.service_seconds,
                        reservation.created_at,
                        reservation.expires_at,
                        reservation.kv_bytes,
                        reservation.capacity_units,
                    )
                except Exception:
                    self._reservations.pop(reservation_id, None)
                    raise
            return RouteDecision(
                reservation_id=reservation_id,
                gpu_id=selected.gpu_id,
                endpoint=selected.endpoint,
                expected_completion_seconds=expected_completion,
                expires_at=expires_at,
                stream_public_key=selected.stream_public_key,
                reserved_kv_bytes=request_kv_bytes,
                capacity_units=request_capacity_units,
            )

    def complete(self, reservation_id: str) -> bool:
        with self._lock:
            completed = self._reservations.pop(reservation_id, None) is not None
            if self._store is not None:
                completed = self._store.complete_reservation(reservation_id) or completed
            return completed

    def detach(self, reservation_id: str) -> Reservation | None:
        """Remove one reservation only from memory for an outer atomic commit."""
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is not None:
                return reservation
            for index, expired in enumerate(self._expired_reservations):
                if expired.reservation_id == reservation_id:
                    return self._expired_reservations.pop(index)
            return None

    def reattach(self, reservation: Reservation) -> None:
        """Restore a detached reservation after an outer transaction rolls back."""
        with self._lock:
            self._reservations[reservation.reservation_id] = reservation

    def detach_gpu(self, gpu_id: str) -> tuple[Reservation, ...]:
        """Remove one GPU's reservations from memory before an atomic state commit."""
        with self._lock:
            active = tuple(reservation for reservation in self._reservations.values() if reservation.gpu_id == gpu_id)
            expired = tuple(reservation for reservation in self._expired_reservations if reservation.gpu_id == gpu_id)
            for reservation in active:
                self._reservations.pop(reservation.reservation_id, None)
            if expired:
                expired_ids = {reservation.reservation_id for reservation in expired}
                self._expired_reservations = [
                    reservation
                    for reservation in self._expired_reservations
                    if reservation.reservation_id not in expired_ids
                ]
            return active + expired

    def renew(self, reservation_id: str, now: float, *, eligible_until: float = float('inf')) -> float | None:
        """Extend a still-live reservation and its durable row atomically."""
        with self._lock:
            self._expire(now)
            current = self._reservations.get(reservation_id)
            if current is None:
                return None
            renewed = replace(
                current,
                expires_at=min(
                    max(current.expires_at, now + self.reservation_ttl_seconds),
                    eligible_until,
                ),
            )
            if self._store is not None and not self._store.renew_reservation(
                reservation_id,
                renewed.expires_at,
                now,
            ):
                self._reservations.pop(reservation_id, None)
                return None
            self._reservations[reservation_id] = renewed
            return renewed.expires_at

    def active_counts(self, now: float) -> dict[str, int]:
        with self._lock:
            self._expire(now)
            return self._active_by_gpu()

    def active_capacity_by_release(self, now: float) -> dict[str, float]:
        with self._lock:
            self._expire(now)
            totals: dict[str, float] = {}
            for reservation in self._reservations.values():
                totals[reservation.release_digest] = totals.get(reservation.release_digest, 0.0) + max(
                    0.0,
                    reservation.capacity_units,
                )
            return totals

    def active_reservations(self, now: float) -> tuple[Reservation, ...]:
        with self._lock:
            self._expire(now)
            return tuple(self._reservations.values())

    def take_expired_reservations(self, now: float) -> tuple[Reservation, ...]:
        """Return each timed-out reservation once for utilization accounting."""
        with self._lock:
            self._expire(now)
            expired = tuple(self._expired_reservations)
            self._expired_reservations.clear()
            return expired

    def reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            return self._reservations.get(reservation_id)

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
                    # Old reservation records did not bind KV or fractional
                    # capacity. Treat them as full/unknown until their short
                    # TTL expires rather than risk overcommitting the GPU.
                    kv_bytes=int(value.get('kv_bytes', 2**63 - 1)),
                    capacity_units=float(value.get('capacity_units', 1.0)),
                )
                for value in reservations
                if float(value['expires_at']) > now
            }

    def checkpoint(self) -> tuple[dict[str, Reservation], list[Reservation]]:
        """Capture volatile routing state so a failed outer commit can be retried."""
        with self._lock:
            return dict(self._reservations), list(self._expired_reservations)

    def restore_checkpoint(self, checkpoint: tuple[dict[str, Reservation], list[Reservation]]) -> None:
        with self._lock:
            reservations, expired = checkpoint
            self._reservations = dict(reservations)
            self._expired_reservations = list(expired)

    def _expire(self, now: float) -> None:
        expired = [key for key, value in self._reservations.items() if value.expires_at <= now]
        for key in expired:
            reservation = self._reservations.pop(key, None)
            if reservation is not None:
                self._expired_reservations.append(reservation)

    def _active_by_gpu(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reservation in self._reservations.values():
            counts[reservation.gpu_id] = counts.get(reservation.gpu_id, 0) + 1
        return counts

    def _active_kv_by_gpu(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for reservation in self._reservations.values():
            totals[reservation.gpu_id] = totals.get(reservation.gpu_id, 0) + reservation.kv_bytes
        return totals
