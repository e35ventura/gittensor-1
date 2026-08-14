"""Automatic verification, challenge, placement, and settlement loops."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.emission_oracle import EmissionObservation

logger = logging.getLogger(__name__)


class EmissionOracle(Protocol):
    def observe(self, now: float | None = None) -> EmissionObservation: ...


class ComputeSupervisor:
    def __init__(
        self,
        control_plane: ComputeControlPlane,
        *,
        emission_oracle: EmissionOracle | None = None,
        clock=time.time,
    ) -> None:
        self.control_plane = control_plane
        self.config = control_plane.config
        self.clock = clock
        self.emission_oracle = emission_oracle
        now = clock()
        self._next_verification = now
        self._next_emission_oracle = now
        self._next_challenge = now
        self._next_control = now
        self._next_settlement = max(
            now,
            self.control_plane._settlement_started_at + self.config.control_loop.settlement_interval_seconds,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name='gittensor-compute-supervisor', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def run_due(self, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else now
        revocation_outcomes = self.control_plane.retry_pending_revocations()
        for gpu_id, outcome in revocation_outcomes.items():
            if outcome != 'revoked':
                logger.error('assignment revocation retry failed for %s: %s', gpu_id, outcome)
        if self.emission_oracle is not None and timestamp >= self._next_emission_oracle:
            try:
                observation = self.emission_oracle.observe(now=timestamp)
                self.control_plane.apply_emission_observation(observation, now=timestamp)
            except Exception as exc:
                self.control_plane.record_emission_oracle_failure(exc)
                logger.exception('emission oracle refresh failed')
            finally:
                self._next_emission_oracle = timestamp + self.config.emission_oracle.refresh_interval_seconds
        if timestamp >= self._next_verification:
            try:
                self.control_plane.refresh_verification(now=timestamp)
            except Exception:
                logger.exception('verification refresh failed')
            finally:
                self._next_verification = timestamp + self.config.control_loop.verification_interval_seconds
        if timestamp >= self._next_challenge:
            try:
                self.control_plane.run_weight_challenges(now=timestamp)
            except Exception:
                logger.exception('weight challenge sample failed')
            finally:
                self._next_challenge = timestamp + self.config.verification.weight_challenge_interval_seconds
        if timestamp >= self._next_control:
            try:
                self.control_plane.tick(now=timestamp, execute=True)
            except Exception:
                logger.exception('fleet control tick failed')
            finally:
                self._next_control = timestamp + self.config.placement.control_interval_seconds
        if timestamp >= self._next_settlement:
            try:
                self.control_plane.settle(now=timestamp)
            except Exception:
                logger.exception('settlement failed')
                self._next_settlement = timestamp + min(
                    30.0,
                    self.config.control_loop.settlement_interval_seconds,
                )
            else:
                self._next_settlement = timestamp + self.config.control_loop.settlement_interval_seconds

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due()
            except Exception:
                logger.exception('compute supervisor iteration failed')
            self._stop.wait(1.0)
