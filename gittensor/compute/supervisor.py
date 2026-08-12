"""Automatic verification, challenge, placement, and settlement loops."""

from __future__ import annotations

import logging
import threading
import time

from gittensor.compute.control_plane import ComputeControlPlane

logger = logging.getLogger(__name__)


class ComputeSupervisor:
    def __init__(self, control_plane: ComputeControlPlane, *, clock=time.time) -> None:
        self.control_plane = control_plane
        self.config = control_plane.config
        self.clock = clock
        now = clock()
        self._next_verification = now
        self._next_challenge = now
        self._next_control = now
        self._next_settlement = now + self.config.control_loop.settlement_interval_seconds
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
            finally:
                self._next_settlement = timestamp + self.config.control_loop.settlement_interval_seconds

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due()
            except Exception:
                logger.exception('compute supervisor iteration failed')
            self._stop.wait(1.0)
