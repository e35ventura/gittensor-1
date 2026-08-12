"""SparkCompute adapter and short-lived eligibility leases."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gittensor.compute.config import VerificationConfig
from gittensor.compute.models import GPURegistration, Release, VerificationLease


@dataclass(frozen=True)
class VerificationOutcome:
    lease: VerificationLease | None
    reason: str | None

    @property
    def accepted(self) -> bool:
        return self.lease is not None


class SparkComputeClient:
    """Read the public status contract exposed by sparkcompute-server."""

    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    def fetch_status(self) -> Sequence[Mapping[str, Any]]:
        headers = {'Accept': 'application/json'}
        if self.config.bearer_token_env:
            token = os.environ.get(self.config.bearer_token_env)
            if token:
                headers['Authorization'] = f'Bearer {token}'
        request = urllib.request.Request(self.config.status_url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode())
        if not isinstance(payload, list):
            raise ValueError('SparkCompute /api/status must return a JSON array')
        return payload


class SparkVerifier:
    """Convert a SparkCompute snapshot into a Gittensor READY lease.

    SparkCompute proves the hardware and continuously checks the model canary.
    Gittensor binds the epoch-acknowledged assignment to an approved release,
    then requires SparkCompute runtime evidence to match that immutable manifest
    before the GPU can become READY again.
    """

    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        registration: GPURegistration,
        release: Release,
        snapshot: Mapping[str, Any] | None,
        now: float,
    ) -> VerificationOutcome:
        hardware = self.evaluate_hardware(registration, snapshot, now)
        if not hardware.accepted:
            return hardware
        assert snapshot is not None
        if registration.release_digest != release.release_digest:
            return VerificationOutcome(None, 'GPU release is not in the approved catalog')
        if registration.canary_release_digest != release.release_digest:
            return VerificationOutcome(None, 'canary binding does not match the approved release digest')

        report = snapshot.get('report') or {}
        if self.config.require_runtime_attestation:
            runtime = report.get('runtime') or {}
            expected = {
                'release_digest': release.release_digest,
                'model_repository': release.model_repository,
                'model_revision': release.model_revision,
                'runtime_digest': release.runtime_digest,
                'runtime_commit': release.runtime_commit,
                'container_image': release.container_image,
                'container_digest': release.container_digest,
                'filesystem_digest': release.filesystem_digest,
            }
            mismatched = sorted(key for key, value in expected.items() if not value or runtime.get(key) != value)
            if mismatched:
                return VerificationOutcome(
                    None,
                    f'runtime attestation does not match approved release: {", ".join(mismatched)}',
                )
            attestation = snapshot.get('attestation') or {}
            if not attestation.get('verified') or not attestation.get('evidence_digest'):
                return VerificationOutcome(None, 'runtime attestation evidence is missing or unverified')

        liveness = snapshot.get('liveness') or {}
        if self.config.require_model_canary:
            if not liveness.get('model_canary_enabled'):
                return VerificationOutcome(None, 'SparkCompute model canaries are disabled')
            if not liveness.get('model_verified'):
                return VerificationOutcome(None, 'SparkCompute model canary is not verified')

        base = hardware.lease
        assert base is not None
        return VerificationOutcome(
            VerificationLease(
                gpu_id=registration.gpu_id,
                hardware_type=base.hardware_type,
                hardware_uuid=base.hardware_uuid,
                driver_version=base.driver_version,
                release_digest=release.release_digest,
                verified_at=max(base.verified_at, now - float(liveness.get('model_age_sec') or 0)),
                last_heartbeat=base.last_heartbeat,
                expires_at=base.expires_at,
                verifier_protocol=base.verifier_protocol,
                verifier_measurement=base.verifier_measurement,
                evidence_digest=str((snapshot.get('attestation') or {}).get('evidence_digest') or ''),
            ),
            None,
        )

    def evaluate_hardware(
        self,
        registration: GPURegistration,
        snapshot: Mapping[str, Any] | None,
        now: float,
    ) -> VerificationOutcome:
        """Verify hardware and the verifier binary independently of model placement."""
        if snapshot is None:
            return VerificationOutcome(None, 'SparkCompute node is absent from /api/status')
        if snapshot.get('verdict') != 'VERIFIED':
            return VerificationOutcome(None, f'SparkCompute verdict is {snapshot.get("verdict", "missing")}')
        verifier = snapshot.get('verifier') or {}
        protocol = str(verifier.get('protocol') or '')
        measurement = str(verifier.get('measurement') or '')
        source_commit = str(verifier.get('source_commit') or '')
        if self.config.require_verifier_measurement:
            if protocol != self.config.expected_verifier_protocol:
                return VerificationOutcome(None, f'unexpected SparkCompute verifier protocol: {protocol or "missing"}')
            if measurement != self.config.expected_verifier_measurement:
                return VerificationOutcome(None, 'SparkCompute verifier measurement does not match configuration')
            if source_commit != self.config.source_commit:
                return VerificationOutcome(None, 'SparkCompute source commit was not proven by the verifier')

        report = snapshot.get('report') or {}
        gpu = report.get('gpu') or {}
        hardware = str(gpu.get('name') or '')
        if self.config.expected_hardware.casefold() not in hardware.casefold():
            return VerificationOutcome(None, f'unexpected hardware: {hardware or "missing"}')
        driver_version = str(gpu.get('driver_version') or '')
        if not driver_version:
            return VerificationOutcome(None, 'driver version is missing')
        hardware_uuid = str(gpu.get('uuid') or '')
        if not hardware_uuid:
            return VerificationOutcome(None, 'GPU UUID is missing')
        liveness = snapshot.get('liveness') or {}
        if not liveness.get('online'):
            return VerificationOutcome(None, 'SparkCompute heartbeat is not live')
        if not liveness.get('gpu_live'):
            return VerificationOutcome(None, 'SparkCompute GPU micro-challenge is not live')
        last_checked = float(snapshot.get('last_checked') or 0)
        if last_checked <= 0 or now - last_checked >= self.config.lease_ttl_seconds:
            return VerificationOutcome(None, 'SparkCompute verification is stale')
        last_heartbeat = now - float(liveness.get('online_age_sec') or 0)
        expires_at = min(last_checked + self.config.lease_ttl_seconds, now + self.config.lease_ttl_seconds)
        return VerificationOutcome(
            VerificationLease(
                gpu_id=registration.gpu_id,
                hardware_type=hardware,
                hardware_uuid=hardware_uuid,
                driver_version=driver_version,
                release_digest=registration.release_digest,
                verified_at=last_checked,
                last_heartbeat=last_heartbeat,
                expires_at=expires_at,
                verifier_protocol=protocol,
                verifier_measurement=measurement,
                evidence_digest=str((snapshot.get('attestation') or {}).get('evidence_digest') or ''),
            ),
            None,
        )


def index_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(snapshot['id']): snapshot for snapshot in snapshots if snapshot.get('id')}
