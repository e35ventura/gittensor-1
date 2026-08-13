"""SparkCompute adapter and short-lived eligibility leases."""

from __future__ import annotations

import hashlib
import math
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import bittensor as bt

from gittensor.compute.config import VerificationConfig
from gittensor.compute.http_json import load_json_value
from gittensor.compute.models import GPURegistration, Release, VerificationLease
from gittensor.compute.safe_http import no_redirect_urlopen

_SHA256_PATTERN = re.compile(r'sha256:[0-9a-f]{64}')


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
        for status_url in tuple(config.status_urls) or (config.status_url,):
            _validate_status_url(status_url)

    def fetch_status(self) -> Sequence[Mapping[str, Any]]:
        """Fetch one or more sharded SparkCompute status feeds concurrently."""
        urls = tuple(self.config.status_urls) or (self.config.status_url,)

        def fetch(status_url: str) -> list[Mapping[str, Any]]:
            request = urllib.request.Request(status_url, headers=self._headers())
            with no_redirect_urlopen(request, timeout=self.config.timeout_seconds) as response:
                if response.status != 200:
                    raise ValueError(f'SparkCompute status endpoint returned HTTP {response.status}')
                content_type = (
                    _single_response_header(response.headers, 'Content-Type').partition(';')[0].strip().casefold()
                )
                if content_type != 'application/json':
                    raise ValueError('SparkCompute status response must use application/json')
                content_length = _optional_single_response_header(response.headers, 'Content-Length')
                declared_length = int(content_length) if content_length is not None else None
                if declared_length is not None and (declared_length < 0 or declared_length > 64 * 1024 * 1024):
                    raise ValueError('SparkCompute status response exceeds 64 MiB')
                body = response.read(64 * 1024 * 1024 + 1)
                if len(body) > 64 * 1024 * 1024:
                    raise ValueError('SparkCompute status response exceeds 64 MiB')
                if declared_length is not None and len(body) != declared_length:
                    raise ValueError('SparkCompute status response ended before Content-Length')
                if self.config.require_status_signature:
                    _verify_status_signature(body, response.headers, self.config.trusted_verifier_public_keys)
                payload = load_json_value(body)
            if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
                raise ValueError('SparkCompute /api/status must return a JSON array of objects')
            return payload

        if len(urls) == 1:
            return fetch(urls[0])
        workers = min(self.config.refresh_workers, len(urls))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='spark-status') as executor:
            shards = list(executor.map(fetch, urls))
        return [snapshot for shard in shards for snapshot in shard]

    def _headers(self) -> dict[str, str]:
        headers = {'Accept': 'application/json'}
        if self.config.bearer_token_env:
            token = os.environ.get(self.config.bearer_token_env)
            if token:
                headers['Authorization'] = f'Bearer {token}'
        return headers


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

        report = _mapping(snapshot.get('report'))
        attestation = _mapping(snapshot.get('attestation'))
        if self.config.require_runtime_attestation:
            runtime = _mapping(report.get('runtime'))
            expected = {
                'release_digest': release.release_digest,
                'model_repository': release.model_repository,
                'model_revision': release.model_revision,
                'tokenizer_repository': release.tokenizer_repository,
                'tokenizer_revision': release.tokenizer_revision,
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
            if attestation.get('verified') is not True or not attestation.get('evidence_digest'):
                return VerificationOutcome(None, 'runtime attestation evidence is missing or unverified')
        stream_public_key = str(attestation.get('stream_public_key') or '')
        if self.config.require_confidential_compute:
            if attestation.get('confidential_compute') is not True:
                return VerificationOutcome(None, 'confidential-compute attestation is missing')
            if attestation.get('data_policy') != 'ephemeral-no-retention-v1':
                return VerificationOutcome(None, 'attested runtime does not enforce the no-retention data policy')
        if self.config.require_stream_proof:
            if release.token_proof_scheme != 'sr25519-response-v1':
                return VerificationOutcome(None, 'release does not use the required attested stream proof scheme')
            try:
                if len(bytes.fromhex(stream_public_key.removeprefix('0x'))) != 32:
                    raise ValueError
            except ValueError:
                return VerificationOutcome(None, 'attested stream proof public key is missing or invalid')

        liveness = _mapping(snapshot.get('liveness'))
        if self.config.require_model_canary:
            if liveness.get('model_canary_enabled') is not True:
                return VerificationOutcome(None, 'SparkCompute model canaries are disabled')
            if liveness.get('model_verified') is not True:
                return VerificationOutcome(None, 'SparkCompute model canary is not verified')
            model_age = _nonnegative_finite(liveness.get('model_age_sec'))
            if model_age is None or model_age >= self.config.lease_ttl_seconds:
                return VerificationOutcome(None, 'SparkCompute model canary is stale or has an invalid age')
        else:
            model_age = 0.0

        base = hardware.lease
        assert base is not None
        model_verified_at = now - model_age
        return VerificationOutcome(
            VerificationLease(
                gpu_id=registration.gpu_id,
                hardware_type=base.hardware_type,
                hardware_uuid=base.hardware_uuid,
                driver_version=base.driver_version,
                release_digest=release.release_digest,
                verified_at=min(base.verified_at, model_verified_at),
                last_heartbeat=base.last_heartbeat,
                expires_at=min(base.expires_at, model_verified_at + self.config.lease_ttl_seconds),
                verifier_protocol=base.verifier_protocol,
                verifier_measurement=base.verifier_measurement,
                evidence_digest=str(attestation.get('evidence_digest') or ''),
                stream_public_key=stream_public_key,
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
        verifier = _mapping(snapshot.get('verifier'))
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

        report = _mapping(snapshot.get('report'))
        gpu = _mapping(report.get('gpu'))
        hardware = str(gpu.get('name') or '')
        if self.config.expected_hardware.casefold() not in hardware.casefold():
            return VerificationOutcome(None, f'unexpected hardware: {hardware or "missing"}')
        driver_version = str(gpu.get('driver_version') or '')
        if not driver_version:
            return VerificationOutcome(None, 'driver version is missing')
        hardware_uuid = str(gpu.get('uuid') or '')
        if not hardware_uuid:
            return VerificationOutcome(None, 'GPU UUID is missing')
        if self.config.require_uniqueness_challenge:
            uniqueness = _mapping(snapshot.get('uniqueness'))
            uniqueness_verified_at = _positive_finite(uniqueness.get('verified_at'))
            if (
                uniqueness.get('verified') is not True
                or not str(uniqueness.get('batch_id') or '')
                or not isinstance(uniqueness.get('batch_size'), int)
                or isinstance(uniqueness.get('batch_size'), bool)
                or int(uniqueness['batch_size']) < self.config.uniqueness_batch_size
                or not _SHA256_PATTERN.fullmatch(str(uniqueness.get('challenge_digest') or ''))
                or uniqueness_verified_at is None
                or uniqueness_verified_at > now
                or now - uniqueness_verified_at >= self.config.uniqueness_challenge_ttl_seconds
            ):
                return VerificationOutcome(None, 'fresh simultaneous GPU uniqueness challenge is missing')
        liveness = _mapping(snapshot.get('liveness'))
        if liveness.get('online') is not True:
            return VerificationOutcome(None, 'SparkCompute heartbeat is not live')
        if liveness.get('gpu_live') is not True:
            return VerificationOutcome(None, 'SparkCompute GPU micro-challenge is not live')
        last_checked = _positive_finite(snapshot.get('last_checked'))
        if last_checked is None or last_checked > now or now - last_checked >= self.config.lease_ttl_seconds:
            return VerificationOutcome(None, 'SparkCompute verification is stale')
        online_age = _nonnegative_finite(liveness.get('online_age_sec'))
        if online_age is None or online_age >= self.config.lease_ttl_seconds:
            return VerificationOutcome(None, 'SparkCompute heartbeat is stale or has an invalid age')
        last_heartbeat = now - online_age
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
                evidence_digest=str(_mapping(snapshot.get('attestation')).get('evidence_digest') or ''),
                stream_public_key='',
            ),
            None,
        )


def index_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for snapshot in snapshots:
        node_id = str(snapshot.get('id') or '').strip()
        if not node_id:
            raise ValueError('SparkCompute status snapshot is missing its node id')
        if node_id in indexed:
            raise ValueError(f'SparkCompute status contains duplicate node id: {node_id}')
        indexed[node_id] = snapshot
    return indexed


def canonical_verifier_status(body: bytes) -> bytes:
    """Bind the exact complete body returned by one SparkCompute status shard."""
    return b'gittensor-spark-verifier-status-v1\n' + hashlib.sha256(body).hexdigest().encode()


def _verify_status_signature(body: bytes, headers: Mapping[str, Any], trusted_keys: Sequence[str]) -> None:
    public_key = _single_response_header(headers, 'X-Gittensor-Verifier-Public-Key')
    signature_hex = _single_response_header(headers, 'X-Gittensor-Verifier-Signature')
    if public_key not in trusted_keys:
        raise ValueError('SparkCompute status signer is not trusted')
    try:
        public_key_bytes = bytes.fromhex(public_key)
        signature = bytes.fromhex(signature_hex.removeprefix('0x'))
    except ValueError:
        raise ValueError('SparkCompute status signature is malformed') from None
    if len(public_key_bytes) != 32 or len(signature) != 64:
        raise ValueError('SparkCompute status signature is malformed')
    if not bt.Keypair(public_key=public_key).verify(canonical_verifier_status(body), signature):
        raise ValueError('SparkCompute status signature is invalid')


def _optional_single_response_header(headers: Mapping[str, Any], name: str) -> str | None:
    get_all = getattr(headers, 'get_all', None)
    if callable(get_all):
        raw_values = get_all(name)
        values = list(raw_values) if isinstance(raw_values, (list, tuple)) else []
        if len(values) > 1:
            raise ValueError(f'SparkCompute status response has duplicate {name} headers')
        return str(values[0]) if values else None
    value = headers.get(name)
    return str(value) if value is not None else None


def _single_response_header(headers: Mapping[str, Any], name: str) -> str:
    value = _optional_single_response_header(headers, name)
    if value is None:
        raise ValueError(f'SparkCompute status response is missing {name}')
    return value


def _positive_finite(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _nonnegative_finite(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validate_status_url(status_url: str) -> None:
    parsed = urlsplit(status_url)
    loopback_http = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', '::1', 'localhost'}
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.scheme != 'https' and not loopback_http)
    ):
        raise ValueError('SparkCompute status URLs must use HTTPS, except on loopback')
