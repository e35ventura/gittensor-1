"""Chutes-style unpredictable byte-range checks for pinned model weights."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Protocol

from gittensor.compute.models import GPURecord, Release


@dataclass(frozen=True)
class WeightChallenge:
    challenge_id: str
    gpu_id: str
    release_digest: str
    model_repository: str
    model_revision: str
    path: str
    start_byte: int
    end_byte: int
    nonce: str
    expected_digest: str
    expires_at: float

    def public_payload(self) -> dict[str, str | int | float]:
        return {
            'challenge_id': self.challenge_id,
            'gpu_id': self.gpu_id,
            'release_digest': self.release_digest,
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'path': self.path,
            'start_byte': self.start_byte,
            'end_byte': self.end_byte,
            'nonce': self.nonce,
            'expires_at': self.expires_at,
        }


class RangeSource(Protocol):
    def fetch(self, repository: str, revision: str, path: str, start: int, end: int) -> bytes: ...


class HuggingFaceRangeSource:
    """Fetch trusted reference bytes from an exact Hugging Face commit."""

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, repository: str, revision: str, path: str, start: int, end: int) -> bytes:
        url = f'https://huggingface.co/{repository}/resolve/{revision}/{path}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            contents = response.read()
        expected_length = end - start
        if len(contents) != expected_length:
            raise ValueError(f'Hugging Face range returned {len(contents)} bytes, expected {expected_length}')
        return contents


class WeightChallengeTransport(Protocol):
    def answer(self, record: GPURecord, challenge: WeightChallenge) -> str: ...


class HTTPWeightChallengeTransport:
    """Send a challenge directly to the assigned miner runtime agent."""

    def __init__(self, timeout_seconds: float, bearer_token: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token

    def answer(self, record: GPURecord, challenge: WeightChallenge) -> str:
        url = f'{record.registration.endpoint.rstrip("/")}/v1/gittensor/challenges/weights'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        request = urllib.request.Request(
            url,
            data=json.dumps(challenge.public_payload(), separators=(',', ':')).encode(),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode())
        digest = str(payload.get('sha256') or '')
        if len(digest) != 64:
            raise ValueError('miner returned an invalid weight challenge digest')
        return digest


class WeightChallengeVerifier:
    """Issue single-use challenges and compare them to independent reference bytes."""

    def __init__(self, source: RangeSource, ttl_seconds: float) -> None:
        self.source = source
        self.ttl_seconds = ttl_seconds
        self._pending: dict[str, WeightChallenge] = {}
        self._lock = threading.Lock()

    def issue(self, gpu_id: str, release: Release, now: float) -> WeightChallenge:
        release.validate_production_manifest()
        path = secrets.choice(sorted(release.weight_files))
        file_size = int(release.weight_files[path])
        length = min(file_size, 256 + secrets.randbelow(min(3840, max(1, file_size - 255))))
        start = secrets.randbelow(file_size - length + 1)
        end = start + length
        reference = self.source.fetch(release.model_repository, release.model_revision, path, start, end)
        nonce = secrets.token_hex(32)
        challenge = WeightChallenge(
            challenge_id=uuid.uuid4().hex,
            gpu_id=gpu_id,
            release_digest=release.release_digest,
            model_repository=release.model_repository,
            model_revision=release.model_revision,
            path=path,
            start_byte=start,
            end_byte=end,
            nonce=nonce,
            expected_digest=hashlib.sha256(bytes.fromhex(nonce) + reference).hexdigest(),
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._pending[challenge.challenge_id] = challenge
        return challenge

    def verify(self, challenge_id: str, gpu_id: str, digest: str, now: float) -> bool:
        with self._lock:
            challenge = self._pending.pop(challenge_id, None)
        if challenge is None or challenge.gpu_id != gpu_id or now >= challenge.expires_at:
            return False
        return secrets.compare_digest(challenge.expected_digest, digest.casefold())

    def cancel(self, challenge_id: str) -> None:
        with self._lock:
            self._pending.pop(challenge_id, None)
