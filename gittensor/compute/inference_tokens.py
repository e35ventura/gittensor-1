"""Short-lived, one-use inference capability tokens for routed reservations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceCapability:
    reservation_id: str
    gpu_id: str
    release_digest: str
    expires_at: float


def issue_inference_capability(
    secret: str,
    *,
    reservation_id: str,
    gpu_id: str,
    release_digest: str,
    expires_at: float,
) -> str:
    if not secret or not reservation_id or not gpu_id or not release_digest:
        raise ValueError('inference capability fields cannot be empty')
    if not math.isfinite(expires_at) or expires_at <= 0:
        raise ValueError('inference capability expiry must be finite and positive')
    payload = json.dumps(
        {
            'version': 1,
            'reservation_id': reservation_id,
            'gpu_id': gpu_id,
            'release_digest': release_digest,
            'expires_at': expires_at,
        },
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).rstrip(b'=')
    signature = hmac.new(secret.encode(), encoded, hashlib.sha256).hexdigest().encode()
    return b'.'.join((encoded, signature)).decode()


def verify_inference_capability(secret: str, token: str, *, now: float) -> InferenceCapability | None:
    if not secret or not token or len(token) > 4096 or not math.isfinite(now):
        return None
    try:
        encoded, signature = token.encode().split(b'.', 1)
        expected = hmac.new(secret.encode(), encoded, hashlib.sha256).hexdigest().encode()
        if not hmac.compare_digest(signature, expected):
            return None
        padding = b'=' * (-len(encoded) % 4)
        payload: Any = json.loads(base64.b64decode(encoded + padding, altchars=b'-_', validate=True))
        if not isinstance(payload, dict) or set(payload) != {
            'version',
            'reservation_id',
            'gpu_id',
            'release_digest',
            'expires_at',
        }:
            return None
        expires_at = float(payload['expires_at'])
        capability = InferenceCapability(
            reservation_id=str(payload['reservation_id']),
            gpu_id=str(payload['gpu_id']),
            release_digest=str(payload['release_digest']),
            expires_at=expires_at,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        payload['version'] != 1
        or not capability.reservation_id
        or not capability.gpu_id
        or not capability.release_digest
        or not math.isfinite(capability.expires_at)
        or now >= capability.expires_at
    ):
        return None
    return capability
