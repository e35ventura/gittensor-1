"""Hotkey authentication and live metagraph ownership resolution."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import bittensor as bt


class AuthenticationError(ValueError):
    """Raised when a miner request cannot prove a current hotkey identity."""


def canonical_request(method: str, path: str, payload: Mapping[str, Any], timestamp: int, nonce: str) -> bytes:
    """Return the exact bytes a miner hotkey must sign."""
    body = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    return f'gittensor-compute-v1\n{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{digest}'.encode()


@dataclass(frozen=True)
class AuthenticatedMiner:
    uid: int
    hotkey: str


class IdentityResolver(Protocol):
    def uid_for_hotkey(self, hotkey: str) -> int | None: ...


class StaticIdentityResolver:
    """Deterministic resolver used by tests and offline validators."""

    def __init__(self, hotkeys: Mapping[str, int]) -> None:
        self.hotkeys = dict(hotkeys)

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        return self.hotkeys.get(hotkey)


class LiveMetagraphResolver:
    """Resolve ownership from the current subnet metagraph, never request input."""

    def __init__(self, netuid: int, network: str, refresh_seconds: float, *, clock=time.time) -> None:
        self.netuid = netuid
        self.network = network
        self.refresh_seconds = refresh_seconds
        self.clock = clock
        self._hotkeys: dict[str, int] = {}
        self._refreshed_at = 0.0
        self._lock = threading.Lock()
        self._subtensor = bt.Subtensor(network=self.network)

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        now = self.clock()
        with self._lock:
            if now - self._refreshed_at >= self.refresh_seconds:
                metagraph = self._subtensor.metagraph(self.netuid, lite=True)
                self._hotkeys = {str(value): uid for uid, value in enumerate(metagraph.hotkeys)}
                self._refreshed_at = now
            return self._hotkeys.get(hotkey)


class NonceStore(Protocol):
    def consume_nonce(self, hotkey: str, nonce: str, expires_at: float) -> bool: ...


class HotkeyAuthenticator:
    """Verify ownership, freshness, signature, and one-time nonce use."""

    def __init__(
        self,
        resolver: IdentityResolver,
        nonce_store: NonceStore,
        ttl_seconds: float,
        *,
        clock=time.time,
    ) -> None:
        self.resolver = resolver
        self.nonce_store = nonce_store
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def authenticate(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        auth: Mapping[str, Any],
    ) -> AuthenticatedMiner:
        hotkey = str(auth.get('hotkey') or '')
        nonce = str(auth.get('nonce') or '')
        signature_hex = str(auth.get('signature') or '')
        raw_timestamp = auth.get('timestamp')
        try:
            if not isinstance(raw_timestamp, (int, str)):
                raise TypeError
            timestamp = int(raw_timestamp)
        except (TypeError, ValueError):
            raise AuthenticationError('auth.timestamp must be an integer') from None
        if not hotkey or not nonce or not signature_hex:
            raise AuthenticationError('auth requires hotkey, timestamp, nonce, and signature')
        now = self.clock()
        if abs(now - timestamp) > self.ttl_seconds:
            raise AuthenticationError('request signature is stale or too far in the future')
        uid = self.resolver.uid_for_hotkey(hotkey)
        if uid is None:
            raise AuthenticationError('hotkey is not registered on the configured subnet')
        try:
            signature = bytes.fromhex(signature_hex.removeprefix('0x'))
        except ValueError:
            raise AuthenticationError('auth.signature must be hexadecimal') from None
        message = canonical_request(method, path, payload, timestamp, nonce)
        if not bt.Keypair(ss58_address=hotkey).verify(message, signature):
            raise AuthenticationError('invalid hotkey signature')
        if not self.nonce_store.consume_nonce(hotkey, nonce, now + self.ttl_seconds):
            raise AuthenticationError('request nonce has already been used')
        return AuthenticatedMiner(uid=uid, hotkey=hotkey)
