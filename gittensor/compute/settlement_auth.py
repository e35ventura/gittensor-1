"""Signed distribution contract for global compute settlements."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import bittensor as bt


def canonical_settlement(
    window_id: str,
    started_at: float,
    ended_at: float,
    hotkey_rewards: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> bytes:
    payload = {
        'window_id': window_id,
        'started_at': started_at,
        'ended_at': ended_at,
        'hotkey_rewards': dict(sorted(hotkey_rewards.items())),
        'metadata': dict(sorted(metadata.items())),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return b'gittensor-compute-settlement-v1\n' + hashlib.sha256(encoded).hexdigest().encode()


class SettlementSigner:
    def __init__(self, keypair: bt.Keypair) -> None:
        self.keypair = keypair

    def sign(
        self,
        window_id: str,
        started_at: float,
        ended_at: float,
        hotkey_rewards: Mapping[str, str],
        metadata: Mapping[str, Any],
    ) -> dict[str, str]:
        signature = self.keypair.sign(canonical_settlement(window_id, started_at, ended_at, hotkey_rewards, metadata))
        return {'signer_hotkey': self.keypair.ss58_address, 'signature': f'0x{signature.hex()}'}


def verify_settlement(settlement: Mapping[str, Any], expected_hotkey: str) -> bool:
    metadata = settlement.get('metadata')
    rewards = settlement.get('hotkey_rewards')
    if not isinstance(metadata, dict) or not isinstance(rewards, dict):
        return False
    signer_hotkey = str(metadata.get('signer_hotkey') or '')
    signature_hex = str(metadata.get('signature') or '')
    if signer_hotkey != expected_hotkey or not signature_hex:
        return False
    signed_metadata = {key: value for key, value in metadata.items() if key not in {'signer_hotkey', 'signature'}}
    try:
        signature = bytes.fromhex(signature_hex.removeprefix('0x'))
        message = canonical_settlement(
            str(settlement['window_id']),
            float(settlement['started_at']),
            float(settlement['ended_at']),
            {str(key): str(value) for key, value in rewards.items()},
            signed_metadata,
        )
        return bt.Keypair(ss58_address=expected_hotkey).verify(message, signature)
    except (KeyError, TypeError, ValueError):
        return False
