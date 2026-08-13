"""Per-chunk stream commitments for the inference gateway."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import bittensor as bt


@dataclass(frozen=True)
class StreamProofContext:
    request_id: str
    created: int
    release_digest: str
    model_id: str
    model_revision: str
    request_digest: str = ''


def canonical_request_digest(request_payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        request_payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


def response_payload_without_proof(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact OpenAI payload covered by the runtime signature."""
    return {str(key): value for key, value in payload.items() if key != 'gittensor_proof'}


def canonical_stream_chunk(
    context: StreamProofContext,
    chunk_index: int,
    response_payload: Mapping[str, Any],
) -> bytes:
    canonical_response = json.dumps(
        response_payload_without_proof(response_payload),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload = {
        'request_id': context.request_id,
        'created': context.created,
        'release_digest': context.release_digest,
        'model_id': context.model_id,
        'model_revision': context.model_revision,
        'request_digest': context.request_digest,
        'chunk_index': chunk_index,
        'response_payload': canonical_response,
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


class SignedStreamVerifier:
    """Verify chunks signed by a runtime key bound into hardware attestation."""

    def __init__(self, public_key_hex: str, context: StreamProofContext) -> None:
        try:
            public_key = bytes.fromhex(public_key_hex.removeprefix('0x'))
        except ValueError:
            raise ValueError('stream proof public key must be hexadecimal') from None
        if len(public_key) != 32:
            raise ValueError('stream proof public key must contain exactly 32 bytes')
        self.keypair = bt.Keypair(public_key=public_key.hex())
        self.context = context
        self.next_index = 0

    def verify(self, chunk_index: int, response_payload: Mapping[str, Any], signature_hex: str) -> bool:
        if chunk_index != self.next_index:
            return False
        try:
            signature = bytes.fromhex(signature_hex.removeprefix('0x'))
        except ValueError:
            return False
        try:
            message = canonical_stream_chunk(self.context, chunk_index, response_payload)
        except (TypeError, ValueError):
            return False
        if not self.keypair.verify(message, signature):
            return False
        self.next_index += 1
        return True


class RuntimeStreamSigner:
    """Reference helper an approved runtime uses to sign each OpenAI chunk."""

    def __init__(self, keypair: bt.Keypair, context: StreamProofContext) -> None:
        self.keypair = keypair
        self.context = context
        self.next_index = 0

    @property
    def public_key_hex(self) -> str:
        public_key = self.keypair.public_key
        if public_key is None:
            raise ValueError('runtime stream signer requires a public key')
        return public_key.hex()

    def sign(self, response_payload: Mapping[str, Any]) -> dict[str, int | str]:
        index = self.next_index
        signature = self.keypair.sign(canonical_stream_chunk(self.context, index, response_payload)).hex()
        self.next_index += 1
        return {'index': index, 'signature': f'0x{signature}'}

    def attach(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = response_payload_without_proof(payload)
        choices = unsigned.get('choices')
        if not isinstance(choices, list):
            raise ValueError('OpenAI response must contain choices')
        return {**unsigned, 'gittensor_proof': self.sign(unsigned)}

    def attach_error(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Sign a model-server client error so an outer miner cannot forge it."""
        unsigned = response_payload_without_proof(payload)
        if 'error' not in unsigned:
            raise ValueError('OpenAI error response must contain error')
        return {**unsigned, 'gittensor_proof': self.sign(unsigned)}

    def terminal(self) -> dict[str, Any]:
        """Create the signed internal event that must precede SSE ``[DONE]``."""
        payload: dict[str, Any] = {'choices': [], 'gittensor_terminal': True}
        return {**payload, 'gittensor_proof': self.sign(payload)}
