"""Per-chunk stream commitments for the inference gateway."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamProofContext:
    request_id: str
    created: int
    release_digest: str
    model_id: str
    model_revision: str


class HMACStreamVerifier:
    """Verify each streamed chunk without retaining prompt or output data.

    This mirrors the role of Chutes CLLMV v2. The session key must be delivered
    to, and bound to, the attested runtime. Weight challenges and release
    attestation prove what is loaded; this commitment proves the response came
    through that verified runtime session.
    """

    def __init__(self, session_key: bytes, context: StreamProofContext) -> None:
        if len(session_key) < 32:
            raise ValueError('stream proof session keys must contain at least 32 bytes')
        self.session_key = session_key
        self.context = context

    def expected_proof(self, chunk_index: int, text: str) -> str:
        payload = {
            'request_id': self.context.request_id,
            'created': self.context.created,
            'release_digest': self.context.release_digest,
            'model_id': self.context.model_id,
            'model_revision': self.context.model_revision,
            'chunk_index': chunk_index,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        }
        message = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        return hmac.new(self.session_key, message, hashlib.sha256).hexdigest()

    def verify(self, chunk_index: int, text: str, proof: str) -> bool:
        return hmac.compare_digest(self.expected_proof(chunk_index, text), proof.casefold())
