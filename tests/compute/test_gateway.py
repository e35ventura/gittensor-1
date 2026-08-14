import io
import json
import time
from unittest.mock import patch

import bittensor as bt
import pytest

from gittensor.compute.gateway import (
    GatewayConfig,
    GatewayError,
    GatewaySession,
    InferenceGateway,
    _miner_http_status_is_failure,
    _read_control_response,
    _verify_runtime_error,
)
from gittensor.compute.inference_verification import (
    RuntimeStreamSigner,
    SignedStreamVerifier,
    StreamProofContext,
    canonical_request_digest,
)


class FakeGateway:
    def __init__(self):
        self.outcomes = []

    def finish(self, session, *, success):
        self.outcomes.append(success)

    def renew(self, reservation_id):
        return time.time() + 60


class BytesResponse(io.BytesIO):
    status = 200

    def __init__(self, value=b'', headers=None):
        super().__init__(value)
        self.headers = headers or {}


def test_gateway_converts_release_selector_to_the_canonical_runtime_model(monkeypatch):
    monkeypatch.setenv('CONTROL_TOKEN', 'control-token')
    gateway = InferenceGateway(
        GatewayConfig(
            control_plane_url='https://control.example',
            control_plane_token_env='CONTROL_TOKEN',
            gateway_token_env='PUBLIC_TOKEN',
            region='us-central',
            require_stream_proof=False,
        )
    )
    gateway._resolve_release = lambda model: {
        'release_digest': 'sha256:release',
        'model_id': 'owner/exact-model',
        'model_revision': 'a' * 40,
    }
    gateway._control_post = lambda path, payload: {
        'reservation_id': 'reservation-1',
        'gpu_id': 'gpu-1',
        'endpoint': 'https://miner.example',
        'expires_at': time.time() + 300,
        'inference_token': 'capability',
    }
    captured = {}

    def open_miner(url, *, body, headers, method, timeout):
        if method == 'GET':
            return BytesResponse(b'{"status":"ok"}')
        captured.update(json.loads(body))
        return BytesResponse()

    with patch('gittensor.compute.gateway.public_https_request', side_effect=open_miner):
        session = gateway.open({'model': 'sha256:release', 'messages': []})
        session.close(success=None)

    assert captured['openai_request']['model'] == 'owner/exact-model'


def _signed_chunk(signer, text, **extra):
    return signer.attach({'choices': [{'delta': {'content': text}}], **extra})


def _public_key_hex(keypair):
    assert keypair.public_key is not None
    return keypair.public_key.hex()


def test_gateway_verifies_every_stream_chunk_before_forwarding():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    signer = RuntimeStreamSigner(keypair, context)
    chunks = [_signed_chunk(signer, 'hello'), _signed_chunk(signer, ' world'), signer.terminal()]
    encoded = b''.join(f'data: {json.dumps(chunk)}\n\n'.encode() for chunk in chunks) + b'data: [DONE]\n\n'
    expected = b''.join(f'data: {json.dumps(chunk)}\n\n'.encode() for chunk in chunks[:-1]) + b'data: [DONE]\n\n'
    gateway = FakeGateway()
    session = GatewaySession(
        gateway=gateway,
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(encoded),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), context),
        started=0,
        connected=0,
        streaming=True,
    )

    assert b''.join(session.iter_verified()) == expected
    session.close(success=True)
    assert gateway.outcomes == [True]


def test_gateway_rejects_unsigned_or_reordered_stream_chunks():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    signer = RuntimeStreamSigner(keypair, context)
    signer.next_index = 1
    chunk = _signed_chunk(signer, 'wrong index')
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(f'data: {json.dumps(chunk)}\n\n'.encode()),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), context),
        started=0,
        connected=0,
        streaming=True,
    )

    with pytest.raises(GatewayError, match='invalid or out of order'):
        list(session.iter_verified())


def test_gateway_accepts_only_request_bound_runtime_signed_client_errors():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext(
        'request-1',
        100,
        'release:1',
        'model',
        'a' * 40,
        canonical_request_digest({'model': 'model'}),
    )
    signed = RuntimeStreamSigner(keypair, context).attach_error({'error': {'message': 'invalid prompt'}})
    body = json.dumps(signed).encode()

    assert _verify_runtime_error(body, _public_key_hex(keypair), context)
    assert not _verify_runtime_error(b'{"error":{"message":"forged"}}', _public_key_hex(keypair), context)
    other = StreamProofContext(**{**context.__dict__, 'request_id': 'request-2'})
    assert not _verify_runtime_error(body, _public_key_hex(keypair), other)


def test_gateway_rejects_oversized_control_plane_responses():
    response = BytesResponse(b'{}', {'Content-Length': str(4 * 1024 * 1024 + 1)})

    with pytest.raises(GatewayError, match='exceeds'):
        _read_control_response(response)


def test_stream_proof_is_bound_to_the_exact_request_body():
    keypair = bt.Keypair.create_from_uri('//Alice')
    signed_context = StreamProofContext(
        'request-1',
        100,
        'release:1',
        'model',
        'a' * 40,
        canonical_request_digest({'messages': [{'role': 'user', 'content': 'one'}]}),
    )
    verifying_context = StreamProofContext(
        'request-1',
        100,
        'release:1',
        'model',
        'a' * 40,
        canonical_request_digest({'messages': [{'role': 'user', 'content': 'two'}]}),
    )
    payload = _signed_chunk(RuntimeStreamSigner(keypair, signed_context), 'answer')
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(json.dumps(payload).encode()),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), verifying_context),
        started=0,
        connected=0,
        streaming=False,
    )

    with pytest.raises(GatewayError, match='invalid or out of order'):
        session.read_verified()


def test_gateway_verifies_non_stream_response():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    payload = _signed_chunk(RuntimeStreamSigner(keypair, context), 'complete')
    body = json.dumps(payload).encode()
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(body),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), context),
        started=0,
        connected=0,
        streaming=False,
    )

    assert session.read_verified() == body


def test_gateway_rejects_response_metadata_tampering():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    payload = _signed_chunk(RuntimeStreamSigner(keypair, context), '', object='chat.completion.chunk')
    payload['choices'][0]['delta']['tool_calls'] = [{'function': {'name': 'stolen'}}]
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(json.dumps(payload).encode()),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), context),
        started=0,
        connected=0,
        streaming=False,
    )

    with pytest.raises(GatewayError, match='invalid or out of order'):
        session.read_verified()


def test_gateway_rejects_truncated_stream_without_signed_terminal_and_done():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    payload = _signed_chunk(RuntimeStreamSigner(keypair, context), 'partial')
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(f'data: {json.dumps(payload)}\n\n'.encode()),
        verifier=SignedStreamVerifier(_public_key_hex(keypair), context),
        started=0,
        connected=0,
        streaming=True,
    )

    with pytest.raises(GatewayError, match='truncated'):
        list(session.iter_verified())


def test_gateway_rejects_response_after_reservation_expires():
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(b'{}'),
        verifier=None,
        started=0,
        connected=0,
        streaming=False,
        reservation_expires_at=time.time() - 1,
    )

    with pytest.raises(GatewayError, match='reservation expired'):
        session.read_verified()


def test_gateway_rejects_oversized_or_malformed_runtime_responses():
    oversized = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(b'{}', {'Content-Length': '3'}),
        verifier=None,
        started=0,
        connected=0,
        streaming=False,
        max_response_bytes=2,
    )
    with pytest.raises(GatewayError, match='exceeds'):
        oversized.read_verified()

    malformed = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(b'not-json'),
        verifier=None,
        started=0,
        connected=0,
        streaming=False,
    )
    with pytest.raises(GatewayError, match='malformed'):
        malformed.read_verified()


def test_gateway_rejects_oversized_stream_lines():
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(b'data: 123456\n'),
        verifier=None,
        started=0,
        connected=0,
        streaming=True,
        max_stream_line_bytes=8,
    )

    with pytest.raises(GatewayError, match='line exceeds'):
        list(session.iter_verified())


def test_gateway_rejects_unsigned_sse_control_fields():
    session = GatewaySession(
        gateway=FakeGateway(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=BytesResponse(b'event: malicious\n'),
        verifier=None,
        started=0,
        connected=0,
        streaming=True,
    )

    with pytest.raises(GatewayError, match='unsigned SSE'):
        list(session.iter_verified())


@pytest.mark.parametrize('status', [400, 402, 403, 406, 413, 415, 422])
def test_client_caused_runtime_http_errors_do_not_penalize_the_gpu(status):
    assert not _miner_http_status_is_failure(status)


@pytest.mark.parametrize('status', [401, 404, 405, 409, 429, 500, 503])
def test_runtime_and_capacity_http_errors_count_against_the_gpu(status):
    assert _miner_http_status_is_failure(status)
