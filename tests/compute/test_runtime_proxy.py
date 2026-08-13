import json
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bittensor as bt
import pytest

from gittensor.compute.gateway import GatewaySession
from gittensor.compute.inference_verification import (
    SignedStreamVerifier,
    StreamProofContext,
    canonical_request_digest,
)
from gittensor.compute.runtime_proxy import (
    RuntimeProxyConfig,
    SigningRuntimeProxy,
    _clean_backend_payload,
    make_handler,
)


def _proxy(tmp_path):
    return SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url='http://127.0.0.1:8001',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        ),
        keypair=bt.Keypair.create_from_seed('11' * 32),
    )


def _headers(proxy, request):
    return {
        'X-Gittensor-Release-Digest': 'sha256:release',
        'X-Gittensor-Model-Id': 'model',
        'X-Gittensor-Model-Revision': 'a' * 40,
        'X-Gittensor-Stream-Public-Key': proxy.public_key_hex,
        'X-Gittensor-Request-Id': 'request-1',
        'X-Gittensor-Created': '100',
        'X-Gittensor-Request-Digest': canonical_request_digest(request),
    }


def test_runtime_proxy_identity_and_response_signature_bind_assignment_and_request(tmp_path):
    proxy = _proxy(tmp_path)
    request = {'model': 'model', 'messages': [{'role': 'user', 'content': 'hello'}]}
    signer = proxy.signer(_headers(proxy, request), request)
    signed = signer.attach({'choices': [{'message': {'role': 'assistant', 'content': 'hi'}}]})
    context = StreamProofContext(
        'request-1',
        100,
        'sha256:release',
        'model',
        'a' * 40,
        canonical_request_digest(request),
    )
    verifier = SignedStreamVerifier(proxy.public_key_hex, context)

    assert proxy.identity()['stream_public_key'] == proxy.public_key_hex
    assert verifier.verify(0, signed, signed['gittensor_proof']['signature'])


def test_runtime_proxy_rejects_wrong_request_digest_or_attested_key(tmp_path):
    proxy = _proxy(tmp_path)
    request = {'model': 'model'}
    headers = _headers(proxy, request)
    headers['X-Gittensor-Request-Digest'] = canonical_request_digest({'model': 'other'})

    with pytest.raises(ValueError, match='identity or digest'):
        proxy.signer(headers, request)

    headers = _headers(proxy, request)
    headers['X-Gittensor-Stream-Public-Key'] = '22' * 32
    with pytest.raises(ValueError, match='active runtime'):
        proxy.signer(headers, request)

    wrong_model = {'model': 'small-model'}
    with pytest.raises(ValueError, match='assigned model'):
        proxy.signer(_headers(proxy, wrong_model), wrong_model)


def test_runtime_proxy_removes_backend_supplied_proof_fields():
    cleaned = _clean_backend_payload(
        {
            'choices': [],
            'gittensor_proof': {'signature': 'forged'},
            'gittensor_terminal': True,
        }
    )

    assert cleaned == {'choices': []}


def test_runtime_proxy_rejects_non_loopback_backend(tmp_path):
    with pytest.raises(ValueError, match='loopback'):
        SigningRuntimeProxy(
            RuntimeProxyConfig(
                backend_url='https://miner.example',
                release_digest='sha256:release',
                model_id='model',
                model_revision='a' * 40,
            )
        )


def test_runtime_proxy_key_matches_bittensor_seed_derivation(tmp_path):
    proxy = _proxy(tmp_path)
    expected = bt.Keypair.create_from_seed('11' * 32).public_key

    assert expected is not None
    assert proxy.public_key_hex == expected.hex()


def test_runtime_proxy_generates_an_in_memory_key_when_no_seed_is_injected(tmp_path):
    proxy = SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url='http://127.0.0.1:8001',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        )
    )

    assert len(proxy.public_key_hex) == 64
    assert not list(tmp_path.iterdir())


def test_runtime_proxy_health_requires_live_backend(tmp_path):
    proxy = _proxy(tmp_path)
    proxy_server, proxy_thread = _start_server(make_handler(proxy))
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(f'http://127.0.0.1:{proxy_server.server_port}/health', timeout=2)
        assert failure.value.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    finally:
        proxy_server.shutdown()
        proxy_thread.join(2)


class _RedirectBackendHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(HTTPStatus.TEMPORARY_REDIRECT.value)
        self.send_header('Location', 'https://example.com/v1/chat/completions')
        self.end_headers()

    def log_message(self, format, *args):
        return


def test_runtime_proxy_never_follows_backend_redirect(tmp_path):
    backend, backend_thread = _start_server(_RedirectBackendHandler)
    proxy = SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url=f'http://127.0.0.1:{backend.server_port}',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        )
    )
    request_payload = {'model': 'model'}
    with pytest.raises(urllib.error.HTTPError) as failure:
        proxy.open_backend(json.dumps(request_payload).encode(), stream=False)
    assert failure.value.code == HTTPStatus.TEMPORARY_REDIRECT.value
    backend.shutdown()
    backend_thread.join(2)


class _BackendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/health':
            self.send_error(HTTPStatus.NOT_FOUND.value)
            return
        body = b'{"status":"ready"}'
        self.send_response(HTTPStatus.OK.value)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        payload = json.loads(self.rfile.read(length))
        if payload.get('reject') is True:
            body = b'{"error":{"message":"invalid prompt"}}'
            self.send_response(HTTPStatus.UNPROCESSABLE_ENTITY.value)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if payload.get('stream'):
            body = (
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            content_type = 'text/event-stream'
        else:
            body = b'{"choices":[{"message":{"role":"assistant","content":"hello"}}]}'
            content_type = 'application/json'
        self.send_response(HTTPStatus.OK.value)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class _GatewayRecorder:
    def finish(self, session, *, success):
        return

    def renew(self, reservation_id):
        return 10**12


def _start_server(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.parametrize('stream', [False, True])
def test_runtime_proxy_http_output_passes_gateway_verification(tmp_path, stream):
    backend, backend_thread = _start_server(_BackendHandler)
    proxy = SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url=f'http://127.0.0.1:{backend.server_port}',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        )
    )
    proxy_server, proxy_thread = _start_server(make_handler(proxy))
    request_payload = {'model': 'model', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream}
    context = StreamProofContext(
        'request-1',
        100,
        'sha256:release',
        'model',
        'a' * 40,
        canonical_request_digest(request_payload),
    )
    request = urllib.request.Request(
        f'http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions',
        data=json.dumps(request_payload).encode(),
        headers={**_headers(proxy, request_payload), 'Content-Type': 'application/json'},
        method='POST',
    )
    response = urllib.request.urlopen(request, timeout=2)
    session = GatewaySession(
        gateway=_GatewayRecorder(),
        route={'gpu_id': 'gpu-1', 'reservation_id': 'reservation-1'},
        response=response,
        verifier=SignedStreamVerifier(proxy.public_key_hex, context),
        started=0,
        connected=0,
        streaming=stream,
    )
    try:
        if stream:
            output = b''.join(session.iter_verified())
            assert b'hello' in output
            assert b' world' in output
            assert output.endswith(b'data: [DONE]\n\n')
        else:
            output = json.loads(session.read_verified())
            assert output['choices'][0]['message']['content'] == 'hello'
    finally:
        session.close(success=True)
        proxy_server.shutdown()
        backend.shutdown()
        proxy_thread.join(2)
        backend_thread.join(2)


def test_runtime_proxy_preserves_bounded_backend_client_errors(tmp_path):
    backend, backend_thread = _start_server(_BackendHandler)
    proxy = SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url=f'http://127.0.0.1:{backend.server_port}',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        )
    )
    proxy_server, proxy_thread = _start_server(make_handler(proxy))
    request_payload = {'model': 'model', 'reject': True}
    request = urllib.request.Request(
        f'http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions',
        data=json.dumps(request_payload).encode(),
        headers={**_headers(proxy, request_payload), 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request, timeout=2)
        assert failure.value.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert json.loads(failure.value.read())['error']['message'] == 'invalid prompt'
    finally:
        proxy_server.shutdown()
        backend.shutdown()
        proxy_thread.join(2)
        backend_thread.join(2)


def test_runtime_proxy_reports_internal_request_binding_failure_as_bad_gateway(tmp_path):
    backend, backend_thread = _start_server(_BackendHandler)
    proxy = SigningRuntimeProxy(
        RuntimeProxyConfig(
            backend_url=f'http://127.0.0.1:{backend.server_port}',
            release_digest='sha256:release',
            model_id='model',
            model_revision='a' * 40,
        )
    )
    proxy_server, proxy_thread = _start_server(make_handler(proxy))
    request_payload = {'model': 'model'}
    headers = _headers(proxy, request_payload)
    headers['X-Gittensor-Request-Digest'] = canonical_request_digest({'model': 'tampered'})
    request = urllib.request.Request(
        f'http://127.0.0.1:{proxy_server.server_port}/v1/chat/completions',
        data=json.dumps(request_payload).encode(),
        headers={**headers, 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request, timeout=2)
        assert failure.value.code == HTTPStatus.BAD_GATEWAY.value
    finally:
        proxy_server.shutdown()
        backend.shutdown()
        proxy_thread.join(2)
        backend_thread.join(2)
