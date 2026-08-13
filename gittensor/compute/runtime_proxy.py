"""Approved-runtime sidecar that signs exact OpenAI responses."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urlsplit

import bittensor as bt

from gittensor.compute.http_json import load_json_object, read_json_object
from gittensor.compute.inference_verification import (
    RuntimeStreamSigner,
    StreamProofContext,
    canonical_request_digest,
)


@dataclass(frozen=True)
class RuntimeProxyConfig:
    backend_url: str
    release_digest: str
    model_id: str
    model_revision: str
    proof_scheme: str = 'sr25519-response-v1'
    backend_timeout_seconds: float = 900.0
    max_request_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 32 * 1024 * 1024
    max_stream_line_bytes: int = 1024 * 1024

    @classmethod
    def from_environment(cls) -> RuntimeProxyConfig:
        required = {
            'release_digest': os.environ.get('GITTENSOR_RELEASE_DIGEST'),
            'model_id': os.environ.get('GITTENSOR_MODEL_ID'),
            'model_revision': os.environ.get('GITTENSOR_MODEL_REVISION'),
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError(f'runtime proxy environment is missing: {", ".join(missing)}')
        return cls(
            backend_url=os.environ.get('GITTENSOR_BACKEND_URL', 'http://127.0.0.1:8001'),
            release_digest=str(required['release_digest']),
            model_id=str(required['model_id']),
            model_revision=str(required['model_revision']),
            proof_scheme=os.environ.get('GITTENSOR_PROOF_SCHEME', 'sr25519-response-v1'),
        )

    def validate(self) -> None:
        parsed = urlsplit(self.backend_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', '::1', 'localhost'}:
            raise ValueError('runtime backend must be a loopback HTTP endpoint')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('runtime backend URL cannot contain userinfo, query, or fragment')
        if self.proof_scheme != 'sr25519-response-v1':
            raise ValueError('runtime proxy supports only sr25519-response-v1')
        if not math.isfinite(self.backend_timeout_seconds) or self.backend_timeout_seconds <= 0:
            raise ValueError('runtime backend timeout must be finite and positive')
        for value in (self.max_request_bytes, self.max_response_bytes, self.max_stream_line_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('runtime proxy byte limits must be positive integers')


class SigningRuntimeProxy:
    """Bind an isolated backend response to an assignment-scoped signing key."""

    def __init__(self, config: RuntimeProxyConfig, *, keypair: bt.Keypair | None = None) -> None:
        config.validate()
        self.config = config
        self.keypair = keypair or bt.Keypair.create_from_seed(os.urandom(32).hex())
        public_key = self.keypair.public_key
        if public_key is None:
            raise ValueError('runtime signing key did not produce a public key')
        self.public_key_hex = public_key.hex()

    def identity(self) -> dict[str, str]:
        return {
            'release_digest': self.config.release_digest,
            'model_id': self.config.model_id,
            'model_revision': self.config.model_revision,
            'stream_public_key': self.public_key_hex,
            'proof_scheme': self.config.proof_scheme,
        }

    def backend_ready(self) -> bool:
        request = urllib.request.Request(
            f'{self.config.backend_url.rstrip("/")}/health',
            headers={'Accept': 'application/json,text/plain'},
            method='GET',
        )
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=min(self.config.backend_timeout_seconds, 10.0)) as response:
                body = response.read(64 * 1024 + 1)
                return int(response.status) == HTTPStatus.OK.value and len(body) <= 64 * 1024
        except (OSError, urllib.error.URLError):
            return False

    def open_backend(self, body: bytes, *, stream: bool) -> Any:
        request = urllib.request.Request(
            f'{self.config.backend_url.rstrip("/")}/v1/chat/completions',
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream' if stream else 'application/json',
            },
            method='POST',
        )
        try:
            return _NO_REDIRECT_OPENER.open(request, timeout=self.config.backend_timeout_seconds)
        except urllib.error.HTTPError as response:
            # Preserve model-server 4xx errors so the public gateway can return a
            # client error without treating an honest GPU as failed. Redirects
            # and 5xx responses remain runtime failures.
            if 400 <= response.code < 500:
                return response
            response.close()
            raise

    def signer(self, headers: Mapping[str, str], request_payload: Mapping[str, Any]) -> RuntimeStreamSigner:
        if request_payload.get('model') != self.config.model_id:
            raise ValueError('runtime request does not target the assigned model')
        normalized_headers = {str(key).casefold(): str(value) for key, value in headers.items()}
        required_headers = {
            'X-Gittensor-Release-Digest': self.config.release_digest,
            'X-Gittensor-Model-Id': self.config.model_id,
            'X-Gittensor-Model-Revision': self.config.model_revision,
            'X-Gittensor-Stream-Public-Key': self.public_key_hex,
        }
        for key, expected in required_headers.items():
            actual = normalized_headers.get(key.casefold(), '')
            if not secrets.compare_digest(actual, expected):
                raise ValueError(f'{key} does not match the active runtime')
        request_id = normalized_headers.get('x-gittensor-request-id', '')
        request_digest = normalized_headers.get('x-gittensor-request-digest', '')
        try:
            created = int(normalized_headers.get('x-gittensor-created', ''))
        except ValueError:
            raise ValueError('X-Gittensor-Created must be an integer') from None
        expected_digest = canonical_request_digest(request_payload)
        if not request_id or created <= 0 or not secrets.compare_digest(request_digest, expected_digest):
            raise ValueError('runtime request identity or digest is invalid')
        return RuntimeStreamSigner(
            self.keypair,
            StreamProofContext(
                request_id=request_id,
                created=created,
                release_digest=self.config.release_digest,
                model_id=self.config.model_id,
                model_revision=self.config.model_revision,
                request_digest=expected_digest,
            ),
        )


def make_handler(proxy: SigningRuntimeProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = 'GittensorSignedRuntime/1'

        def do_GET(self) -> None:
            if self.path == '/health':
                if proxy.backend_ready():
                    self._send_json(HTTPStatus.OK, {'status': 'ready'})
                else:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {'status': 'backend_unavailable'})
            elif self.path == '/v1/gittensor/runtime':
                self._send_json(HTTPStatus.OK, proxy.identity())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {'error': 'not_found'})

        def do_POST(self) -> None:
            if self.path != '/v1/chat/completions':
                self._send_json(HTTPStatus.NOT_FOUND, {'error': 'not_found'})
                return
            backend = None
            stream_started = False
            try:
                body, payload = self._read_request()
                signer = proxy.signer({key: value for key, value in self.headers.items()}, payload)
                stream = payload.get('stream') is True
                backend = proxy.open_backend(body, stream=stream)
                if int(backend.status) >= 300:
                    error_body = backend.read(proxy.config.max_response_bytes + 1)
                    if len(error_body) > proxy.config.max_response_bytes:
                        raise ValueError('backend error response exceeds the runtime proxy limit')
                    content_type = str(backend.headers.get('Content-Type', '')).partition(';')[0].strip().casefold()
                    if content_type != 'application/json':
                        raise ValueError('backend error response must use application/json')
                    error_payload = signer.attach_error(load_json_object(error_body))
                    encoded_error = json.dumps(error_payload, separators=(',', ':'), allow_nan=False).encode()
                    self._send_bytes(HTTPStatus(int(backend.status)), encoded_error, 'application/json')
                    return
                content_type = str(backend.headers.get('Content-Type', '')).partition(';')[0].strip().casefold()
                if stream:
                    if content_type != 'text/event-stream':
                        raise ValueError('streaming backend response must use text/event-stream')
                    stream_started = True
                    self._proxy_stream(backend, signer)
                else:
                    if content_type != 'application/json':
                        raise ValueError('backend response must use application/json')
                    self._proxy_response(backend, signer)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if stream_started:
                    self.close_connection = True
                else:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {'error': 'runtime_contract_failure', 'message': str(exc)})
            except urllib.error.HTTPError as exc:
                if stream_started:
                    self.close_connection = True
                else:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {'error': 'backend_failure', 'message': str(exc.reason)})
            except Exception as exc:
                if stream_started:
                    self.close_connection = True
                else:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {'error': 'runtime_failure', 'message': str(exc)})
            finally:
                if backend is not None:
                    backend.close()

        def _read_request(self) -> tuple[bytes, dict[str, Any]]:
            payload = read_json_object(self.rfile, self.headers, proxy.config.max_request_bytes)
            body = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
            if 'stream' in payload and not isinstance(payload['stream'], bool):
                raise ValueError('stream must be a boolean')
            return body, payload

        def _proxy_response(self, backend: Any, signer: RuntimeStreamSigner) -> None:
            body = backend.read(proxy.config.max_response_bytes + 1)
            if len(body) > proxy.config.max_response_bytes:
                raise ValueError('backend response exceeds the runtime proxy limit')
            payload = load_json_object(body)
            signed = signer.attach(_clean_backend_payload(payload))
            encoded = json.dumps(signed, separators=(',', ':'), allow_nan=False).encode()
            self._send_bytes(HTTPStatus.OK, encoded, 'application/json')

        def _proxy_stream(self, backend: Any, signer: RuntimeStreamSigner) -> None:
            self.send_response(HTTPStatus.OK.value)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            total_bytes = 0
            while line := backend.readline(proxy.config.max_stream_line_bytes + 1):
                if len(line) > proxy.config.max_stream_line_bytes:
                    raise ValueError('backend stream line exceeds the runtime proxy limit')
                total_bytes += len(line)
                if total_bytes > proxy.config.max_response_bytes:
                    raise ValueError('backend stream exceeds the runtime proxy limit')
                if line in {b'\n', b'\r\n'}:
                    continue
                if not line.startswith(b'data:'):
                    raise ValueError('backend stream contains unsupported SSE control fields')
                value = line[5:].strip()
                if value == b'[DONE]':
                    terminal = signer.terminal()
                    self.wfile.write(f'data: {json.dumps(terminal, separators=(",", ":"))}\n\n'.encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                    return
                payload = load_json_object(value)
                signed = signer.attach(_clean_backend_payload(payload))
                self.wfile.write(f'data: {json.dumps(signed, separators=(",", ":"))}\n\n'.encode())
                self.wfile.flush()
            raise ValueError('backend stream ended without [DONE]')

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            self._send_bytes(status, json.dumps(payload, separators=(',', ':')).encode(), 'application/json')

        def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _clean_backend_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if key not in {'gittensor_proof', 'gittensor_terminal'}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the Gittensor signed-runtime proxy')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = args or build_parser().parse_args()
    proxy = SigningRuntimeProxy(RuntimeProxyConfig.from_environment())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(proxy))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
