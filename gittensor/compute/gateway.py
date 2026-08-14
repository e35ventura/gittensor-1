"""OpenAI-compatible inference gateway with verified fastest-finish routing."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from gittensor.compute.http_json import load_json_object, read_json_object
from gittensor.compute.inference_verification import (
    SignedStreamVerifier,
    StreamProofContext,
    canonical_request_digest,
)
from gittensor.compute.request_capacity import estimate_request_capacity, normalize_openai_request
from gittensor.compute.safe_http import no_redirect_urlopen, public_https_request, validate_https_or_loopback_origin

_MAX_CONTROL_RESPONSE_BYTES = 4 * 1024 * 1024


class GatewayError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


class SessionGateway(Protocol):
    def finish(self, session: GatewaySession, *, success: bool | None) -> None: ...

    def renew(self, reservation_id: str) -> float: ...


@dataclass(frozen=True)
class GatewayConfig:
    control_plane_url: str
    control_plane_token_env: str
    gateway_token_env: str
    region: str
    request_timeout_seconds: float = 900.0
    control_timeout_seconds: float = 10.0
    catalog_ttl_seconds: float = 30.0
    estimated_tokens_per_second: float = 20.0
    estimated_fixed_seconds: float = 0.25
    max_request_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 32 * 1024 * 1024
    max_stream_line_bytes: int = 1024 * 1024
    reservation_renew_interval_seconds: float = 30.0
    rtt_probe_ttl_seconds: float = 30.0
    rtt_probe_timeout_seconds: float = 3.0
    require_stream_proof: bool = True

    @classmethod
    def load(cls, path: str | Path) -> GatewayConfig:
        payload = load_json_object(Path(path).read_bytes())
        return cls(**payload)


class InferenceGateway:
    def __init__(self, config: GatewayConfig) -> None:
        validate_https_or_loopback_origin(config.control_plane_url, 'gateway control_plane_url')
        numeric_values = (
            config.request_timeout_seconds,
            config.control_timeout_seconds,
            config.catalog_ttl_seconds,
            config.estimated_tokens_per_second,
            config.estimated_fixed_seconds,
            config.reservation_renew_interval_seconds,
            config.rtt_probe_ttl_seconds,
            config.rtt_probe_timeout_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in numeric_values):
            raise ValueError('gateway timeouts and inference estimates must be finite and positive')
        byte_limits = (
            config.max_request_bytes,
            config.max_response_bytes,
            config.max_stream_line_bytes,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in byte_limits):
            raise ValueError('gateway byte limits must be integers')
        if any(value < 1 for value in byte_limits):
            raise ValueError('gateway byte limits must be positive')
        self.config = config
        self.control_token = os.environ.get(config.control_plane_token_env)
        if not self.control_token:
            raise ValueError(f'{config.control_plane_token_env} must be set')
        self._catalog: list[dict[str, Any]] = []
        self._catalog_at = 0.0
        self._lock = threading.Lock()
        self._active_by_gpu: dict[str, int] = {}
        self._rtt_by_endpoint: dict[str, tuple[float, float]] = {}

    def open(self, request_payload: Mapping[str, Any]) -> GatewaySession:
        model = str(request_payload.get('model') or '')
        if not model:
            raise GatewayError(400, 'model is required')
        if 'stream' in request_payload and not isinstance(request_payload['stream'], bool):
            raise GatewayError(400, 'stream must be a boolean')
        release = self._resolve_release(model)
        runtime_request = dict(request_payload)
        # A release digest is a valid public selector, but the approved runtime
        # must always receive its canonical model ID. Never let a caller choose a
        # different model inside a multi-model backend.
        runtime_request['model'] = release['model_id']
        try:
            runtime_request = normalize_openai_request(runtime_request)
        except ValueError as exc:
            raise GatewayError(400, str(exc)) from exc
        estimated_input_tokens, max_output_tokens, expected_seconds = self._estimate_request(
            runtime_request,
            release,
        )
        route = self._control_post(
            '/v1/route',
            {
                'release_digest': release['release_digest'],
                'requester_region': self.config.region,
                'expected_service_seconds': expected_seconds,
                'estimated_input_tokens': estimated_input_tokens,
                'max_output_tokens': max_output_tokens,
            },
        )
        stream_public_key = str(route.get('stream_public_key') or '')
        if self.config.require_stream_proof and not stream_public_key:
            self._report_route_failure(route, release['release_digest'], expected_seconds, time.monotonic())
            raise GatewayError(502, 'selected runtime has no attested stream proof key')
        endpoint = str(route['endpoint']).rstrip('/')
        try:
            measured_rtt_ms = self._measure_network_rtt(endpoint)
        except (OSError, ValueError, GatewayError) as exc:
            self._report_route_failure(route, release['release_digest'], expected_seconds, time.monotonic())
            raise GatewayError(502, f'selected miner health probe failed: {exc}') from exc
        request_id = uuid.uuid4().hex
        created = int(time.time())
        miner_payload = {
            'request_id': request_id,
            'created': created,
            'release_digest': release['release_digest'],
            'openai_request': runtime_request,
        }
        request_digest = canonical_request_digest(miner_payload['openai_request'])
        context = StreamProofContext(
            request_id=request_id,
            created=created,
            release_digest=release['release_digest'],
            model_id=release['model_id'],
            model_revision=release['model_revision'],
            request_digest=request_digest,
        )
        path = '/v1/gittensor/inference'
        inference_token = str(route.get('inference_token') or '')
        if not inference_token:
            self._report_route_failure(route, release['release_digest'], expected_seconds, time.monotonic())
            raise GatewayError(502, 'selected runtime has no assignment-scoped inference token')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {inference_token}',
        }
        started = time.monotonic()
        session = GatewaySession(
            gateway=self,
            route=route,
            response=None,
            verifier=None,
            started=started,
            connected=started,
            streaming=bool(runtime_request.get('stream')),
            expected_service_seconds=expected_seconds,
            release_digest=release['release_digest'],
            reservation_expires_at=float(route['expires_at']),
            max_response_bytes=self.config.max_response_bytes,
            max_stream_line_bytes=self.config.max_stream_line_bytes,
            measured_rtt_ms=measured_rtt_ms,
        )
        session.start_renewal(self.config.reservation_renew_interval_seconds)
        try:
            response = public_https_request(
                f'{endpoint}{path}',
                body=json.dumps(miner_payload, separators=(',', ':')).encode(),
                headers=headers,
                method='POST',
                timeout=self.config.request_timeout_seconds,
            )
            if not 200 <= response.status < 300:
                error_body = response.read(self.config.max_response_bytes + 1)
                status = response.status
                response.close()
                session.stop_renewal()
                verified_client_error = (
                    len(error_body) <= self.config.max_response_bytes
                    and not _miner_http_status_is_failure(status)
                    and _verify_runtime_error(error_body, stream_public_key, context)
                )
                if verified_client_error:
                    self._complete_reservation(str(route['reservation_id']))
                else:
                    self._report_route_failure(route, release['release_digest'], expected_seconds, started)
                message = _runtime_error_message(error_body) if verified_client_error else 'runtime request failed'
                public_status = status if verified_client_error else HTTPStatus.BAD_GATEWAY.value
                raise GatewayError(public_status, message or f'miner returned HTTP {status}')
        except GatewayError:
            raise
        except (OSError, ValueError) as exc:
            session.stop_renewal()
            self._report_route_failure(route, release['release_digest'], expected_seconds, started)
            raise GatewayError(502, f'miner inference connection failed: {exc}') from exc
        connected = time.monotonic()
        if session.reservation_lost:
            response.close()
            session.stop_renewal()
            self._report_route_failure(route, release['release_digest'], expected_seconds, started)
            raise GatewayError(502, 'inference reservation expired while connecting to the runtime')
        with self._lock:
            gpu_id = str(route['gpu_id'])
            self._active_by_gpu[gpu_id] = self._active_by_gpu.get(gpu_id, 0) + 1
        verifier = SignedStreamVerifier(stream_public_key, context) if stream_public_key else None
        session.response = response
        session.verifier = verifier
        session.connected = connected
        return session

    def finish(self, session: GatewaySession, *, success: bool | None) -> None:
        gpu_id = str(session.route['gpu_id'])
        with self._lock:
            active = max(0, self._active_by_gpu.get(gpu_id, 1) - 1)
            if active:
                self._active_by_gpu[gpu_id] = active
            else:
                self._active_by_gpu.pop(gpu_id, None)
        total_seconds = max(0.0, time.monotonic() - session.started)
        service_seconds = max(0.0, total_seconds - session.measured_rtt_ms / 1000.0)
        if success is None:
            self._complete_reservation(str(session.route['reservation_id']))
            return
        try:
            self._control_post(
                '/v1/observations',
                {
                    'reservation_id': str(session.route['reservation_id']),
                    'gpu_id': gpu_id,
                    'requester_region': self.config.region,
                    'measured_rtt_ms': session.measured_rtt_ms,
                    'service_seconds': service_seconds,
                    'success': success,
                    'observed_active_slots': active,
                    'remaining_work_seconds': 0.0,
                    'expected_service_seconds': session.expected_service_seconds,
                    'release_digest': session.release_digest,
                },
            )
        except GatewayError:
            self._complete_reservation(str(session.route['reservation_id']))

    def _report_route_failure(
        self,
        route: Mapping[str, Any],
        release_digest: str,
        expected_service_seconds: float,
        started: float,
    ) -> None:
        try:
            self._control_post(
                '/v1/observations',
                {
                    'reservation_id': str(route['reservation_id']),
                    'gpu_id': str(route['gpu_id']),
                    'requester_region': self.config.region,
                    'measured_rtt_ms': 0.0,
                    'service_seconds': 0.0,
                    'success': False,
                    'observed_active_slots': 0,
                    'remaining_work_seconds': 0.0,
                    'expected_service_seconds': expected_service_seconds,
                    'release_digest': release_digest,
                },
            )
        except GatewayError:
            self._complete_reservation(str(route['reservation_id']))

    def renew(self, reservation_id: str) -> float:
        payload = self._control_post('/v1/reservations/renew', {'reservation_id': reservation_id})
        try:
            expires_at = float(payload['expires_at'])
        except (KeyError, TypeError, ValueError):
            raise GatewayError(502, 'control plane returned an invalid reservation renewal') from None
        if not math.isfinite(expires_at) or expires_at <= time.time():
            raise GatewayError(502, 'control plane returned an expired reservation renewal')
        return expires_at

    def _resolve_release(self, model: str) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._catalog_at >= self.config.catalog_ttl_seconds:
            payload = self._control_get('/v1/catalog')
            releases = payload.get('releases')
            if not isinstance(releases, list):
                raise GatewayError(502, 'control plane returned an invalid release catalog')
            if not all(isinstance(item, dict) for item in releases):
                raise GatewayError(502, 'control plane returned an invalid release catalog')
            self._catalog = [dict(item) for item in releases]
            self._catalog_at = now
        exact_digest = [release for release in self._catalog if release['release_digest'] == model]
        if exact_digest:
            return exact_digest[0]
        matches = [release for release in self._catalog if release['model_id'] == model]
        if not matches:
            raise GatewayError(404, f'unknown model: {model}')
        if len(matches) > 1:
            raise GatewayError(409, 'model has multiple approved revisions; request its exact release digest')
        return matches[0]

    def _estimate_request(
        self,
        payload: Mapping[str, Any],
        release: Mapping[str, Any],
    ) -> tuple[int, int, float]:
        try:
            request_overhead_tokens = release['request_overhead_tokens']
            max_context_tokens = release['max_context_tokens']
        except KeyError:
            raise GatewayError(502, 'release catalog is missing certified context capacity') from None
        if (
            not isinstance(request_overhead_tokens, int)
            or isinstance(request_overhead_tokens, bool)
            or request_overhead_tokens < 0
            or not isinstance(max_context_tokens, int)
            or isinstance(max_context_tokens, bool)
            or max_context_tokens < 1
        ):
            raise GatewayError(502, 'release catalog contains invalid context capacity')
        try:
            capacity = estimate_request_capacity(
                payload,
                request_overhead_tokens=request_overhead_tokens,
                max_context_tokens=max_context_tokens,
            )
        except ValueError as exc:
            raise GatewayError(400, str(exc)) from exc
        estimate = self.config.estimated_fixed_seconds + capacity.context_tokens / max(
            0.001, self.config.estimated_tokens_per_second
        )
        return (
            capacity.input_tokens,
            capacity.output_tokens,
            min(estimate, self.config.request_timeout_seconds),
        )

    def _measure_network_rtt(self, endpoint: str) -> float:
        now = time.monotonic()
        with self._lock:
            cached = self._rtt_by_endpoint.get(endpoint)
        if cached is not None and now - cached[1] < self.config.rtt_probe_ttl_seconds:
            return cached[0]
        started = time.monotonic()
        response = public_https_request(
            f'{endpoint}/health',
            body=None,
            headers={'Accept': 'application/json'},
            method='GET',
            timeout=self.config.rtt_probe_timeout_seconds,
        )
        try:
            if response.status != HTTPStatus.OK.value:
                raise GatewayError(502, 'selected miner health probe failed')
            body = response.read(64 * 1024 + 1)
            if len(body) > 64 * 1024:
                raise GatewayError(502, 'selected miner health response exceeds 64 KiB')
        finally:
            response.close()
        measured = max(0.001, (time.monotonic() - started) * 1000.0)
        with self._lock:
            self._rtt_by_endpoint[endpoint] = (measured, now)
        return measured

    def _control_get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f'{self.config.control_plane_url.rstrip("/")}{path}',
            headers={'Authorization': f'Bearer {self.control_token}', 'Accept': 'application/json'},
        )
        try:
            with no_redirect_urlopen(request, timeout=self.config.control_timeout_seconds) as response:
                return _read_control_response(response)
        except urllib.error.HTTPError as exc:
            raise GatewayError(exc.code, _http_error_message(exc)) from exc

    def _control_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f'{self.config.control_plane_url.rstrip("/")}{path}',
            data=json.dumps(payload, separators=(',', ':')).encode(),
            headers={
                'Authorization': f'Bearer {self.control_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST',
        )
        try:
            with no_redirect_urlopen(request, timeout=self.config.control_timeout_seconds) as response:
                return _read_control_response(response)
        except urllib.error.HTTPError as exc:
            raise GatewayError(exc.code, _http_error_message(exc)) from exc

    def _complete_reservation(self, reservation_id: str) -> None:
        try:
            self._control_post('/v1/reservations/complete', {'reservation_id': reservation_id})
        except GatewayError:
            pass


@dataclass
class GatewaySession:
    gateway: SessionGateway
    route: Mapping[str, Any]
    response: Any
    verifier: SignedStreamVerifier | None
    started: float
    connected: float
    streaming: bool
    expected_service_seconds: float = 0.0
    release_digest: str = ''
    reservation_expires_at: float = 0.0
    max_response_bytes: int = 32 * 1024 * 1024
    max_stream_line_bytes: int = 1024 * 1024
    measured_rtt_ms: float = 0.0
    closed: bool = False
    terminal_verified: bool = False
    reservation_lost: bool = False
    _reservation_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _renewal_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _renewal_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start_renewal(self, configured_interval: float) -> None:
        remaining = self.reservation_expires_at - time.time()
        interval = min(configured_interval, max(1.0, remaining / 3))
        self._renewal_thread = threading.Thread(
            target=self._renew_until_closed,
            args=(interval,),
            name=f'gittensor-reservation-{self.route["reservation_id"]}',
            daemon=True,
        )
        self._renewal_thread.start()

    def iter_verified(self) -> Iterator[bytes]:
        if not self.streaming:
            raise ValueError('session is not streaming')
        saw_done = False
        suppress_terminal_separator = False
        total_bytes = 0
        while line := self._readline():
            if len(line) > self.max_stream_line_bytes:
                raise GatewayError(502, 'runtime stream line exceeds the gateway limit')
            total_bytes += len(line)
            if total_bytes > self.max_response_bytes:
                raise GatewayError(502, 'runtime stream exceeds the gateway response limit')
            self._require_live_reservation()
            if saw_done and line not in {b'\n', b'\r\n'}:
                raise GatewayError(502, 'runtime sent content after the stream terminator')
            if suppress_terminal_separator and line in {b'\n', b'\r\n'}:
                suppress_terminal_separator = False
                continue
            if line.startswith(b'data:'):
                value = line[5:].strip()
                if value == b'[DONE]':
                    if saw_done:
                        raise GatewayError(502, 'runtime sent more than one stream terminator')
                    if self.verifier is not None and not self.terminal_verified:
                        raise GatewayError(502, 'runtime stream ended without a signed terminal proof')
                    saw_done = True
                elif value:
                    if saw_done:
                        raise GatewayError(502, 'runtime sent data after the stream terminator')
                    try:
                        payload = load_json_object(value)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        raise GatewayError(502, 'runtime returned malformed stream JSON') from exc
                    terminal = self._verify_payload(payload)
                    if terminal:
                        self.terminal_verified = True
                        suppress_terminal_separator = True
                        continue
            elif line not in {b'\n', b'\r\n'}:
                raise GatewayError(502, 'runtime stream contains unsigned SSE control fields')
            yield line
        if self.verifier is not None and (not self.terminal_verified or not saw_done):
            raise GatewayError(502, 'runtime stream was truncated before verified completion')
        self._require_live_reservation()

    def read_verified(self) -> bytes:
        if self.streaming:
            raise ValueError('session is streaming')
        content_length = self.response.headers.get('Content-Length')
        try:
            declared_length = int(content_length) if content_length is not None else None
        except ValueError:
            raise GatewayError(502, 'runtime returned an invalid Content-Length') from None
        if declared_length is not None and (declared_length < 0 or declared_length > self.max_response_bytes):
            raise GatewayError(502, 'runtime response exceeds the gateway limit')
        body = self.response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise GatewayError(502, 'runtime response exceeds the gateway limit')
        self._require_live_reservation()
        try:
            payload = load_json_object(body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise GatewayError(502, 'runtime returned malformed response JSON') from exc
        if self._verify_payload(payload):
            raise GatewayError(502, 'non-stream response cannot be a stream terminal event')
        return body

    def close(self, *, success: bool | None) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_renewal()
        self.response.close()
        self.gateway.finish(self, success=success)

    def _readline(self) -> bytes:
        try:
            return self.response.readline(self.max_stream_line_bytes + 1)
        except OSError as exc:
            raise GatewayError(502, f'runtime stream read failed: {exc}') from exc

    def stop_renewal(self) -> None:
        self._renewal_stop.set()
        thread = self._renewal_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)

    def _renew_until_closed(self, interval: float) -> None:
        reservation_id = str(self.route['reservation_id'])
        while not self._renewal_stop.wait(interval):
            try:
                renewed_until = self.gateway.renew(reservation_id)
                with self._reservation_lock:
                    self.reservation_expires_at = renewed_until
            except Exception as exc:
                with self._reservation_lock:
                    expired = time.time() >= self.reservation_expires_at
                not_found = isinstance(exc, GatewayError) and exc.status == HTTPStatus.NOT_FOUND
                if not_found or expired:
                    with self._reservation_lock:
                        self.reservation_lost = True
                    if self.response is not None:
                        self.response.close()
                    return

    def _require_live_reservation(self) -> None:
        with self._reservation_lock:
            reservation_lost = self.reservation_lost
            expires_at = self.reservation_expires_at
        if reservation_lost or (expires_at > 0 and time.time() >= expires_at):
            raise GatewayError(502, 'inference reservation expired before the response completed')

    def _verify_payload(self, payload: Mapping[str, Any]) -> bool:
        if self.verifier is None:
            return False
        proof = payload.get('gittensor_proof')
        if not isinstance(proof, dict):
            raise GatewayError(502, 'runtime response is missing its attested stream proof')
        choices = payload.get('choices')
        if not isinstance(choices, list):
            raise GatewayError(502, 'runtime response does not contain OpenAI-compatible choices')
        if not self.verifier.verify(
            int(proof.get('index', -1)),
            payload,
            str(proof.get('signature') or ''),
        ):
            raise GatewayError(502, 'runtime stream proof is invalid or out of order')
        terminal = payload.get('gittensor_terminal') is True
        if terminal and choices:
            raise GatewayError(502, 'runtime terminal proof must not contain response choices')
        if self.terminal_verified:
            raise GatewayError(502, 'runtime sent more than one terminal proof')
        return terminal


def make_handler(gateway: InferenceGateway, bearer_token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = 'GittensorInferenceGateway/1'

        def do_GET(self) -> None:
            if self.path == '/health':
                self._send(HTTPStatus.OK, {'status': 'ok'})
                return
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not_found'})

        def do_POST(self) -> None:
            authorization = self.headers.get('Authorization')
            if not isinstance(authorization, str) or not secrets.compare_digest(
                authorization,
                f'Bearer {bearer_token}',
            ):
                self._send(HTTPStatus.UNAUTHORIZED, {'error': 'unauthorized'})
                return
            session: GatewaySession | None = None
            headers_sent = False
            runtime_response_received = False
            try:
                payload = self._read_json()
                if self.path != '/v1/chat/completions':
                    self._send(HTTPStatus.NOT_FOUND, {'error': 'not_found'})
                    return
                session = gateway.open(payload)
                if session.streaming:
                    iterator = session.iter_verified()
                    first = next(iterator, b'')
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    headers_sent = True
                    if first:
                        self.wfile.write(first)
                    for line in iterator:
                        self.wfile.write(line)
                        self.wfile.flush()
                    runtime_response_received = True
                else:
                    body = session.read_verified()
                    runtime_response_received = True
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    headers_sent = True
                    self.wfile.write(body)
                session.close(success=True)
            except GatewayError as exc:
                if session is not None:
                    session.close(success=False)
                if headers_sent:
                    error = json.dumps({'error': {'message': exc.message, 'type': 'verification_error'}})
                    self.wfile.write(f'data: {error}\n\n'.encode())
                    self.wfile.flush()
                else:
                    self._send(HTTPStatus(exc.status), {'error': exc.message})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                if session is not None:
                    session.close(success=False)
                if not headers_sent:
                    self._send(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
            except (BrokenPipeError, ConnectionResetError) as exc:
                if session is not None:
                    session.close(success=True if runtime_response_received else None)
                if not headers_sent:
                    self._send(HTTPStatus.BAD_GATEWAY, {'error': str(exc)})
            except Exception as exc:
                if session is not None:
                    session.close(success=False)
                if not headers_sent:
                    self._send(HTTPStatus.BAD_GATEWAY, {'error': str(exc)})

        def _read_json(self) -> dict[str, Any]:
            return read_json_object(self.rfile, self.headers, gateway.config.max_request_bytes)

        def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, separators=(',', ':')).encode()
            self.send_response(status.value)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _http_error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = load_json_object(error.read(_MAX_CONTROL_RESPONSE_BYTES + 1))
        return str(payload.get('message') or payload.get('error') or error.reason)
    except Exception:
        return str(error.reason)


def _read_control_response(response: Any) -> dict[str, Any]:
    content_length = response.headers.get('Content-Length')
    if content_length is not None:
        try:
            declared_length = int(content_length)
            if declared_length < 0 or declared_length > _MAX_CONTROL_RESPONSE_BYTES:
                raise GatewayError(502, 'control-plane response exceeds 4 MiB')
        except ValueError:
            raise GatewayError(502, 'control-plane response has an invalid Content-Length') from None
    body = response.read(_MAX_CONTROL_RESPONSE_BYTES + 1)
    if len(body) > _MAX_CONTROL_RESPONSE_BYTES:
        raise GatewayError(502, 'control-plane response exceeds 4 MiB')
    try:
        payload = load_json_object(body)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GatewayError(502, 'control-plane response is not valid JSON') from exc
    return payload


def _miner_http_status_is_failure(status: int) -> bool:
    """Return whether a status is ineligible for attested client-error handling."""
    client_error_statuses = {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.PAYMENT_REQUIRED,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.NOT_ACCEPTABLE,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }
    return status not in client_error_statuses


def _verify_runtime_error(body: bytes, public_key: str, context: StreamProofContext) -> bool:
    if not public_key:
        return False
    try:
        payload = load_json_object(body)
        if 'error' not in payload:
            return False
        proof = payload.get('gittensor_proof')
        if not isinstance(proof, dict):
            return False
        return SignedStreamVerifier(public_key, context).verify(
            int(proof.get('index', -1)),
            payload,
            str(proof.get('signature') or ''),
        )
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _runtime_error_message(body: bytes) -> str:
    try:
        payload = load_json_object(body)
        error = payload.get('error')
        if isinstance(error, dict):
            return str(error.get('message') or 'runtime rejected the request')
        return str(error or 'runtime rejected the request')
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return 'runtime rejected the request'


def build_parser() -> argparse.ArgumentParser:
    from gittensor.compute.entrypoints import gateway_parser

    return gateway_parser()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or build_parser().parse_args()
    config = GatewayConfig.load(args.config)
    gateway_token = os.environ.get(config.gateway_token_env)
    if not gateway_token:
        raise SystemExit(f'{config.gateway_token_env} must be set')
    gateway = InferenceGateway(config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(gateway, gateway_token))
    print(f'gittensor inference gateway listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
