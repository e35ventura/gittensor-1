"""Signed miner agent for executing global Gepetto assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Protocol

import bittensor as bt

from gittensor.compute.auth import (
    AuthenticationError,
    MinerRequestSigner,
    ValidatorCommandAuthenticator,
)
from gittensor.compute.http_json import load_json_object, read_json_object
from gittensor.compute.inference_tokens import InferenceCapability, verify_inference_capability
from gittensor.compute.inference_verification import canonical_request_digest
from gittensor.compute.miner_runtime import ContainerRuntimeConfig, ContainerRuntimeManager, RuntimeManager
from gittensor.compute.models import AssignmentCommand, GPUState, RuntimeEvidence
from gittensor.compute.request_capacity import estimate_request_capacity, normalize_openai_request
from gittensor.compute.safe_http import no_redirect_urlopen, validate_https_or_loopback_origin

_MAX_CONTROL_RESPONSE_BYTES = 64 * 1024


class AgentStateStore(Protocol):
    def consume_nonce(self, hotkey: str, nonce: str, expires_at: float) -> bool: ...

    def load(self) -> dict[str, Any] | None: ...

    def save(self, state: Mapping[str, Any]) -> None: ...


class AcknowledgementClient(Protocol):
    def acknowledge(
        self,
        gpu_id: str,
        epoch: int,
        state: GPUState,
        evidence: RuntimeEvidence | None = None,
    ) -> None: ...


class InferenceCapacityUnavailable(RuntimeError):
    """The signed assignment's certified runtime concurrency is full."""


class SQLiteAgentStateStore:
    """Durable assignment and replay state for one miner agent."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS command_nonces (
                    hotkey TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY(hotkey, nonce)
                );
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    def consume_nonce(self, hotkey: str, nonce: str, expires_at: float) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM command_nonces WHERE expires_at <= ?', (time.time(),))
            try:
                connection.execute(
                    'INSERT INTO command_nonces(hotkey, nonce, expires_at) VALUES(?, ?, ?)',
                    (hotkey, nonce, expires_at),
                )
            except sqlite3.IntegrityError:
                connection.execute('ROLLBACK')
                return False
            connection.execute('COMMIT')
        return True

    def load(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute('SELECT value FROM agent_state WHERE id = 1').fetchone()
        return json.loads(row['value']) if row else None

    def save(self, state: Mapping[str, Any]) -> None:
        value = json.dumps(state, sort_keys=True, separators=(',', ':'))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_state(id, value, updated_at) VALUES(1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (value, time.time()),
            )


class ControlPlaneAcknowledgementClient:
    """Report lifecycle progress using the miner's SN74 hotkey."""

    def __init__(
        self,
        control_plane_url: str,
        signer: MinerRequestSigner,
        timeout_seconds: float,
    ) -> None:
        validate_https_or_loopback_origin(control_plane_url, 'miner control_plane_url')
        self.control_plane_url = control_plane_url.rstrip('/')
        self.signer = signer
        self.timeout_seconds = timeout_seconds

    def acknowledge(
        self,
        gpu_id: str,
        epoch: int,
        state: GPUState,
        evidence: RuntimeEvidence | None = None,
    ) -> None:
        path = '/v1/assignments/ack'
        payload: dict[str, Any] = {'gpu_id': gpu_id, 'epoch': epoch, 'state': state.value}
        if evidence is not None:
            payload['evidence'] = asdict(evidence)
        signed = self.signer.sign('POST', path, payload)
        request = urllib.request.Request(
            f'{self.control_plane_url}{path}',
            data=json.dumps(signed, separators=(',', ':')).encode(),
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            method='POST',
        )
        with no_redirect_urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(_MAX_CONTROL_RESPONSE_BYTES + 1)
            if len(body) > _MAX_CONTROL_RESPONSE_BYTES:
                raise ValueError('control-plane acknowledgement response exceeds 64 KiB')
            result = load_json_object(body)
        accepted_states = {
            GPUState.LOADING: {GPUState.LOADING.value, GPUState.RUNTIME_VERIFY.value, GPUState.READY.value},
            GPUState.RUNTIME_VERIFY: {GPUState.RUNTIME_VERIFY.value, GPUState.READY.value},
        }
        if result.get('state') not in accepted_states.get(state, {state.value}):
            raise ValueError('control plane did not accept the exact assignment lifecycle state')


@dataclass(frozen=True)
class MinerAgentConfig:
    gpu_id: str
    expected_validator_hotkey: str
    control_plane_url: str
    miner_wallet_name: str
    miner_wallet_hotkey: str
    miner_wallet_path: str | None
    miner_identity_hotkey: str
    database_path: str
    weights_root: str
    gpu_device: str = '0'
    container_engine: str = 'docker'
    container_name: str = 'gittensor-runtime'
    internal_network: str = 'gittensor-inference-internal'
    model_downloader_binary: str = 'hf'
    runtime_port: int = 18000
    runtime_container_port: int = 8000
    runtime_health_path: str = '/health'
    command_ttl_seconds: float = 30.0
    control_plane_timeout_seconds: float = 10.0
    acknowledgement_retry_seconds: float = 30.0
    drain_timeout_seconds: float = 300.0
    inference_timeout_seconds: float = 900.0
    max_request_bytes: int = 8 * 1024 * 1024
    runtime_pids_limit: int = 4096
    runtime_shm_size: str = '16g'

    @classmethod
    def load(cls, path: str | Path) -> MinerAgentConfig:
        payload = load_json_object(Path(path).read_bytes())
        return cls(**payload)


class MinerRuntimeAgent:
    """Persist, execute, and prove monotonic validator assignments."""

    def __init__(
        self,
        config: MinerAgentConfig,
        runtime: RuntimeManager,
        store: AgentStateStore,
        authenticator: ValidatorCommandAuthenticator,
        acknowledgements: AcknowledgementClient,
    ) -> None:
        numeric_values = (
            config.command_ttl_seconds,
            config.control_plane_timeout_seconds,
            config.acknowledgement_retry_seconds,
            config.drain_timeout_seconds,
            config.inference_timeout_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in numeric_values):
            raise ValueError('miner agent timeouts must be finite and positive')
        if not isinstance(config.max_request_bytes, int) or isinstance(config.max_request_bytes, bool):
            raise ValueError('miner max_request_bytes must be an integer')
        if config.max_request_bytes < 1:
            raise ValueError('miner max_request_bytes must be positive')
        validate_https_or_loopback_origin(config.control_plane_url, 'miner control_plane_url')
        if (
            not isinstance(config.runtime_pids_limit, int)
            or isinstance(config.runtime_pids_limit, bool)
            or config.runtime_pids_limit < 1
        ):
            raise ValueError('miner runtime_pids_limit must be a positive integer')
        self.config = config
        self.runtime = runtime
        self.store = store
        self.authenticator = authenticator
        self.acknowledgements = acknowledgements
        self._lock = threading.RLock()
        self._runtime_lock = threading.Lock()
        self._command: AssignmentCommand | None = None
        self._state = GPUState.REGISTERED
        self._last_error: str | None = None
        self._stream_public_key = ''
        self._worker: threading.Thread | None = None
        self._revocation_worker: threading.Thread | None = None
        self._accepting_inference = False
        self._active_inference = 0
        self._active_kv_bytes = 0
        self._revoked_epoch = 0
        self._revocation_draining = False
        self._drained = threading.Condition(self._lock)
        if saved := store.load():
            command = saved.get('command')
            self._command = AssignmentCommand(**command) if command else None
            self._state = GPUState(saved.get('state', GPUState.REGISTERED.value))
            self._last_error = saved.get('last_error')
            self._stream_public_key = str(saved.get('stream_public_key') or '')
            self._revoked_epoch = int(saved.get('revoked_epoch', 0))
            self._revocation_draining = bool(saved.get('revocation_draining', False))

    def authenticate(self, method: str, path: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> None:
        self.authenticator.authenticate(method, path, payload, headers)

    def authenticate_inference(self, authorization: str) -> InferenceCapability:
        with self._lock:
            command = self._command
            if command is None or command.epoch <= self._revoked_epoch or not authorization.startswith('Bearer '):
                raise AuthenticationError('inference token is invalid or stale')
            token = authorization.removeprefix('Bearer ')
            capability = verify_inference_capability(command.assignment_token, token, now=time.time())
            if (
                capability is None
                or capability.gpu_id != command.gpu_id
                or capability.release_digest != command.release_digest
            ):
                raise AuthenticationError('inference token is invalid or stale')
            nonce_key = f'inference:{command.gpu_id}:{command.epoch}'
            if not self.store.consume_nonce(nonce_key, capability.reservation_id, capability.expires_at):
                raise AuthenticationError('inference token has already been used')
            return capability

    def accept_assignment(self, command: AssignmentCommand) -> None:
        if command.gpu_id != self.config.gpu_id:
            raise ValueError('assignment targets a different GPU')
        if command.miner_hotkey != self.config.miner_identity_hotkey:
            raise ValueError('assignment targets a different miner hotkey')
        with self._lock:
            if command.epoch <= self._revoked_epoch:
                raise ValueError('assignment epoch was revoked')
            if self._revocation_draining:
                raise ValueError('revoked runtime has not finished draining')
            if self._command is not None:
                if command.epoch < self._command.epoch:
                    raise ValueError('assignment epoch is stale')
                if command.epoch == self._command.epoch and command != self._command:
                    raise ValueError('assignment epoch cannot be redefined')
                if command.epoch > self._command.epoch + 1:
                    raise ValueError('assignment epoch skipped a required transition')
                if command == self._command and self._worker and self._worker.is_alive():
                    return
            elif command.epoch != self._revoked_epoch + 1:
                raise ValueError('the first assignment epoch must follow the latest revocation')
            self._command = command
            self._state = GPUState.DRAINING
            self._revocation_draining = False
            self._last_error = None
            self._stream_public_key = ''
            self._persist()
            self._start_worker(command)

    def revoke_assignment(self, gpu_id: str, epoch: int, reason: str) -> None:
        """Persistently stop new inference for the exact signed assignment epoch."""
        reason = reason.strip()
        if not reason or len(reason) > 600:
            raise ValueError('revocation reason must contain 1 to 600 characters')
        if gpu_id != self.config.gpu_id:
            raise ValueError('revocation targets a different GPU')
        with self._lock:
            command = self._command
            if command is None:
                if epoch <= self._revoked_epoch:
                    return
                self._revoked_epoch = epoch
                self._revocation_draining = False
                self._accepting_inference = False
                self._state = GPUState.QUARANTINED
                self._last_error = f'assignment epoch revoked before delivery: {reason}'
                self._persist()
                return
            if epoch != command.epoch:
                raise ValueError('revocation epoch does not match the active assignment')
            if self._revoked_epoch == epoch:
                if not self._revocation_draining:
                    return
                if self._revocation_worker is not None and self._revocation_worker.is_alive():
                    return
            self._revoked_epoch = epoch
            self._revocation_draining = True
            self._accepting_inference = False
            self._state = GPUState.QUARANTINED
            self._last_error = f'assignment revoked: {reason}'
            self._persist()
            self._revocation_worker = threading.Thread(
                target=self._drain_revoked_assignment,
                args=(command,),
                name=f'gittensor-revoke-{command.epoch}',
                daemon=True,
            )
            self._revocation_worker.start()

    def resume(self) -> None:
        with self._lock:
            if self._command is None:
                return
            if self._command.epoch <= self._revoked_epoch:
                self._accepting_inference = False
                self._revocation_draining = True
                self._revocation_worker = threading.Thread(
                    target=self._drain_revoked_assignment,
                    args=(self._command,),
                    name=f'gittensor-restore-revoke-{self._command.epoch}',
                    daemon=True,
                )
                self._revocation_worker.start()
                return
            if self._state == GPUState.RUNTIME_VERIFY:
                self._worker = threading.Thread(
                    target=self._restore_runtime,
                    args=(self._command,),
                    name=f'gittensor-restore-{self._command.epoch}',
                    daemon=True,
                )
                self._worker.start()
            else:
                self._start_worker(self._command)

    def answer_weight_challenge(self, payload: Mapping[str, Any], now: float | None = None) -> str:
        timestamp = time.time() if now is None else now
        with self._lock:
            command = self._command
            if command is None or self._state != GPUState.RUNTIME_VERIFY:
                raise ValueError('runtime is not ready for model weight challenges')
            if payload.get('gpu_id') != command.gpu_id or payload.get('release_digest') != command.release_digest:
                raise ValueError('weight challenge does not match the active assignment')
            if timestamp >= float(payload['expires_at']):
                raise ValueError('weight challenge has expired')
            with self._runtime_lock:
                contents = self.runtime.read_weight_range(
                    command,
                    str(payload['path']),
                    int(payload['start_byte']),
                    int(payload['end_byte']),
                )
        nonce = bytes.fromhex(str(payload['nonce']))
        return hashlib.sha256(nonce + contents).hexdigest()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                'gpu_id': self.config.gpu_id,
                'state': self._state.value,
                'epoch': self._command.epoch if self._command else 0,
                'release_digest': self._command.release_digest if self._command else None,
                'revoked_epoch': self._revoked_epoch,
                'revocation_draining': self._revocation_draining,
                'last_error': self._last_error,
                'accepting_inference': self._accepting_inference,
                'active_inference': self._active_inference,
                'active_kv_bytes': self._active_kv_bytes,
            }

    def open_inference(self, payload: Mapping[str, Any], capability: InferenceCapability) -> Any:
        with self._lock:
            command = self._command
            if (
                command is None
                or command.epoch <= self._revoked_epoch
                or self._state != GPUState.RUNTIME_VERIFY
                or not self._accepting_inference
            ):
                raise ValueError('runtime is not accepting inference')
            if self._active_inference >= command.certified_slots:
                raise InferenceCapacityUnavailable('runtime certified concurrency is full')
            if capability.gpu_id != command.gpu_id or capability.release_digest != command.release_digest:
                raise AuthenticationError('inference capability does not match the active assignment')
            if self._active_kv_bytes + capability.reserved_kv_bytes > command.kv_cache_capacity_bytes:
                raise InferenceCapacityUnavailable('runtime certified KV-cache capacity is full')
            if payload.get('release_digest') != command.release_digest:
                raise ValueError('inference request does not match the active release')
            request_payload = payload.get('openai_request')
            if not isinstance(request_payload, dict):
                raise ValueError('inference request must contain an openai_request object')
            if request_payload.get('model') != command.model_id:
                raise ValueError('inference request does not target the assigned model')
            try:
                request_payload = normalize_openai_request(request_payload)
                request_capacity = estimate_request_capacity(
                    request_payload,
                    request_overhead_tokens=command.request_overhead_tokens,
                    max_context_tokens=command.max_context_tokens,
                )
            except ValueError as exc:
                raise ValueError(f'inference request capacity is invalid: {exc}') from exc
            required_kv_bytes = request_capacity.context_tokens * command.kv_bytes_per_token
            if required_kv_bytes > capability.reserved_kv_bytes:
                raise AuthenticationError('inference request exceeds its signed KV reservation')
            created = int(payload.get('created') or 0)
            if created <= 0:
                raise ValueError('inference request must contain a positive created timestamp')
            stream_public_key = self.runtime.stream_public_key()
            if not stream_public_key:
                raise ValueError('runtime stream signing key is unavailable')
            self._active_inference += 1
            self._active_kv_bytes += capability.reserved_kv_bytes
            request_digest = canonical_request_digest(request_payload)
        try:
            request = urllib.request.Request(
                self.runtime.inference_url(),
                data=json.dumps(request_payload, separators=(',', ':')).encode(),
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream' if request_payload.get('stream') else 'application/json',
                    'X-Gittensor-Request-Id': str(payload.get('request_id') or ''),
                    'X-Gittensor-Created': str(created),
                    'X-Gittensor-Release-Digest': command.release_digest,
                    'X-Gittensor-Model-Id': command.model_id,
                    'X-Gittensor-Model-Revision': command.model_revision,
                    'X-Gittensor-Request-Digest': request_digest,
                    'X-Gittensor-Stream-Public-Key': stream_public_key,
                },
                method='POST',
            )
            try:
                return no_redirect_urlopen(request, timeout=self.config.inference_timeout_seconds)
            except urllib.error.HTTPError as response:
                if 400 <= response.code < 500:
                    return response
                response.close()
                raise
        except Exception:
            self.close_inference(capability.reserved_kv_bytes)
            raise

    def close_inference(self, reserved_kv_bytes: int) -> None:
        with self._lock:
            self._active_inference = max(0, self._active_inference - 1)
            self._active_kv_bytes = max(0, self._active_kv_bytes - max(0, reserved_kv_bytes))
            if self._active_inference == 0:
                self._drained.notify_all()

    def _start_worker(self, command: AssignmentCommand) -> None:
        self._worker = threading.Thread(
            target=self._execute,
            args=(command,),
            name=f'gittensor-assignment-{command.epoch}',
            daemon=True,
        )
        self._worker.start()

    def _drain_revoked_assignment(self, command: AssignmentCommand) -> None:
        try:
            with self._lock:
                deadline = time.monotonic() + self.config.drain_timeout_seconds
                while self._active_inference > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('revoked runtime did not drain before the assignment deadline')
                    self._drained.wait(remaining)
            with self._runtime_lock:
                self.runtime.drain()
            with self._lock:
                if self._command == command and command.epoch <= self._revoked_epoch:
                    self._revocation_draining = False
                    self._persist()
        except Exception as exc:
            with self._lock:
                if self._command == command and command.epoch <= self._revoked_epoch:
                    self._last_error = f'assignment revoked; runtime drain failed: {exc}'
                    self._persist()

    def _execute(self, command: AssignmentCommand) -> None:
        try:
            with self._lock:
                self._accepting_inference = False
                deadline = time.monotonic() + self.config.drain_timeout_seconds
                while self._active_inference > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('runtime did not drain active inference before assignment deadline')
                    self._drained.wait(remaining)
            with self._runtime_lock:
                self.runtime.drain()
            self._acknowledge_with_retry(command, GPUState.LOADING)
            with self._lock:
                if self._command != command or command.epoch <= self._revoked_epoch:
                    return
                self._state = GPUState.LOADING
                self._persist()
            with self._runtime_lock:
                evidence = self.runtime.load(command)
            self._acknowledge_with_retry(command, GPUState.RUNTIME_VERIFY, evidence)
            with self._lock:
                if self._command != command or command.epoch <= self._revoked_epoch:
                    return
                self._state = GPUState.RUNTIME_VERIFY
                self._last_error = None
                self._stream_public_key = evidence.stream_public_key
                self._accepting_inference = True
                self._persist()
        except Exception as exc:
            with self._lock:
                if self._command == command and command.epoch > self._revoked_epoch:
                    self._last_error = str(exc)
                    self._persist()

    def _restore_runtime(self, command: AssignmentCommand) -> None:
        try:
            with self._runtime_lock:
                evidence = self.runtime.load(command)
            self._acknowledge_with_retry(command, GPUState.RUNTIME_VERIFY, evidence)
            with self._lock:
                if self._command == command and command.epoch > self._revoked_epoch:
                    self._stream_public_key = evidence.stream_public_key
                    self._accepting_inference = True
                    self._last_error = None
                    self._persist()
        except Exception as exc:
            with self._lock:
                if self._command == command and command.epoch > self._revoked_epoch:
                    self._accepting_inference = False
                    self._last_error = f'runtime restore failed: {exc}'
                    self._persist()

    def _acknowledge_with_retry(
        self,
        command: AssignmentCommand,
        state: GPUState,
        evidence: RuntimeEvidence | None = None,
    ) -> None:
        deadline = time.monotonic() + self.config.acknowledgement_retry_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.acknowledgements.acknowledge(command.gpu_id, command.epoch, state, evidence)
                return
            except (OSError, ValueError, urllib.error.HTTPError) as exc:
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(f'control-plane acknowledgement failed: {last_error}')

    def _persist(self) -> None:
        self.store.save(
            {
                'command': asdict(self._command) if self._command else None,
                'state': self._state.value,
                'last_error': self._last_error,
                'stream_public_key': self._stream_public_key,
                'revoked_epoch': self._revoked_epoch,
                'revocation_draining': self._revocation_draining,
            }
        )


def make_handler(agent: MinerRuntimeAgent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = 'GittensorMinerAgent/1'

        def do_GET(self) -> None:
            if self.path == '/health':
                ready = agent.status()['accepting_inference'] is True
                self._send(
                    HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                    {'status': 'ready' if ready else 'unavailable'},
                )
                return
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not_found'})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                headers = {key: value for key, value in self.headers.items()}
                capability: InferenceCapability | None = None
                if self.path == '/v1/gittensor/inference':
                    capability = agent.authenticate_inference(self.headers.get('Authorization', ''))
                else:
                    agent.authenticate('POST', self.path, payload, headers)
                if self.path == '/v1/gittensor/assignments':
                    command = AssignmentCommand(**payload)
                    agent.accept_assignment(command)
                    self._send(HTTPStatus.ACCEPTED, {'accepted': True, 'epoch': command.epoch})
                    return
                if self.path == '/v1/gittensor/revocations':
                    required = {'gpu_id', 'epoch', 'reason'}
                    if payload.keys() != required:
                        raise ValueError('revocation requires exactly gpu_id, epoch, and reason')
                    agent.revoke_assignment(str(payload['gpu_id']), int(payload['epoch']), str(payload['reason']))
                    self._send(HTTPStatus.OK, {'revoked': True, 'epoch': int(payload['epoch'])})
                    return
                if self.path == '/v1/gittensor/challenges/weights':
                    digest = agent.answer_weight_challenge(payload)
                    self._send(HTTPStatus.OK, {'sha256': digest})
                    return
                if self.path == '/v1/gittensor/inference':
                    assert capability is not None
                    self._proxy_inference(payload, capability)
                    return
                self._send(HTTPStatus.NOT_FOUND, {'error': 'not_found'})
            except AuthenticationError as exc:
                self._send(HTTPStatus.UNAUTHORIZED, {'error': 'unauthorized', 'message': str(exc)})
            except InferenceCapacityUnavailable as exc:
                self._send(HTTPStatus.TOO_MANY_REQUESTS, {'error': 'capacity_unavailable', 'message': str(exc)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {'error': 'invalid_request', 'message': str(exc)})
            except Exception as exc:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'agent_failure', 'message': str(exc)})

        def _read_json(self) -> dict[str, Any]:
            return read_json_object(self.rfile, self.headers, agent.config.max_request_bytes)

        def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, separators=(',', ':')).encode()
            self.send_response(status.value)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy_inference(self, payload: Mapping[str, Any], capability: InferenceCapability) -> None:
            response = agent.open_inference(payload, capability)
            try:
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                response.close()
                agent.close_inference(capability.reserved_kv_bytes)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    from gittensor.compute.entrypoints import miner_parser

    return miner_parser()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or build_parser().parse_args()
    config = MinerAgentConfig.load(args.config)
    wallet = bt.Wallet(
        name=config.miner_wallet_name,
        hotkey=config.miner_wallet_hotkey,
        path=config.miner_wallet_path,
    )
    if wallet.hotkey.ss58_address != config.miner_identity_hotkey:
        raise SystemExit('configured miner_identity_hotkey does not match the loaded wallet hotkey')
    store = SQLiteAgentStateStore(config.database_path)
    runtime = ContainerRuntimeManager(
        ContainerRuntimeConfig(
            engine=config.container_engine,
            gpu_device=config.gpu_device,
            container_name=config.container_name,
            weights_root=Path(config.weights_root),
            runtime_port=config.runtime_port,
            runtime_container_port=config.runtime_container_port,
            health_path=config.runtime_health_path,
            internal_network=config.internal_network,
            model_downloader_binary=config.model_downloader_binary,
            pids_limit=config.runtime_pids_limit,
            shm_size=config.runtime_shm_size,
        )
    )
    agent = MinerRuntimeAgent(
        config,
        runtime,
        store,
        ValidatorCommandAuthenticator(
            config.expected_validator_hotkey,
            store,
            config.command_ttl_seconds,
        ),
        ControlPlaneAcknowledgementClient(
            config.control_plane_url,
            MinerRequestSigner(wallet.hotkey),
            config.control_plane_timeout_seconds,
        ),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent))
    agent.resume()
    print(f'gittensor miner agent listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
