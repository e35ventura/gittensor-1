"""Authenticated HTTP service for the durable compute control plane."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gittensor.compute.artifacts import CosignReleaseVerifier
from gittensor.compute.assignment import HTTPAssignmentExecutor
from gittensor.compute.auth import AuthenticationError, HotkeyAuthenticator, LiveMetagraphResolver
from gittensor.compute.config import load_compute_config
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPUState, Release, RoutingObservation, RuntimeEvidence
from gittensor.compute.routing import CapacityUnavailable
from gittensor.compute.storage import SQLiteStateStore
from gittensor.compute.supervisor import ComputeSupervisor
from gittensor.compute.weight_challenges import (
    HTTPWeightChallengeTransport,
    HuggingFaceRangeSource,
    WeightChallengeVerifier,
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f'cannot encode {type(value).__name__}')


def make_handler(
    control_plane: ComputeControlPlane,
    bearer_token: str | None,
    miner_authenticator: HotkeyAuthenticator | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'gittensor-compute/0.2'

        def log_message(self, format_string: str, *args: Any) -> None:
            # Request bodies are never logged. The default request-line log is safe.
            super().log_message(format_string, *args)

        def do_GET(self) -> None:
            if self.path == '/health':
                self._send(HTTPStatus.OK, {'status': 'ok'})
                return
            if not self._authorized():
                return
            if self.path == '/v1/status':
                self._send(HTTPStatus.OK, control_plane.status())
                return
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                miner = None
                if self.path in {'/v1/gpus', '/v1/gpus/telemetry', '/v1/assignments/ack'}:
                    if miner_authenticator is None:
                        raise AuthenticationError('miner hotkey authentication is not configured')
                    auth = payload.pop('auth', None)
                    if not isinstance(auth, dict):
                        raise AuthenticationError('request requires an auth object')
                    miner = miner_authenticator.authenticate('POST', self.path, payload, auth)
                elif not self._authorized():
                    return
                if self.path == '/v1/releases':
                    self._require_keys(
                        payload,
                        required={
                            'release_digest',
                            'model_id',
                            'runtime_digest',
                            'model_repository',
                            'model_revision',
                            'tokenizer_revision',
                            'container_image',
                            'container_digest',
                            'filesystem_digest',
                            'runtime_commit',
                            'weight_files',
                        },
                        optional={'token_proof_scheme', 'minimum_replicas', 'placement_weight'},
                    )
                    release = Release(**payload)
                    release.validate_production_manifest()
                    control_plane.register_release(release)
                    self._send(HTTPStatus.CREATED, {'status': 'approved'})
                elif self.path == '/v1/gpus':
                    self._require_keys(
                        payload,
                        required={'gpu_id', 'spark_node_id', 'endpoint', 'region'},
                        optional=set(),
                    )
                    assert miner is not None
                    control_plane.register_miner_gpu(
                        miner_uid=miner.uid,
                        miner_hotkey=miner.hotkey,
                        **payload,
                    )
                    self._send(HTTPStatus.CREATED, {'status': 'registered'})
                elif self.path == '/v1/gpus/telemetry':
                    self._require_keys(
                        payload,
                        required={'gpu_id', 'active_slots', 'remaining_work_seconds'},
                        optional=set(),
                    )
                    assert miner is not None
                    control_plane.update_gpu_telemetry(miner_hotkey=miner.hotkey, **payload)
                    self._send(HTTPStatus.OK, {'status': 'recorded'})
                elif self.path == '/v1/assignments/ack':
                    self._require_keys(
                        payload,
                        required={'gpu_id', 'epoch', 'state'},
                        optional={'evidence'},
                    )
                    assert miner is not None
                    record = control_plane.gpus[payload['gpu_id']]
                    if record.registration.miner_hotkey != miner.hotkey:
                        raise AuthenticationError('hotkey does not own this GPU')
                    evidence_payload = payload.get('evidence')
                    evidence = RuntimeEvidence(**evidence_payload) if evidence_payload else None
                    state = control_plane.acknowledge_assignment(
                        payload['gpu_id'],
                        int(payload['epoch']),
                        GPUState(payload['state']),
                        evidence,
                    )
                    self._send(HTTPStatus.OK, {'state': state.value})
                elif self.path == '/v1/verification/refresh':
                    self._require_keys(payload, required=set(), optional=set())
                    self._send(HTTPStatus.OK, {'gpus': control_plane.refresh_verification()})
                elif self.path == '/v1/control/tick':
                    self._require_keys(payload, required=set(), optional=set())
                    self._send(HTTPStatus.OK, control_plane.tick())
                elif self.path == '/v1/funding':
                    self._require_keys(payload, required={'max_budget_per_hour'}, optional=set())
                    self._send(
                        HTTPStatus.OK,
                        control_plane.update_budget(float(payload['max_budget_per_hour'])),
                    )
                elif self.path == '/v1/route':
                    self._require_keys(
                        payload,
                        required={'release_digest', 'requester_region', 'expected_service_seconds'},
                        optional=set(),
                    )
                    self._send(HTTPStatus.CREATED, control_plane.route(**payload))
                elif self.path == '/v1/reservations/complete':
                    self._require_keys(payload, required={'reservation_id'}, optional=set())
                    completed = control_plane.complete_reservation(payload['reservation_id'])
                    self._send(HTTPStatus.OK, {'completed': completed})
                elif self.path == '/v1/observations':
                    self._require_keys(
                        payload,
                        required={
                            'gpu_id',
                            'requester_region',
                            'measured_rtt_ms',
                            'service_seconds',
                            'success',
                            'observed_active_slots',
                            'remaining_work_seconds',
                        },
                        optional=set(),
                    )
                    control_plane.record_routing_observation(RoutingObservation(**payload))
                    self._send(HTTPStatus.OK, {'status': 'recorded'})
                elif self.path == '/v1/settlement':
                    self._require_keys(payload, required=set(), optional=set())
                    result, miner_rewards = control_plane.settle()
                    self._send(
                        HTTPStatus.OK,
                        {'settlement': result, 'miner_rewards': miner_rewards},
                    )
                else:
                    self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})
            except CapacityUnavailable as exc:
                self._send(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {'error': 'capacity_unavailable', 'message': str(exc)},
                )
            except AuthenticationError as exc:
                self._send(HTTPStatus.UNAUTHORIZED, {'error': 'unauthorized', 'message': str(exc)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {'error': 'invalid_request', 'message': str(exc)})
            except Exception as exc:
                self._send(HTTPStatus.BAD_GATEWAY, {'error': 'upstream_failure', 'message': str(exc)})

        def _authorized(self) -> bool:
            if bearer_token is None:
                return True
            if self.headers.get('Authorization') == f'Bearer {bearer_token}':
                return True
            self._send(HTTPStatus.UNAUTHORIZED, {'error': 'unauthorized'})
            return False

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 64 * 1024:
                raise ValueError('request body exceeds 64 KiB')
            payload = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(payload, dict):
                raise ValueError('request body must be a JSON object')
            return payload

        @staticmethod
        def _require_keys(
            payload: dict[str, Any],
            *,
            required: set[str],
            optional: set[str],
        ) -> None:
            missing = required - payload.keys()
            unexpected = payload.keys() - required - optional
            if missing:
                raise ValueError(f'missing fields: {", ".join(sorted(missing))}')
            if unexpected:
                raise ValueError(f'unexpected fields: {", ".join(sorted(unexpected))}')

        def _send(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, default=_json_default, separators=(',', ':')).encode()
            self.send_response(status.value)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the Gittensor compute control plane')
    parser.add_argument('--config', required=True, help='path to compute JSON configuration')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--token-env', default='GITTENSOR_COMPUTE_TOKEN')
    parser.add_argument(
        '--allow-insecure-local',
        action='store_true',
        help='allow startup without an API token when bound to a loopback address',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    token = os.environ.get(args.token_env)
    loopback = args.host in {'127.0.0.1', '::1', 'localhost'}
    if not token and not (args.allow_insecure_local and loopback):
        raise SystemExit(f'{args.token_env} must be set unless --allow-insecure-local is used on loopback')
    config = load_compute_config(args.config)
    if not config.identity.spark_node_owners_path:
        raise SystemExit('identity.spark_node_owners_path must point to the verifier enrollment map')
    owners_path = Path(config.identity.spark_node_owners_path)
    try:
        spark_node_owners = json.loads(owners_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'failed to load SparkCompute ownership map: {exc}') from exc
    if not isinstance(spark_node_owners, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in spark_node_owners.items()
    ):
        raise SystemExit('SparkCompute ownership map must be a JSON object of node_id -> hotkey')
    store = SQLiteStateStore(config.state.database_path)
    assignment_token = (
        os.environ.get(config.assignment.bearer_token_env) if config.assignment.bearer_token_env else None
    )
    assignment_executor = HTTPAssignmentExecutor(config.assignment.request_timeout_seconds, assignment_token)
    weight_verifier = WeightChallengeVerifier(
        HuggingFaceRangeSource(config.verification.timeout_seconds),
        config.verification.weight_challenge_ttl_seconds,
    )
    weight_transport = HTTPWeightChallengeTransport(config.assignment.request_timeout_seconds, assignment_token)
    artifact_verifier = (
        CosignReleaseVerifier(
            config.verification.cosign_binary,
            config.verification.cosign_public_key_path,
        )
        if config.verification.require_signed_containers
        else None
    )
    control_plane = ComputeControlPlane(
        config,
        store=store,
        assignment_executor=assignment_executor,
        weight_verifier=weight_verifier,
        weight_transport=weight_transport,
        spark_node_owners=spark_node_owners,
        release_artifact_verifier=artifact_verifier,
    )
    resolver = LiveMetagraphResolver(
        config.identity.netuid,
        config.identity.network,
        config.identity.metagraph_refresh_seconds,
    )
    authenticator = HotkeyAuthenticator(resolver, store, config.identity.signature_ttl_seconds)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(control_plane, token, authenticator),
    )
    supervisor = ComputeSupervisor(control_plane)
    supervisor.start()
    print(f'gittensor compute control plane listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
        server.server_close()


if __name__ == '__main__':
    main()
