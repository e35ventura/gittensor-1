"""Minimal HTTP service for the compute control plane reference implementation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from gittensor.compute.config import load_compute_config
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.routing import CapacityUnavailable


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f'cannot encode {type(value).__name__}')


def make_handler(control_plane: ComputeControlPlane, bearer_token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'gittensor-compute/0.1'

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
            if not self._authorized():
                return
            try:
                payload = self._read_json()
                if self.path == '/v1/releases':
                    self._require_keys(
                        payload,
                        required={'release_digest', 'model_id', 'runtime_digest'},
                        optional={'minimum_replicas', 'placement_weight'},
                    )
                    control_plane.register_release(Release(**payload))
                    self._send(HTTPStatus.CREATED, {'status': 'approved'})
                elif self.path == '/v1/gpus':
                    self._require_keys(
                        payload,
                        required={
                            'gpu_id',
                            'spark_node_id',
                            'miner_uid',
                            'endpoint',
                            'region',
                            'release_digest',
                            'canary_release_digest',
                            'certified_slots',
                        },
                        optional={'performance_class', 'latency_by_region_ms'},
                    )
                    control_plane.register_gpu(GPURegistration(**payload))
                    self._send(HTTPStatus.CREATED, {'status': 'registered'})
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
                elif self.path == '/v1/assignments/begin':
                    self._require_keys(payload, required={'gpu_id', 'release_digest'}, optional=set())
                    control_plane.begin_assignment(**payload)
                    self._send(HTTPStatus.ACCEPTED, {'state': 'DRAINING'})
                elif self.path == '/v1/assignments/bind-canary':
                    self._require_keys(payload, required={'gpu_id', 'release_digest'}, optional=set())
                    control_plane.bind_runtime_canary(**payload)
                    self._send(HTTPStatus.ACCEPTED, {'state': 'RUNTIME_VERIFY'})
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
    control_plane = ComputeControlPlane(load_compute_config(args.config))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(control_plane, token))
    print(f'gittensor compute control plane listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
