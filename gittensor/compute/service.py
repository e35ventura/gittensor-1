"""Authenticated HTTP service for the durable compute control plane."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import bittensor as bt

from gittensor.compute.artifacts import CosignReleaseVerifier
from gittensor.compute.assignment import HTTPAssignmentExecutor
from gittensor.compute.auth import AuthenticationError, HotkeyAuthenticator, HotkeyRequestSigner, LiveMetagraphResolver
from gittensor.compute.config import load_compute_config
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.emission_oracle import SubnetEmissionOracle
from gittensor.compute.http_json import load_json_object, read_json_object
from gittensor.compute.models import GPUState, Release, RoutingObservation, RuntimeEvidence
from gittensor.compute.routing import CapacityUnavailable
from gittensor.compute.settlement_auth import SettlementSigner
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
    gateway_token: str | None = None,
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
            if self.path == '/v1/settlements/latest':
                settlement = control_plane.latest_settlement()
                if settlement is None:
                    self._send(HTTPStatus.NOT_FOUND, {'error': 'no_fresh_settlement'})
                else:
                    self._send(HTTPStatus.OK, settlement)
                return
            if self.path == '/v1/status':
                if not self._authorized(bearer_token):
                    return
                self._send(HTTPStatus.OK, control_plane.status())
                return
            if self.path == '/v1/catalog':
                if not self._authorized(gateway_token, allow_operator=True):
                    return
                self._send(HTTPStatus.OK, {'releases': control_plane.catalog()})
                return
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                miner = None
                if self.path in {'/v1/gpus', '/v1/assignments/ack'}:
                    if miner_authenticator is None:
                        raise AuthenticationError('miner hotkey authentication is not configured')
                    auth = payload.pop('auth', None)
                    if not isinstance(auth, dict):
                        raise AuthenticationError('request requires an auth object')
                    miner = miner_authenticator.authenticate('POST', self.path, payload, auth)
                elif self.path in {
                    '/v1/route',
                    '/v1/reservations/complete',
                    '/v1/reservations/renew',
                    '/v1/observations',
                }:
                    if not self._authorized(gateway_token, allow_operator=True):
                        return
                elif not self._authorized(bearer_token):
                    return
                if self.path == '/v1/gpus/disable':
                    self._require_keys(payload, required={'gpu_id', 'reason'}, optional=set())
                    control_plane.disable_gpu(payload['gpu_id'], payload['reason'])
                    self._send(HTTPStatus.OK, {'status': 'disabled'})
                elif self.path == '/v1/gpus/enable':
                    self._require_keys(payload, required={'gpu_id'}, optional=set())
                    control_plane.enable_gpu(payload['gpu_id'])
                    self._send(HTTPStatus.OK, {'status': 'enabled'})
                elif self.path == '/v1/releases/revoke':
                    self._require_keys(payload, required={'release_digest', 'reason'}, optional=set())
                    control_plane.revoke_release(payload['release_digest'], payload['reason'])
                    self._send(HTTPStatus.OK, {'status': 'revoked'})
                elif self.path == '/v1/releases':
                    self._require_keys(
                        payload,
                        required={
                            'release_digest',
                            'model_id',
                            'runtime_digest',
                            'model_repository',
                            'model_revision',
                            'tokenizer_repository',
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
                    self._send(HTTPStatus.OK, control_plane.tick(execute=True))
                elif self.path == '/v1/funding':
                    self._require_keys(
                        payload,
                        required={'max_budget_per_hour'},
                        optional={'subnet_miner_emission_value_per_hour'},
                    )
                    budget = payload['max_budget_per_hour']
                    self._send(
                        HTTPStatus.OK,
                        control_plane.update_budget(
                            float(budget) if budget is not None else None,
                            subnet_miner_emission_value_per_hour=(
                                float(payload['subnet_miner_emission_value_per_hour'])
                                if payload.get('subnet_miner_emission_value_per_hour') is not None
                                else None
                            ),
                        ),
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
                elif self.path == '/v1/reservations/renew':
                    self._require_keys(payload, required={'reservation_id'}, optional=set())
                    expires_at = control_plane.renew_reservation(payload['reservation_id'])
                    if expires_at is None:
                        self._send(HTTPStatus.NOT_FOUND, {'error': 'reservation_not_live'})
                    else:
                        self._send(HTTPStatus.OK, {'renewed': True, 'expires_at': expires_at})
                elif self.path == '/v1/observations':
                    self._require_keys(
                        payload,
                        required={
                            'reservation_id',
                            'gpu_id',
                            'requester_region',
                            'measured_rtt_ms',
                            'service_seconds',
                            'success',
                            'observed_active_slots',
                            'remaining_work_seconds',
                            'expected_service_seconds',
                            'release_digest',
                        },
                        optional=set(),
                    )
                    control_plane.record_routing_observation(RoutingObservation(**payload))
                    self._send(HTTPStatus.OK, {'status': 'recorded'})
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

        def _authorized(self, required_token: str | None, *, allow_operator: bool = False) -> bool:
            if required_token is None:
                return True
            authorization = self.headers.get('Authorization')
            if isinstance(authorization, str) and secrets.compare_digest(authorization, f'Bearer {required_token}'):
                return True
            if (
                allow_operator
                and bearer_token is not None
                and isinstance(authorization, str)
                and secrets.compare_digest(authorization, f'Bearer {bearer_token}')
            ):
                return True
            self._send(HTTPStatus.UNAUTHORIZED, {'error': 'unauthorized'})
            return False

        def _read_json(self) -> dict[str, Any]:
            return read_json_object(self.rfile, self.headers, 64 * 1024)

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
    from gittensor.compute.entrypoints import control_plane_parser

    return control_plane_parser()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or build_parser().parse_args()
    token = os.environ.get(args.token_env)
    gateway_token = os.environ.get(args.gateway_token_env)
    loopback = args.host in {'127.0.0.1', '::1', 'localhost'}
    if not token and not (args.allow_insecure_local and loopback):
        raise SystemExit(f'{args.token_env} must be set unless --allow-insecure-local is used on loopback')
    if not gateway_token and not (args.allow_insecure_local and loopback):
        raise SystemExit(f'{args.gateway_token_env} must be set unless --allow-insecure-local is used on loopback')
    config = load_compute_config(args.config)
    if not config.identity.spark_node_owners_path:
        raise SystemExit('identity.spark_node_owners_path must point to the verifier enrollment map')
    owners_path = Path(config.identity.spark_node_owners_path)
    try:
        spark_node_owners = load_json_object(owners_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'failed to load SparkCompute ownership map: {exc}') from exc
    if (
        not isinstance(spark_node_owners, dict)
        or not spark_node_owners
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in spark_node_owners.items())
    ):
        raise SystemExit('SparkCompute ownership map must be a non-empty JSON object of node_id -> hotkey')
    store = SQLiteStateStore(config.state.database_path)
    validator_wallet = bt.Wallet(
        name=config.assignment.validator_wallet_name,
        hotkey=config.assignment.validator_wallet_hotkey,
        path=config.assignment.validator_wallet_path,
    )
    command_signer = HotkeyRequestSigner(validator_wallet.hotkey)
    assignment_executor = HTTPAssignmentExecutor(config.assignment.request_timeout_seconds, command_signer)
    weight_verifier = WeightChallengeVerifier(
        HuggingFaceRangeSource(config.verification.timeout_seconds),
        config.verification.weight_challenge_ttl_seconds,
    )
    weight_transport = HTTPWeightChallengeTransport(config.assignment.request_timeout_seconds, command_signer)
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
        settlement_signer=SettlementSigner(validator_wallet.hotkey),
    )
    resolver = LiveMetagraphResolver(
        config.identity.netuid,
        config.identity.network,
        config.identity.metagraph_refresh_seconds,
    )
    authenticator = HotkeyAuthenticator(resolver, store, config.identity.signature_ttl_seconds)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(control_plane, token, authenticator, gateway_token),
    )
    emission_oracle = (
        SubnetEmissionOracle(
            config.emission_oracle,
            netuid=config.identity.netuid,
            network=config.identity.network,
            target_currency=config.fleet.target_price_currency,
        )
        if config.emission_oracle.enabled
        else None
    )
    supervisor = ComputeSupervisor(control_plane, emission_oracle=emission_oracle)
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
