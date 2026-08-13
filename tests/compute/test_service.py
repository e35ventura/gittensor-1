import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import bittensor as bt

from gittensor.compute.auth import HotkeyAuthenticator, StaticIdentityResolver, canonical_request
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.service import make_handler
from gittensor.compute.settlement_auth import SettlementSigner, verify_settlement
from gittensor.compute.storage import SQLiteStateStore

from .test_control_plane import Clock, _config


def _post(port, path, payload, token=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    body = json.dumps(payload, separators=(',', ':'))
    headers = {'Content-Type': 'application/json'}
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'
    connection.request('POST', path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def _get(port, path, token=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    headers = {'Authorization': f'Bearer {token}'} if token is not None else {}
    connection.request('GET', path, headers=headers)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def test_registration_uses_signed_metagraph_uid_and_accepts_no_assignment_fields(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    authenticator = HotkeyAuthenticator(
        StaticIdentityResolver({keypair.ss58_address: 17}),
        store,
        60,
        clock=lambda: 100,
    )
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(control, None, authenticator))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            'gpu_id': 'gpu-1',
            'spark_node_id': 'node-1',
            'endpoint': 'https://gpu-1',
            'region': 'us-east',
        }
        nonce = 'registration-nonce'
        signature = keypair.sign(canonical_request('POST', '/v1/gpus', payload, 100, nonce)).hex()
        status, _ = _post(
            server.server_port,
            '/v1/gpus',
            {
                **payload,
                'auth': {
                    'hotkey': keypair.ss58_address,
                    'timestamp': 100,
                    'nonce': nonce,
                    'signature': signature,
                },
            },
        )

        registration = control.gpus['gpu-1'].registration
        assert status == 201
        assert registration.miner_uid == 17
        assert registration.miner_hotkey == keypair.ss58_address
        assert registration.release_digest == ''
        assert registration.canary_release_digest == ''
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_registration_rejects_miner_selected_uid_release_and_concurrency(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    authenticator = HotkeyAuthenticator(
        StaticIdentityResolver({keypair.ss58_address: 17}),
        store,
        60,
        clock=lambda: 100,
    )
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(control, None, authenticator))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            'gpu_id': 'gpu-1',
            'spark_node_id': 'node-1',
            'endpoint': 'https://gpu-1',
            'region': 'us-east',
            'miner_uid': 999,
            'release_digest': 'attacker-release',
            'certified_slots': 999,
        }
        nonce = f'invalid-{time.time_ns()}'
        signature = keypair.sign(canonical_request('POST', '/v1/gpus', payload, 100, nonce)).hex()
        status, response = _post(
            server.server_port,
            '/v1/gpus',
            {
                **payload,
                'auth': {
                    'hotkey': keypair.ss58_address,
                    'timestamp': 100,
                    'nonce': nonce,
                    'signature': signature,
                },
            },
        )

        assert status == 400
        assert response['error'] == 'invalid_request'
        assert 'unexpected fields' in response['message']
        assert 'gpu-1' not in control.gpus
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_credential_cannot_call_operator_endpoints():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    server = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        make_handler(control, 'operator-secret', gateway_token='gateway-secret'),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        forbidden, _ = _post(
            server.server_port,
            '/v1/funding',
            {'max_budget_per_hour': 3.9},
            token='gateway-secret',
        )
        allowed, _ = _post(
            server.server_port,
            '/v1/funding',
            {'max_budget_per_hour': 3.9},
            token='operator-secret',
        )

        assert forbidden == 401
        assert allowed == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_operator_control_tick_executes_gepetto_assignments():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    server = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        make_handler(control, 'operator-secret'),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.object(control, 'tick', wraps=control.tick) as tick:
            status, _ = _post(
                server.server_port,
                '/v1/control/tick',
                {},
                token='operator-secret',
            )

        assert status == 200
        tick.assert_called_once_with(execute=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_operator_can_disable_enable_gpu_and_revoke_release():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=1,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    server = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        make_handler(control, 'operator-secret', gateway_token='gateway-secret'),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        forbidden, _ = _post(
            server.server_port,
            '/v1/gpus/disable',
            {'gpu_id': 'gpu-1', 'reason': 'test'},
            token='gateway-secret',
        )
        disabled, _ = _post(
            server.server_port,
            '/v1/gpus/disable',
            {'gpu_id': 'gpu-1', 'reason': 'test'},
            token='operator-secret',
        )
        enabled, _ = _post(
            server.server_port,
            '/v1/gpus/enable',
            {'gpu_id': 'gpu-1'},
            token='operator-secret',
        )
        revoked, _ = _post(
            server.server_port,
            '/v1/releases/revoke',
            {'release_digest': 'release:1', 'reason': 'test'},
            token='operator-secret',
        )

        assert forbidden == 401
        assert disabled == 200
        assert enabled == 200
        assert revoked == 200
        assert 'release:1' not in control.releases
        assert control.gpus['gpu-1'].registration.release_digest == ''
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_public_latest_settlement_feed_is_signed(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    now = time.time()
    control = ComputeControlPlane(
        _config(),
        clock=Clock(now - 100),
        store=store,
        settlement_signer=SettlementSigner(keypair),
    )
    control.settle(now=now)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(control, 'operator-secret'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, settlement = _get(server.server_port, '/v1/settlements/latest')

        assert status == 200
        assert verify_settlement(settlement, keypair.ss58_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
