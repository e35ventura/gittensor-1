import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import bittensor as bt

from gittensor.compute.auth import HotkeyAuthenticator, StaticIdentityResolver, canonical_request
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.service import make_handler
from gittensor.compute.storage import SQLiteStateStore

from .test_control_plane import Clock, _config


def _post(port, path, payload):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    body = json.dumps(payload, separators=(',', ':'))
    connection.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
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
