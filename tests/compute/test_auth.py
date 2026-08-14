import time

import bittensor as bt
import pytest

from gittensor.compute.auth import (
    AuthenticationError,
    HotkeyAuthenticator,
    HotkeyRequestSigner,
    StaticIdentityResolver,
    ValidatorCommandAuthenticator,
    canonical_request,
)
from gittensor.compute.storage import SQLiteStateStore


def test_hotkey_signature_resolves_uid_and_rejects_replay(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    resolver = StaticIdentityResolver({keypair.ss58_address: 17})
    now = int(time.time())
    payload = {'gpu_id': 'gpu-1'}
    nonce = 'one-time-nonce'
    signature = keypair.sign(canonical_request('POST', '/v1/gpus', payload, now, nonce)).hex()
    authenticator = HotkeyAuthenticator(resolver, store, 60, clock=lambda: now)
    auth = {
        'hotkey': keypair.ss58_address,
        'timestamp': now,
        'nonce': nonce,
        'signature': signature,
    }

    identity = authenticator.authenticate('POST', '/v1/gpus', payload, auth)

    assert identity.uid == 17
    assert identity.hotkey == keypair.ss58_address
    with pytest.raises(AuthenticationError, match='already been used'):
        authenticator.authenticate('POST', '/v1/gpus', payload, auth)


def test_signature_is_bound_to_exact_request_payload(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    resolver = StaticIdentityResolver({keypair.ss58_address: 17})
    now = int(time.time())
    signed_payload = {'gpu_id': 'gpu-1'}
    nonce = 'payload-bound'
    signature = keypair.sign(canonical_request('POST', '/v1/gpus', signed_payload, now, nonce)).hex()

    with pytest.raises(AuthenticationError, match='invalid hotkey signature'):
        HotkeyAuthenticator(resolver, store, 60, clock=lambda: now).authenticate(
            'POST',
            '/v1/gpus',
            {'gpu_id': 'gpu-2'},
            {
                'hotkey': keypair.ss58_address,
                'timestamp': now,
                'nonce': nonce,
                'signature': signature,
            },
        )


def test_validator_command_signature_is_role_bound_and_replay_safe(tmp_path):
    keypair = bt.Keypair.create_from_uri('//Alice')
    now = int(time.time())
    payload = {'gpu_id': 'gpu-1', 'epoch': 3}
    signer = HotkeyRequestSigner(keypair, clock=lambda: now)
    headers = signer.headers('POST', '/v1/gittensor/assignments', payload)
    authenticator = ValidatorCommandAuthenticator(
        keypair.ss58_address,
        SQLiteStateStore(tmp_path / 'agent.sqlite3'),
        30,
        clock=lambda: now,
    )

    authenticator.authenticate('POST', '/v1/gittensor/assignments', payload, headers)

    with pytest.raises(AuthenticationError, match='already been used'):
        authenticator.authenticate('POST', '/v1/gittensor/assignments', payload, headers)


def test_validator_command_rejects_different_validator(tmp_path):
    trusted = bt.Keypair.create_from_uri('//Alice')
    attacker = bt.Keypair.create_from_uri('//Bob')
    now = int(time.time())
    payload = {'gpu_id': 'gpu-1'}
    headers = HotkeyRequestSigner(attacker, clock=lambda: now).headers('POST', '/command', payload)
    authenticator = ValidatorCommandAuthenticator(
        trusted.ss58_address,
        SQLiteStateStore(tmp_path / 'agent.sqlite3'),
        30,
        clock=lambda: now,
    )

    with pytest.raises(AuthenticationError, match='configured hotkey'):
        authenticator.authenticate('POST', '/command', payload, headers)
