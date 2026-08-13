import hashlib
import io
import time
import urllib.error
from dataclasses import asdict
from email.message import Message
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import bittensor as bt
import pytest

from gittensor.compute.auth import HotkeyRequestSigner, ValidatorCommandAuthenticator
from gittensor.compute.inference_tokens import issue_inference_capability
from gittensor.compute.miner_agent import (
    InferenceCapacityUnavailable,
    MinerAgentConfig,
    MinerRuntimeAgent,
    SQLiteAgentStateStore,
)
from gittensor.compute.models import AssignmentCommand, GPUState, RuntimeEvidence


def _command(epoch=1):
    return AssignmentCommand(
        gpu_id='gpu-1',
        miner_hotkey='miner',
        epoch=epoch,
        release_digest='sha256:release',
        model_id='model',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_repository='owner/model',
        tokenizer_revision='b' * 40,
        runtime_digest='sha256:runtime',
        runtime_commit='c' * 40,
        container_image='registry/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
        weight_files={'model.safetensors': 64},
        token_proof_scheme='sr25519-response-v1',
        certified_slots=4,
        assignment_token='assignment-secret',
    )


class MemoryRuntime:
    def __init__(self):
        self.current = None
        self.drains = 0
        self.contents = b'w' * 64
        self.stream_key = '11' * 32

    def drain(self):
        self.drains += 1

    def load(self, command):
        self.current = command
        return RuntimeEvidence(
            release_digest=command.release_digest,
            model_repository=command.model_repository,
            model_revision=command.model_revision,
            tokenizer_repository=command.tokenizer_repository,
            tokenizer_revision=command.tokenizer_revision,
            runtime_digest=command.runtime_digest,
            runtime_commit=command.runtime_commit,
            container_image=command.container_image,
            container_digest=command.container_digest,
            filesystem_digest=command.filesystem_digest,
            stream_public_key=self.stream_key,
        )

    def read_weight_range(self, command, path, start, end):
        assert command == self.current
        assert path == 'model.safetensors'
        return self.contents[start:end]

    def inference_url(self):
        return 'http://127.0.0.1:18000/v1/chat/completions'

    def stream_public_key(self):
        return self.stream_key


class RecordingAcknowledgements:
    def __init__(self):
        self.states = []

    def acknowledge(self, gpu_id, epoch, state, evidence=None):
        self.states.append((gpu_id, epoch, state, evidence))


def _config(tmp_path, validator_hotkey):
    return MinerAgentConfig(
        gpu_id='gpu-1',
        expected_validator_hotkey=validator_hotkey,
        control_plane_url='https://control.example',
        miner_wallet_name='default',
        miner_wallet_hotkey='default',
        miner_wallet_path=None,
        miner_identity_hotkey='miner',
        database_path=str(tmp_path / 'agent.sqlite3'),
        weights_root=str(tmp_path / 'weights'),
    )


def _wait_for_state(agent, expected):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if agent.status()['state'] == expected.value:
            return
        time.sleep(0.01)
    raise AssertionError(f'agent did not reach {expected.value}: {agent.status()}')


def test_agent_executes_monotonic_assignment_and_answers_local_weight_challenge(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    runtime = MemoryRuntime()
    acknowledgements = RecordingAcknowledgements()
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        acknowledgements,
    )
    command = _command()

    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)

    assert runtime.drains == 1
    assert [item[2] for item in acknowledgements.states] == [GPUState.LOADING, GPUState.RUNTIME_VERIFY]
    payload = {
        'gpu_id': 'gpu-1',
        'release_digest': command.release_digest,
        'path': 'model.safetensors',
        'start_byte': 7,
        'end_byte': 31,
        'nonce': '11' * 32,
        'expires_at': time.time() + 30,
    }
    assert agent.answer_weight_challenge(payload) == hashlib.sha256(bytes.fromhex('11' * 32) + b'w' * 24).hexdigest()


def test_agent_rejects_epoch_redefinition_and_skips(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    agent.accept_assignment(command)

    with pytest.raises(ValueError, match='cannot be redefined'):
        agent.accept_assignment(AssignmentCommand(**{**asdict(command), 'release_digest': 'sha256:other'}))
    with pytest.raises(ValueError, match='skipped'):
        agent.accept_assignment(_command(epoch=3))


def test_agent_authenticates_validator_headers_and_rejects_replay(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    now = int(time.time())
    payload = asdict(_command())
    headers = HotkeyRequestSigner(validator, clock=lambda: now).headers('POST', '/v1/gittensor/assignments', payload)
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30, clock=lambda: now),
        RecordingAcknowledgements(),
    )

    agent.authenticate('POST', '/v1/gittensor/assignments', payload, headers)

    with pytest.raises(ValueError, match='already been used'):
        agent.authenticate('POST', '/v1/gittensor/assignments', payload, headers)


def test_inference_uses_rotating_assignment_token_not_validator_hotkey(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    agent.accept_assignment(command)
    token = issue_inference_capability(
        command.assignment_token,
        reservation_id='reservation-1',
        gpu_id=command.gpu_id,
        release_digest=command.release_digest,
        expires_at=time.time() + 30,
    )

    agent.authenticate_inference(f'Bearer {token}')
    with pytest.raises(ValueError, match='invalid or stale'):
        agent.authenticate_inference('Bearer wrong')
    with pytest.raises(ValueError, match='already been used'):
        agent.authenticate_inference(f'Bearer {token}')


def test_miner_agent_enforces_signed_certified_concurrency(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    command = type(command)(**{**asdict(command), 'certified_slots': 1})
    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)
    payload = {
        'request_id': 'request-1',
        'created': int(time.time()),
        'release_digest': command.release_digest,
        'openai_request': {'model': command.model_id},
    }
    response = MagicMock()

    with patch('gittensor.compute.miner_agent.no_redirect_urlopen', return_value=response):
        assert agent.open_inference(payload) is response
        with pytest.raises(InferenceCapacityUnavailable, match='certified concurrency'):
            agent.open_inference(payload)
        agent.close_inference()


def test_miner_agent_rejects_a_different_model_inside_a_valid_release_request(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)

    with pytest.raises(ValueError, match='assigned model'):
        agent.open_inference(
            {
                'request_id': 'request-1',
                'created': int(time.time()),
                'release_digest': command.release_digest,
                'openai_request': {'model': 'small-model'},
            }
        )

    assert agent.status()['active_inference'] == 0


def test_miner_agent_preserves_runtime_client_error_without_losing_concurrency_accounting(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)
    payload = {
        'request_id': 'request-1',
        'created': int(time.time()),
        'release_digest': command.release_digest,
        'openai_request': {'model': command.model_id},
    }
    headers = Message()
    failure = urllib.error.HTTPError(
        agent.runtime.inference_url(),
        HTTPStatus.UNPROCESSABLE_ENTITY.value,
        'invalid prompt',
        headers,
        io.BytesIO(b'{"error":"invalid prompt"}'),
    )

    with patch('gittensor.compute.miner_agent.no_redirect_urlopen', side_effect=failure):
        response = agent.open_inference(payload)
        assert response.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert agent.status()['active_inference'] == 1
        response.close()
        agent.close_inference()

    assert agent.status()['active_inference'] == 0


def test_signed_assignment_revocation_persistently_invalidates_issued_capabilities(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    runtime = MemoryRuntime()
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    command = _command()
    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)
    token = issue_inference_capability(
        command.assignment_token,
        reservation_id='reservation-after-disable',
        gpu_id=command.gpu_id,
        release_digest=command.release_digest,
        expires_at=time.time() + 30,
    )

    agent.revoke_assignment(command.gpu_id, command.epoch, 'operator quarantine')

    with pytest.raises(ValueError, match='invalid or stale'):
        agent.authenticate_inference(f'Bearer {token}')
    restored = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    with pytest.raises(ValueError, match='invalid or stale'):
        restored.authenticate_inference(f'Bearer {token}')
    assert restored.status()['state'] == GPUState.QUARANTINED.value


def test_revocation_arriving_before_assignment_tombstones_that_epoch(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )

    agent.revoke_assignment('gpu-1', 1, 'operator quarantine')

    with pytest.raises(ValueError, match='was revoked'):
        agent.accept_assignment(_command())
    restored = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )
    with pytest.raises(ValueError, match='was revoked'):
        restored.accept_assignment(_command())

    restored.accept_assignment(_command(epoch=2))
    _wait_for_state(restored, GPUState.RUNTIME_VERIFY)
    assert restored.status()['epoch'] == 2


def test_revocation_before_assignment_rejects_a_different_gpu_identity(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        MemoryRuntime(),
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        RecordingAcknowledgements(),
    )

    with pytest.raises(ValueError, match='different GPU'):
        agent.revoke_assignment('gpu-attacker', 1, 'operator quarantine')

    assert agent.status()['revoked_epoch'] == 0


def test_agent_restart_reacknowledges_rotated_runtime_key_before_serving(tmp_path):
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    first_runtime = MemoryRuntime()
    first_acknowledgements = RecordingAcknowledgements()
    agent = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        first_runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        first_acknowledgements,
    )
    command = _command()
    agent.accept_assignment(command)
    _wait_for_state(agent, GPUState.RUNTIME_VERIFY)

    restored_runtime = MemoryRuntime()
    restored_runtime.stream_key = '22' * 32
    original_load = restored_runtime.load

    def load_with_rotated_key(command):
        evidence = original_load(command)
        return RuntimeEvidence(**{**asdict(evidence), 'stream_public_key': restored_runtime.stream_key})

    restored_runtime.load = load_with_rotated_key
    restored_runtime.stream_public_key = lambda: restored_runtime.stream_key
    restored_acknowledgements = RecordingAcknowledgements()
    restored = MinerRuntimeAgent(
        _config(tmp_path, validator.ss58_address),
        restored_runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        restored_acknowledgements,
    )

    restored.resume()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not restored.status()['accepting_inference']:
        time.sleep(0.01)

    assert restored.status()['accepting_inference'] is True
    assert restored_acknowledgements.states[0][2] == GPUState.RUNTIME_VERIFY
    assert restored_acknowledgements.states[0][3].stream_public_key == '22' * 32
