import time

import bittensor as bt

from gittensor.compute.auth import ValidatorCommandAuthenticator
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.miner_agent import MinerRuntimeAgent, SQLiteAgentStateStore
from gittensor.compute.models import GPUState, Release, RuntimeEvidence

from .test_control_plane import Clock, _config, _snapshot
from .test_miner_agent import MemoryRuntime
from .test_miner_agent import _config as _miner_config


class DirectAcknowledgements:
    def __init__(self, control):
        self.control = control

    def acknowledge(self, gpu_id, epoch, state, evidence=None):
        self.control.acknowledge_assignment(gpu_id, epoch, state, evidence, now=102)


class DirectAssignmentExecutor:
    def __init__(self):
        self.agent: MinerRuntimeAgent | None = None

    def dispatch(self, record, command):
        assert self.agent is not None
        self.agent.accept_assignment(command)

    def revoke(self, record, epoch, reason):
        assert self.agent is not None
        self.agent.revoke_assignment(record.registration.gpu_id, epoch, reason)


def test_global_gepetto_executes_through_miner_agent_and_reaches_routable_ready(tmp_path):
    clock = Clock(100)
    executor = DirectAssignmentExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='miner',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://miner.example',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)
    validator = bt.Keypair.create_from_uri('//Alice')
    store = SQLiteAgentStateStore(tmp_path / 'agent.sqlite3')
    runtime = MemoryRuntime()
    agent = MinerRuntimeAgent(
        _miner_config(tmp_path, validator.ss58_address),
        runtime,
        store,
        ValidatorCommandAuthenticator(validator.ss58_address, store, 30),
        DirectAcknowledgements(control),
    )
    executor.agent = agent

    control.tick(now=101, execute=True)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and agent.status()['state'] != GPUState.RUNTIME_VERIFY.value:
        time.sleep(0.01)
    assert agent.status()['state'] == GPUState.RUNTIME_VERIFY.value

    record = control.gpus['gpu-1']
    release = control.releases[record.registration.release_digest]
    evidence = RuntimeEvidence(
        release_digest=release.release_digest,
        model_repository=release.model_repository,
        model_revision=release.model_revision,
        tokenizer_repository=release.tokenizer_repository,
        tokenizer_revision=release.tokenizer_revision,
        runtime_digest=release.runtime_digest,
        runtime_commit=release.runtime_commit,
        container_image=release.container_image,
        container_digest=release.container_digest,
        filesystem_digest=release.filesystem_digest,
        stream_public_key='11' * 32,
    )
    assert runtime.current is not None
    assert runtime.load(runtime.current) == evidence
    assert control.refresh_verification([_snapshot(1, last_checked=103)], now=103) == {'gpu-1': 'READY'}

    route = control.route('release:1', 'us-east', expected_service_seconds=1, now=104)
    assert route.gpu_id == 'gpu-1'
    assert route.endpoint == 'https://miner.example'

    control.disable_gpu('gpu-1', 'operator quarantine', now=105)

    assert agent.status()['state'] == GPUState.QUARANTINED.value
    assert agent.status()['accepting_inference'] is False
    assert control.renew_reservation(route.reservation_id, now=105) is None
