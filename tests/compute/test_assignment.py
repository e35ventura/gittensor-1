import pytest

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPUState, Release, RuntimeEvidence

from .test_control_plane import Clock, _config, _snapshot


class RecordingExecutor:
    def __init__(self):
        self.commands = []

    def dispatch(self, record, command):
        self.commands.append(command)


def test_global_gepetto_dispatches_and_enforces_assignment_lifecycle():
    clock = Clock(100)
    executor = RecordingExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    release = Release('release:1', 'model', 'runtime')
    control.register_release(release)
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='hotkey-7',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    assert control.refresh_verification([_snapshot(1)], now=100) == {'gpu-1': 'REGISTERED'}

    control.tick(now=101, execute=True)

    assert len(executor.commands) == 1
    assert executor.commands[0].release_digest == 'release:1'
    assert executor.commands[0].epoch == 1
    assert control.gpus['gpu-1'].state == GPUState.DRAINING
    control.acknowledge_assignment('gpu-1', 1, GPUState.LOADING, now=102)
    control.acknowledge_assignment(
        'gpu-1',
        1,
        GPUState.RUNTIME_VERIFY,
        RuntimeEvidence(
            release_digest=release.release_digest,
            model_repository=release.model_repository,
            model_revision=release.model_revision,
            runtime_digest=release.runtime_digest,
            runtime_commit=release.runtime_commit,
            container_image=release.container_image,
            container_digest=release.container_digest,
            filesystem_digest=release.filesystem_digest,
        ),
        now=103,
    )
    assert control.refresh_verification([_snapshot(1, last_checked=104)], now=104) == {'gpu-1': 'READY'}


def test_spark_node_must_be_enrolled_to_the_signing_hotkey():
    control = ComputeControlPlane(
        _config(),
        clock=Clock(100),
        spark_node_owners={'node-1': 'owner-hotkey'},
    )

    with pytest.raises(ValueError, match='not enrolled'):
        control.register_miner_gpu(
            miner_uid=7,
            miner_hotkey='attacker-hotkey',
            gpu_id='gpu-1',
            spark_node_id='node-1',
            endpoint='https://gpu-1',
            region='us-east',
        )
