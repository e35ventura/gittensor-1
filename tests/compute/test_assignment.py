import threading

import pytest

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPUState, Release, RuntimeEvidence

from .test_control_plane import Clock, _config, _snapshot


class RecordingExecutor:
    def __init__(self):
        self.commands = []

    def dispatch(self, record, command):
        self.commands.append(command)


class FailingOnceExecutor:
    def __init__(self):
        self.calls = 0

    def dispatch(self, record, command):
        self.calls += 1
        if self.calls == 1:
            raise OSError('temporary failure')


class BlockingExecutor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def dispatch(self, record, command):
        self.started.set()
        self.release.wait(2)


class BlockingRevocationExecutor(BlockingExecutor):
    def __init__(self):
        super().__init__()
        self.revocations = []

    def revoke(self, record, epoch, reason):
        self.revocations.append((record.registration.gpu_id, epoch, reason))


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
            tokenizer_repository=release.tokenizer_repository,
            tokenizer_revision=release.tokenizer_revision,
            runtime_digest=release.runtime_digest,
            runtime_commit=release.runtime_commit,
            container_image=release.container_image,
            container_digest=release.container_digest,
            filesystem_digest=release.filesystem_digest,
            stream_public_key='11' * 32,
        ),
        now=103,
    )
    assert (
        control.acknowledge_assignment(
            'gpu-1',
            1,
            GPUState.RUNTIME_VERIFY,
            RuntimeEvidence(
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
            ),
            now=103,
        )
        == GPUState.RUNTIME_VERIFY
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


def test_failed_assignment_delivery_is_retried_with_same_epoch():
    clock = Clock(100)
    executor = FailingOnceExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='hotkey-7',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)

    control.tick(now=101, execute=True)
    assert not control.gpus['gpu-1'].assignment_dispatched
    assert control.gpus['gpu-1'].assignment_epoch == 1

    control.tick(now=102, execute=True)
    assert control.gpus['gpu-1'].assignment_dispatched
    assert control.gpus['gpu-1'].assignment_epoch == 1
    assert executor.calls == 2


def test_slow_assignment_delivery_does_not_hold_control_plane_lock():
    clock = Clock(100)
    executor = BlockingExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='hotkey-7',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)
    worker = threading.Thread(target=lambda: control.tick(now=101, execute=True))
    worker.start()
    assert executor.started.wait(1)

    assert control.status(now=101)['registered_gpus'] == 1

    executor.release.set()
    worker.join(2)
    assert not worker.is_alive()


def test_disable_racing_with_assignment_delivery_revokes_the_late_command():
    clock = Clock(100)
    executor = BlockingRevocationExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='hotkey-7',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)
    worker = threading.Thread(target=lambda: control.tick(now=101, execute=True))
    worker.start()
    assert executor.started.wait(1)

    control.disable_gpu('gpu-1', 'operator quarantine', now=102)
    with pytest.raises(ValueError, match='revocation has not been acknowledged'):
        control.enable_gpu('gpu-1', now=102)
    executor.release.set()
    worker.join(2)

    assert not worker.is_alive()
    assert executor.revocations == [
        ('gpu-1', 1, 'administratively disabled: operator quarantine'),
        ('gpu-1', 1, 'administratively disabled: operator quarantine'),
    ]
    assert control.gpus['gpu-1'].administratively_disabled
    assert not control.gpus['gpu-1'].assignment_dispatched


def test_execute_mode_refuses_to_claim_transitions_without_an_assignment_executor():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_miner_gpu(
        miner_uid=7,
        miner_hotkey='hotkey-7',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)

    with pytest.raises(RuntimeError, match='assignment executor'):
        control.tick(now=101, execute=True)

    assert control.gpus['gpu-1'].registration.release_digest == ''
    assert control.gpus['gpu-1'].state == GPUState.REGISTERED
