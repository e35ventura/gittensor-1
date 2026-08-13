from gittensor.compute.miner_runtime import ContainerRuntimeConfig, ContainerRuntimeManager

from .test_miner_agent import _command


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, arguments, *, check=True):
        self.calls.append((list(arguments), check))


class StagingRunner(RecordingRunner):
    def run(self, arguments, *, check=True):
        super().run(arguments, check=check)
        if arguments[:2] == ['hf', 'download']:
            target = arguments[arguments.index('--local-dir') + 1]
            from pathlib import Path

            directory = Path(target)
            directory.mkdir(parents=True)
            if directory.name == 'model':
                (directory / 'model.safetensors').write_bytes(b'w' * 64)


def test_container_runtime_uses_exact_digest_and_ephemeral_security_controls(tmp_path):
    command = _command()
    release_root = tmp_path / command.release_digest.replace(':', '_') / 'model'
    release_root.mkdir(parents=True)
    (release_root / 'model.safetensors').write_bytes(b'w' * 64)
    runner = RecordingRunner()
    runtime = ContainerRuntimeManager(
        ContainerRuntimeConfig(
            engine='docker',
            gpu_device='0',
            container_name='gittensor-runtime-gpu-0',
            weights_root=tmp_path,
            runtime_port=18000,
        ),
        runner,
    )
    runtime._wait_until_healthy = lambda: None
    runtime._verify_runtime_identity = lambda command: '11' * 32

    evidence = runtime.load(command)

    image = f'{command.container_image}@{command.container_digest}'
    assert runner.calls[0][0] == ['docker', 'pull', image]
    assert runner.calls[1][0] == ['docker', 'image', 'inspect', image]
    assert runner.calls[3][0] == ['docker', 'network', 'create', '--internal', 'gittensor-inference-internal']
    run = runner.calls[4][0]
    assert image == run[-1]
    assert '--read-only' in run
    assert ['--log-driver', 'none'] == run[run.index('--log-driver') : run.index('--log-driver') + 2]
    assert ['--cap-drop', 'ALL'] == run[run.index('--cap-drop') : run.index('--cap-drop') + 2]
    assert ['--pids-limit', '4096'] == run[run.index('--pids-limit') : run.index('--pids-limit') + 2]
    assert ['--shm-size', '16g'] == run[run.index('--shm-size') : run.index('--shm-size') + 2]
    assert ['--restart', 'no'] == run[run.index('--restart') : run.index('--restart') + 2]
    assert ['--network', 'gittensor-inference-internal'] == run[run.index('--network') : run.index('--network') + 2]
    assert not any('/run/secrets/gittensor-stream-seed' in argument for argument in run)
    assert not any('GITTENSOR_ASSIGNMENT_TOKEN=' in argument for argument in run)
    assert len(evidence.stream_public_key) == 64
    assert evidence.release_digest == command.release_digest


def test_weight_read_cannot_escape_release_directory(tmp_path):
    command = _command()
    release_root = tmp_path / command.release_digest.replace(':', '_') / 'model'
    release_root.mkdir(parents=True)
    (release_root / 'model.safetensors').write_bytes(b'w' * 64)
    runtime = ContainerRuntimeManager(
        ContainerRuntimeConfig('docker', '0', 'runtime', tmp_path, 18000),
        RecordingRunner(),
    )
    runtime.current = command

    assert runtime.read_weight_range(command, 'model.safetensors', 3, 9) == b'w' * 6


def test_runtime_stages_exact_model_and_tokenizer_revisions_before_start(tmp_path):
    command = _command()
    runner = StagingRunner()
    runtime = ContainerRuntimeManager(
        ContainerRuntimeConfig('docker', '0', 'runtime', tmp_path, 18000),
        runner,
    )
    runtime._wait_until_healthy = lambda: None
    runtime._verify_runtime_identity = lambda command: '11' * 32

    runtime.load(command)

    downloads = [call for call, _ in runner.calls if call[:2] == ['hf', 'download']]
    assert downloads[0][2:5] == [command.model_repository, '--revision', command.model_revision]
    assert downloads[1][2:5] == [command.tokenizer_repository, '--revision', command.tokenizer_revision]
    release_root = tmp_path / command.release_digest.replace(':', '_')
    assert (release_root / 'model' / 'model.safetensors').is_file()
    assert (release_root / 'tokenizer').is_dir()
