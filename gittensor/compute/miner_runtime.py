"""Miner-side execution of validator-approved model and runtime assignments."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from gittensor.compute.http_json import load_json_object
from gittensor.compute.models import AssignmentCommand, RuntimeEvidence
from gittensor.compute.safe_http import no_redirect_urlopen


class CommandRunner(Protocol):
    def run(self, arguments: Sequence[str], *, check: bool = True) -> None: ...


class SubprocessCommandRunner:
    def run(self, arguments: Sequence[str], *, check: bool = True) -> None:
        subprocess.run(
            list(arguments),
            check=check,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )


class RuntimeManager(Protocol):
    def drain(self) -> None: ...

    def load(self, command: AssignmentCommand) -> RuntimeEvidence: ...

    def read_weight_range(self, command: AssignmentCommand, path: str, start: int, end: int) -> bytes: ...

    def inference_url(self) -> str: ...

    def stream_public_key(self) -> str: ...


@dataclass(frozen=True)
class ContainerRuntimeConfig:
    engine: str
    gpu_device: str
    container_name: str
    weights_root: Path
    runtime_port: int
    runtime_container_port: int = 8000
    health_path: str = '/health'
    health_timeout_seconds: float = 180.0
    internal_network: str = 'gittensor-inference-internal'
    model_downloader_binary: str = 'hf'
    pids_limit: int = 4096
    shm_size: str = '16g'


class ContainerRuntimeManager:
    """Run the exact image digest with an ephemeral, read-only container."""

    def __init__(self, config: ContainerRuntimeConfig, runner: CommandRunner | None = None) -> None:
        if config.engine not in {'docker', 'podman'}:
            raise ValueError('container engine must be docker or podman')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config.container_name):
            raise ValueError('container_name contains unsupported characters')
        if not isinstance(config.pids_limit, int) or isinstance(config.pids_limit, bool) or config.pids_limit < 1:
            raise ValueError('pids_limit must be a positive integer')
        if not re.fullmatch(r'[1-9][0-9]*[kmgt]?', config.shm_size, re.IGNORECASE):
            raise ValueError('shm_size must be a positive container size')
        self.config = config
        self.runner = runner or SubprocessCommandRunner()
        self.current: AssignmentCommand | None = None
        self._stream_public_key = ''

    def drain(self) -> None:
        self.runner.run([self.config.engine, 'rm', '--force', self.config.container_name], check=False)
        self.current = None
        self._stream_public_key = ''

    def load(self, command: AssignmentCommand) -> RuntimeEvidence:
        self._ensure_model_staged(command)
        weights = self._release_weights(command)
        tokenizer = weights.parent / 'tokenizer'
        self._verify_weight_manifest(command, weights)
        image = f'{command.container_image}@{command.container_digest}'
        self.runner.run([self.config.engine, 'pull', image])
        self.runner.run([self.config.engine, 'image', 'inspect', image])
        self.runner.run([self.config.engine, 'rm', '--force', self.config.container_name], check=False)
        self.runner.run(
            [self.config.engine, 'network', 'create', '--internal', self.config.internal_network],
            check=False,
        )
        self.runner.run(
            [
                self.config.engine,
                'run',
                '--detach',
                '--name',
                self.config.container_name,
                '--gpus',
                f'device={self.config.gpu_device}',
                '--read-only',
                '--tmpfs',
                '/tmp:rw,noexec,nosuid,size=1g',
                '--security-opt',
                'no-new-privileges',
                '--cap-drop',
                'ALL',
                '--pids-limit',
                str(self.config.pids_limit),
                '--shm-size',
                self.config.shm_size,
                '--log-driver',
                'none',
                '--restart',
                'no',
                '--network',
                self.config.internal_network,
                '--publish',
                f'127.0.0.1:{self.config.runtime_port}:{self.config.runtime_container_port}',
                '--mount',
                f'type=bind,src={weights},dst=/models,readonly',
                '--mount',
                f'type=bind,src={tokenizer},dst=/tokenizer,readonly',
                '--env',
                f'GITTENSOR_RELEASE_DIGEST={command.release_digest}',
                '--env',
                f'GITTENSOR_MODEL_ID={command.model_id}',
                '--env',
                f'GITTENSOR_MODEL_REPOSITORY={command.model_repository}',
                '--env',
                f'GITTENSOR_MODEL_REVISION={command.model_revision}',
                '--env',
                f'GITTENSOR_TOKENIZER_REPOSITORY={command.tokenizer_repository}',
                '--env',
                f'GITTENSOR_TOKENIZER_REVISION={command.tokenizer_revision}',
                '--env',
                'HF_HUB_OFFLINE=1',
                '--env',
                'TRANSFORMERS_OFFLINE=1',
                '--env',
                f'GITTENSOR_PROOF_SCHEME={command.token_proof_scheme}',
                image,
            ]
        )
        self._wait_until_healthy()
        stream_public_key = self._verify_runtime_identity(command)
        self.current = command
        self._stream_public_key = stream_public_key
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
            stream_public_key=stream_public_key,
        )

    def read_weight_range(self, command: AssignmentCommand, path: str, start: int, end: int) -> bytes:
        if self.current != command:
            raise ValueError('challenged release is not the active runtime assignment')
        if path not in command.weight_files or start < 0 or end <= start or end > int(command.weight_files[path]):
            raise ValueError('weight challenge range is outside the approved manifest')
        root = self._release_weights(command).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError('weight challenge path escapes the approved model directory')
        with target.open('rb') as handle:
            handle.seek(start)
            contents = handle.read(end - start)
        if len(contents) != end - start:
            raise ValueError('weight file ended before the requested challenge range')
        return contents

    def inference_url(self) -> str:
        return f'http://127.0.0.1:{self.config.runtime_port}/v1/chat/completions'

    def stream_public_key(self) -> str:
        return self._stream_public_key

    def _release_weights(self, command: AssignmentCommand) -> Path:
        safe_digest = command.release_digest.replace(':', '_')
        return (self.config.weights_root / safe_digest / 'model').resolve()

    def _verify_runtime_identity(self, command: AssignmentCommand) -> str:
        url = f'http://127.0.0.1:{self.config.runtime_port}/v1/gittensor/runtime'
        with no_redirect_urlopen(url, timeout=2) as response:
            body = response.read(64 * 1024 + 1)
            if response.status != 200:
                raise ValueError('assigned runtime identity endpoint is not ready')
            content_type = str(response.headers.get('Content-Type', '')).partition(';')[0].strip().casefold()
            if content_type != 'application/json':
                raise ValueError('assigned runtime identity endpoint must use application/json')
        if len(body) > 64 * 1024:
            raise ValueError('assigned runtime identity response exceeds 64 KiB')
        payload = load_json_object(body)
        expected = {
            'release_digest': command.release_digest,
            'model_id': command.model_id,
            'model_revision': command.model_revision,
            'proof_scheme': command.token_proof_scheme,
        }
        mismatched = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatched:
            raise ValueError(f'assigned runtime identity does not match: {", ".join(sorted(mismatched))}')
        stream_public_key = str(payload.get('stream_public_key') or '')
        try:
            if len(bytes.fromhex(stream_public_key.removeprefix('0x'))) != 32:
                raise ValueError
        except ValueError:
            raise ValueError('assigned runtime stream public key is invalid') from None
        return stream_public_key

    def _ensure_model_staged(self, command: AssignmentCommand) -> None:
        root = self._release_weights(command).parent
        model_root = root / 'model'
        if model_root.is_dir():
            self._verify_weight_manifest(command, model_root)
            return
        if root.exists():
            raise ValueError(f'incomplete approved model directory must be removed by the operator: {root}')
        partial = root.with_name(f'{root.name}.partial')
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            self.runner.run(
                [
                    self.config.model_downloader_binary,
                    'download',
                    command.model_repository,
                    '--revision',
                    command.model_revision,
                    '--local-dir',
                    str(partial / 'model'),
                ]
            )
            if (
                command.tokenizer_repository == command.model_repository
                and command.tokenizer_revision == command.model_revision
            ):
                (partial / 'tokenizer').symlink_to('model', target_is_directory=True)
            else:
                self.runner.run(
                    [
                        self.config.model_downloader_binary,
                        'download',
                        command.tokenizer_repository,
                        '--revision',
                        command.tokenizer_revision,
                        '--local-dir',
                        str(partial / 'tokenizer'),
                    ]
                )
            self._verify_weight_manifest(command, partial / 'model')
            os.replace(partial, root)
        except Exception:
            if partial.exists():
                shutil.rmtree(partial)
            raise

    def _verify_weight_manifest(self, command: AssignmentCommand, root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f'approved model directory is missing: {root}')
        for relative, expected_size in command.weight_files.items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f'approved weight file is missing: {relative}')
            if target.stat().st_size != int(expected_size):
                raise ValueError(f'approved weight file size does not match manifest: {relative}')

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.config.health_timeout_seconds
        url = f'http://127.0.0.1:{self.config.runtime_port}{self.config.health_path}'
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with no_redirect_urlopen(url, timeout=2) as response:
                    body = response.read(64 * 1024 + 1)
                    if len(body) > 64 * 1024:
                        raise ValueError('runtime health response exceeds 64 KiB')
                    payload = load_json_object(body or b'{}')
                if response.status < 300 and payload.get('status') in {'ok', 'ready'}:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1)
        raise RuntimeError(f'assigned runtime did not become healthy: {last_error}')
