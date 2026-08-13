import subprocess
from dataclasses import replace
from unittest.mock import patch

import pytest

from gittensor.compute.artifacts import CosignReleaseVerifier
from gittensor.compute.models import Release


def _release():
    release = Release(
        release_digest='pending',
        model_id='model',
        runtime_digest=f'sha256:{"1" * 64}',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_repository='owner/model',
        tokenizer_revision='b' * 40,
        container_image='registry.example/runtime',
        container_digest=f'sha256:{"2" * 64}',
        filesystem_digest=f'sha256:{"3" * 64}',
        runtime_commit='c' * 40,
        weight_files={'model.safetensors': 100},
    )
    return replace(release, release_digest=release.computed_release_digest())


def test_cosign_verifies_exact_container_digest_before_admission():
    release = _release()
    with patch('gittensor.compute.artifacts.subprocess.run') as run:
        CosignReleaseVerifier('cosign', '/keys/runtime.pub').verify(release)

    run.assert_called_once_with(
        [
            'cosign',
            'verify',
            '--key',
            '/keys/runtime.pub',
            f'registry.example/runtime@sha256:{"2" * 64}',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )


def test_cosign_failure_rejects_release():
    with patch(
        'gittensor.compute.artifacts.subprocess.run',
        side_effect=subprocess.CalledProcessError(1, 'cosign'),
    ):
        with pytest.raises(ValueError, match='signature verification failed'):
            CosignReleaseVerifier('cosign', '/keys/runtime.pub').verify(_release())
