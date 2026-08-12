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
        runtime_digest='sha256:runtime',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_revision='b' * 40,
        container_image='registry.example/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
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
            'registry.example/runtime@sha256:container',
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
