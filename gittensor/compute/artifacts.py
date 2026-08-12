"""Admission checks for signed immutable runtime artifacts."""

from __future__ import annotations

import subprocess
from typing import Protocol

from gittensor.compute.models import Release


class ReleaseArtifactVerifier(Protocol):
    def verify(self, release: Release) -> None: ...


class CosignReleaseVerifier:
    """Require a valid Cosign signature for the exact container digest."""

    def __init__(self, binary: str, public_key_path: str, timeout_seconds: float = 30.0) -> None:
        self.binary = binary
        self.public_key_path = public_key_path
        self.timeout_seconds = timeout_seconds

    def verify(self, release: Release) -> None:
        release.validate_production_manifest()
        image = f'{release.container_image}@{release.container_digest}'
        try:
            subprocess.run(
                [self.binary, 'verify', '--key', self.public_key_path, image],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f'container signature verification failed for {image}') from exc
