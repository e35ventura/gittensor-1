from dataclasses import replace

from gittensor.compute.config import VerificationConfig
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.verification import SparkVerifier


def _config():
    return VerificationConfig(
        status_url='http://sparkcompute/api/status',
        timeout_seconds=1,
        lease_ttl_seconds=90,
        expected_hardware='RTX 5090',
        require_model_canary=True,
        bearer_token_env=None,
        source_repository='https://github.com/gittensor-ai-lab/sparkcompute',
        source_commit='abc',
    )


def _registration(canary_digest='release:1'):
    return GPURegistration(
        gpu_id='gpu-1',
        spark_node_id='node-1',
        miner_uid=7,
        endpoint='https://miner.example',
        region='us-east',
        release_digest='release:1',
        canary_release_digest=canary_digest,
        certified_slots=4,
    )


def _snapshot(**liveness_overrides):
    liveness = {
        'online': True,
        'online_age_sec': 2,
        'gpu_live': True,
        'model_canary_enabled': True,
        'model_verified': True,
        'model_age_sec': 3,
    }
    liveness.update(liveness_overrides)
    return {
        'id': 'node-1',
        'verdict': 'VERIFIED',
        'last_checked': 95,
        'report': {
            'gpu': {
                'name': 'NVIDIA GeForce RTX 5090',
                'uuid': 'GPU-unique-1',
                'driver_version': '580.1',
            }
        },
        'liveness': liveness,
    }


def test_sparkcompute_verified_snapshot_issues_short_lived_release_bound_lease():
    outcome = SparkVerifier(_config()).evaluate(
        _registration(),
        Release('release:1', 'model', 'runtime'),
        _snapshot(),
        now=100,
    )
    assert outcome.accepted
    assert outcome.lease is not None
    assert outcome.lease.hardware_type == 'NVIDIA GeForce RTX 5090'
    assert outcome.lease.hardware_uuid == 'GPU-unique-1'
    assert outcome.lease.driver_version == '580.1'
    assert outcome.lease.release_digest == 'release:1'
    assert outcome.lease.expires_at == 185


def test_canary_must_be_enabled_verified_and_bound_to_exact_release():
    verifier = SparkVerifier(_config())
    release = Release('release:1', 'model', 'runtime')
    assert not verifier.evaluate(_registration('release:old'), release, _snapshot(), 100).accepted
    assert not verifier.evaluate(_registration(), release, _snapshot(model_canary_enabled=False), 100).accepted
    assert not verifier.evaluate(_registration(), release, _snapshot(model_verified=False), 100).accepted


def test_verifier_measurement_and_runtime_manifest_fail_closed():
    config = replace(
        _config(),
        require_verifier_measurement=True,
        expected_verifier_protocol='sparkcompute-v1',
        expected_verifier_measurement='measurement-1',
        require_runtime_attestation=True,
    )
    release = Release(
        'release:1',
        'model',
        'runtime',
        model_repository='owner/model',
        model_revision='a' * 40,
        runtime_commit='b' * 40,
        container_image='registry.example/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
    )
    snapshot = _snapshot()
    snapshot['verifier'] = {
        'protocol': 'sparkcompute-v1',
        'measurement': 'measurement-1',
        'source_commit': 'abc',
    }
    snapshot['report']['runtime'] = {
        'release_digest': 'release:1',
        'model_repository': 'owner/model',
        'model_revision': 'a' * 40,
        'runtime_digest': 'runtime',
        'runtime_commit': 'b' * 40,
        'container_image': 'registry.example/runtime',
        'container_digest': 'sha256:container',
        'filesystem_digest': 'sha256:filesystem',
    }
    snapshot['attestation'] = {'verified': True, 'evidence_digest': 'sha256:evidence'}

    assert SparkVerifier(config).evaluate(_registration(), release, snapshot, 100).accepted
    snapshot['verifier']['measurement'] = 'wrong'
    rejected = SparkVerifier(config).evaluate(_registration(), release, snapshot, 100)
    assert not rejected.accepted
    assert rejected.reason is not None
    assert 'measurement' in rejected.reason
