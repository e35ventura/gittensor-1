from dataclasses import replace
from email.message import Message
from unittest.mock import MagicMock, patch

import bittensor as bt
import pytest

from gittensor.compute.config import ComputeConfig, VerificationConfig
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.verification import (
    SparkComputeClient,
    SparkVerifier,
    canonical_verifier_status,
    index_snapshots,
)


def _config():
    return VerificationConfig(
        status_url='http://127.0.0.1:9090/api/status',
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


@pytest.mark.parametrize('model_age', [90, -1, float('nan'), float('inf'), 'invalid'])
def test_model_canary_age_must_be_recent_finite_and_nonnegative(model_age):
    outcome = SparkVerifier(_config()).evaluate(
        _registration(),
        Release('release:1', 'model', 'runtime'),
        _snapshot(model_age_sec=model_age),
        now=100,
    )

    assert not outcome.accepted
    assert outcome.reason is not None
    assert 'model canary' in outcome.reason


@pytest.mark.parametrize('field,value', [('last_checked', float('nan')), ('last_checked', 101)])
def test_hardware_verification_timestamp_must_be_finite_and_not_in_the_future(field, value):
    snapshot = _snapshot()
    snapshot[field] = value

    outcome = SparkVerifier(_config()).evaluate_hardware(_registration(), snapshot, now=100)

    assert not outcome.accepted
    assert outcome.reason == 'SparkCompute verification is stale'


@pytest.mark.parametrize('online_age', [90, -1, float('nan'), float('inf'), 'invalid'])
def test_heartbeat_age_must_be_recent_finite_and_nonnegative(online_age):
    outcome = SparkVerifier(_config()).evaluate_hardware(
        _registration(),
        _snapshot(online_age_sec=online_age),
        now=100,
    )

    assert not outcome.accepted
    assert outcome.reason == 'SparkCompute heartbeat is stale or has an invalid age'


def test_verifier_measurement_and_runtime_manifest_fail_closed():
    config = replace(
        _config(),
        require_verifier_measurement=True,
        expected_verifier_protocol='sparkcompute-v1',
        expected_verifier_measurement=f'sha256:{"1" * 64}',
        require_runtime_attestation=True,
        require_stream_proof=True,
        require_confidential_compute=True,
    )
    release = Release(
        'release:1',
        'model',
        'runtime',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_repository='owner/model',
        tokenizer_revision='a' * 40,
        runtime_commit='b' * 40,
        container_image='registry.example/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
    )
    snapshot = _snapshot()
    snapshot['verifier'] = {
        'protocol': 'sparkcompute-v1',
        'measurement': f'sha256:{"1" * 64}',
        'source_commit': 'abc',
    }
    snapshot['report']['runtime'] = {
        'release_digest': 'release:1',
        'model_repository': 'owner/model',
        'model_revision': 'a' * 40,
        'tokenizer_repository': 'owner/model',
        'tokenizer_revision': 'a' * 40,
        'runtime_digest': 'runtime',
        'runtime_commit': 'b' * 40,
        'container_image': 'registry.example/runtime',
        'container_digest': 'sha256:container',
        'filesystem_digest': 'sha256:filesystem',
    }
    snapshot['attestation'] = {
        'verified': True,
        'evidence_digest': 'sha256:evidence',
        'stream_public_key': '11' * 32,
        'confidential_compute': True,
        'data_policy': 'ephemeral-no-retention-v1',
    }

    assert SparkVerifier(config).evaluate(_registration(), release, snapshot, 100).accepted
    snapshot['verifier']['measurement'] = 'wrong'
    rejected = SparkVerifier(config).evaluate(_registration(), release, snapshot, 100)
    assert not rejected.accepted
    assert rejected.reason is not None
    assert 'measurement' in rejected.reason


def test_sharded_status_feeds_are_fetched_and_merged_concurrently():
    config = replace(
        _config(),
        status_url='',
        status_urls=('https://verifier-a/status', 'https://verifier-b/status'),
        refresh_workers=2,
    )
    responses = {
        'https://verifier-a/status': b'[{"id":"node-a"}]',
        'https://verifier-b/status': b'[{"id":"node-b"}]',
    }

    def open_request(request, timeout):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {'Content-Type': 'application/json'}
        response.read.return_value = responses[request.full_url]
        return response

    with patch('gittensor.compute.verification.no_redirect_urlopen', side_effect=open_request) as urlopen:
        snapshots = SparkComputeClient(config).fetch_status()

    assert {snapshot['id'] for snapshot in snapshots} == {'node-a', 'node-b'}
    assert urlopen.call_count == 2


def test_status_client_requires_valid_trusted_signature_in_strict_mode():
    signer = bt.Keypair.create_from_uri('//Verifier')
    public_key = signer.public_key
    assert public_key is not None
    public_key_hex = public_key.hex()
    body = b'[{"id":"node-a"}]'
    config = replace(
        _config(),
        require_status_signature=True,
        trusted_verifier_public_keys=(public_key_hex,),
    )

    def response_for(payload, signature):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {
            'Content-Type': 'application/json',
            'X-Gittensor-Verifier-Public-Key': public_key_hex,
            'X-Gittensor-Verifier-Signature': f'0x{signature.hex()}',
        }
        response.read.return_value = payload
        return response

    signature = signer.sign(canonical_verifier_status(body))
    with patch('gittensor.compute.verification.no_redirect_urlopen', return_value=response_for(body, signature)):
        assert SparkComputeClient(config).fetch_status() == [{'id': 'node-a'}]

    tampered = b'[{"id":"node-attacker"}]'
    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response_for(tampered, signature),
        ),
        pytest.raises(ValueError, match='signature is invalid'),
    ):
        SparkComputeClient(config).fetch_status()


def test_status_client_rejects_untrusted_signer_duplicate_keys_and_wrong_media_type():
    trusted = bt.Keypair.create_from_uri('//Verifier')
    attacker = bt.Keypair.create_from_uri('//Attacker')
    trusted_public_key = trusted.public_key
    assert trusted_public_key is not None
    trusted_public_key_hex = trusted_public_key.hex()
    config = replace(
        _config(),
        require_status_signature=True,
        trusted_verifier_public_keys=(trusted_public_key_hex,),
    )
    duplicate = b'[{"id":"node-a","id":"node-b"}]'

    def response(body, signer, content_type='application/json'):
        signer_public_key = signer.public_key
        assert signer_public_key is not None
        result = MagicMock()
        result.__enter__.return_value = result
        result.status = 200
        result.headers = {
            'Content-Type': content_type,
            'X-Gittensor-Verifier-Public-Key': signer_public_key.hex(),
            'X-Gittensor-Verifier-Signature': f'0x{signer.sign(canonical_verifier_status(body)).hex()}',
        }
        result.read.return_value = body
        return result

    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response(b'[]', attacker),
        ),
        pytest.raises(ValueError, match='signer is not trusted'),
    ):
        SparkComputeClient(config).fetch_status()
    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response(duplicate, trusted),
        ),
        pytest.raises(ValueError, match='duplicate JSON key'),
    ):
        SparkComputeClient(config).fetch_status()
    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response(b'[]', trusted, 'text/plain'),
        ),
        pytest.raises(ValueError, match='application/json'),
    ):
        SparkComputeClient(config).fetch_status()


def test_status_client_rejects_duplicate_signature_headers_and_truncated_body():
    signer = bt.Keypair.create_from_uri('//Verifier')
    public_key = signer.public_key
    assert public_key is not None
    public_key_hex = public_key.hex()
    body = b'[]'
    config = replace(
        _config(),
        require_status_signature=True,
        trusted_verifier_public_keys=(public_key_hex,),
    )

    def response_for(*, duplicate_signature=False, declared_length=None):
        result = MagicMock()
        result.__enter__.return_value = result
        result.status = 200
        headers = Message()
        headers.add_header('Content-Type', 'application/json')
        headers.add_header('X-Gittensor-Verifier-Public-Key', public_key_hex)
        signature = f'0x{signer.sign(canonical_verifier_status(body)).hex()}'
        headers.add_header('X-Gittensor-Verifier-Signature', signature)
        if duplicate_signature:
            headers.add_header('X-Gittensor-Verifier-Signature', signature)
        if declared_length is not None:
            headers.add_header('Content-Length', str(declared_length))
        result.headers = headers
        result.read.return_value = body
        return result

    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response_for(duplicate_signature=True),
        ),
        pytest.raises(ValueError, match='duplicate X-Gittensor-Verifier-Signature'),
    ):
        SparkComputeClient(config).fetch_status()
    with (
        patch(
            'gittensor.compute.verification.no_redirect_urlopen',
            return_value=response_for(declared_length=len(body) + 1),
        ),
        pytest.raises(ValueError, match='ended before Content-Length'),
    ):
        SparkComputeClient(config).fetch_status()


def test_status_index_rejects_missing_and_duplicate_node_ids():
    with pytest.raises(ValueError, match='missing its node id'):
        index_snapshots([{}])
    with pytest.raises(ValueError, match='duplicate node id'):
        index_snapshots([{'id': 'node-1'}, {'id': 'node-1'}])


def test_compute_config_rejects_duplicate_status_urls():
    from .test_control_plane import _config

    config = _config()
    object.__setattr__(config.verification, 'status_url', '')
    object.__setattr__(config.verification, 'status_urls', ('https://verifier/status',) * 2)

    with pytest.raises(ValueError, match='must be unique'):
        ComputeConfig.from_mapping(
            {
                'fleet': config.fleet.__dict__,
                'autoscaling': config.autoscaling.__dict__,
                'verification': config.verification.__dict__,
                'router': config.router.__dict__,
                'placement': config.placement.__dict__,
            }
        )


def test_strict_compute_config_requires_immutable_verifier_identifiers():
    from .test_control_plane import _config

    config = _config()
    base = {
        'fleet': config.fleet.__dict__,
        'autoscaling': config.autoscaling.__dict__,
        'verification': {
            **config.verification.__dict__,
            'require_verifier_measurement': True,
            'require_status_signature': True,
            'trusted_verifier_public_keys': ('1' * 64,),
            'source_commit': 'not-a-commit',
            'expected_verifier_measurement': 'not-a-digest',
        },
        'router': config.router.__dict__,
        'placement': config.placement.__dict__,
    }

    with pytest.raises(ValueError, match='source_commit'):
        ComputeConfig.from_mapping(base)
    base['verification']['source_commit'] = 'a' * 40
    with pytest.raises(ValueError, match='expected_verifier_measurement'):
        ComputeConfig.from_mapping(base)
    base['verification']['expected_verifier_measurement'] = f'sha256:{"0" * 64}'
    with pytest.raises(ValueError, match='replaced with the deployed digest'):
        ComputeConfig.from_mapping(base)


def test_strict_compute_config_requires_trusted_status_signing_keys():
    from .test_control_plane import _config

    config = _config()
    verification = {
        **config.verification.__dict__,
        'require_status_signature': True,
        'trusted_verifier_public_keys': (),
    }
    base = {
        'fleet': config.fleet.__dict__,
        'autoscaling': config.autoscaling.__dict__,
        'verification': verification,
        'router': config.router.__dict__,
        'placement': config.placement.__dict__,
    }

    with pytest.raises(ValueError, match='at least one trusted public key'):
        ComputeConfig.from_mapping(base)
    verification['trusted_verifier_public_keys'] = ('0' * 64,)
    with pytest.raises(ValueError, match='nonzero lowercase'):
        ComputeConfig.from_mapping(base)
    verification['trusted_verifier_public_keys'] = ('1' * 64, '1' * 64)
    with pytest.raises(ValueError, match='must be unique'):
        ComputeConfig.from_mapping(base)


def test_compute_config_normalizes_json_sequences_to_immutable_tuples():
    from .test_control_plane import _config

    config = _config()
    verification = {**config.verification.__dict__, 'status_urls': ['https://verifier.example/status']}
    verification['trusted_verifier_public_keys'] = ['1' * 64]
    loaded = ComputeConfig.from_mapping(
        {
            'fleet': config.fleet.__dict__,
            'autoscaling': config.autoscaling.__dict__,
            'verification': verification,
            'router': config.router.__dict__,
            'placement': config.placement.__dict__,
            'emission_oracle': {'tao_usd_price_sources': ['coinbase', 'coingecko']},
        }
    )

    assert loaded.verification.status_urls == ('https://verifier.example/status',)
    assert loaded.verification.trusted_verifier_public_keys == ('1' * 64,)
    assert loaded.emission_oracle.tao_usd_price_sources == ('coinbase', 'coingecko')


@pytest.mark.parametrize('strict_field', ['require_runtime_attestation', 'require_uniqueness_challenge'])
def test_strict_evidence_modes_require_a_pinned_verifier(strict_field):
    from .test_control_plane import _config

    config = _config()
    verification = {**config.verification.__dict__, strict_field: True, 'require_verifier_measurement': False}

    with pytest.raises(ValueError, match='pinned verifier measurement'):
        ComputeConfig.from_mapping(
            {
                'fleet': config.fleet.__dict__,
                'autoscaling': config.autoscaling.__dict__,
                'verification': verification,
                'router': config.router.__dict__,
                'placement': config.placement.__dict__,
            }
        )


def test_sparkcompute_status_url_requires_https_except_on_loopback():
    insecure = replace(_config(), status_url='http://verifier.example/api/status')
    with pytest.raises(ValueError, match='must use HTTPS'):
        SparkComputeClient(insecure)

    SparkComputeClient(replace(_config(), status_url='http://127.0.0.1:9090/api/status'))


def test_verifier_rejects_truthy_strings_and_malformed_nested_reports():
    verifier = SparkVerifier(_config())
    release = Release('release:1', 'model', 'runtime')
    snapshot = _snapshot(online='true')
    assert not verifier.evaluate(_registration(), release, snapshot, now=100).accepted

    snapshot = _snapshot()
    snapshot['report'] = []
    outcome = verifier.evaluate(_registration(), release, snapshot, now=100)
    assert not outcome.accepted
    assert outcome.reason == 'unexpected hardware: missing'


def test_strict_hardware_verification_requires_fresh_simultaneous_uniqueness_evidence():
    config = replace(_config(), require_uniqueness_challenge=True)
    snapshot = _snapshot()
    release = Release('release:1', 'model', 'runtime')

    missing = SparkVerifier(config).evaluate(_registration(), release, snapshot, now=100)
    assert not missing.accepted
    assert missing.reason == 'fresh simultaneous GPU uniqueness challenge is missing'

    snapshot['uniqueness'] = {
        'verified': True,
        'batch_id': 'batch-1',
        'batch_size': 3,
        'verified_at': 99,
        'challenge_digest': f'sha256:{"1" * 64}',
    }
    assert SparkVerifier(config).evaluate(_registration(), release, snapshot, now=100).accepted

    snapshot['uniqueness']['verified_at'] = 1
    assert not SparkVerifier(config).evaluate(_registration(), release, snapshot, now=1_000).accepted
