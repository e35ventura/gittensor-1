import hashlib
from dataclasses import replace

import bittensor as bt
import pytest

from gittensor.compute.inference_verification import (
    RuntimeStreamSigner,
    SignedStreamVerifier,
    StreamProofContext,
    canonical_stream_chunk,
)
from gittensor.compute.models import Release
from gittensor.compute.weight_challenges import WeightChallengeVerifier


class MemoryRangeSource:
    def __init__(self, contents):
        self.contents = contents

    def fetch(self, repository, revision, path, start, end):
        return self.contents[path][start:end]


def _release(contents):
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
        weight_files={path: len(value) for path, value in contents.items()},
    )
    return replace(release, release_digest=release.computed_release_digest())


def test_random_weight_range_is_checked_against_independent_reference():
    contents = {'model-00001.safetensors': b'known-weights' * 1_000}
    release = _release(contents)
    verifier = WeightChallengeVerifier(MemoryRangeSource(contents), ttl_seconds=30)
    challenge = verifier.issue('gpu-1', release, now=100)
    selected = contents[challenge.path][challenge.start_byte : challenge.end_byte]

    digest = hashlib.sha256(bytes.fromhex(challenge.nonce) + selected).hexdigest()
    assert verifier.verify(challenge.challenge_id, 'gpu-1', digest, now=101)
    assert not verifier.verify(challenge.challenge_id, 'gpu-1', digest, now=102)


def test_production_manifest_rejects_noncanonical_artifacts_and_escaping_paths():
    release = _release({'../escape.safetensors': b'bad'})

    with pytest.raises(ValueError, match='stay inside'):
        release.validate_production_manifest()


def test_attested_stream_signature_rejects_reordering_and_wrong_content():
    keypair = bt.Keypair.create_from_uri('//Alice')
    assert keypair.public_key is not None
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    verifier = SignedStreamVerifier(keypair.public_key.hex(), context)
    first_payload = {'choices': [{'delta': {'content': 'hello'}}]}
    second_payload = {'choices': [{'delta': {'content': ' world'}}]}
    first = keypair.sign(canonical_stream_chunk(context, 0, first_payload)).hex()
    second = keypair.sign(canonical_stream_chunk(context, 1, second_payload)).hex()

    assert verifier.verify(0, first_payload, first)
    assert not verifier.verify(2, second_payload, second)
    assert verifier.verify(1, second_payload, second)
    wrong_payload = {'choices': [{'delta': {'content': 'wrong'}}]}
    assert not SignedStreamVerifier(keypair.public_key.hex(), context).verify(0, wrong_payload, first)


def test_runtime_signer_and_gateway_verifier_share_the_exact_chunk_contract():
    keypair = bt.Keypair.create_from_uri('//Alice')
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    signer = RuntimeStreamSigner(keypair, context)
    verifier = SignedStreamVerifier(signer.public_key_hex, context)

    payload = signer.attach({'choices': [{'delta': {'content': 'hello'}}]})
    proof = payload['gittensor_proof']

    assert isinstance(proof, dict)
    assert verifier.verify(proof['index'], payload, proof['signature'])
