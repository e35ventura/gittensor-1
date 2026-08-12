import hashlib
from dataclasses import replace

from gittensor.compute.inference_verification import HMACStreamVerifier, StreamProofContext
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
        runtime_digest='sha256:runtime',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_revision='b' * 40,
        container_image='registry.example/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
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


def test_stream_proof_binds_chunk_to_release_model_and_request():
    context = StreamProofContext('request-1', 100, 'release:1', 'model', 'a' * 40)
    verifier = HMACStreamVerifier(b's' * 32, context)
    proof = verifier.expected_proof(0, 'hello')

    assert verifier.verify(0, 'hello', proof)
    assert not verifier.verify(0, 'different', proof)
    assert not HMACStreamVerifier(
        b's' * 32,
        StreamProofContext('request-1', 100, 'release:2', 'model', 'a' * 40),
    ).verify(0, 'hello', proof)
