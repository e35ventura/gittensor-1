import pytest

from gittensor.compute.request_capacity import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    estimate_request_capacity,
    normalize_openai_request,
)


def test_normalization_injects_an_enforceable_default_output_cap():
    normalized = normalize_openai_request({'model': 'model', 'messages': []})
    capacity = estimate_request_capacity(
        normalized,
        request_overhead_tokens=0,
        max_context_tokens=4_096,
    )

    assert normalized['max_tokens'] == DEFAULT_MAX_OUTPUT_TOKENS
    assert 'max_completion_tokens' not in normalized
    assert capacity.output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_normalization_canonicalizes_the_completion_token_alias():
    normalized = normalize_openai_request({'model': 'model', 'messages': [], 'max_completion_tokens': 37})

    assert normalized['max_tokens'] == 37
    assert 'max_completion_tokens' not in normalized


def test_normalization_rejects_ambiguous_output_limits():
    with pytest.raises(ValueError, match='cannot both be set'):
        normalize_openai_request(
            {
                'model': 'model',
                'max_tokens': 1,
                'max_completion_tokens': 100_000,
            }
        )


@pytest.mark.parametrize('choices', [0, 2, True, 1.5, '2', None])
def test_normalization_rejects_unreserved_completion_multiplicity(choices):
    with pytest.raises(ValueError, match='n must be 1'):
        normalize_openai_request({'model': 'model', 'n': choices})
