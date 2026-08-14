"""Deterministic, conservative OpenAI request capacity estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MAX_OUTPUT_TOKENS = 256


@dataclass(frozen=True)
class RequestCapacity:
    input_tokens: int
    output_tokens: int

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def normalize_openai_request(
    payload: Mapping[str, Any],
    *,
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Return one enforceable request shape for capacity-sensitive fields.

    The approved runtimes use ``max_tokens`` as their canonical output cap.
    Accept the newer alias at the public gateway, but never forward two
    potentially conflicting limits or an uncapped request. Multiple choices are
    rejected until routing can reserve their aggregate KV-cache footprint.
    """
    if (
        not isinstance(default_max_output_tokens, int)
        or isinstance(default_max_output_tokens, bool)
        or default_max_output_tokens < 1
    ):
        raise ValueError('default_max_output_tokens must be a positive integer')

    normalized = dict(payload)
    max_tokens = normalized.pop('max_tokens', None)
    max_completion_tokens = normalized.pop('max_completion_tokens', None)
    if max_tokens is not None and max_completion_tokens is not None:
        raise ValueError('max_tokens and max_completion_tokens cannot both be set')
    output_tokens = (
        max_tokens
        if max_tokens is not None
        else max_completion_tokens
        if max_completion_tokens is not None
        else default_max_output_tokens
    )
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 1:
        raise ValueError('output token limit must be a positive integer')

    choices = normalized.get('n', 1)
    if not isinstance(choices, int) or isinstance(choices, bool) or choices != 1:
        raise ValueError('n must be 1; multiple completions are not supported')

    normalized['max_tokens'] = output_tokens
    return normalized


def estimate_request_capacity(
    payload: Mapping[str, Any],
    *,
    request_overhead_tokens: int,
    max_context_tokens: int,
) -> RequestCapacity:
    """Return a tokenizer-independent upper bound for request context.

    A valid tokenizer cannot emit more ordinary tokens than the UTF-8 bytes it
    consumes. The approved release's explicit overhead covers chat templates
    and other special tokens that are not represented directly in request JSON.
    """
    if (
        not isinstance(request_overhead_tokens, int)
        or isinstance(request_overhead_tokens, bool)
        or request_overhead_tokens < 0
    ):
        raise ValueError('request_overhead_tokens must be a non-negative integer')
    if not isinstance(max_context_tokens, int) or isinstance(max_context_tokens, bool) or max_context_tokens < 1:
        raise ValueError('max_context_tokens must be a positive integer')
    normalized = normalize_openai_request(payload)
    output_tokens = normalized['max_tokens']
    sizing_payload = {
        key: value
        for key, value in normalized.items()
        if key not in {'model', 'stream', 'max_tokens', 'max_completion_tokens'}
    }
    try:
        encoded_prompt = json.dumps(
            sizing_payload,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('request body must contain valid JSON values') from exc
    capacity = RequestCapacity(
        input_tokens=max(1, len(encoded_prompt) + request_overhead_tokens),
        output_tokens=output_tokens,
    )
    if capacity.context_tokens > max_context_tokens:
        raise ValueError('request exceeds the approved release context limit')
    return capacity
