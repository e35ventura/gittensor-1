"""Deterministic, conservative OpenAI request capacity estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RequestCapacity:
    input_tokens: int
    output_tokens: int

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
    output_tokens = payload.get('max_tokens')
    if output_tokens is None:
        output_tokens = payload.get('max_completion_tokens', 256)
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    sizing_payload = {
        key: value
        for key, value in payload.items()
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
