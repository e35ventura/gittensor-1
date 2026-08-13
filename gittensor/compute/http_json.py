"""Strict JSON request parsing shared by compute HTTP services."""

from __future__ import annotations

import json
from typing import Any, Protocol


class ReadableBody(Protocol):
    def read(self, size: int | None = -1, /) -> bytes: ...


class HTTPHeaders(Protocol):
    def get(self, name: str, failobj: Any = None) -> Any: ...

    def get_all(self, name: str, failobj: Any = None) -> list[str] | None: ...


def load_json_value(body: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f'non-standard JSON constant is not allowed: {value}')

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'duplicate JSON key is not allowed: {key}')
            value[key] = item
        return value

    return json.loads(body, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)


def load_json_object(body: bytes) -> dict[str, Any]:
    payload = load_json_value(body)
    if not isinstance(payload, dict):
        raise ValueError('request body must be a JSON object')
    return payload


def read_json_object(stream: ReadableBody, headers: HTTPHeaders, max_bytes: int) -> dict[str, Any]:
    if headers.get('Transfer-Encoding') is not None:
        raise ValueError('Transfer-Encoding is not supported')
    content_lengths = headers.get_all('Content-Length') or []
    if len(content_lengths) != 1 or ',' in content_lengths[0]:
        raise ValueError('exactly one Content-Length header is required')
    content_type = str(headers.get('Content-Type', '')).partition(';')[0].strip().casefold()
    if content_type != 'application/json':
        raise ValueError('Content-Type must be application/json')
    try:
        length = int(content_lengths[0])
    except ValueError:
        raise ValueError('Content-Length must be an integer') from None
    if length < 1 or length > max_bytes:
        raise ValueError('request body size is invalid')
    body = stream.read(length)
    if len(body) != length:
        raise ValueError('request body ended before Content-Length')
    return load_json_object(body)
