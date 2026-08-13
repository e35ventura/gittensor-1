import io
from email.message import Message

import pytest

from gittensor.compute.http_json import load_json_object, read_json_object


def _headers(**values):
    headers = Message()
    for key, value in values.items():
        headers.add_header(key.replace('_', '-'), value)
    return headers


def test_strict_json_rejects_non_object_and_nonstandard_numbers():
    with pytest.raises(ValueError, match='JSON object'):
        load_json_object(b'[]')
    with pytest.raises(ValueError, match='non-standard'):
        load_json_object(b'{"value":NaN}')
    with pytest.raises(ValueError, match='non-standard'):
        load_json_object(b'{"value":Infinity}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        load_json_object(b'{"value":1,"value":2}')


def test_http_json_requires_unambiguous_fixed_json_body():
    body = b'{"ok":true}'
    headers = _headers(Content_Type='application/json', Content_Length=str(len(body)))
    assert read_json_object(io.BytesIO(body), headers, 100) == {'ok': True}

    headers.add_header('Content-Length', str(len(body)))
    with pytest.raises(ValueError, match='exactly one'):
        read_json_object(io.BytesIO(body), headers, 100)

    chunked = _headers(Content_Type='application/json', Content_Length=str(len(body)), Transfer_Encoding='chunked')
    with pytest.raises(ValueError, match='Transfer-Encoding'):
        read_json_object(io.BytesIO(body), chunked, 100)


def test_http_json_rejects_wrong_media_type_and_short_body():
    body = b'{"ok":true}'
    with pytest.raises(ValueError, match='Content-Type'):
        read_json_object(
            io.BytesIO(body),
            _headers(Content_Type='text/plain', Content_Length=str(len(body))),
            100,
        )
    with pytest.raises(ValueError, match='ended before'):
        read_json_object(
            io.BytesIO(body),
            _headers(Content_Type='application/json', Content_Length=str(len(body) + 1)),
            100,
        )
