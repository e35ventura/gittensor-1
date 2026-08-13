import socket
from unittest.mock import MagicMock, patch

import pytest

from gittensor.compute.safe_http import public_https_get_follow_redirects, public_https_request


def test_https_egress_rejects_private_dns_answers(monkeypatch):
    monkeypatch.setattr(
        socket,
        'getaddrinfo',
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))],
    )

    with pytest.raises(ValueError, match='public IP'):
        public_https_request(
            'https://miner.example/v1/gittensor/assignments',
            method='POST',
            body=b'{}',
            headers={},
            timeout=1,
        )


def test_https_egress_rejects_userinfo_and_localhost():
    with pytest.raises(ValueError, match='userinfo'):
        public_https_request(
            'https://user:password@miner.example/path',
            method='POST',
            body=None,
            headers={},
            timeout=1,
        )
    with pytest.raises(ValueError, match='localhost'):
        public_https_request(
            'https://localhost/path',
            method='POST',
            body=None,
            headers={},
            timeout=1,
        )


def test_public_get_follows_only_bounded_https_redirects():
    redirect = MagicMock(status=302, headers={'Location': 'https://cdn.example/weights'})
    final = MagicMock(status=206, headers={'Content-Range': 'bytes 0-9/100'})
    with patch(
        'gittensor.compute.safe_http.public_https_request',
        side_effect=[redirect, final],
    ) as request:
        response = public_https_get_follow_redirects(
            'https://huggingface.co/model',
            headers={'Range': 'bytes=0-9'},
            timeout=1,
        )

    assert response is final
    redirect.close.assert_called_once()
    assert request.call_args_list[1].args[0] == 'https://cdn.example/weights'


def test_public_get_rejects_redirect_downgrade():
    redirect = MagicMock(status=302, headers={'Location': 'http://cdn.example/weights'})
    with (
        patch('gittensor.compute.safe_http.public_https_request', return_value=redirect),
        pytest.raises(ValueError, match='must be HTTPS'),
    ):
        public_https_get_follow_redirects(
            'https://huggingface.co/model',
            headers={'Range': 'bytes=0-9'},
            timeout=1,
        )
    redirect.close.assert_called_once()
