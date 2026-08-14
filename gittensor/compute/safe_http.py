"""HTTPS egress pinned to a prevalidated public IP to prevent DNS rebinding."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import threading
import urllib.request
from typing import Mapping
from urllib.parse import urljoin, urlsplit


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        self._ssl_context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._ssl_context)
        self._address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw_socket, server_hostname=self.host)


class ManagedHTTPResponse:
    """Close both the response and its pinned connection."""

    def __init__(self, response: http.client.HTTPResponse, connection: _PinnedHTTPSConnection) -> None:
        self._response = response
        self._connection = connection
        self._lock = threading.Lock()

    @property
    def status(self) -> int:
        return self._response.status

    @property
    def headers(self) -> http.client.HTTPMessage:
        return self._response.headers

    def read(self, amount: int | None = None) -> bytes:
        return self._response.read(amount)

    def readline(self, limit: int = -1) -> bytes:
        return self._response.readline(limit)

    def close(self) -> None:
        with self._lock:
            self._response.close()
            self._connection.close()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def no_redirect_urlopen(request: urllib.request.Request | str, *, timeout: float):
    """Open one configured URL without allowing credential or data redirects."""
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


def validate_https_or_loopback_origin(url: str, field_name: str) -> None:
    """Require a TLS origin, permitting cleartext only for explicit local development."""
    parsed = urlsplit(url)
    loopback_http = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', '::1', 'localhost'}
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {'', '/'}
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != 'https' and not loopback_http)
    ):
        raise ValueError(f'{field_name} must be an HTTPS origin, except on loopback')


def public_https_request(
    url: str,
    *,
    method: str,
    body: bytes | None,
    headers: Mapping[str, str],
    timeout: float,
) -> ManagedHTTPResponse:
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('endpoint must be an absolute public HTTPS URL without userinfo')
    hostname = parsed.hostname.casefold()
    if hostname == 'localhost' or hostname.endswith('.localhost'):
        raise ValueError('endpoint cannot resolve to localhost')
    port = parsed.port or 443
    try:
        addresses = sorted(
            {str(item[4][0]) for item in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)}
        )
    except socket.gaierror as exc:
        raise ValueError('endpoint hostname could not be resolved') from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('endpoint must resolve only to public IP addresses')
    connection = _PinnedHTTPSConnection(parsed.hostname, port, addresses[0], timeout)
    path = parsed.path or '/'
    if parsed.query:
        path = f'{path}?{parsed.query}'
    try:
        connection.request(method, path, body=body, headers=dict(headers))
        return ManagedHTTPResponse(connection.getresponse(), connection)
    except Exception:
        connection.close()
        raise


def public_https_get_follow_redirects(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
    max_redirects: int = 5,
) -> ManagedHTTPResponse:
    """Follow bounded GET redirects only when every hop is public HTTPS."""
    current = url
    for redirect_count in range(max_redirects + 1):
        response = public_https_request(
            current,
            method='GET',
            body=None,
            headers=headers,
            timeout=timeout,
        )
        if response.status not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get('Location')
        response.close()
        if redirect_count >= max_redirects:
            raise ValueError('public HTTPS redirect limit exceeded')
        if not isinstance(location, str) or not location:
            raise ValueError('public HTTPS redirect is missing Location')
        current = urljoin(current, location)
        parsed = urlsplit(current)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('public redirect target must be HTTPS without userinfo')
    raise AssertionError('unreachable redirect loop')
