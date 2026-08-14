"""Execute global Gepetto assignments through miner runtime agents."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from gittensor.compute.auth import ValidatorRequestSigner
from gittensor.compute.http_json import load_json_object
from gittensor.compute.models import AssignmentCommand, GPURecord
from gittensor.compute.safe_http import public_https_request

_MAX_AGENT_RESPONSE_BYTES = 64 * 1024


class AssignmentExecutor(Protocol):
    def dispatch(self, record: GPURecord, command: AssignmentCommand) -> None: ...


class HTTPAssignmentExecutor:
    """Send an epoch-bound desired release to the miner's deployment agent."""

    def __init__(self, timeout_seconds: float, signer: ValidatorRequestSigner) -> None:
        self.timeout_seconds = timeout_seconds
        self.signer = signer

    def dispatch(self, record: GPURecord, command: AssignmentCommand) -> None:
        endpoint = record.registration.endpoint.rstrip('/')
        path = '/v1/gittensor/assignments'
        url = f'{endpoint}{path}'
        payload = asdict(command)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        headers.update(self.signer.headers('POST', path, payload))
        response = public_https_request(
            url,
            body=json.dumps(payload, separators=(',', ':')).encode(),
            headers=headers,
            method='POST',
            timeout=self.timeout_seconds,
        )
        try:
            if response.status >= 300:
                raise ValueError(f'miner runtime agent returned HTTP {response.status}')
            body = response.read(_MAX_AGENT_RESPONSE_BYTES + 1)
            if len(body) > _MAX_AGENT_RESPONSE_BYTES:
                raise ValueError('miner runtime agent response exceeds 64 KiB')
            payload = load_json_object(body)
        finally:
            response.close()
        if payload.get('accepted') is not True or int(payload.get('epoch', -1)) != command.epoch:
            raise ValueError('miner runtime agent did not accept the exact assignment epoch')

    def revoke(self, record: GPURecord, epoch: int, reason: str) -> None:
        """Invalidate one assignment at its miner agent using validator authentication."""
        endpoint = record.registration.endpoint.rstrip('/')
        path = '/v1/gittensor/revocations'
        url = f'{endpoint}{path}'
        payload = {'gpu_id': record.registration.gpu_id, 'epoch': epoch, 'reason': reason}
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        headers.update(self.signer.headers('POST', path, payload))
        response = public_https_request(
            url,
            body=json.dumps(payload, separators=(',', ':')).encode(),
            headers=headers,
            method='POST',
            timeout=self.timeout_seconds,
        )
        try:
            if response.status >= 300:
                raise ValueError(f'miner runtime agent returned HTTP {response.status}')
            body = response.read(_MAX_AGENT_RESPONSE_BYTES + 1)
            if len(body) > _MAX_AGENT_RESPONSE_BYTES:
                raise ValueError('miner runtime agent response exceeds 64 KiB')
            result = load_json_object(body)
        finally:
            response.close()
        if result.get('revoked') is not True or int(result.get('epoch', -1)) != epoch:
            raise ValueError('miner runtime agent did not revoke the exact assignment epoch')
