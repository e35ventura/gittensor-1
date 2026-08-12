"""Execute global Gepetto assignments through miner runtime agents."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import Protocol

from gittensor.compute.models import AssignmentCommand, GPURecord


class AssignmentExecutor(Protocol):
    def dispatch(self, record: GPURecord, command: AssignmentCommand) -> None: ...


class HTTPAssignmentExecutor:
    """Send an epoch-bound desired release to the miner's deployment agent."""

    def __init__(self, timeout_seconds: float, bearer_token: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token

    def dispatch(self, record: GPURecord, command: AssignmentCommand) -> None:
        url = f'{record.registration.endpoint.rstrip("/")}/v1/gittensor/assignments'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        request = urllib.request.Request(
            url,
            data=json.dumps(asdict(command), separators=(',', ':')).encode(),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode())
        if payload.get('accepted') is not True or int(payload.get('epoch', -1)) != command.epoch:
            raise ValueError('miner runtime agent did not accept the exact assignment epoch')
