"""Load finalized compute settlement into the validator emission round."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit

import bittensor as bt

from gittensor.compute.http_json import load_json_object
from gittensor.compute.safe_http import no_redirect_urlopen
from gittensor.compute.settlement_auth import verify_settlement
from gittensor.compute.storage import SQLiteStateStore
from gittensor.constants import COMPUTE_EMISSION_SHARE, MAX_COMPUTE_EMISSION_SHARE


@dataclass(frozen=True)
class ComputeAllocation:
    scores: dict[int, float]
    emission_share: float
    reserved_emission_share: float


_MAX_SETTLEMENT_BYTES = 1024 * 1024
_MAX_FUTURE_SKEW_SECONDS = 60.0


def load_compute_allocation(
    hotkeys: Sequence[str],
    *,
    database_path: str | None = None,
    max_age_seconds: float = 7_200,
) -> ComputeAllocation | None:
    settlement_url = os.environ.get('GITTENSOR_COMPUTE_SETTLEMENT_URL')
    if settlement_url:
        parsed_url = urlsplit(settlement_url)
        if (
            parsed_url.scheme != 'https'
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            bt.logging.error('GITTENSOR_COMPUTE_SETTLEMENT_URL must use HTTPS')
            return _recycled_compute_allocation()
        expected_hotkey = os.environ.get('GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY')
        if not expected_hotkey:
            bt.logging.error('GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY is required for the remote settlement feed')
            return _recycled_compute_allocation()
        try:
            with no_redirect_urlopen(settlement_url, timeout=10) as response:
                content_length = response.headers.get('Content-Length')
                if content_length is not None and int(content_length) > _MAX_SETTLEMENT_BYTES:
                    raise ValueError('settlement response exceeds 1 MiB')
                body = response.read(_MAX_SETTLEMENT_BYTES + 1)
                if len(body) > _MAX_SETTLEMENT_BYTES:
                    raise ValueError('settlement response exceeds 1 MiB')
                settlement = load_json_object(body)
        except (OSError, ValueError) as exc:
            bt.logging.warning(f'compute settlement feed is unavailable: {exc}')
            return _recycled_compute_allocation()
        if not isinstance(settlement, dict) or not verify_settlement(settlement, expected_hotkey):
            bt.logging.error('compute settlement feed signature is invalid')
            return _recycled_compute_allocation()
        try:
            now = time.time()
            started_at = float(settlement['started_at'])
            ended_at = float(settlement['ended_at'])
            fresh = (
                math.isfinite(started_at)
                and math.isfinite(ended_at)
                and started_at < ended_at
                and ended_at >= now - max_age_seconds
                and ended_at <= now + _MAX_FUTURE_SKEW_SECONDS
            )
        except (KeyError, TypeError, ValueError):
            fresh = False
        if not fresh:
            bt.logging.warning('compute settlement feed returned a stale settlement')
            return _recycled_compute_allocation(settlement)
        return _allocation_from_settlement(hotkeys, settlement)
    path = database_path or os.environ.get('GITTENSOR_COMPUTE_DB')
    if not path:
        return None
    if not Path(path).exists():
        bt.logging.warning(f'compute settlement database does not exist: {path}')
        return _recycled_compute_allocation()
    settlement = SQLiteStateStore(path).latest_settlement(max_age_seconds)
    if settlement is None:
        bt.logging.warning('no fresh compute settlement is available; compute slice will recycle')
        return _recycled_compute_allocation()
    return _allocation_from_settlement(hotkeys, settlement)


def _allocation_from_settlement(
    hotkeys: Sequence[str],
    settlement: dict[str, object],
) -> ComputeAllocation:
    uid_by_hotkey = {str(hotkey): uid for uid, hotkey in enumerate(hotkeys)}
    scores: dict[int, float] = {}
    rewards = settlement.get('hotkey_rewards')
    metadata = settlement.get('metadata')
    if not isinstance(rewards, dict) or not isinstance(metadata, dict):
        return ComputeAllocation({}, 0.0, COMPUTE_EMISSION_SHARE)
    for hotkey, raw_amount in rewards.items():
        uid = uid_by_hotkey.get(hotkey)
        if uid is None:
            continue
        try:
            amount = Decimal(str(raw_amount))
        except (InvalidOperation, ValueError):
            continue
        if amount.is_finite() and amount > 0:
            numeric_amount = float(amount)
            if numeric_amount < float('inf'):
                scores[uid] = scores.get(uid, 0.0) + numeric_amount
    raw_emission_share = metadata.get('compute_emission_share')
    emission_share_valid = raw_emission_share is not None
    try:
        emission_share = float(raw_emission_share) if raw_emission_share is not None else 0.0
    except (TypeError, ValueError):
        emission_share = 0.0
    if not math.isfinite(emission_share) or emission_share < 0:
        emission_share_valid = False
        emission_share = 0.0
    raw_reserved_share = metadata.get('compute_reserved_emission_share')
    try:
        reserved_share = float(
            raw_reserved_share
            if raw_reserved_share is not None
            else emission_share
            if emission_share_valid
            else COMPUTE_EMISSION_SHARE
        )
    except (TypeError, ValueError):
        reserved_share = COMPUTE_EMISSION_SHARE
    if not math.isfinite(reserved_share) or reserved_share < 0:
        reserved_share = COMPUTE_EMISSION_SHARE
    emission_share = min(MAX_COMPUTE_EMISSION_SHARE, emission_share)
    reserved_share = min(MAX_COMPUTE_EMISSION_SHARE, max(emission_share, reserved_share))
    return ComputeAllocation(scores, emission_share, reserved_share)


def _recycled_compute_allocation(settlement: dict[str, object] | None = None) -> ComputeAllocation:
    """Reserve the last stated compute slice, or its baseline, for recycle."""
    if settlement is not None:
        allocation = _allocation_from_settlement((), settlement)
        if allocation.reserved_emission_share > 0:
            return ComputeAllocation({}, 0.0, allocation.reserved_emission_share)
    return ComputeAllocation({}, 0.0, COMPUTE_EMISSION_SHARE)
