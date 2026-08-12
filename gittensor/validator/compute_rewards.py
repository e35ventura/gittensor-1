"""Load finalized compute settlement into the validator emission round."""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import bittensor as bt

from gittensor.compute.storage import SQLiteStateStore


def load_compute_scores(
    hotkeys: Sequence[str],
    *,
    database_path: str | None = None,
    max_age_seconds: float = 7_200,
) -> dict[int, float] | None:
    """Map the latest durable hotkey settlement onto current metagraph UIDs.

    ``None`` means compute emissions are not configured. An empty dictionary
    means they are configured but no fresh eligible settlement exists, so the
    compute slice recycles rather than being paid from stale state.
    """
    path = database_path or os.environ.get('GITTENSOR_COMPUTE_DB')
    if not path:
        return None
    if not Path(path).exists():
        bt.logging.warning(f'compute settlement database does not exist: {path}')
        return {}
    settlement = SQLiteStateStore(path).latest_settlement(max_age_seconds)
    if settlement is None:
        bt.logging.warning('no fresh compute settlement is available; compute slice will recycle')
        return {}
    uid_by_hotkey = {str(hotkey): uid for uid, hotkey in enumerate(hotkeys)}
    scores: dict[int, float] = {}
    for hotkey, raw_amount in settlement['hotkey_rewards'].items():
        uid = uid_by_hotkey.get(hotkey)
        if uid is None:
            continue
        try:
            amount = Decimal(str(raw_amount))
        except InvalidOperation:
            continue
        if amount > 0:
            scores[uid] = scores.get(uid, 0.0) + float(amount)
    return scores
