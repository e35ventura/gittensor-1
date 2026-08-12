import time

import pytest

from gittensor.compute.storage import SQLiteStateStore
from gittensor.constants import COMPUTE_EMISSION_SHARE, ISSUES_TREASURY_UID, RECYCLE_UID
from gittensor.validator.compute_rewards import load_compute_scores
from gittensor.validator.emission_allocation import blend_emission_pools


def test_finalized_hotkey_settlement_enters_validator_weights(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.record_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': '2.5'},
        {'window_budget': '2.5'},
    )

    scores = load_compute_scores(
        ['recycle', 'miner-hotkey'] + ['unused'] * (ISSUES_TREASURY_UID - 2) + ['treasury'],
        database_path=str(path),
        max_age_seconds=1_000,
    )
    miner_uids = {RECYCLE_UID, 1, ISSUES_TREASURY_UID}
    rewards = blend_emission_pools({}, {}, miner_uids, compute_scores=scores)

    assert scores == {1: 2.5}
    assert rewards[sorted(miner_uids).index(1)] == pytest.approx(COMPUTE_EMISSION_SHARE)
    assert rewards.sum() == pytest.approx(1.0)


def test_missing_compute_settlement_recycles_compute_slice(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    SQLiteStateStore(path)
    scores = load_compute_scores(['recycle'], database_path=str(path), max_age_seconds=1_000)

    rewards = blend_emission_pools({}, {}, {RECYCLE_UID}, compute_scores=scores)

    assert scores == {}
    assert rewards[0] == pytest.approx(0.9)
