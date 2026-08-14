import io
import json
import time

import pytest

from gittensor.compute.settlement_auth import SettlementSigner
from gittensor.compute.storage import SQLiteStateStore
from gittensor.constants import COMPUTE_EMISSION_SHARE, ISSUES_TREASURY_UID, RECYCLE_UID
from gittensor.validator.compute_rewards import load_compute_allocation
from gittensor.validator.emission_allocation import blend_emission_pools


def test_finalized_hotkey_settlement_enters_validator_weights(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.finalize_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': '2.5'},
        {'window_budget': '2.5', 'compute_emission_share': 0.1},
        {},
    )

    allocation = load_compute_allocation(
        ['recycle', 'miner-hotkey'] + ['unused'] * (ISSUES_TREASURY_UID - 2) + ['treasury'],
        database_path=str(path),
        max_age_seconds=1_000,
    )
    miner_uids = {RECYCLE_UID, 1, ISSUES_TREASURY_UID}
    assert allocation is not None
    rewards = blend_emission_pools(
        {},
        {},
        miner_uids,
        compute_scores=allocation.scores,
        compute_emission_share=allocation.emission_share,
        compute_reserved_emission_share=allocation.reserved_emission_share,
    )

    assert allocation.scores == {1: 2.5}
    assert rewards[sorted(miner_uids).index(1)] == pytest.approx(0.1)
    assert rewards.sum() == pytest.approx(1.0)


def test_missing_compute_settlement_recycles_compute_slice(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    SQLiteStateStore(path)
    allocation = load_compute_allocation(['recycle'], database_path=str(path), max_age_seconds=1_000)

    assert allocation is not None
    rewards = blend_emission_pools(
        {},
        {},
        {RECYCLE_UID},
        compute_scores=allocation.scores,
        compute_emission_share=allocation.emission_share,
        compute_reserved_emission_share=allocation.reserved_emission_share,
    )

    assert allocation.scores == {}
    assert allocation.emission_share == 0.0
    assert allocation.reserved_emission_share == COMPUTE_EMISSION_SHARE
    assert rewards[0] == pytest.approx(0.9)


def test_target_scaled_compute_share_enters_validator_weights(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.finalize_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': '3.9'},
        {'window_budget': '3.9', 'compute_emission_share': 0.15},
        {},
    )
    hotkeys = ['recycle', 'miner-hotkey'] + ['unused'] * (ISSUES_TREASURY_UID - 2) + ['treasury']

    allocation = load_compute_allocation(hotkeys, database_path=str(path), max_age_seconds=1_000)
    assert allocation is not None
    miner_uids = {RECYCLE_UID, 1, ISSUES_TREASURY_UID}
    rewards = blend_emission_pools(
        {},
        {},
        miner_uids,
        compute_scores=allocation.scores,
        compute_emission_share=allocation.emission_share,
        compute_reserved_emission_share=allocation.reserved_emission_share,
    )

    assert allocation.emission_share == pytest.approx(0.15)
    assert rewards[sorted(miner_uids).index(1)] == pytest.approx(0.15)
    assert rewards.sum() == pytest.approx(1.0)


def test_unpaid_scarcity_budget_recycles_instead_of_expanding_oss_rewards(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.finalize_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': '1.3'},
        {
            'compute_emission_share': 0.05,
            'compute_reserved_emission_share': 0.10,
        },
        {},
    )
    hotkeys = ['recycle', 'miner-hotkey'] + ['unused'] * (ISSUES_TREASURY_UID - 2) + ['treasury']
    allocation = load_compute_allocation(hotkeys, database_path=str(path), max_age_seconds=1_000)
    assert allocation is not None
    miner_uids = {RECYCLE_UID, 1, ISSUES_TREASURY_UID}

    rewards = blend_emission_pools(
        {},
        {},
        miner_uids,
        compute_scores=allocation.scores,
        compute_emission_share=allocation.emission_share,
        compute_reserved_emission_share=allocation.reserved_emission_share,
    )

    assert rewards[sorted(miner_uids).index(1)] == pytest.approx(0.05)
    assert rewards[sorted(miner_uids).index(RECYCLE_UID)] == pytest.approx(0.85)
    assert rewards.sum() == pytest.approx(1.0)


class SettlementResponse(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_remote_validator_feed_requires_valid_signature(monkeypatch):
    import bittensor as bt

    keypair = bt.Keypair.create_from_uri('//Alice')
    now = time.time()
    metadata = {'compute_emission_share': 0.15}
    settlement = {
        'window_id': 'window',
        'started_at': now - 100,
        'ended_at': now,
        'hotkey_rewards': {'miner-hotkey': '3.9'},
        'metadata': {
            **metadata,
            **SettlementSigner(keypair).sign('window', now - 100, now, {'miner-hotkey': '3.9'}, metadata),
        },
    }
    monkeypatch.setenv('GITTENSOR_COMPUTE_SETTLEMENT_URL', 'https://control.example/v1/settlements/latest')
    monkeypatch.setenv('GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY', keypair.ss58_address)
    monkeypatch.setattr(
        'gittensor.validator.compute_rewards.no_redirect_urlopen',
        lambda *args, **kwargs: SettlementResponse(json.dumps(settlement).encode()),
    )

    allocation = load_compute_allocation(['recycle', 'miner-hotkey'])

    assert allocation is not None
    assert allocation.scores == {1: 3.9}
    assert allocation.emission_share == pytest.approx(0.15)


def test_non_finite_settlement_values_cannot_poison_validator_weights(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.finalize_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': 'NaN'},
        {'compute_emission_share': float('nan')},
        {},
    )

    allocation = load_compute_allocation(['recycle', 'miner-hotkey'], database_path=str(path))

    assert allocation is not None
    assert allocation.scores == {}
    assert allocation.emission_share == 0.0
    assert allocation.reserved_emission_share == COMPUTE_EMISSION_SHARE


def test_compute_settlement_cannot_exceed_validator_emission_cap(tmp_path):
    path = tmp_path / 'compute.sqlite3'
    store = SQLiteStateStore(path)
    now = time.time()
    store.finalize_settlement(
        'window-1',
        now - 100,
        now,
        {'miner-hotkey': '1'},
        {'compute_emission_share': 0.90},
        {},
    )

    allocation = load_compute_allocation(['recycle', 'miner-hotkey'], database_path=str(path))

    assert allocation is not None
    assert allocation.emission_share == pytest.approx(0.50)


def test_invalid_remote_settlement_recycles_baseline_compute_slice(monkeypatch):
    monkeypatch.setenv('GITTENSOR_COMPUTE_SETTLEMENT_URL', 'http://insecure.example/settlement')

    allocation = load_compute_allocation(['recycle'])

    assert allocation is not None
    assert allocation.scores == {}
    assert allocation.emission_share == 0.0
    assert allocation.reserved_emission_share == COMPUTE_EMISSION_SHARE


def test_future_dated_remote_settlement_is_recycled(monkeypatch):
    import bittensor as bt

    keypair = bt.Keypair.create_from_uri('//Alice')
    now = time.time()
    metadata = {'compute_emission_share': 0.15}
    settlement = {
        'window_id': 'future-window',
        'started_at': now + 100,
        'ended_at': now + 200,
        'hotkey_rewards': {'miner-hotkey': '3.9'},
        'metadata': {
            **metadata,
            **SettlementSigner(keypair).sign(
                'future-window',
                now + 100,
                now + 200,
                {'miner-hotkey': '3.9'},
                metadata,
            ),
        },
    }
    monkeypatch.setenv('GITTENSOR_COMPUTE_SETTLEMENT_URL', 'https://control.example/settlement')
    monkeypatch.setenv('GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY', keypair.ss58_address)
    monkeypatch.setattr(
        'gittensor.validator.compute_rewards.no_redirect_urlopen',
        lambda *args, **kwargs: SettlementResponse(json.dumps(settlement).encode()),
    )

    allocation = load_compute_allocation(['recycle', 'miner-hotkey'])

    assert allocation is not None
    assert allocation.scores == {}
    assert allocation.emission_share == 0.0
    assert allocation.reserved_emission_share == pytest.approx(0.15)
