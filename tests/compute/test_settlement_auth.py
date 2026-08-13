import bittensor as bt

from gittensor.compute.settlement_auth import SettlementSigner, verify_settlement


def test_settlement_signature_binds_window_rewards_and_emission_share():
    keypair = bt.Keypair.create_from_uri('//Alice')
    settlement = {
        'window_id': '100:200',
        'started_at': 100.0,
        'ended_at': 200.0,
        'hotkey_rewards': {'miner': '2.6'},
        'metadata': {'window_budget': '2.6', 'compute_emission_share': 0.1},
    }
    settlement['metadata'].update(
        SettlementSigner(keypair).sign(
            settlement['window_id'],
            settlement['started_at'],
            settlement['ended_at'],
            settlement['hotkey_rewards'],
            settlement['metadata'],
        )
    )

    assert verify_settlement(settlement, keypair.ss58_address)
    settlement['hotkey_rewards']['miner'] = '9.9'
    assert not verify_settlement(settlement, keypair.ss58_address)


def test_wrong_hotkey_cannot_validate_settlement():
    alice = bt.Keypair.create_from_uri('//Alice')
    bob = bt.Keypair.create_from_uri('//Bob')
    metadata = {'compute_emission_share': 0.1}
    started_at = 100.0
    ended_at = 200.0
    settlement = {
        'window_id': 'window',
        'started_at': started_at,
        'ended_at': ended_at,
        'hotkey_rewards': {},
        'metadata': {
            **metadata,
            **SettlementSigner(alice).sign('window', started_at, ended_at, {}, metadata),
        },
    }

    assert not verify_settlement(settlement, bob.ss58_address)
