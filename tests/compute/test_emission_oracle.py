from types import SimpleNamespace

import pytest

from gittensor.compute.config import EmissionOracleConfig
from gittensor.compute.emission_oracle import SubnetEmissionOracle, _parse_tao_usd_price


class Value:
    def __init__(self, value):
        self.value = value


class Substrate:
    def __init__(self, events):
        self.events = events

    def get_events(self, block_hash):
        if block_hash == 'block-640':
            return [_event()]
        if block_hash == 'block-1000':
            return self.events
        return []

    def get_block(self, *, block_number):
        timestamps = {640: 1_000_000, 1000: 5_320_000}
        return {
            'extrinsics': [
                {
                    'call': {
                        'call_module': 'Timestamp',
                        'call_function': 'set',
                        'call_args': [{'name': 'now', 'value': timestamps[block_number]}],
                    }
                }
            ]
        }

    def get_chain_finalised_head(self):
        return 'block-1100'

    def get_block_number(self, block_hash):
        assert block_hash == 'block-1100'
        return 1100


class Subtensor:
    def __init__(self, events):
        self.substrate = Substrate(events)

    def get_metagraph_info(self, netuid, mechid=0, block=None):
        assert netuid == 74
        assert mechid == 0
        assert block == 1100
        return SimpleNamespace(last_step=1000, block=1100, moving_price=0.004, tempo=360)

    def get_mechanism_count(self, netuid, block=None):
        return 1

    def get_block_hash(self, block):
        return f'block-{block}'


def _event(emissions=None):
    return {
        'event': {
            'module_id': 'SubtensorModule',
            'event_id': 'IncentiveAlphaEmittedToMiners',
            'attributes': {'netuid': 74, 'emissions': emissions or [100_000_000_000, 47_600_000_000]},
        }
    }


def _oracle(events, prices, *, currency='USD'):
    return SubnetEmissionOracle(
        EmissionOracleConfig(),
        netuid=74,
        network='finney',
        target_currency=currency,
        subtensor=Subtensor(events),
        price_fetcher=lambda source, timeout: prices[source],
        clock=lambda: 5_320,
    )


def test_oracle_prices_exact_miner_event_with_epoch_price_and_dual_usd_sources():
    oracle = _oracle(
        [_event()],
        {
            'coinbase': {'data': {'amount': '200'}},
            'coingecko': {'bittensor': {'usd': 202, 'last_updated_at': 5_320}},
        },
    )

    observation = oracle.observe(now=5_320)

    assert observation.epoch_block == 1000
    assert observation.epoch_seconds == 4_320
    assert observation.miner_alpha == pytest.approx(147.6)
    assert observation.alpha_tao_price == 0.004
    assert observation.price_block == 1100
    assert observation.tao_currency_price == 201
    assert observation.value_per_hour == pytest.approx(98.892)


def test_oracle_rejects_ambiguous_or_bad_miner_emission_events():
    with pytest.raises(RuntimeError, match='exactly one'):
        _oracle([], {}).observe(now=5_320)
    with pytest.raises(RuntimeError, match='values are invalid'):
        _oracle([_event([-1])], {}).observe(now=5_320)


def test_oracle_rejects_divergent_or_missing_usd_sources():
    prices = {
        'coinbase': {'data': {'amount': '200'}},
        'coingecko': {'bittensor': {'usd': 300, 'last_updated_at': 5_320}},
    }
    with pytest.raises(RuntimeError, match='diverge'):
        _oracle([_event()], prices).observe(now=5_320)

    def one_source(source, timeout):
        if source == 'coingecko':
            raise OSError('offline')
        return prices[source]

    oracle = _oracle([_event()], prices)
    oracle.price_fetcher = one_source
    with pytest.raises(RuntimeError, match='not enough'):
        oracle.observe(now=5_320)


def test_tao_target_needs_no_external_price_provider():
    observation = _oracle([_event()], {}, currency='TAO').observe(now=5_320)

    assert observation.tao_currency_price == 1
    assert observation.value_per_hour == pytest.approx(0.492)


def test_coingecko_timestamp_must_be_fresh():
    with pytest.raises(ValueError, match='stale'):
        _parse_tao_usd_price('coingecko', {'bittensor': {'usd': 200, 'last_updated_at': 1}}, 1_000)
