"""Realized SN74 miner-emission value oracle."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import bittensor as bt

from gittensor.compute.config import EmissionOracleConfig
from gittensor.compute.http_json import load_json_object
from gittensor.compute.safe_http import public_https_request

_PRICE_URLS = {
    'coinbase': 'https://api.coinbase.com/v2/prices/TAO-USD/spot',
    'coingecko': 'https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd',
}
_MAX_PRICE_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class EmissionObservation:
    value_per_hour: float
    currency: str
    epoch_block: int
    epoch_started_block: int
    epoch_seconds: float
    miner_alpha: float
    alpha_tao_price: float
    price_block: int
    tao_currency_price: float
    source: str
    observed_at: float


class SubnetEmissionOracle:
    """Price the latest realized on-chain miner epoch in the target unit."""

    def __init__(
        self,
        config: EmissionOracleConfig,
        *,
        netuid: int,
        network: str,
        target_currency: str,
        subtensor: Any | None = None,
        price_fetcher: Callable[[str, float], Mapping[str, Any]] | None = None,
        clock=time.time,
    ) -> None:
        self.config = config
        self.netuid = netuid
        self.target_currency = target_currency
        self.subtensor = subtensor or bt.Subtensor(network=network)
        self.price_fetcher = price_fetcher or _fetch_price_payload
        self.clock = clock

    def observe(self, now: float | None = None) -> EmissionObservation:
        timestamp = self.clock() if now is None else now
        finalized_hash = self.subtensor.substrate.get_chain_finalised_head()
        finalized_block = self.subtensor.substrate.get_block_number(finalized_hash)
        if not isinstance(finalized_block, int) or isinstance(finalized_block, bool) or finalized_block < 1:
            raise RuntimeError('finalized chain head is unavailable')
        current = self.subtensor.get_metagraph_info(self.netuid, mechid=0, block=finalized_block)
        if current is None:
            raise RuntimeError('subnet metagraph information is unavailable')
        if self.subtensor.get_mechanism_count(self.netuid, block=current.block) > 1:
            raise RuntimeError('multiple on-chain incentive mechanisms require a mechanism-specific oracle')
        epoch_block = int(current.last_step)
        if epoch_block < 1:
            raise RuntimeError('subnet has not completed an emission epoch')
        epoch_started_block = self._previous_epoch_block(epoch_block, int(current.tempo))
        epoch_time = self._block_timestamp(epoch_block)
        epoch_started_time = self._block_timestamp(epoch_started_block)
        epoch_seconds = epoch_time - epoch_started_time
        if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise RuntimeError('subnet epoch duration is invalid')
        age = timestamp - epoch_time
        if age < -120 or age > self.config.max_epoch_age_seconds:
            raise RuntimeError('latest subnet miner-emission event is stale or future-dated')
        miner_alpha = self._miner_alpha(epoch_block)
        alpha_tao_price = float(current.moving_price)
        if not math.isfinite(alpha_tao_price) or alpha_tao_price <= 0:
            raise RuntimeError('finalized alpha-to-TAO price is invalid')
        tao_per_hour = miner_alpha * alpha_tao_price * 3600.0 / epoch_seconds
        tao_currency_price, price_source = self._tao_currency_price(timestamp)
        value_per_hour = tao_per_hour * tao_currency_price
        if not math.isfinite(value_per_hour) or value_per_hour <= 0:
            raise RuntimeError('subnet miner-emission value is invalid')
        return EmissionObservation(
            value_per_hour=value_per_hour,
            currency=self.target_currency,
            epoch_block=epoch_block,
            epoch_started_block=epoch_started_block,
            epoch_seconds=epoch_seconds,
            miner_alpha=miner_alpha,
            alpha_tao_price=alpha_tao_price,
            price_block=finalized_block,
            tao_currency_price=tao_currency_price,
            source=f'chain:IncentiveAlphaEmittedToMiners+{price_source}',
            observed_at=timestamp,
        )

    def _block_timestamp(self, block: int) -> float:
        body = self.subtensor.substrate.get_block(block_number=block)
        extrinsics = body.get('extrinsics') if isinstance(body, Mapping) else None
        if not isinstance(extrinsics, list):
            raise RuntimeError('on-chain block body is invalid')
        matches: list[int] = []
        for extrinsic in extrinsics:
            value = getattr(extrinsic, 'value', extrinsic)
            call = value.get('call') if isinstance(value, Mapping) else None
            if not isinstance(call, Mapping):
                continue
            if call.get('call_module') != 'Timestamp' or call.get('call_function') != 'set':
                continue
            arguments = call.get('call_args')
            if not isinstance(arguments, list):
                continue
            for argument in arguments:
                if isinstance(argument, Mapping) and argument.get('name') == 'now':
                    raw = argument.get('value')
                    if isinstance(raw, int) and not isinstance(raw, bool):
                        matches.append(raw)
        if len(matches) != 1 or matches[0] <= 0:
            raise RuntimeError('on-chain block timestamp is invalid')
        return matches[0] / 1000.0

    def _previous_epoch_block(self, epoch_block: int, tempo: int) -> int:
        if tempo < 1:
            raise RuntimeError('subnet tempo is invalid')
        expected = epoch_block - tempo
        if expected < 1:
            raise RuntimeError('previous subnet emission epoch is unavailable')
        exact_matches = self._miner_emissions(expected)
        if len(exact_matches) == 1:
            return expected
        if len(exact_matches) > 1:
            raise RuntimeError('previous subnet miner-emission epoch is ambiguous')
        # Epoch execution can move a few blocks around a runtime upgrade. Search a
        # bounded interval, preferring the normal exact-tempo boundary.
        offsets = [value for distance in range(1, 17) for value in (-distance, distance)]
        matches = [
            candidate for offset in offsets if (candidate := expected + offset) > 0 and self._has_miner_event(candidate)
        ]
        if len(matches) != 1:
            raise RuntimeError('previous subnet miner-emission epoch is unavailable or ambiguous')
        return matches[0]

    def _has_miner_event(self, block: int) -> bool:
        return len(self._miner_emissions(block)) == 1

    def _miner_alpha(self, block: int) -> float:
        matches = self._miner_emissions(block)
        if len(matches) != 1:
            raise RuntimeError('expected exactly one subnet miner-emission event')
        emissions = matches[0]
        if not emissions or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in emissions
        ):
            raise RuntimeError('miner-emission values are invalid')
        total_rao = sum(emissions)
        if total_rao <= 0:
            raise RuntimeError('subnet miner emissions are zero')
        return total_rao / 1_000_000_000.0

    def _miner_emissions(self, block: int) -> list[list[int]]:
        events = self.subtensor.substrate.get_events(self.subtensor.get_block_hash(block))
        matches: list[list[int]] = []
        for record in events:
            value = getattr(record, 'value', record)
            event = value.get('event', {}) if isinstance(value, Mapping) else {}
            attributes = event.get('attributes', {}) if isinstance(event, Mapping) else {}
            if (
                event.get('module_id') == 'SubtensorModule'
                and event.get('event_id') == 'IncentiveAlphaEmittedToMiners'
                and isinstance(attributes, Mapping)
                and attributes.get('netuid') == self.netuid
            ):
                emissions = attributes.get('emissions')
                if not isinstance(emissions, list):
                    raise RuntimeError('miner-emission event payload is invalid')
                matches.append(emissions)
        return matches

    def _tao_currency_price(self, now: float) -> tuple[float, str]:
        if self.target_currency == 'TAO':
            return 1.0, 'TAO'
        prices: list[tuple[str, float]] = []
        for source in self.config.tao_usd_price_sources:
            try:
                payload = self.price_fetcher(source, self.config.request_timeout_seconds)
                price = _parse_tao_usd_price(source, payload, now)
            except Exception:
                continue
            prices.append((source, price))
        if len(prices) < self.config.minimum_price_sources:
            raise RuntimeError('not enough independent TAO/USD price sources are available')
        values = [price for _, price in prices]
        median = statistics.median(values)
        if median <= 0 or (max(values) - min(values)) / median > self.config.maximum_price_divergence:
            raise RuntimeError('TAO/USD price sources diverge beyond the configured limit')
        return median, '+'.join(source for source, _ in prices)


def _fetch_price_payload(source: str, timeout: float) -> Mapping[str, Any]:
    response = public_https_request(
        _PRICE_URLS[source],
        method='GET',
        body=None,
        headers={'Accept': 'application/json', 'User-Agent': 'gittensor-compute/1'},
        timeout=timeout,
    )
    try:
        if response.status != 200:
            raise ValueError(f'price endpoint returned HTTP {response.status}')
        body = response.read(_MAX_PRICE_RESPONSE_BYTES + 1)
        if len(body) > _MAX_PRICE_RESPONSE_BYTES:
            raise ValueError('price response exceeds 64 KiB')
        payload = load_json_object(body)
    finally:
        response.close()
    return payload


def _parse_tao_usd_price(source: str, payload: Mapping[str, Any], now: float) -> float:
    if source == 'coinbase':
        data = payload.get('data')
        raw = data.get('amount') if isinstance(data, Mapping) else None
    elif source == 'coingecko':
        data = payload.get('bittensor')
        raw = data.get('usd') if isinstance(data, Mapping) else None
        updated = data.get('last_updated_at') if isinstance(data, Mapping) else None
        if updated is not None and (
            not isinstance(updated, (int, float))
            or isinstance(updated, bool)
            or not math.isfinite(float(updated))
            or abs(now - float(updated)) > 300
        ):
            raise ValueError('CoinGecko price timestamp is stale')
    else:
        raise ValueError('unknown TAO/USD price source')
    if raw is None:
        raise ValueError('TAO/USD price is missing')
    price = float(raw)
    if not math.isfinite(price) or price <= 0:
        raise ValueError('TAO/USD price is invalid')
    return price
