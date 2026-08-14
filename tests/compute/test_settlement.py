from decimal import Decimal

import pytest

from gittensor.compute.settlement import funding_plan, settle_ready_seconds


def test_sublinear_scarcity_premium_preserves_marginal_supply_incentive_and_dilutes_above_target():
    funding = funding_plan(4, 0.65, 2.60)
    assert funding.funded_target == 4
    assert funding.funding_shortfall == 0

    two = settle_ready_seconds(funding, 3600, {'a': 3600, 'b': 3600})
    four = settle_ready_seconds(funding, 3600, {str(i): 3600 for i in range(4)})
    eight = settle_ready_seconds(funding, 3600, {str(i): 3600 for i in range(8)})

    one = settle_ready_seconds(funding, 3600, {'only': 3600})

    assert float(one.gpu_rewards['only']) == pytest.approx(1.30)
    assert float(two.gpu_rewards['a']) == pytest.approx(0.65 * 2**0.5)
    assert four.gpu_rewards['0'] == Decimal('0.65')
    assert eight.gpu_rewards['0'] == Decimal('0.325')
    assert sum(one.gpu_rewards.values()) < sum(two.gpu_rewards.values()) < sum(four.gpu_rewards.values())
    assert two.distributed_budget < two.window_budget
    assert sum(eight.gpu_rewards.values()) == Decimal('2.600')


def test_funding_guard_reports_shortfall_and_zero_supply_does_not_carry_budget():
    funding = funding_plan(10, 0.65, 2.60)
    assert funding.funded_target == 4
    assert funding.funding_shortfall == 6

    result = settle_ready_seconds(funding, 900, {})
    assert result.gpu_rewards == {}
    assert result.unspent_budget == result.window_budget == Decimal('0.650')
