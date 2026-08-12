from decimal import Decimal

from gittensor.compute.settlement import funding_plan, settle_ready_seconds


def test_lium_style_target_price_dilutes_above_target_and_pays_more_below():
    funding = funding_plan(4, 0.65, 2.60)
    assert funding.funded_target == 4
    assert funding.funding_shortfall == 0

    two = settle_ready_seconds(funding, 3600, {'a': 3600, 'b': 3600})
    four = settle_ready_seconds(funding, 3600, {str(i): 3600 for i in range(4)})
    eight = settle_ready_seconds(funding, 3600, {str(i): 3600 for i in range(8)})

    assert two.gpu_rewards['a'] == Decimal('1.30')
    assert four.gpu_rewards['0'] == Decimal('0.65')
    assert eight.gpu_rewards['0'] == Decimal('0.325')
    assert sum(eight.gpu_rewards.values()) == Decimal('2.600')


def test_funding_guard_reports_shortfall_and_zero_supply_does_not_carry_budget():
    funding = funding_plan(10, 0.65, 2.60)
    assert funding.funded_target == 4
    assert funding.funding_shortfall == 6

    result = settle_ready_seconds(funding, 900, {})
    assert result.gpu_rewards == {}
    assert result.unspent_budget == result.window_budget == Decimal('0.650')
