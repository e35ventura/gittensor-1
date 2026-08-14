from gittensor.compute.autoscaler import FleetAutoscaler
from gittensor.compute.config import AutoscalingConfig


def test_target_scales_up_after_sustained_utilization_and_down_one_per_cooldown():
    scaler = FleetAutoscaler(
        AutoscalingConfig(
            ewma_alpha=1.0,
            utilization_up=0.75,
            utilization_down=0.25,
            sustain_up_seconds=10,
            sustain_down_seconds=20,
            cooldown_seconds=20,
        ),
        floor=4,
        initial_target=4,
    )

    first = scaler.update(active_gpu_equivalents=3, rejected_gpu_equivalents=0, funded_target=4, now=0)
    assert first.desired_target == 4
    assert first.utilization == 0.75
    assert first.required_target == 5
    assert first.supply_shortage == 0

    scaled = scaler.update(active_gpu_equivalents=3, rejected_gpu_equivalents=0, funded_target=4, now=10)
    assert scaled.desired_target == 5
    assert scaled.changed

    scaler.update(active_gpu_equivalents=0, rejected_gpu_equivalents=0, funded_target=5, now=40)
    down = scaler.update(active_gpu_equivalents=0, rejected_gpu_equivalents=0, funded_target=5, now=60)
    assert down.desired_target == 4
    assert down.changed


def test_rejected_requests_add_concurrent_demand_and_target_has_no_ceiling():
    scaler = FleetAutoscaler(
        AutoscalingConfig(
            ewma_alpha=1.0,
            utilization_up=0.75,
            utilization_down=0.25,
            sustain_up_seconds=1,
            sustain_down_seconds=30,
            cooldown_seconds=30,
        ),
        floor=4,
        initial_target=4,
    )
    scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=26, funded_target=4, now=0)
    decision = scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=26, funded_target=4, now=1)
    assert decision.concurrent_demand == 30
    assert decision.desired_target == 41
    assert decision.supply_shortage == 36


def test_funding_shortfall_does_not_make_target_grow_without_new_demand():
    scaler = FleetAutoscaler(
        AutoscalingConfig(
            ewma_alpha=1.0,
            utilization_up=0.75,
            utilization_down=0.25,
            sustain_up_seconds=1,
            sustain_down_seconds=30,
            cooldown_seconds=30,
        ),
        floor=4,
        initial_target=4,
    )

    scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=0, funded_target=4, now=0)
    first = scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=0, funded_target=4, now=1)
    assert first.desired_target == 6

    scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=0, funded_target=4, now=2)
    stable = scaler.update(active_gpu_equivalents=4, rejected_gpu_equivalents=0, funded_target=4, now=3)

    assert stable.desired_target == 6
    assert not stable.changed
    assert stable.reason == 'desired target already covers measured demand; funding is constrained'
