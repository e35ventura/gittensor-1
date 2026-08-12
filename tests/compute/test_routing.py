import pytest

from gittensor.compute.routing import CapacityUnavailable, FastestFinishRouter, RoutingGPU


def _gpu(index, rtt_ms=10):
    return RoutingGPU(
        gpu_id=f'gpu-{index}',
        endpoint=f'https://gpu-{index}',
        release_digest='release:1',
        performance_class='rtx-5090',
        certified_slots=4,
        reported_active_slots=0,
        remaining_work_seconds=0,
        service_seconds=1,
        rtt_ms=rtt_ms,
    )


def test_router_spreads_across_idle_equivalent_gpus_before_stacking():
    router = FastestFinishRouter(reservation_ttl_seconds=60)
    gpus = [_gpu(index) for index in range(4)]
    first_wave = [router.route(gpus, 'release:1', now=index).gpu_id for index in range(4)]
    assert set(first_wave) == {f'gpu-{index}' for index in range(4)}

    fifth = router.route(gpus, 'release:1', now=5)
    assert fifth.gpu_id == 'gpu-0'


def test_router_returns_capacity_error_instead_of_queueing():
    router = FastestFinishRouter(reservation_ttl_seconds=60)
    gpu = _gpu(0)
    for index in range(4):
        router.route([gpu], 'release:1', now=index)
    with pytest.raises(CapacityUnavailable):
        router.route([gpu], 'release:1', now=5)
