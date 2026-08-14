import pytest

from gittensor.compute.routing import CapacityUnavailable, FastestFinishRouter, RoutingGPU


def _gpu(index, rtt_ms=10):
    return RoutingGPU(
        gpu_id=f'gpu-{index}',
        endpoint=f'https://gpu-{index}',
        release_digest='release:1',
        performance_class='rtx-5090',
        certified_slots=4,
        observed_active_slots=0,
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


def test_expected_completion_beats_idle_status():
    router = FastestFinishRouter(reservation_ttl_seconds=60)
    local = RoutingGPU(
        gpu_id='local',
        endpoint='https://local',
        release_digest='release:1',
        performance_class='rtx-5090',
        certified_slots=4,
        observed_active_slots=1,
        remaining_work_seconds=0.1,
        service_seconds=0.1,
        rtt_ms=10,
    )
    remote = RoutingGPU(
        gpu_id='remote',
        endpoint='https://remote',
        release_digest='release:1',
        performance_class='rtx-5090',
        certified_slots=4,
        observed_active_slots=0,
        remaining_work_seconds=0,
        service_seconds=0.1,
        rtt_ms=5_000,
    )

    decision = router.route([local, remote], 'release:1', now=0)

    assert decision.gpu_id == 'local'
    assert decision.expected_completion_seconds == pytest.approx(0.21)


def test_free_concurrency_slot_does_not_add_serial_wait_time_and_long_requests_keep_their_slot():
    router = FastestFinishRouter(reservation_ttl_seconds=10)
    gpu = _gpu(0)
    first = router.route([gpu], 'release:1', now=0)
    second = router.route([gpu], 'release:1', now=0.5)

    assert first.expires_at == 31
    assert second.expected_completion_seconds == pytest.approx(1.01)


def test_live_reservation_can_be_renewed_but_expired_one_cannot():
    router = FastestFinishRouter(reservation_ttl_seconds=5)
    gpu = _gpu(0)
    reservation = router.route([gpu], 'release:1', now=0)

    assert reservation.expires_at == 31
    assert router.renew(reservation.reservation_id, now=4) == 31
    assert router.renew(reservation.reservation_id, now=30) == 35
    assert router.active_counts(now=34) == {'gpu-0': 1}
    assert router.renew(reservation.reservation_id, now=35) is None
    assert router.active_counts(now=35) == {}


def test_router_admits_by_both_release_concurrency_and_reserved_kv_cache():
    router = FastestFinishRouter(reservation_ttl_seconds=60)
    gpu = RoutingGPU(
        gpu_id='gpu-0',
        endpoint='https://gpu-0',
        release_digest='release:1',
        performance_class='rtx-5090',
        certified_slots=4,
        observed_active_slots=0,
        remaining_work_seconds=0,
        service_seconds=1,
        rtt_ms=10,
        kv_cache_capacity_bytes=1_000,
    )

    first = router.route(
        [gpu],
        'release:1',
        now=0,
        request_kv_bytes=600,
        request_capacity_units=0.6,
    )

    assert first.reserved_kv_bytes == 600
    assert first.capacity_units == pytest.approx(0.6)
    with pytest.raises(CapacityUnavailable):
        router.route(
            [gpu],
            'release:1',
            now=1,
            request_kv_bytes=600,
            request_capacity_units=0.6,
        )
