import pytest

from gittensor.compute.config import ComputeConfig
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.routing import CapacityUnavailable


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def _config():
    return ComputeConfig.from_mapping(
        {
            'fleet': {
                'floor': 4,
                'initial_target': 4,
                'certified_slots_per_gpu': 4,
                'target_price_per_gpu_hour': 0.65,
                'max_budget_per_hour': 2.6,
            },
            'autoscaling': {
                'ewma_alpha': 1.0,
                'utilization_up': 0.75,
                'utilization_down': 0.25,
                'sustain_up_seconds': 120,
                'sustain_down_seconds': 900,
                'cooldown_seconds': 900,
            },
            'verification': {
                'status_url': 'http://sparkcompute/api/status',
                'timeout_seconds': 1,
                'lease_ttl_seconds': 500,
                'expected_hardware': 'RTX 5090',
                'require_model_canary': True,
                'bearer_token_env': None,
                'source_repository': 'https://github.com/gittensor-ai-lab/sparkcompute',
                'source_commit': 'abc',
            },
            'router': {'reservation_ttl_seconds': 500, 'default_rtt_ms': 50},
            'placement': {'minimum_residency_seconds': 0, 'control_interval_seconds': 60},
        }
    )


def _snapshot(index, last_checked=100, hardware_uuid=None):
    return {
        'id': f'node-{index}',
        'verdict': 'VERIFIED',
        'last_checked': last_checked,
        'report': {
            'gpu': {
                'name': 'NVIDIA GeForce RTX 5090',
                'uuid': hardware_uuid or f'GPU-unique-{index}',
                'driver_version': '580.1',
            }
        },
        'liveness': {
            'online': True,
            'online_age_sec': 1,
            'gpu_live': True,
            'model_canary_enabled': True,
            'model_verified': True,
            'model_age_sec': 1,
        },
    }


def test_end_to_end_verification_routing_scaling_funding_and_settlement():
    clock = Clock(100)
    control = ComputeControlPlane(_config(), clock=clock)
    control.register_release(Release('release:1', 'model', 'runtime', minimum_replicas=1))
    for index in range(4):
        control.register_gpu(
            GPURegistration(
                gpu_id=f'gpu-{index}',
                spark_node_id=f'node-{index}',
                miner_uid=index,
                endpoint=f'https://gpu-{index}',
                region='us-east',
                release_digest='release:1',
                canary_release_digest='release:1',
                certified_slots=4,
            )
        )

    assert set(control.refresh_verification([_snapshot(index) for index in range(4)]).values()) == {'READY'}

    reservations = [
        control.route('release:1', 'us-east', expected_service_seconds=10, now=100 + index) for index in range(16)
    ]
    assert {reservation.gpu_id for reservation in reservations[:4]} == {f'gpu-{index}' for index in range(4)}
    with pytest.raises(CapacityUnavailable):
        control.route('release:1', 'us-east', expected_service_seconds=10, now=117)

    first_tick = control.tick(now=117)
    assert first_tick.autoscaling.desired_target == 4
    second_tick = control.tick(now=237)
    assert second_tick.autoscaling.desired_target == 6
    assert second_tick.funding.funded_target == 4
    assert second_tick.funding.funding_shortfall == 2

    funding = control.update_budget(3.9, now=237)
    assert funding.funded_target == 6
    assert funding.funding_shortfall == 0

    settlement, miner_rewards = control.settle(now=337)
    assert settlement.effective_ready_gpus == 4
    assert set(miner_rewards) == {0, 1, 2, 3}
    assert sum(miner_rewards.values()) == settlement.window_budget


def test_settlement_prorates_budget_when_funded_target_changes_mid_window():
    clock = Clock(100)
    control = ComputeControlPlane(_config(), clock=clock)
    control.autoscaler.desired_target = 6
    control.update_budget(3.9, now=200)

    result, miner_rewards = control.settle(now=300)

    expected_target_seconds = 4 * 100 + 6 * 100
    assert result.window_budget == result.funding.target_price_per_gpu_hour * expected_target_seconds / 3600
    assert result.unspent_budget == result.window_budget
    assert miner_rewards == {}


def test_dynamic_budget_cannot_defund_the_baseline_fleet():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    with pytest.raises(ValueError, match='configured GPU floor'):
        control.update_budget(2.59, now=100)


def test_duplicate_sparkcompute_gpu_uuid_quarantines_every_claim():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
    for index in range(2):
        control.register_gpu(
            GPURegistration(
                gpu_id=f'gpu-{index}',
                spark_node_id=f'node-{index}',
                miner_uid=index,
                endpoint=f'https://gpu-{index}',
                region='us-east',
                release_digest='release:1',
                canary_release_digest='release:1',
                certified_slots=4,
            )
        )

    outcomes = control.refresh_verification([_snapshot(index, hardware_uuid='GPU-duplicate') for index in range(2)])
    assert set(outcomes.values()) == {'duplicate SparkCompute GPU UUID is registered more than once'}
    assert control.status(now=100)['ready_gpus'] == 0


def test_assignment_change_requires_new_canary_binding_and_verification():
    clock = Clock(100)
    control = ComputeControlPlane(_config(), clock=clock)
    control.register_release(Release('release:1', 'model-1', 'runtime-1'))
    control.register_release(Release('release:2', 'model-2', 'runtime-2'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=1,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)])

    control.begin_assignment('gpu-1', 'release:2', now=110)
    rejected = control.refresh_verification([_snapshot(1, last_checked=110)], now=110)
    assert 'canary binding' in rejected['gpu-1']

    control.bind_runtime_canary('gpu-1', 'release:2')
    accepted = control.refresh_verification([_snapshot(1, last_checked=111)], now=111)
    assert accepted == {'gpu-1': 'READY'}
