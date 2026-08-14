from dataclasses import replace
from decimal import Decimal
from typing import cast

import pytest

from gittensor.compute.config import ComputeConfig
from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.emission_oracle import EmissionObservation
from gittensor.compute.models import (
    GPURegistration,
    GPUState,
    Release,
    RoutingObservation,
    RuntimeEvidence,
)
from gittensor.compute.routing import CapacityUnavailable


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class AcceptingAssignmentExecutor:
    def dispatch(self, record, command):
        return None


class RevocationRecordingExecutor(AcceptingAssignmentExecutor):
    def __init__(self):
        self.revocations = []

    def revoke(self, record, epoch, reason):
        self.revocations.append((record.registration.gpu_id, epoch, reason))


def _config():
    return ComputeConfig.from_mapping(
        {
            'fleet': {
                'floor': 4,
                'initial_target': 4,
                'certified_slots_per_gpu': 4,
                'target_price_per_gpu_hour': 0.65,
                'subnet_miner_emission_value_per_hour': 26.0,
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
                'status_url': 'http://127.0.0.1:9090/api/status',
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


def _emission_observation(value, observed_at):
    return EmissionObservation(
        value_per_hour=value,
        currency='USD',
        epoch_block=1_000,
        epoch_started_block=640,
        epoch_seconds=4_320,
        miner_alpha=147.6,
        alpha_tao_price=0.004,
        price_block=1_100,
        tao_currency_price=200,
        source='chain+coinbase+coingecko',
        observed_at=observed_at,
    )


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
    assert sum(miner_rewards.values()) == settlement.distributed_budget
    assert settlement.distributed_budget < settlement.window_budget


def test_completed_requests_still_count_toward_window_utilization():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(index) for index in range(4)], now=100)

    for index in range(72):
        started = 100 + index / 2
        reservation = control.route('release:1', 'us-east', expected_service_seconds=10, now=started)
        assert control.complete_reservation(reservation.reservation_id, now=started + 10)

    first = control.tick(now=160)
    for index in range(144):
        started = 160 + index / 2
        reservation = control.route('release:1', 'us-east', expected_service_seconds=10, now=started)
        assert control.complete_reservation(reservation.reservation_id, now=started + 10)
    second = control.tick(now=280)

    assert first.autoscaling.concurrent_demand == pytest.approx(3)
    assert second.autoscaling.concurrent_demand == pytest.approx(3)
    assert second.autoscaling.desired_target == 5


def test_expired_reservations_still_count_toward_window_utilization():
    config = _config()
    object.__setattr__(config.router, 'reservation_ttl_seconds', 10)
    object.__setattr__(config.router, 'maximum_service_seconds', 20)
    control = ComputeControlPlane(config, clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    control.route('release:1', 'us-east', expected_service_seconds=1, now=100)

    tick = control.tick(now=131)

    assert tick.autoscaling.concurrent_demand == pytest.approx(0.25)


def test_release_capacity_controls_concurrency_instead_of_the_gpu_registration_ceiling():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    release = Release('release:serial', 'serial-model', 'runtime', max_concurrency=1)
    control.register_release(release)
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=1,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest=release.release_digest,
            canary_release_digest=release.release_digest,
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)

    control.route(release.release_digest, 'us-east', 10, now=100)

    with pytest.raises(CapacityUnavailable):
        control.route(release.release_digest, 'us-east', 10, now=101)


def test_release_kv_budget_can_fill_before_its_concurrency_limit():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    release = Release(
        'release:kv',
        'kv-model',
        'runtime',
        max_concurrency=4,
        max_context_tokens=100,
        kv_bytes_per_token=10,
        kv_cache_capacity_bytes=1_000,
    )
    control.register_release(release)
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=1,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest=release.release_digest,
            canary_release_digest=release.release_digest,
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)

    route = control.route(
        release.release_digest,
        'us-east',
        10,
        estimated_input_tokens=30,
        max_output_tokens=30,
        now=100,
    )

    assert route.reserved_kv_bytes == 600
    with pytest.raises(CapacityUnavailable):
        control.route(
            release.release_digest,
            'us-east',
            10,
            estimated_input_tokens=30,
            max_output_tokens=30,
            now=101,
        )


def test_mixed_releases_contribute_comparable_gpu_equivalent_demand():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    releases = [
        Release('release:four', 'model-four', 'runtime', max_concurrency=4),
        Release('release:two', 'model-two', 'runtime', max_concurrency=2),
    ]
    for index, release in enumerate(releases):
        control.register_release(release)
        control.register_gpu(
            GPURegistration(
                gpu_id=f'gpu-{index}',
                spark_node_id=f'node-{index}',
                miner_uid=index,
                endpoint=f'https://gpu-{index}',
                region='us-east',
                release_digest=release.release_digest,
                canary_release_digest=release.release_digest,
                certified_slots=4,
            )
        )
    control.refresh_verification([_snapshot(0), _snapshot(1)], now=100)

    control.route(releases[0].release_digest, 'us-east', 10, now=100)
    control.route(releases[1].release_digest, 'us-east', 10, now=100)
    tick = control.tick(now=110)

    assert tick.autoscaling.concurrent_demand == pytest.approx(0.75)


def test_supply_shortage_is_distinct_from_wrong_release_placement():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    release_a = Release('release:a', 'model-a', 'runtime')
    release_b = Release('release:b', 'model-b', 'runtime')
    control.register_release(release_a)
    control.register_release(release_b)
    for index in range(4):
        control.register_gpu(
            GPURegistration(
                gpu_id=f'gpu-{index}',
                spark_node_id=f'node-{index}',
                miner_uid=index,
                endpoint=f'https://gpu-{index}',
                region='us-east',
                release_digest=release_a.release_digest,
                canary_release_digest=release_a.release_digest,
                certified_slots=4,
            )
        )
    control.refresh_verification([_snapshot(index) for index in range(4)], now=100)

    with pytest.raises(CapacityUnavailable):
        control.route(release_b.release_digest, 'us-east', 10, now=100)
    tick = control.tick(now=110)

    assert tick.autoscaling.supply_shortage == 0
    assert tick.autoscaling.required_target == 4
    assert tick.placement.shortages[release_b.release_digest] > 0
    assert release_b.release_digest in tick.placement.replica_counts


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


def test_uncapped_target_funding_scales_automatically_with_utilization():
    config = _config()
    object.__setattr__(config.fleet, 'max_budget_per_hour', None)
    control = ComputeControlPlane(config, clock=Clock(100))
    control.autoscaler.desired_target = 6

    plan = control._funding_plan()

    assert plan.funded_target == 6
    assert plan.max_budget_per_hour == plan.target_price_per_gpu_hour * 6


def test_subnet_emission_ceiling_surfaces_a_funding_shortfall():
    config = _config()
    object.__setattr__(config.fleet, 'max_budget_per_hour', None)
    object.__setattr__(config.fleet, 'max_compute_emission_share', 0.10)
    control = ComputeControlPlane(config, clock=Clock(100))
    control.autoscaler.desired_target = 8

    plan = control._funding_plan()

    assert plan.funded_target == 4
    assert plan.funding_shortfall == 4


def test_fractional_emission_budget_is_not_discarded_at_gpu_boundary():
    config = _config()
    object.__setattr__(config.fleet, 'max_budget_per_hour', None)
    object.__setattr__(config.fleet, 'subnet_miner_emission_value_per_hour', 25.0)
    object.__setattr__(config.fleet, 'max_compute_emission_share', 0.10)
    control = ComputeControlPlane(config, clock=Clock(100))
    control.autoscaler.desired_target = 8

    plan = control._funding_plan()
    control.funding = plan
    result, _ = control.settle(now=3_700)

    assert plan.funded_target == 3
    assert plan.max_budget_per_hour == Decimal('2.50')
    assert control._current_compute_emission_share() == pytest.approx(0.10)
    assert result.window_budget == Decimal('2.50')


def test_target_price_converts_through_live_subnet_emission_value():
    config = _config()
    object.__setattr__(config.fleet, 'max_budget_per_hour', None)
    control = ComputeControlPlane(config, clock=Clock(100))

    initial = control._current_compute_emission_share()
    control.update_budget(None, now=100, subnet_miner_emission_value_per_hour=13.0)

    assert initial == pytest.approx(0.10)
    assert control._current_compute_emission_share() == pytest.approx(0.20)


def test_low_subnet_emission_value_surfaces_floor_funding_shortfall():
    control = ComputeControlPlane(_config(), clock=Clock(100))

    plan = control.update_budget(None, now=100, subnet_miner_emission_value_per_hour=5.0)

    assert plan.funded_target == 3
    assert plan.funding_shortfall == 1


def test_ready_requires_spark_attestation_to_match_runtime_reported_stream_key():
    config = _config()
    object.__setattr__(config.verification, 'require_runtime_attestation', True)
    object.__setattr__(config.verification, 'require_stream_proof', True)
    control = ComputeControlPlane(config, clock=Clock(100), assignment_executor=AcceptingAssignmentExecutor())
    release = Release(
        'release:1',
        'model',
        'runtime',
        model_repository='owner/model',
        model_revision='a' * 40,
        tokenizer_repository='owner/model',
        tokenizer_revision='a' * 40,
        runtime_commit='b' * 40,
        container_image='registry/runtime',
        container_digest='sha256:container',
        filesystem_digest='sha256:filesystem',
    )
    control.register_release(release)
    control.register_miner_gpu(
        miner_uid=1,
        miner_hotkey='hotkey-1',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)
    control.tick(now=101, execute=True)
    control.acknowledge_assignment('gpu-1', 1, GPUState.LOADING, now=102)
    control.acknowledge_assignment(
        'gpu-1',
        1,
        GPUState.RUNTIME_VERIFY,
        RuntimeEvidence(
            release_digest=release.release_digest,
            model_repository=release.model_repository,
            model_revision=release.model_revision,
            tokenizer_repository=release.tokenizer_repository,
            tokenizer_revision=release.tokenizer_revision,
            runtime_digest=release.runtime_digest,
            runtime_commit=release.runtime_commit,
            container_image=release.container_image,
            container_digest=release.container_digest,
            filesystem_digest=release.filesystem_digest,
            stream_public_key='11' * 32,
        ),
        now=103,
    )
    snapshot = _snapshot(1, last_checked=104)
    snapshot['report']['runtime'] = {
        'release_digest': release.release_digest,
        'model_repository': release.model_repository,
        'model_revision': release.model_revision,
        'tokenizer_repository': release.tokenizer_repository,
        'tokenizer_revision': release.tokenizer_revision,
        'runtime_digest': release.runtime_digest,
        'runtime_commit': release.runtime_commit,
        'container_image': release.container_image,
        'container_digest': release.container_digest,
        'filesystem_digest': release.filesystem_digest,
    }
    snapshot['attestation'] = {
        'verified': True,
        'evidence_digest': 'sha256:evidence',
        'stream_public_key': '22' * 32,
    }

    rejected = control.refresh_verification([snapshot], now=104)

    assert 'stream key' in rejected['gpu-1']
    assert control.gpus['gpu-1'].state == GPUState.QUARANTINED


def test_verifier_rejection_revokes_assignment_and_every_serving_capability():
    executor = RevocationRecordingExecutor()
    control = ComputeControlPlane(_config(), clock=Clock(100), assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    record = control.gpus['gpu-1']
    record.assignment_epoch = 3
    record.assignment_dispatched = True
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=101)
    old_assignment_token = record.assignment_token

    outcome = control.refresh_verification([], now=102)

    assert 'absent' in outcome['gpu-1']
    assert record.state == GPUState.QUARANTINED
    assert record.registration.release_digest == ''
    assert record.assignment_token != old_assignment_token
    assert control.renew_reservation(reservation.reservation_id, now=102) is None
    assert executor.revocations == [('gpu-1', 3, 'SparkCompute node is absent from /api/status')]


def test_route_capability_and_reservation_cannot_outlive_verification_lease():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)

    route = control.route('release:1', 'us-east', 900, now=101)

    assert route.expires_at == 599
    assert control.renew_reservation(route.reservation_id, now=590) == 599
    assert control.renew_reservation(route.reservation_id, now=599) is None


def test_runtime_key_rotation_revokes_assignment_and_requires_gepetto_readmission():
    config = _config()
    object.__setattr__(config.verification, 'require_runtime_attestation', True)
    object.__setattr__(config.verification, 'require_stream_proof', True)
    executor = RevocationRecordingExecutor()
    control = ComputeControlPlane(config, clock=Clock(100), assignment_executor=executor)
    release = Release('release:1', 'model', 'runtime')
    control.register_release(release)
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
    record = control.gpus['gpu-1']
    record.assignment_epoch = 1
    record.assignment_dispatched = True
    record.state = GPUState.RUNTIME_VERIFY
    record.runtime_stream_public_key = '11' * 32
    record.state = GPUState.READY
    record.lease = control.verifier.evaluate_hardware(record.registration, _snapshot(1), 100).lease
    old_assignment_token = record.assignment_token
    reservation = control.route('release:1', 'us-east', 1, now=100)
    evidence = RuntimeEvidence(
        release_digest=release.release_digest,
        model_repository=release.model_repository,
        model_revision=release.model_revision,
        tokenizer_repository=release.tokenizer_repository,
        tokenizer_revision=release.tokenizer_revision,
        runtime_digest=release.runtime_digest,
        runtime_commit=release.runtime_commit,
        container_image=release.container_image,
        container_digest=release.container_digest,
        filesystem_digest=release.filesystem_digest,
        stream_public_key='22' * 32,
    )

    state = control.acknowledge_assignment('gpu-1', 1, GPUState.RUNTIME_VERIFY, evidence, now=101)

    assert state == GPUState.QUARANTINED
    assert record.state == GPUState.QUARANTINED
    assert record.lease is None
    assert record.registration.release_digest == ''
    assert record.registration.canary_release_digest == ''
    assert record.runtime_stream_public_key == ''
    assert record.assignment_token != old_assignment_token
    assert executor.revocations == [('gpu-1', 1, 'runtime signing key rotated; new assignment is required')]
    assert control.renew_reservation(reservation.reservation_id, now=101) is None
    with pytest.raises(CapacityUnavailable):
        control.route('release:1', 'us-east', 1, now=101)


def test_operator_disable_immediately_revokes_capacity_and_requires_fresh_readmission():
    executor = RevocationRecordingExecutor()
    control = ComputeControlPlane(_config(), clock=Clock(100), assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    control.gpus['gpu-1'].assignment_dispatched = True
    control.gpus['gpu-1'].assignment_epoch = 1
    reservation = control.route('release:1', 'us-east', 1, now=101)

    control.disable_gpu('gpu-1', 'suspected compromise', now=102)

    record = control.gpus['gpu-1']
    assert record.administratively_disabled
    assert record.state == GPUState.QUARANTINED
    assert record.lease is None
    assert executor.revocations == [('gpu-1', 1, 'administratively disabled: suspected compromise')]
    with pytest.raises(ValueError, match='disabled GPU'):
        control.acknowledge_assignment('gpu-1', 1, GPUState.LOADING, now=102)
    assert control.renew_reservation(reservation.reservation_id, now=102) is None
    with pytest.raises(CapacityUnavailable):
        control.route('release:1', 'us-east', 1, now=102)
    assert 'administratively disabled' in control.refresh_verification([_snapshot(1)], now=103)['gpu-1']

    control.enable_gpu('gpu-1', now=104)

    assert not record.administratively_disabled
    assert record.state == GPUState.REGISTERED
    assert record.registration.release_digest == ''
    assert record.lease is None


def test_release_revocation_invalidates_every_assignment_and_reservation():
    executor = RevocationRecordingExecutor()
    control = ComputeControlPlane(_config(), clock=Clock(100), assignment_executor=executor)
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    control.gpus['gpu-1'].assignment_dispatched = True
    control.gpus['gpu-1'].assignment_epoch = 1
    reservation = control.route('release:1', 'us-east', 1, now=101)

    control.revoke_release('release:1', 'bad runtime image', now=102)

    record = control.gpus['gpu-1']
    assert 'release:1' not in control.releases
    assert record.state == GPUState.QUARANTINED
    assert record.registration.release_digest == ''
    assert record.lease is None
    assert executor.revocations == [('gpu-1', 1, 'release revoked: bad runtime image')]
    assert control.renew_reservation(reservation.reservation_id, now=102) is None
    with pytest.raises(CapacityUnavailable, match='not approved'):
        control.route('release:1', 'us-east', 1, now=102)


def test_dynamic_budget_cannot_defund_the_baseline_fleet():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    with pytest.raises(ValueError, match='configured GPU floor'):
        control.update_budget(2.59, now=100)


@pytest.mark.parametrize('estimate', [0, -1, 901, float('nan'), float('inf')])
def test_route_rejects_service_estimates_outside_physical_limits(estimate):
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))

    with pytest.raises(ValueError, match='configured maximum'):
        control.route('release:1', 'us-east', estimate, now=100)


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


def test_assignment_requires_new_canary_binding_and_verification():
    clock = Clock(100)
    executor = AcceptingAssignmentExecutor()
    control = ComputeControlPlane(_config(), clock=clock, assignment_executor=executor)
    control.register_release(Release('release:1', 'model-1', 'runtime-1'))
    control.register_release(Release('release:2', 'model-2', 'runtime-2'))
    control.register_miner_gpu(
        miner_uid=1,
        miner_hotkey='hotkey-1',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)])

    control.tick(now=110, execute=True)
    rejected = control.refresh_verification([_snapshot(1, last_checked=110)], now=110)
    assert rejected == {'gpu-1': 'DRAINING'}
    assert control.status(now=110)['ready_gpus'] == 0

    control.acknowledge_assignment('gpu-1', 1, GPUState.LOADING, now=110)
    release = control.releases[control.gpus['gpu-1'].registration.release_digest]
    control.acknowledge_assignment(
        'gpu-1',
        1,
        GPUState.RUNTIME_VERIFY,
        RuntimeEvidence(
            release_digest=release.release_digest,
            model_repository=release.model_repository,
            model_revision=release.model_revision,
            tokenizer_repository=release.tokenizer_repository,
            tokenizer_revision=release.tokenizer_revision,
            runtime_digest=release.runtime_digest,
            runtime_commit=release.runtime_commit,
            container_image=release.container_image,
            container_digest=release.container_digest,
            filesystem_digest=release.filesystem_digest,
            stream_public_key='11' * 32,
        ),
        now=110,
    )
    accepted = control.refresh_verification([_snapshot(1, last_checked=111)], now=111)
    assert accepted == {'gpu-1': 'READY'}


def test_offline_transition_does_not_satisfy_global_gepetto_replica_supply():
    control = ComputeControlPlane(_config(), clock=Clock(100), assignment_executor=AcceptingAssignmentExecutor())
    control.register_release(Release('release:1', 'model-1', 'runtime-1', minimum_replicas=1))
    control.register_miner_gpu(
        miner_uid=1,
        miner_hotkey='hotkey-1',
        gpu_id='gpu-1',
        spark_node_id='node-1',
        endpoint='https://gpu-1',
        region='us-east',
    )
    control.refresh_verification([_snapshot(1)], now=100)
    control.tick(now=101, execute=True)
    assert control.gpus['gpu-1'].state == GPUState.DRAINING

    control.refresh_verification([], now=102)
    tick = control.tick(now=103)

    assert tick.placement.assignments == {}
    assert tick.placement.replica_counts == {}


def test_gateway_observation_updates_authoritative_routing_telemetry():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 5, now=100.5)

    control.record_routing_observation(
        RoutingObservation(
            reservation_id=reservation.reservation_id,
            gpu_id='gpu-1',
            requester_region='us-east',
            measured_rtt_ms=12,
            service_seconds=2.5,
            success=True,
            observed_active_slots=2,
            remaining_work_seconds=1.25,
            expected_service_seconds=5.0,
            release_digest='release:1',
        ),
        now=101,
    )

    record = control.gpus['gpu-1']
    assert record.measured_rtt_by_region_ms == {'us-east': 12}
    assert record.service_seconds_ewma_by_release == {'release:1': 2.5}
    assert record.request_estimate_ratio_ewma_by_release == {'release:1': pytest.approx(0.9)}
    assert record.gateway_active_slots == 2
    assert record.gateway_remaining_work_seconds == 1.25


def test_routing_observation_requires_a_strict_boolean_outcome():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 5, now=100.5)

    with pytest.raises(ValueError, match='must be a boolean'):
        control.record_routing_observation(
            RoutingObservation(
                reservation_id=reservation.reservation_id,
                gpu_id='gpu-1',
                requester_region='us-east',
                measured_rtt_ms=12,
                service_seconds=2.5,
                success=cast(bool, 1),
                observed_active_slots=0,
                remaining_work_seconds=0,
                expected_service_seconds=5,
                release_digest='release:1',
            ),
            now=101,
        )


def test_router_keeps_current_request_size_after_calibration():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 5, now=100.5)
    control.record_routing_observation(
        RoutingObservation(
            reservation_id=reservation.reservation_id,
            gpu_id='gpu-1',
            requester_region='us-east',
            measured_rtt_ms=0,
            service_seconds=10,
            success=True,
            observed_active_slots=0,
            remaining_work_seconds=0,
            expected_service_seconds=5,
            release_digest='release:1',
        ),
        now=101,
    )

    short = control.route('release:1', 'us-east', 1, now=102)
    control.complete_reservation(short.reservation_id)
    long = control.route('release:1', 'us-east', 10, now=103)

    assert long.expected_completion_seconds > short.expected_completion_seconds * 9


def test_repeated_real_inference_failures_quarantine_until_new_verification():
    config = _config()
    object.__setattr__(config.router, 'failure_quarantine_threshold', 2)
    control = ComputeControlPlane(config, clock=Clock(100))
    control.register_release(Release('release:1', 'model', 'runtime'))
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
    control.refresh_verification([_snapshot(1)], now=100)

    for index in range(2):
        reservation = control.route('release:1', 'us-east', 5, now=101 + index)
        control.record_routing_observation(
            RoutingObservation(
                reservation_id=reservation.reservation_id,
                gpu_id='gpu-1',
                requester_region='us-east',
                measured_rtt_ms=10,
                service_seconds=1,
                success=False,
                observed_active_slots=0,
                remaining_work_seconds=0,
                expected_service_seconds=5,
                release_digest='release:1',
            ),
            now=101.5 + index,
        )

    assert control.gpus['gpu-1'].state == GPUState.QUARANTINED
    refreshed_hardware = control.refresh_verification([_snapshot(1, last_checked=102)], now=103)
    assert refreshed_hardware == {'gpu-1': 'REGISTERED'}
    assert control.gpus['gpu-1'].registration.release_digest == ''
    with pytest.raises(CapacityUnavailable):
        control.route('release:1', 'us-east', 1, now=103)
    assert control.gpus['gpu-1'].consecutive_inference_failures == 2


def test_automatic_emission_oracle_funds_scaled_target_only_while_fresh():
    clock = Clock(100)
    config = replace(
        _config(),
        fleet=replace(_config().fleet, max_budget_per_hour=None),
        emission_oracle=replace(
            _config().emission_oracle,
            enabled=True,
            max_refresh_staleness_seconds=300,
        ),
    )
    control = ComputeControlPlane(config, clock=clock)
    control.autoscaler.desired_target = 8

    assert control.status(now=100)['funded_target'] == 4

    control.apply_emission_observation(_emission_observation(26, 100), now=100)
    assert control.status(now=100)['funded_target'] == 8

    clock.now = 500
    status = control.status(now=500)
    assert status['funded_target'] == 4
    assert status['funding_shortfall'] == 4
    assert status['emission_oracle']['fresh'] is False
    assert control._funded_target_seconds == 2_800


def test_manual_emission_override_is_disabled_with_automatic_oracle():
    config = replace(_config(), emission_oracle=replace(_config().emission_oracle, enabled=True))
    control = ComputeControlPlane(config, clock=Clock(100))

    with pytest.raises(ValueError, match='manual emission values are disabled'):
        control.update_budget(None, subnet_miner_emission_value_per_hour=30, now=100)
