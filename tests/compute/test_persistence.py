import sqlite3
from dataclasses import replace

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.emission_oracle import EmissionObservation
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.routing import CapacityUnavailable
from gittensor.compute.storage import SQLiteStateStore

from .test_control_plane import Clock, RevocationRecordingExecutor, _config, _snapshot


class RejectingSettlementStore(SQLiteStateStore):
    def finalize_settlement(self, *args, **kwargs):
        return False


class CrashingSecurityStore(SQLiteStateStore):
    fail_security_commit = False

    def save_state_and_delete_gpu_reservations(self, state, gpu_ids, reservation_ids=()):
        if not self.fail_security_commit or not gpu_ids:
            return super().save_state_and_delete_gpu_reservations(state, gpu_ids, reservation_ids)
        normalized_gpu_ids = tuple(sorted(set(gpu_ids)))
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            placeholders = ','.join('?' for _ in normalized_gpu_ids)
            connection.execute(
                f'DELETE FROM reservations WHERE gpu_id IN ({placeholders})',
                normalized_gpu_ids,
            )
            connection.execute('ROLLBACK')
        raise RuntimeError('simulated crash before security state commit')


class CrashingCompletionStore(SQLiteStateStore):
    fail_completion = False

    def save_state_and_complete_reservation(self, state, reservation_id, gpu_ids=(), expired_reservation_ids=()):
        if not self.fail_completion:
            return super().save_state_and_complete_reservation(
                state,
                reservation_id,
                gpu_ids,
                expired_reservation_ids,
            )
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM reservations WHERE reservation_id = ?', (reservation_id,))
            connection.execute('ROLLBACK')
        raise RuntimeError('simulated crash before completion checkpoint')


class CrashingTickStore(SQLiteStateStore):
    fail_tick = False

    def save_state_and_delete_gpu_reservations(self, state, gpu_ids, reservation_ids=()):
        if not self.fail_tick or not reservation_ids:
            return super().save_state_and_delete_gpu_reservations(state, gpu_ids, reservation_ids)
        normalized_ids = tuple(sorted(set(reservation_ids)))
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            placeholders = ','.join('?' for _ in normalized_ids)
            connection.execute(
                f'DELETE FROM reservations WHERE reservation_id IN ({placeholders})',
                normalized_ids,
            )
            connection.execute('ROLLBACK')
        raise RuntimeError('simulated crash before tick checkpoint')


def test_state_database_is_owner_only(tmp_path):
    path = tmp_path / 'state.sqlite3'
    SQLiteStateStore(path)

    assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_reservations_migrate_with_fail_closed_capacity(tmp_path):
    path = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE reservations (
                reservation_id TEXT PRIMARY KEY,
                gpu_id TEXT NOT NULL,
                release_digest TEXT NOT NULL,
                service_seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            'INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)',
            ('legacy', 'gpu-1', 'release:1', 10, 100, 200),
        )

    rows = SQLiteStateStore(path).load_reservations()

    assert rows[0]['kv_bytes'] == 2**63 - 1
    assert rows[0]['capacity_units'] == 1


def test_restart_restores_registrations_leases_reservations_and_accounting(tmp_path):
    clock = Clock(100)
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=clock, store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)

    restored = ComputeControlPlane(_config(), clock=clock, store=store)

    status = restored.status(now=102)
    assert status['registered_gpus'] == 1
    assert status['ready_gpus'] == 1
    assert status['gpus']['gpu-1']['active_reservations'] == 1
    assert restored.complete_reservation(reservation.reservation_id)


def test_reservation_renewal_survives_restart(tmp_path):
    clock = Clock(100)
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=clock, store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)

    assert control.renew_reservation(reservation.reservation_id, now=590) == 599
    restored = ComputeControlPlane(_config(), clock=Clock(590), store=store)

    status = restored.status(now=599)
    assert status['gpus']['gpu-1']['active_reservations'] == 0
    assert status['gpus']['gpu-1']['state'] == 'QUARANTINED'


def test_administrative_disable_survives_restart_and_cannot_be_bypassed_by_registration(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    registration = GPURegistration(
        gpu_id='gpu-1',
        spark_node_id='node-1',
        miner_uid=7,
        miner_hotkey='hotkey-7',
        endpoint='https://gpu-1',
        region='us-east',
        release_digest='release:1',
        canary_release_digest='release:1',
        certified_slots=4,
    )
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(registration)
    control.refresh_verification([_snapshot(1)], now=100)
    control.disable_gpu('gpu-1', 'operator quarantine', now=101)

    restored = ComputeControlPlane(_config(), clock=Clock(102), store=store)

    record = restored.gpus['gpu-1']
    assert record.administratively_disabled
    assert record.lease is None
    assert record.revocation_reason == 'administratively disabled: operator quarantine'
    try:
        restored.register_gpu(registration, now=102)
    except ValueError as exc:
        assert 'administratively disabled' in str(exc)
    else:
        raise AssertionError('registration bypassed the persisted operator quarantine')


def test_quarantine_and_reservation_invalidation_commit_atomically(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)

    control.refresh_verification([], now=102)
    restored = ComputeControlPlane(_config(), clock=Clock(102), store=store)

    record = restored.gpus['gpu-1']
    assert record.state.value == 'QUARANTINED'
    assert record.registration.release_digest == ''
    assert restored.router.reservation(reservation.reservation_id) is None


def test_security_transaction_rolls_back_reservation_delete_if_state_commit_crashes(tmp_path):
    store = CrashingSecurityStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)
    store.fail_security_commit = True

    try:
        control.refresh_verification([], now=102)
    except RuntimeError as exc:
        assert 'simulated crash' in str(exc)
    else:
        raise AssertionError('simulated security commit failure was accepted')

    restarted_store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    restored = ComputeControlPlane(_config(), clock=Clock(102), store=restarted_store)
    assert restored.gpus['gpu-1'].state.value == 'READY'
    assert restored.router.reservation(reservation.reservation_id) is not None


def test_completion_and_utilization_checkpoint_commit_atomically(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)

    assert control.complete_reservation(reservation.reservation_id, now=111)
    restored = ComputeControlPlane(_config(), clock=Clock(111), store=store)

    assert restored.router.reservation(reservation.reservation_id) is None
    assert restored._completed_capacity_seconds == {'release:1': 2.5}


def test_completion_transaction_rollback_restores_reservation_and_demand(tmp_path):
    store = CrashingCompletionStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 10, now=101)
    store.fail_completion = True

    try:
        control.complete_reservation(reservation.reservation_id, now=111)
    except RuntimeError as exc:
        assert 'simulated crash' in str(exc)
    else:
        raise AssertionError('simulated completion failure was accepted')

    assert control.router.reservation(reservation.reservation_id) is not None
    assert control._completed_capacity_seconds == {}
    restored = ComputeControlPlane(_config(), clock=Clock(111), store=SQLiteStateStore(store.path))
    assert restored.router.reservation(reservation.reservation_id) is not None
    assert restored._completed_capacity_seconds == {}


def test_late_completion_counts_occupancy_only_until_reservation_expiry(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    config = replace(
        _config(),
        router=replace(_config().router, reservation_ttl_seconds=10, maximum_service_seconds=20),
    )
    control = ComputeControlPlane(config, clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=100)

    assert reservation.expires_at == 131
    assert control.complete_reservation(reservation.reservation_id, now=200)
    assert control._completed_capacity_seconds == {'release:1': 7.75}


def test_completion_after_status_expiry_consumes_demand_and_row_exactly_once(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    config = replace(
        _config(),
        router=replace(_config().router, reservation_ttl_seconds=10, maximum_service_seconds=20),
    )
    control = ComputeControlPlane(config, clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=100)

    control.status(now=131)
    assert control.complete_reservation(reservation.reservation_id, now=200)
    assert control._completed_capacity_seconds == {'release:1': 7.75}

    restored = ComputeControlPlane(config, clock=Clock(200), store=store)
    assert restored.router.reservation(reservation.reservation_id) is None
    assert restored._completed_capacity_seconds == {'release:1': 7.75}


def test_quarantine_after_status_expiry_consumes_demand_and_row_exactly_once(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    config = replace(
        _config(),
        router=replace(_config().router, reservation_ttl_seconds=10, maximum_service_seconds=20),
    )
    control = ComputeControlPlane(config, clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=100)

    control.status(now=131)
    control.refresh_verification([], now=132)

    restored = ComputeControlPlane(config, clock=Clock(132), store=store)
    assert restored.router.reservation(reservation.reservation_id) is None
    assert restored._completed_capacity_seconds == {'release:1': 7.75}


def test_rejected_demand_survives_restart_before_control_tick(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))

    try:
        control.route('release:1', 'us-east', 10, now=101)
    except CapacityUnavailable:
        pass

    restored = ComputeControlPlane(_config(), clock=Clock(101), store=store)
    assert restored._rejected_capacity_seconds == {'release:1': 2.5}


def test_expired_reservation_is_not_deleted_before_tick_checkpoints_its_demand(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    config = replace(
        _config(),
        router=replace(_config().router, reservation_ttl_seconds=10, maximum_service_seconds=20),
    )
    control = ComputeControlPlane(config, clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=100)

    control.status(now=131)
    restarted_before_tick = ComputeControlPlane(config, clock=Clock(131), store=store)
    assert restarted_before_tick.router.reservation(reservation.reservation_id) is not None

    tick = restarted_before_tick.tick(now=131)
    restarted_after_tick = ComputeControlPlane(config, clock=Clock(131), store=store)
    assert tick.autoscaling.concurrent_demand == 0.25
    assert restarted_after_tick.router.reservation(reservation.reservation_id) is None


def test_failed_tick_commit_restores_expired_reservation_and_demand_for_retry(tmp_path):
    store = CrashingTickStore(tmp_path / 'state.sqlite3')
    config = replace(
        _config(),
        router=replace(_config().router, reservation_ttl_seconds=10, maximum_service_seconds=20),
    )
    control = ComputeControlPlane(config, clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)
    reservation = control.route('release:1', 'us-east', 1, now=100)
    store.fail_tick = True

    try:
        control.tick(now=131)
    except RuntimeError as exc:
        assert 'simulated crash' in str(exc)
    else:
        raise AssertionError('simulated tick failure was accepted')

    assert control.router.reservation(reservation.reservation_id) is not None
    assert control._completed_capacity_seconds == {}
    assert control._last_tick_at == 100
    restored = ComputeControlPlane(config, clock=Clock(131), store=SQLiteStateStore(store.path))
    assert restored.router.reservation(reservation.reservation_id) is not None

    store.fail_tick = False
    tick = control.tick(now=131)
    assert tick.autoscaling.concurrent_demand == 0.25
    assert control.router.reservation(reservation.reservation_id) is None


def test_pending_assignment_revocation_survives_restart_and_retries(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.gpus['gpu-1'].assignment_epoch = 1
    control.disable_gpu('gpu-1', 'operator quarantine', now=101)
    assert control.retry_pending_revocations() == {'gpu-1': 'revocation transport is unavailable'}
    control.refresh_verification([_snapshot(1, last_checked=102)], now=102)
    assert control.gpus['gpu-1'].revocation_reason == 'administratively disabled: operator quarantine'

    executor = RevocationRecordingExecutor()
    restored = ComputeControlPlane(_config(), clock=Clock(102), store=store, assignment_executor=executor)

    assert restored.gpus['gpu-1'].revocation_pending
    assert restored.retry_pending_revocations() == {'gpu-1': 'revoked'}
    assert executor.revocations == [('gpu-1', 1, 'administratively disabled: operator quarantine')]
    assert not restored.gpus['gpu-1'].revocation_pending


def test_restart_turns_ambiguous_disabled_assignment_delivery_into_pending_revocation(tmp_path):
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    record = control.gpus['gpu-1']
    record.assignment_epoch = 1
    record.assignment_dispatch_in_flight = True
    control.disable_gpu('gpu-1', 'operator quarantine', now=101)
    record.revocation_pending = False
    control._persist()

    restored = ComputeControlPlane(_config(), clock=Clock(102), store=store)

    assert restored.gpus['gpu-1'].revocation_pending
    assert not restored.gpus['gpu-1'].assignment_dispatch_in_flight


def test_release_digest_cannot_be_redefined():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model-a', 'runtime-a'))

    try:
        control.register_release(Release('release:1', 'model-b', 'runtime-b'))
    except ValueError as exc:
        assert 'immutable' in str(exc)
    else:
        raise AssertionError('conflicting release was accepted')


def test_settlement_and_accounting_checkpoint_are_committed_together(tmp_path):
    clock = Clock(100)
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=clock, store=store)
    control.register_release(Release('release:1', 'model', 'runtime'))
    control.register_gpu(
        GPURegistration(
            gpu_id='gpu-1',
            spark_node_id='node-1',
            miner_uid=7,
            miner_hotkey='hotkey-7',
            endpoint='https://gpu-1',
            region='us-east',
            release_digest='release:1',
            canary_release_digest='release:1',
            certified_slots=4,
        )
    )
    control.refresh_verification([_snapshot(1)], now=100)

    first, _ = control.settle(now=200)
    restored = ComputeControlPlane(_config(), clock=Clock(200), store=store)
    second, _ = restored.settle(now=300)

    assert first.total_ready_seconds == 100
    assert second.total_ready_seconds == 100
    latest = store.latest_settlement(1_000, now=300)
    assert latest is not None
    assert latest['metadata']['compute_emission_share'] == 0.05
    assert latest['metadata']['compute_reserved_emission_share'] == 0.1
    settlements = store._connect().execute('SELECT COUNT(*) FROM settlements').fetchone()[0]
    assert settlements == 2


def test_failed_settlement_commit_preserves_in_memory_accounting(tmp_path):
    store = RejectingSettlementStore(tmp_path / 'state.sqlite3')
    control = ComputeControlPlane(_config(), clock=Clock(100), store=store)

    try:
        control.settle(now=200)
    except RuntimeError as exc:
        assert 'already finalized' in str(exc)
    else:
        raise AssertionError('failed settlement commit was accepted')

    assert control._funded_target_seconds == 400
    assert control._funded_budget_seconds == 260
    assert control._compute_emission_share_seconds == 10
    assert control._settlement_started_at == 100


def test_restart_accounts_oracle_expiry_at_the_historical_boundary(tmp_path):
    clock = Clock(100)
    store = SQLiteStateStore(tmp_path / 'state.sqlite3')
    base = _config()
    config = replace(
        base,
        fleet=replace(base.fleet, max_budget_per_hour=None),
        emission_oracle=replace(base.emission_oracle, enabled=True, max_refresh_staleness_seconds=300),
    )
    control = ComputeControlPlane(config, clock=clock, store=store)
    control.autoscaler.desired_target = 8
    control.apply_emission_observation(
        EmissionObservation(
            value_per_hour=26,
            currency='USD',
            epoch_block=1000,
            epoch_started_block=640,
            epoch_seconds=4320,
            miner_alpha=147.6,
            alpha_tao_price=0.004,
            price_block=1100,
            tao_currency_price=200,
            source='chain+coinbase+coingecko',
            observed_at=100,
        ),
        now=100,
    )
    control.status(now=200)

    restored = ComputeControlPlane(config, clock=Clock(500), store=store)

    assert restored.funding.funded_target == 4
    assert restored._funded_target_seconds == 2800
