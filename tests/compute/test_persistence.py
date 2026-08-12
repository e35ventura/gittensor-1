from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.models import GPURegistration, Release
from gittensor.compute.storage import SQLiteStateStore

from .test_control_plane import Clock, _config, _snapshot


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


def test_release_digest_cannot_be_redefined():
    control = ComputeControlPlane(_config(), clock=Clock(100))
    control.register_release(Release('release:1', 'model-a', 'runtime-a'))

    try:
        control.register_release(Release('release:1', 'model-b', 'runtime-b'))
    except ValueError as exc:
        assert 'immutable' in str(exc)
    else:
        raise AssertionError('conflicting release was accepted')
