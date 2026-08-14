from typing import cast

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.emission_oracle import EmissionObservation
from gittensor.compute.supervisor import ComputeSupervisor

from .test_control_plane import _config


class PartiallyFailingControl:
    def __init__(self):
        self.config = _config()
        self.calls = []
        self._settlement_started_at = 0

    def refresh_verification(self, now):
        self.calls.append('verification')
        raise RuntimeError('verifier unavailable')

    def retry_pending_revocations(self):
        self.calls.append('revocations')
        return {}

    def run_weight_challenges(self, now):
        self.calls.append('challenges')

    def tick(self, now, execute):
        self.calls.append('control')

    def settle(self, now):
        self.calls.append('settlement')


def test_failed_verification_does_not_starve_other_control_loops():
    control = PartiallyFailingControl()
    supervisor = ComputeSupervisor(cast(ComputeControlPlane, control), clock=lambda: 0)

    supervisor.run_due(now=4_000)

    assert control.calls == ['revocations', 'verification', 'challenges', 'control', 'settlement']


class FailingOracle:
    def observe(self, now=None):
        raise RuntimeError('price unavailable')


class OracleAwareControl(PartiallyFailingControl):
    def record_emission_oracle_failure(self, error):
        self.calls.append(f'oracle_failure:{error}')

    def apply_emission_observation(self, observation: EmissionObservation, now=None):
        self.calls.append('oracle')


def test_failed_emission_oracle_does_not_starve_control_loops():
    control = OracleAwareControl()
    supervisor = ComputeSupervisor(
        cast(ComputeControlPlane, control),
        emission_oracle=FailingOracle(),
        clock=lambda: 0,
    )

    supervisor.run_due(now=4_000)

    assert control.calls == [
        'revocations',
        'oracle_failure:price unavailable',
        'verification',
        'challenges',
        'control',
        'settlement',
    ]
