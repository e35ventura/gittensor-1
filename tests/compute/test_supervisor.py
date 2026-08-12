from typing import cast

from gittensor.compute.control_plane import ComputeControlPlane
from gittensor.compute.supervisor import ComputeSupervisor

from .test_control_plane import _config


class PartiallyFailingControl:
    def __init__(self):
        self.config = _config()
        self.calls = []

    def refresh_verification(self, now):
        self.calls.append('verification')
        raise RuntimeError('verifier unavailable')

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

    assert control.calls == ['verification', 'challenges', 'control', 'settlement']
