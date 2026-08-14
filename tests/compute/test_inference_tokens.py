import time

from gittensor.compute.inference_tokens import issue_inference_capability, verify_inference_capability


def test_inference_capability_binds_route_and_rejects_tampering_and_expiry():
    now = time.time()
    token = issue_inference_capability(
        'assignment-secret',
        reservation_id='reservation-1',
        gpu_id='gpu-1',
        release_digest='release:1',
        expires_at=now + 30,
    )

    capability = verify_inference_capability('assignment-secret', token, now=now)
    assert capability is not None
    assert capability.reservation_id == 'reservation-1'
    assert verify_inference_capability('wrong-secret', token, now=now) is None
    assert verify_inference_capability('assignment-secret', f'{token}0', now=now) is None
    assert verify_inference_capability('assignment-secret', token, now=now + 30) is None
