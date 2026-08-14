from gittensor.compute.models import GPURecord, GPURegistration, GPUState, Release
from gittensor.compute.placement import GlobalGepetto, ReleaseDemand


def _gpu(index, release):
    return GPURecord(
        registration=GPURegistration(
            gpu_id=f'gpu-{index}',
            spark_node_id=f'node-{index}',
            miner_uid=index,
            endpoint=f'https://gpu-{index}',
            region='us-east',
            release_digest=release,
            canary_release_digest=release,
            certified_slots=4,
        ),
        state=GPUState.READY,
        assignment_started_at=0,
    )


def test_global_gepetto_assigns_all_supply_from_one_subnet_level_map():
    releases = [
        Release('a', 'model-a', 'runtime-a', minimum_replicas=1),
        Release('b', 'model-b', 'runtime-b', minimum_replicas=1),
    ]
    gpus = [_gpu(0, 'a'), _gpu(1, 'a'), _gpu(2, 'b'), _gpu(3, 'b')]
    plan = GlobalGepetto(minimum_residency_seconds=0).plan(
        gpus,
        releases,
        [ReleaseDemand('a', 1), ReleaseDemand('b', 5)],
        now=100,
    )
    assert set(plan.assignments) == {f'gpu-{index}' for index in range(4)}
    assert plan.replica_counts == {'a': 1, 'b': 3}
    assert len(plan.transitions) == 1


def test_minimum_residency_prevents_assignment_thrashing():
    gpu = _gpu(0, 'a')
    gpu.assignment_started_at = 95
    plan = GlobalGepetto(minimum_residency_seconds=60).plan(
        [gpu],
        [Release('a', 'model-a', 'runtime-a'), Release('b', 'model-b', 'runtime-b')],
        [ReleaseDemand('a', 0), ReleaseDemand('b', 10)],
        now=100,
    )
    assert plan.assignments == {'gpu-0': 'a'}
    assert plan.transitions == ()
