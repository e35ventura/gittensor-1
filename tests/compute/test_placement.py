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


def test_missing_minimum_replica_preempts_assignment_residency():
    gpu = _gpu(0, 'a')
    gpu.assignment_started_at = 95
    plan = GlobalGepetto(minimum_residency_seconds=900).plan(
        [gpu],
        [
            Release('a', 'model-a', 'runtime-a'),
            Release('b', 'model-b', 'runtime-b', minimum_replicas=1),
        ],
        [],
        now=100,
    )

    assert plan.assignments == {'gpu-0': 'b'}
    assert plan.replica_counts == {'a': 0, 'b': 1}
    assert len(plan.transitions) == 1
    assert plan.transitions[0].to_release == 'b'


def test_demand_switch_requires_a_persisted_sustained_shortage():
    releases = [Release('a', 'model-a', 'runtime-a'), Release('b', 'model-b', 'runtime-b')]
    gpus = [_gpu(0, 'a'), _gpu(1, 'a')]
    gepetto = GlobalGepetto(
        minimum_residency_seconds=0,
        switch_sustain_seconds=120,
        target_utilization=0.75,
    )

    first = gepetto.plan(gpus, releases, [ReleaseDemand('b', 1)], now=100)

    assert first.transitions == ()
    assert first.deferred == {'b': 'placement shortage is not yet sustained'}
    restored = GlobalGepetto(
        minimum_residency_seconds=0,
        switch_sustain_seconds=120,
        target_utilization=0.75,
    )
    restored.restore_state(gepetto.export_state())
    still_waiting = restored.plan(gpus, releases, [ReleaseDemand('b', 1)], now=219)
    switched = restored.plan(gpus, releases, [ReleaseDemand('b', 1)], now=220)

    assert still_waiting.transitions == ()
    assert switched.transitions
    assert all(transition.to_release == 'b' for transition in switched.transitions)


def test_gepetto_defers_a_switch_that_cannot_repay_its_load_cost():
    releases = [
        Release('a', 'model-a', 'runtime-a'),
        Release('b', 'model-b', 'runtime-b', estimated_load_seconds=20),
    ]
    plan = GlobalGepetto(
        minimum_residency_seconds=0,
        planning_horizon_seconds=10,
        target_utilization=0.75,
    ).plan([_gpu(0, 'a')], releases, [ReleaseDemand('b', 1)], now=100)

    assert plan.assignments == {'gpu-0': 'a'}
    assert plan.transitions == ()
    assert plan.deferred == {'b': 'switching cost exceeds the planning-horizon benefit'}


def test_minimum_replica_repair_bypasses_demand_hysteresis_and_switch_cost():
    releases = [
        Release('a', 'model-a', 'runtime-a'),
        Release('b', 'model-b', 'runtime-b', minimum_replicas=1, estimated_load_seconds=10_000),
    ]
    plan = GlobalGepetto(
        minimum_residency_seconds=0,
        switch_sustain_seconds=10_000,
        planning_horizon_seconds=1,
        target_utilization=0.75,
    ).plan([_gpu(0, 'a')], releases, [], now=100)

    assert plan.assignments == {'gpu-0': 'b'}
    assert len(plan.transitions) == 1


def test_only_the_minimum_replica_gap_bypasses_switch_cost():
    releases = [
        Release('a', 'model-a', 'runtime-a'),
        Release('b', 'model-b', 'runtime-b', minimum_replicas=1, estimated_load_seconds=10_000),
    ]
    plan = GlobalGepetto(
        minimum_residency_seconds=0,
        planning_horizon_seconds=1,
        target_utilization=0.75,
    ).plan([_gpu(index, 'a') for index in range(3)], releases, [ReleaseDemand('b', 10)], now=100)

    assert len(plan.transitions) == 1
    assert plan.replica_counts == {'a': 2, 'b': 1}
    assert plan.deferred == {'b': 'switching cost exceeds the planning-horizon benefit'}
