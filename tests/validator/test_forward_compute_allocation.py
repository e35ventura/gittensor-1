import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from gittensor.validator.forward import forward


def test_forward_installs_exact_emission_allocation_without_ema_distortion():
    validator = SimpleNamespace(
        step=0,
        metagraph=SimpleNamespace(hotkeys=['recycle', 'miner']),
        evaluation_cache=MagicMock(),
        bulk_store_evaluation=AsyncMock(),
        update_scores=MagicMock(),
    )
    exact_rewards = np.array([0.9, 0.1])
    with (
        patch('gittensor.validator.forward.get_all_uids', return_value={0, 1}),
        patch('gittensor.validator.forward.load_master_repo_weights', return_value={}),
        patch('gittensor.validator.forward.load_programming_language_weights', return_value={}),
        patch(
            'gittensor.validator.forward.load_token_config',
            return_value=SimpleNamespace(language_configs={}),
        ),
        patch('gittensor.validator.forward.oss_contributions', AsyncMock(return_value=({}, set(), set()))),
        patch('gittensor.validator.forward.issue_discovery', AsyncMock()),
        patch('gittensor.validator.forward.issue_competitions', AsyncMock()),
        patch('gittensor.validator.forward.build_maintainer_uids_by_repo', return_value={}),
        patch('gittensor.validator.forward.load_compute_allocation', return_value=None),
        patch('gittensor.validator.forward.blend_emission_pools', return_value=exact_rewards),
        patch('gittensor.validator.forward.asyncio.sleep', AsyncMock()),
    ):
        asyncio.run(forward(cast(Any, validator)))

    validator.update_scores.assert_called_once()
    call = validator.update_scores.call_args
    assert np.array_equal(call.args[0], exact_rewards)
    assert call.args[1] == {0, 1}
    assert call.kwargs['alpha_override'] == 1.0
