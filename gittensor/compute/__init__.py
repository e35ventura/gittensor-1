"""Global compute sub-subnet control plane.

The package separates the core responsibilities:

* SparkCompute and Gittensor release checks decide whether a GPU is eligible.
* The autoscaler and funding guard decide the paid target.
* Global Gepetto decides which release each GPU should run.
* The router atomically reserves the eligible GPU expected to finish first.
* Durable settlement maps verified GPU-seconds into validator weights.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gittensor.compute.autoscaler import AutoscaleDecision, FleetAutoscaler
    from gittensor.compute.config import ComputeConfig, load_compute_config
    from gittensor.compute.control_plane import ComputeControlPlane
    from gittensor.compute.models import GPURegistration, GPUState, Release
    from gittensor.compute.routing import CapacityUnavailable, RouteDecision
    from gittensor.compute.settlement import FundingPlan, SettlementResult

__all__ = [
    'AutoscaleDecision',
    'CapacityUnavailable',
    'ComputeConfig',
    'ComputeControlPlane',
    'FleetAutoscaler',
    'FundingPlan',
    'GPURegistration',
    'GPUState',
    'Release',
    'RouteDecision',
    'SettlementResult',
    'load_compute_config',
]

_EXPORTS = {
    'AutoscaleDecision': ('gittensor.compute.autoscaler', 'AutoscaleDecision'),
    'CapacityUnavailable': ('gittensor.compute.routing', 'CapacityUnavailable'),
    'ComputeConfig': ('gittensor.compute.config', 'ComputeConfig'),
    'ComputeControlPlane': ('gittensor.compute.control_plane', 'ComputeControlPlane'),
    'FleetAutoscaler': ('gittensor.compute.autoscaler', 'FleetAutoscaler'),
    'FundingPlan': ('gittensor.compute.settlement', 'FundingPlan'),
    'GPURegistration': ('gittensor.compute.models', 'GPURegistration'),
    'GPUState': ('gittensor.compute.models', 'GPUState'),
    'Release': ('gittensor.compute.models', 'Release'),
    'RouteDecision': ('gittensor.compute.routing', 'RouteDecision'),
    'SettlementResult': ('gittensor.compute.settlement', 'SettlementResult'),
    'load_compute_config': ('gittensor.compute.config', 'load_compute_config'),
}


def __getattr__(name: str) -> Any:
    """Keep package import lightweight so standalone CLI parsing stays isolated."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
