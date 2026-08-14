"""Lightweight console entrypoints that parse arguments before importing Bittensor."""

from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType


def control_plane_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the Gittensor compute control plane')
    parser.add_argument('--config', required=True, help='path to compute JSON configuration')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--token-env', default='GITTENSOR_COMPUTE_TOKEN')
    parser.add_argument('--gateway-token-env', default='GITTENSOR_GATEWAY_CONTROL_TOKEN')
    parser.add_argument(
        '--allow-insecure-local',
        action='store_true',
        help='allow startup without an API token when bound to a loopback address',
    )
    return parser


def miner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run one signed Gittensor miner runtime agent')
    parser.add_argument('--config', required=True, help='path to the miner agent JSON configuration')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8781)
    return parser


def gateway_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the verified Gittensor inference gateway')
    parser.add_argument('--config', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8782)
    return parser


def runtime_proxy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the Gittensor signed-runtime proxy')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    return parser


def _import_without_cli_side_effects(module_name: str) -> ModuleType:
    original = sys.argv
    sys.argv = [original[0]]
    try:
        return importlib.import_module(module_name)
    finally:
        sys.argv = original


def control_plane_main() -> None:
    args = control_plane_parser().parse_args()
    _import_without_cli_side_effects('gittensor.compute.service').main(args)


def miner_main() -> None:
    args = miner_parser().parse_args()
    _import_without_cli_side_effects('gittensor.compute.miner_agent').main(args)


def gateway_main() -> None:
    args = gateway_parser().parse_args()
    _import_without_cli_side_effects('gittensor.compute.gateway').main(args)


def runtime_proxy_main() -> None:
    args = runtime_proxy_parser().parse_args()
    _import_without_cli_side_effects('gittensor.compute.runtime_proxy').main(args)
