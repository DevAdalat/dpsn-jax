from typing import Any, Dict, Optional, Sequence

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import linen as nn


def create_mesh(devices: Optional[Sequence[jax.Device]] = None) -> Mesh:
    """
    Create a mesh for TPU/GPU sharding.

    If devices are provided, uses them. Otherwise, uses all available devices.
    Defaults to a single 'data' axis mesh if no specific topology is requested.
    """
    if devices is None:
        devices = jax.devices()

    num_devices = len(devices)

    mesh = Mesh(devices, axis_names=("data",))
    return mesh


def get_dpsn_sharding_specs() -> Dict[str, PartitionSpec]:
    """
    Define sharding specs for DPSN-R model parameters.

    Returns a dictionary mapping parameter path patterns to PartitionSpecs.
    """
    return {
        "embedding": PartitionSpec(),
        "pool/embeddings": PartitionSpec("data", None),
        "controller": PartitionSpec(),
        "halt_predictor": PartitionSpec(),
        "phase_classifier": PartitionSpec(),
        "state_accumulator": PartitionSpec(),
        "integrator": PartitionSpec(),
    }


def shard_params(
    params: Dict[str, Any], mesh: Mesh, specs: Dict[str, PartitionSpec]
) -> Dict[str, Any]:
    """
    Apply sharding to parameters based on specs.

    This function should be used to distribute parameters to devices.
    """
    return {}


def with_sharding_constraint(x, mesh, axis_names):
    """Wrapper for jax.lax.with_sharding_constraint using NamedSharding."""
    sharding = NamedSharding(mesh, PartitionSpec(*axis_names))
    return jax.lax.with_sharding_constraint(x, sharding)
