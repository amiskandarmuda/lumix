"""Physical mappings between dense Lumix layers and Clements meshes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from lumix.functional.clements_convert import (
    clements_fit_metrics,
    decompose_unitary_to_clements,
    unitary_linear_matrix_from_params,
)


@dataclass(frozen=True)
class PhysicalMappingResult:
    clements_params: dict[str, jnp.ndarray]
    target_matrix: jnp.ndarray
    loss: jnp.ndarray
    relative_frobenius_error: jnp.ndarray
    max_abs_error: jnp.ndarray


def unitary_linear_to_clements_params(unitary_params: dict[str, jnp.ndarray], *, width: int) -> PhysicalMappingResult:
    """Map square ``UnitaryLinear`` params to physical Clements mesh phases."""

    left_width = int(unitary_params["left_re"].shape[0])
    right_width = int(unitary_params["right_re"].shape[0])
    if left_width != width or right_width != width:
        raise ValueError("physical Clements mapping requires square UnitaryLinear params")

    target = unitary_linear_matrix_from_params(unitary_params, width)
    clements_params = decompose_unitary_to_clements(target)
    metrics = clements_fit_metrics(target, clements_params, depth=width)
    return PhysicalMappingResult(
        clements_params=clements_params,
        target_matrix=target,
        loss=metrics["loss"],
        relative_frobenius_error=metrics["relative_frobenius_error"],
        max_abs_error=metrics["max_abs_error"],
    )
