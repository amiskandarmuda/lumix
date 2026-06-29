"""Target-matrix loading and passivity handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lumix.inverse_design.specs import DeviceDesignSpec, MatrixObjectiveSpec, PortCounts, TopologyRegionSpec
from lumix.inverse_design.template import InverseDesignTemplate


def from_matrix_array(
    matrix: Any,
    *,
    device: DeviceDesignSpec,
    topology: TopologyRegionSpec | None = None,
    objective: MatrixObjectiveSpec | None = None,
) -> InverseDesignTemplate:
    """Create an inverse-design template from an in-memory complex matrix."""

    objective = MatrixObjectiveSpec() if objective is None else objective
    target = prepare_target_matrix(matrix, objective=objective)
    return InverseDesignTemplate(
        target_matrix=target,
        device=device,
        topology=TopologyRegionSpec() if topology is None else topology,
        objective=objective,
    )


def from_matrix_file(
    path: str | Path,
    *,
    device: DeviceDesignSpec,
    topology: TopologyRegionSpec | None = None,
    objective: MatrixObjectiveSpec | None = None,
    array_key: str | None = None,
) -> InverseDesignTemplate:
    """Create an inverse-design template from a .npy or array-only .npz file."""

    matrix = load_matrix_file(path, array_key=array_key)
    return from_matrix_array(matrix, device=device, topology=topology, objective=objective)


def load_matrix_file(path: str | Path, *, array_key: str | None = None) -> np.ndarray:
    matrix_path = Path(path)
    loaded = np.load(matrix_path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        if array_key is None:
            if len(keys) != 1:
                raise ValueError("Multi-array .npz inputs require explicit array_key.")
            array_key = keys[0]
        if array_key not in loaded:
            raise KeyError(f"array_key {array_key!r} not found in {matrix_path}.")
        return np.asarray(loaded[array_key])
    return np.asarray(loaded)


def prepare_target_matrix(matrix: Any, *, objective: MatrixObjectiveSpec) -> np.ndarray:
    target = np.asarray(matrix, dtype=np.complex128)
    if target.ndim != 2:
        raise ValueError(f"Target Matrix must be 2D, got shape {target.shape}.")
    if 0 in target.shape:
        raise ValueError(f"Target Matrix dimensions must be non-empty, got shape {target.shape}.")
    singular_max = float(np.max(np.linalg.svd(target, compute_uv=False)))
    if singular_max > 1.0 + float(objective.passivity.atol):
        if objective.passivity.mode == "reject":
            raise ValueError(
                f"Target Matrix is non-passive: largest singular value is {singular_max:.6g}."
            )
        target = target / singular_max
    return target


def port_counts_for_matrix(matrix: np.ndarray) -> PortCounts:
    n_output, n_input = matrix.shape
    return PortCounts(n_input=int(n_input), n_output=int(n_output))
