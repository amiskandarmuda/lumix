"""Inverse-design template object."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lumix.inverse_design.layers import LayerSelector, extract_layer_matrix, load_checkpoint_params
from lumix.inverse_design.specs import (
    DeviceDesignSpec,
    MatrixObjectiveSpec,
    OptimizationScope,
    PortCounts,
    TopologyRegionSpec,
)


@dataclass(frozen=True)
class InverseDesignTemplate:
    """Inspectable conversion result for a target matrix and physical device spec."""

    target_matrix: np.ndarray
    device: DeviceDesignSpec
    topology: TopologyRegionSpec
    objective: MatrixObjectiveSpec

    @property
    def port_counts(self) -> PortCounts:
        n_output, n_input = self.target_matrix.shape
        return PortCounts(n_input=int(n_input), n_output=int(n_output))

    @cached_property
    def design_region(self):
        from lumix.inverse_design.tidy3d_builder import build_design_region

        return build_design_region(self)

    @cached_property
    def base_simulation(self):
        from lumix.inverse_design.tidy3d_builder import build_base_simulation

        return build_base_simulation(self)

    @cached_property
    def excitation_plan(self):
        from lumix.inverse_design.tidy3d_builder import MatrixExcitationPlan

        return MatrixExcitationPlan(self)

    @property
    def initial_design_params(self):
        from lumix.inverse_design.tidy3d_builder import initial_design_params

        return initial_design_params(self.design_region)

    def initial_optimization_params(self, *, scope: OptimizationScope = "matrix"):
        from lumix.inverse_design.tidy3d_builder import initial_optimization_params

        return initial_optimization_params(self, scope=scope)

    def base_simulation_with_params(self, params):
        from lumix.inverse_design.tidy3d_builder import build_simulation_with_design_params

        return build_simulation_with_design_params(self, params)

    def apply_design_mask(self, params):
        from lumix.inverse_design.tidy3d_builder import apply_design_mask

        return apply_design_mask(self, params)

    def preflight(self):
        from lumix.inverse_design.tidy3d_builder import PreflightInspection

        return PreflightInspection(self)

    def to_tidy3d_inverse_design(self, *, task_name: str, verbose: bool = True):
        from lumix.inverse_design.tidy3d_builder import to_tidy3d_inverse_design

        return to_tidy3d_inverse_design(self, task_name=task_name, verbose=verbose)


def from_lumix_checkpoint(
    *,
    checkpoint_path: str | Path,
    model: Any,
    sample_x: Any,
    layer: LayerSelector,
    device: DeviceDesignSpec,
    topology: TopologyRegionSpec | None = None,
    objective: MatrixObjectiveSpec | None = None,
) -> InverseDesignTemplate:
    from lumix.inverse_design.matrix import from_matrix_array

    params = load_checkpoint_params(checkpoint_path=checkpoint_path, model=model, sample_x=sample_x)
    matrix = extract_layer_matrix(model=model, params=params, sample_x=sample_x, selector=layer)
    return from_matrix_array(
        matrix,
        device=device,
        topology=TopologyRegionSpec() if topology is None else topology,
        objective=MatrixObjectiveSpec() if objective is None else objective,
    )
