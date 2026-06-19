"""Flax modules for Clements nearest-neighbor optical meshes."""

from flax import linen as nn
import jax.numpy as jnp

from lumix.functional.clements import build_clements_spec, clements_pair, init_clements
from lumix.functional.routing import RoutingLimit, routing_leakage


class ClementsLinear(nn.Module):
    """Nearest-neighbor Clements mesh optical linear layer.

    If `routing_limit` is set, the layer reconstructs the mesh transfer matrix
    and reports fractional nonlocal routing power to the Flax `metrics`
    collection as `routing_leakage`.
    """

    width: int
    depth: int | None = None
    hadamard: bool = False
    routing_limit: RoutingLimit | None = None

    @nn.compact
    def __call__(self, values):
        depth = self.width if self.depth is None else self.depth
        spec = self.variable(
            "constants",
            "spec",
            lambda: build_clements_spec(self.width, depth),
        ).value
        params = self.param(
            "clements",
            lambda key: init_clements(key, self.width, depth, hadamard=self.hadamard),
        )
        if self.routing_limit is not None:
            basis = jnp.eye(self.width, dtype=values.dtype)
            matrix = clements_pair(
                basis,
                params["theta"],
                params["phi"],
                params["gamma"],
                spec=spec,
                hadamard=self.hadamard,
            ).T
            self.sow("metrics", "routing_leakage", routing_leakage(matrix, self.routing_limit))
        return clements_pair(
            values,
            params["theta"],
            params["phi"],
            params["gamma"],
            spec=spec,
            hadamard=self.hadamard,
        )
