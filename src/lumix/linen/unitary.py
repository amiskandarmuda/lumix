"""Flax modules for dense unitary optical linear transforms."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

from lumix.functional.routing import RoutingLimit, routing_leakage
from lumix.functional.unitary import combine_complex_parts, isometric_matrix, unitary_linear


class UnitaryLinear(nn.Module):
    """Dense unitary or isometric optical linear layer.

    If `routing_limit` is set, the layer reports fractional nonlocal routing
    power to the Flax `metrics` collection as `routing_leakage`. The forward
    transform remains unitary/isometric; routing locality is not enforced by
    masking the matrix.
    """

    width: int
    out_features: int | None = None
    init_scale: float = 1e-2
    routing_limit: RoutingLimit | None = None

    def _complex_param(self, name: str, size: int) -> jnp.ndarray:
        real = self.param(
            f"{name}_re",
            lambda key: self.init_scale * random.normal(key, (size, size), dtype=jnp.float32),
        )
        imag = self.param(
            f"{name}_im",
            lambda key: self.init_scale * random.normal(key, (size, size), dtype=jnp.float32),
        )
        return combine_complex_parts(real, imag)

    @nn.compact
    def __call__(self, values):
        output_features = self.width if self.out_features is None else self.out_features
        input_features = values.shape[-1]
        matrix = isometric_matrix(
            self._complex_param("left", output_features),
            self._complex_param("right", input_features),
            output_features,
            input_features,
        )
        self.sow("lumix_inverse_design", "matrix", matrix)
        if self.routing_limit is not None:
            self.sow("metrics", "routing_leakage", routing_leakage(matrix, self.routing_limit))
        return unitary_linear(values, matrix)
