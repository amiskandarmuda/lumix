"""Flax modules for passive subunitary optical linear transforms."""

from flax import linen as nn

import jax.numpy as jnp
from jax import random
from flax.linen import nowrap

from lumix.functional.routing import RoutingLimit, routing_leakage
from lumix.functional.unitary import combine_complex_parts
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear, subunitary_matrix


class SubUnitaryLinear(nn.Module):
    """Dense passive subunitary optical linear layer.

    Singular values are constrained by `insertion_loss_db`. If `routing_limit`
    is set, the layer reports fractional nonlocal routing power to the Flax
    `metrics` collection as `routing_leakage`. Because the metric is
    fractional, adding uniform loss cannot reduce it.
    """

    width: int
    out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2
    singular_bias: float = 3.0
    routing_limit: RoutingLimit | None = None

    @nowrap
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
        rank = min(output_features, input_features)
        singular_min, singular_max = insertion_loss_bounds(self.insertion_loss_db)
        singular_raw = self.param(
            "singular_raw",
            lambda key: jnp.full((rank,), self.singular_bias, dtype=jnp.float32),
        )
        matrix = subunitary_matrix(
            self._complex_param("left", output_features),
            self._complex_param("right", input_features),
            singular_raw,
            singular_min,
            singular_max,
            output_features,
            input_features,
        )
        self.sow("lumix_inverse_design", "matrix", matrix)
        if self.routing_limit is not None:
            self.sow("metrics", "routing_leakage", routing_leakage(matrix, self.routing_limit))
        return subunitary_linear(values, matrix)
