from flax import linen as nn

import jax.numpy as jnp
from jax import random

from lumix.functional.unitary import complex_matrix
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear, subunitary_matrix


class SubUnitaryLinear(nn.Module):
    width: int
    out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2
    singular_bias: float = 3.0

    @nn.compact
    def __call__(self, values):
        output_features = self.width if self.out_features is None else self.out_features
        input_features = values.shape[-1]
        rank = min(output_features, input_features)
        singular_min, singular_max = insertion_loss_bounds(self.insertion_loss_db)
        singular_min = self.param("singular_min", lambda key: jnp.asarray(singular_min, dtype=jnp.float32))
        singular_max = self.param("singular_max", lambda key: jnp.asarray(singular_max, dtype=jnp.float32))
        left_re = self.param(
            "left_re",
            lambda key: self.init_scale * random.normal(key, (output_features, output_features), dtype=jnp.float32),
        )
        left_im = self.param(
            "left_im",
            lambda key: self.init_scale * random.normal(key, (output_features, output_features), dtype=jnp.float32),
        )
        right_re = self.param(
            "right_re",
            lambda key: self.init_scale * random.normal(key, (input_features, input_features), dtype=jnp.float32),
        )
        right_im = self.param(
            "right_im",
            lambda key: self.init_scale * random.normal(key, (input_features, input_features), dtype=jnp.float32),
        )
        singular_raw = self.param(
            "singular_raw",
            lambda key: jnp.full((rank,), self.singular_bias, dtype=jnp.float32),
        )
        matrix = subunitary_matrix(
            complex_matrix(left_re, left_im),
            complex_matrix(right_re, right_im),
            singular_raw,
            singular_min,
            singular_max,
            output_features,
            input_features,
        )
        return subunitary_linear(values, matrix)
