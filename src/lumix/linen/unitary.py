import jax.numpy as jnp
from jax import random
from flax import linen as nn

from lumix.functional.unitary import complex_matrix, semiunitary_matrix, unitary_linear


class UnitaryLinear(nn.Module):
    width: int
    out_features: int | None = None
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values):
        output_features = self.width if self.out_features is None else self.out_features
        input_features = values.shape[-1]
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
        matrix = semiunitary_matrix(
            complex_matrix(left_re, left_im),
            complex_matrix(right_re, right_im),
            output_features,
            input_features,
        )
        return unitary_linear(values, matrix)
