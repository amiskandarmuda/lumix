import jax.numpy as jnp
from jax import random
from flax import linen as nn

from lumix.functional.unitary import combine_complex_parts, isometric_matrix, unitary_linear


class UnitaryLinear(nn.Module):
    width: int
    out_features: int | None = None
    init_scale: float = 1e-2

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
        return unitary_linear(values, matrix)
