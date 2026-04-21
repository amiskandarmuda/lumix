from flax import linen as nn

import jax.numpy as jnp
from jax import random
from flax.linen import nowrap

from lumix.functional.unitary import combine_complex_parts
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear, subunitary_matrix


class SubUnitaryLinear(nn.Module):
    width: int
    out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2
    singular_bias: float = 3.0

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
        singular_min = self.param("singular_min", lambda key: jnp.asarray(singular_min, dtype=jnp.float32))
        singular_max = self.param("singular_max", lambda key: jnp.asarray(singular_max, dtype=jnp.float32))
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
        return subunitary_linear(values, matrix)
