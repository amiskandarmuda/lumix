from flax import linen as nn

import jax.numpy as jnp
from jax import random

from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear


class SubUnitaryLinear(nn.Module):
    width: int
    out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values):
        output_features = self.width if self.out_features is None else self.out_features
        input_features = values.shape[-1]
        singular_min, singular_max = insertion_loss_bounds(self.insertion_loss_db)
        singular_min = self.param("singular_min", lambda key: jnp.asarray(singular_min, dtype=jnp.float32))
        singular_max = self.param("singular_max", lambda key: jnp.asarray(singular_max, dtype=jnp.float32))
        raw_re = self.param(
            "raw_re",
            lambda key: self.init_scale * random.normal(key, (output_features, input_features), dtype=jnp.float32),
        )
        raw_im = self.param(
            "raw_im",
            lambda key: self.init_scale * random.normal(key, (output_features, input_features), dtype=jnp.float32),
        )
        raw = raw_re + 1j * raw_im
        raw = raw + jnp.asarray(0.0, dtype=raw.dtype) * (singular_min + singular_max)
        return subunitary_linear(values, raw)
