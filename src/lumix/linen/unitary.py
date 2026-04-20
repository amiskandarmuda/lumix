import jax.numpy as jnp
from jax import random
from flax import linen as nn

from lumix.functional.unitary import unitary_linear


class UnitaryLinear(nn.Module):
    width: int
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values):
        raw_re = self.param(
            "raw_re",
            lambda key: self.init_scale * random.normal(key, (self.width, self.width), dtype=jnp.float32),
        )
        raw_im = self.param(
            "raw_im",
            lambda key: self.init_scale * random.normal(key, (self.width, self.width), dtype=jnp.float32),
        )
        raw = raw_re + 1j * raw_im
        return unitary_linear(values, raw)
