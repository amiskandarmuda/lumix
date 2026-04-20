from flax import linen as nn

import jax.numpy as jnp
from jax import random

from lumix.functional.subunitary import resolve_insertion_loss, subunitary_linear
from lumix.functional.unitary import init_unitary


class SubUnitaryLinear(nn.Module):
    width: int
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2
    loss_bias: float = -4.0

    @nn.compact
    def __call__(self, values):
        raw = self.param(
            "raw",
            lambda key: init_unitary(key, self.width, init_scale=self.init_scale),
        )
        loss_raw = None
        if isinstance(self.insertion_loss_db, tuple):
            loss_raw = self.param(
                "loss_raw",
                lambda key: jnp.asarray(
                    self.loss_bias + self.init_scale * random.normal(key, (), dtype=jnp.float32),
                    dtype=jnp.float32,
                ),
            )
        loss_db = resolve_insertion_loss(loss_raw, self.insertion_loss_db)
        return subunitary_linear(
            values,
            raw,
            loss_db,
        )
