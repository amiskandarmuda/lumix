from flax import linen as nn
from jax import random

from lumix.functional.clements import clements_pair, init_clements


class ClementsLinear(nn.Module):
    width: int

    @nn.compact
    def __call__(self, values):
        params = self.param(
            "clements",
            lambda key: init_clements(key, self.width, self.width),
        )
        return clements_pair(values, params["theta"], params["phi"], params["gamma"])
