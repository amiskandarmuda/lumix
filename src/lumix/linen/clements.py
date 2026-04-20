from flax import linen as nn

from lumix.functional.clements import clements_pair, init_clements


class ClementsLinear(nn.Module):
    width: int
    depth: int | None = None
    hadamard: bool = False

    @nn.compact
    def __call__(self, values):
        depth = self.width if self.depth is None else self.depth
        params = self.param(
            "clements",
            lambda key: init_clements(key, self.width, depth, hadamard=self.hadamard),
        )
        return clements_pair(
            values,
            params["theta"],
            params["phi"],
            params["gamma"],
            hadamard=self.hadamard,
        )
