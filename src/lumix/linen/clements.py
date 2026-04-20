from flax import linen as nn

from lumix.functional.clements import build_clements_spec, clements_pair, init_clements


class ClementsLinear(nn.Module):
    width: int
    depth: int | None = None
    hadamard: bool = False

    @nn.compact
    def __call__(self, values):
        depth = self.width if self.depth is None else self.depth
        spec = self.variable(
            "constants",
            "spec",
            lambda: build_clements_spec(self.width, depth),
        ).value
        params = self.param(
            "clements",
            lambda key: init_clements(key, self.width, depth, hadamard=self.hadamard),
        )
        return clements_pair(
            values,
            params["theta"],
            params["phi"],
            params["gamma"],
            spec=spec,
            hadamard=self.hadamard,
        )
