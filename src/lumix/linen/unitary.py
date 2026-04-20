from flax import linen as nn

from lumix.functional.unitary import init_unitary, unitary_linear


class UnitaryLinear(nn.Module):
    width: int
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values):
        raw = self.param(
            "raw",
            lambda key: init_unitary(key, self.width, init_scale=self.init_scale),
        )
        return unitary_linear(values, raw)
