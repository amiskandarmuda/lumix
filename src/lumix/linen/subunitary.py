from flax import linen as nn

from lumix.functional.subunitary import init_subunitary, subunitary_linear


class SubUnitaryLinear(nn.Module):
    width: int
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values):
        raw = self.param(
            "raw",
            lambda key: init_subunitary(key, self.width, init_scale=self.init_scale),
        )
        return subunitary_linear(
            values,
            raw,
            self.insertion_loss_db,
        )
