import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.linen.clements import ClementsLinear
from lumix.linen.readout import PowerReadout
from lumix.linen.williamson import WilliamsonNonlinearity


class SmokeNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = ClementsLinear(width=16)(values)
        values = WilliamsonNonlinearity()(values)
        values = ClementsLinear(width=16)(values)
        return PowerReadout(classes=self.classes)(values)


def test_smoke_forward_shape():
    model = SmokeNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(0), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)
