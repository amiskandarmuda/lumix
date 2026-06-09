import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.linen.clements import ClementsLinear
from lumix.linen.readout import CoherentIQReadout, ProbabilityReadout
from lumix.linen.williamson import WilliamsonNonlinearity


class SmokeNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = ClementsLinear(width=16)(values)
        values = WilliamsonNonlinearity()(values)
        values = ClementsLinear(width=16)(values)
        return ProbabilityReadout(classes=self.classes)(values)


def test_smoke_forward_shape():
    model = SmokeNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(0), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)


def test_clements_apply_is_jittable_with_runtime_variables():
    model = SmokeNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(1), values)

    @jax.jit
    def apply_fn(runtime_variables, runtime_values):
        return model.apply(runtime_variables, runtime_values)

    probs = apply_fn(variables, values)
    assert probs.shape == (4, 10)


def test_coherent_iq_readout_returns_dual_quadratures():
    model = CoherentIQReadout(out_features=4, local_oscillator_phase=0.0, mix=False)
    values = jnp.asarray([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=jnp.complex64)

    variables = model.init(jax.random.key(2), values)
    quadratures = model.apply(variables, values)

    assert quadratures.shape == (1, 4)
    assert jnp.allclose(quadratures, jnp.asarray([[1.0, 3.0, 2.0, -4.0]], dtype=jnp.float32))
