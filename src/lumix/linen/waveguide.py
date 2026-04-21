from collections.abc import Sequence

from flax import linen as nn
from flax.linen import nowrap

import jax.numpy as jnp

from lumix.functional.waveguide import _validated_waveguide_inputs, waveguide_linear, waveguide_propagator


class FixedWaveguideArray(nn.Module):
    delta: Sequence[float]
    kappa: Sequence[float]
    length: float = 1.0

    @nowrap
    def _constants(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        delta_vector = self.variable(
            "constants",
            "delta",
            lambda: _validated_waveguide_inputs(self.delta, self.kappa, self.length)[0],
        ).value
        kappa_vector = self.variable(
            "constants",
            "kappa",
            lambda: _validated_waveguide_inputs(self.delta, self.kappa, self.length)[1],
        ).value
        propagator = self.variable(
            "constants",
            "propagator",
            lambda: waveguide_propagator(self.delta, self.kappa, self.length),
        ).value
        return delta_vector, kappa_vector, propagator

    @nn.compact
    def __call__(self, values):
        delta_vector, _, propagator = self._constants()
        if values.shape[-1] != delta_vector.shape[0]:
            raise ValueError("values width must match waveguide width")
        return waveguide_linear(values, propagator)
