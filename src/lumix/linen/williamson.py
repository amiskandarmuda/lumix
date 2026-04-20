import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.functional.williamson import williamson_response


class WilliamsonNonlinearity(nn.Module):
    tap: float = 0.1
    gain: float = 0.05 * jnp.pi
    bias: float = 1.0 * jnp.pi
    train_gain: bool = False
    train_bias: bool = False

    @nn.compact
    def __call__(self, values):
        gain = self.param("gain", lambda key: jnp.asarray(self.gain))
        bias = self.param("bias", lambda key: jnp.asarray(self.bias))

        if not self.train_gain:
            gain = jax.lax.stop_gradient(gain)
        if not self.train_bias:
            bias = jax.lax.stop_gradient(bias)

        return williamson_response(values, gain, bias, self.tap)
