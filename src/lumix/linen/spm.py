import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.functional.spm import spm_response


class SPMNonlinearity(nn.Module):
    gain: float = 1.0
    train_gain: bool = False

    @nn.compact
    def __call__(self, values):
        gain = self.param("gain", lambda key: jnp.asarray(self.gain, dtype=jnp.float32))

        if not self.train_gain:
            gain = jax.lax.stop_gradient(gain)

        return spm_response(values, gain)
