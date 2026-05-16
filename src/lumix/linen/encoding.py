from flax import linen as nn
import jax.numpy as jnp

from lumix.functional.encoding import encode_amplitude, encode_complex, encode_phase


class InformationEncoder(nn.Module):
    mode: str = "phase"
    normalize: bool = False
    input_range: tuple[float, float] = (0.0, 1.0)
    phase_range: tuple[float, float] = (0.0, jnp.pi)
    amplitude_range: tuple[float, float] = (0.0, 1.0)
    amplitude: float = 1.0
    clip: bool = False

    @nn.compact
    def __call__(self, values):
        if self.mode == "phase":
            return encode_phase(
                values,
                phase_range=self.phase_range,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        if self.mode == "amplitude":
            return encode_amplitude(
                values,
                amplitude_range=self.amplitude_range,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        if self.mode == "complex":
            return encode_complex(
                values,
                phase_range=self.phase_range,
                amplitude=self.amplitude,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        raise ValueError("mode must be one of 'phase', 'amplitude', or 'complex'")
