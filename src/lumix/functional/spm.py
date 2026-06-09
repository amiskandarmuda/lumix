import jax.numpy as jnp


def spm_response(values: jnp.ndarray, gain: float | jnp.ndarray) -> jnp.ndarray:
    """Return the lossless self-phase modulation response.

    This matches Neuroptica's ``SPMActivation``:
    ``Z_out = Z_in * exp(-1j * gain * abs(Z_in)^2)``.
    """

    power = values.real * values.real + values.imag * values.imag
    return values * jnp.exp(-1j * gain * power)
