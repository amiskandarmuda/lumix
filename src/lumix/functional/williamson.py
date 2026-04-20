import jax.numpy as jnp


def williamson_response(
    values: jnp.ndarray,
    gain: jnp.ndarray,
    bias: jnp.ndarray,
    tap: float,
) -> jnp.ndarray:
    power = jnp.real(jnp.conj(values) * values)
    phase = 0.5 * (gain * power + bias)
    return 1j * jnp.sqrt(1.0 - tap) * jnp.exp(-1j * phase) * jnp.cos(phase) * values
