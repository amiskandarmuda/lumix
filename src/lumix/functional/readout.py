import jax.numpy as jnp


def channel_power(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.conj(values) * values)


def class_probs(power: jnp.ndarray, classes: int, eps: float = 1e-7) -> jnp.ndarray:
    logits = power[..., :classes]
    normalizer = jnp.clip(jnp.sum(logits, axis=-1, keepdims=True), eps, None)
    probs = logits / normalizer
    return jnp.clip(probs, eps, 1.0)
