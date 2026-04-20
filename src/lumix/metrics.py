import jax.numpy as jnp


def accuracy(probabilities_target: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(probabilities_target, axis=-1) == jnp.argmax(predictions, axis=-1))
