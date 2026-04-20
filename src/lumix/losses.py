import jax.numpy as jnp
import optax


def cross_entropy(probabilities_target: jnp.ndarray, probabilities_predicted: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    clipped_probabilities = jnp.clip(probabilities_predicted, eps, 1.0)
    return -jnp.mean(jnp.sum(probabilities_target * jnp.log(clipped_probabilities), axis=-1))


def cross_entropy_logits(probabilities_target: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy(logits, probabilities_target).mean()
