import jax.numpy as jnp


def intensity(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.conj(values) * values)


def select_classes(values: jnp.ndarray, classes: int) -> jnp.ndarray:
    return values[..., :classes]


def class_logits(values: jnp.ndarray, classes: int) -> jnp.ndarray:
    return select_classes(values, classes)


def normalize_classes(values: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    normalizer = jnp.clip(jnp.sum(values, axis=-1, keepdims=True), eps, None)
    probs = values / normalizer
    return jnp.clip(probs, eps, 1.0)


def class_probs(values: jnp.ndarray, classes: int, eps: float = 1e-7) -> jnp.ndarray:
    logits = class_logits(values, classes)
    return normalize_classes(logits, eps=eps)
