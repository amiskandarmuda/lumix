import jax.numpy as jnp


def intensity(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.conj(values) * values)


def coherent_iq(
    values: jnp.ndarray,
    out_features: int,
    local_oscillator_phase: float = 0.0,
) -> jnp.ndarray:
    if out_features < 1:
        raise ValueError("out_features must be at least 1")
    if out_features % 2 != 0:
        raise ValueError("coherent_iq requires an even out_features")

    half = out_features // 2
    selected = values[..., :half]
    phase = jnp.asarray(local_oscillator_phase, dtype=selected.real.dtype)
    i_rot = selected * jnp.exp(-1j * phase)
    q_rot = selected * jnp.exp(-1j * (phase + jnp.pi / 2.0))
    return jnp.concatenate([i_rot.real, q_rot.real], axis=-1).astype(jnp.float32)


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
