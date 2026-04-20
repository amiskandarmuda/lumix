from collections.abc import Sequence

import jax.numpy as jnp
from jax import random


LossBound = tuple[float | None, float | None]
LossSpec = float | LossBound


def insertion_loss_amplitude(loss_db: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.power(10.0, -jnp.asarray(loss_db, dtype=jnp.float32) / 20.0)


def insertion_loss_bounds(insertion_loss_db: LossSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    if isinstance(insertion_loss_db, Sequence) and not isinstance(insertion_loss_db, (str, bytes)):
        if len(insertion_loss_db) != 2:
            raise ValueError("insertion_loss_db bounds must contain exactly two values")
        minimum_loss_db, maximum_loss_db = insertion_loss_db
    else:
        minimum_loss_db = float(insertion_loss_db)
        maximum_loss_db = float(insertion_loss_db)

    if minimum_loss_db is None and maximum_loss_db is None:
        raise ValueError("at least one insertion-loss bound must be set")
    if minimum_loss_db is not None and minimum_loss_db < 0.0:
        raise ValueError("minimum insertion loss must be non-negative")
    if maximum_loss_db is not None and maximum_loss_db < 0.0:
        raise ValueError("maximum insertion loss must be non-negative")
    if minimum_loss_db is not None and maximum_loss_db is not None and maximum_loss_db < minimum_loss_db:
        raise ValueError("maximum insertion loss must be greater than or equal to minimum insertion loss")

    singular_min = (
        insertion_loss_amplitude(maximum_loss_db)
        if maximum_loss_db is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    singular_max = (
        insertion_loss_amplitude(minimum_loss_db)
        if minimum_loss_db is not None
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    return singular_min, singular_max


def project_subunitary_to_bounds(
    raw: jnp.ndarray,
    singular_min: float | jnp.ndarray,
    singular_max: float | jnp.ndarray,
) -> jnp.ndarray:
    left_vectors, singular_values, right_vectors = jnp.linalg.svd(raw, full_matrices=False)
    clipped = jnp.clip(
        singular_values,
        jnp.asarray(singular_min, dtype=jnp.float32),
        jnp.asarray(singular_max, dtype=jnp.float32),
    )
    return (left_vectors * clipped[None, :]) @ right_vectors


def init_subunitary(
    key,
    out_features: int,
    in_features: int,
    init_scale: float = 1e-2,
    insertion_loss_db: LossSpec = 0.0,
) -> jnp.ndarray:
    real_key, imag_key = random.split(key)
    real = random.normal(real_key, (out_features, in_features), dtype=jnp.float32)
    imag = random.normal(imag_key, (out_features, in_features), dtype=jnp.float32)
    raw = init_scale * (real + 1j * imag)
    singular_min, singular_max = insertion_loss_bounds(insertion_loss_db)
    return project_subunitary_to_bounds(raw, singular_min, singular_max)


def subunitary_matrix(raw: jnp.ndarray) -> jnp.ndarray:
    return raw


def subunitary_linear(
    values: jnp.ndarray,
    raw: jnp.ndarray,
) -> jnp.ndarray:
    return values @ subunitary_matrix(raw).T
