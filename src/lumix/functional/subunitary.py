from collections.abc import Sequence

import jax
import jax.numpy as jnp

from lumix.functional.unitary import unitary_matrix


LossBound = tuple[float | None, float | None]
LossSpec = float | LossBound


def insertion_loss_amplitude(loss_db: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.power(10.0, -jnp.asarray(loss_db, dtype=jnp.float32) / 20.0)


def _unpack_insertion_loss_bounds(insertion_loss_db: LossSpec) -> tuple[float | None, float | None]:
    if isinstance(insertion_loss_db, Sequence) and not isinstance(insertion_loss_db, (str, bytes)):
        if len(insertion_loss_db) != 2:
            raise ValueError("insertion_loss_db bounds must contain exactly two values")
        return insertion_loss_db

    fixed_loss_db = float(insertion_loss_db)
    return fixed_loss_db, fixed_loss_db


def _validate_insertion_loss_bounds(minimum_loss_db: float | None, maximum_loss_db: float | None) -> None:
    if minimum_loss_db is None and maximum_loss_db is None:
        raise ValueError("at least one insertion-loss bound must be set")
    if minimum_loss_db is not None and minimum_loss_db < 0.0:
        raise ValueError("minimum insertion loss must be non-negative")
    if maximum_loss_db is not None and maximum_loss_db < 0.0:
        raise ValueError("maximum insertion loss must be non-negative")
    if minimum_loss_db is not None and maximum_loss_db is not None and maximum_loss_db < minimum_loss_db:
        raise ValueError("maximum insertion loss must be greater than or equal to minimum insertion loss")


def insertion_loss_bounds(insertion_loss_db: LossSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    minimum_loss_db, maximum_loss_db = _unpack_insertion_loss_bounds(insertion_loss_db)
    _validate_insertion_loss_bounds(minimum_loss_db, maximum_loss_db)

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


def bounded_singular_values(
    singular_raw: jnp.ndarray,
    singular_min: float | jnp.ndarray,
    singular_max: float | jnp.ndarray,
) -> jnp.ndarray:
    lower = jnp.asarray(singular_min, dtype=jnp.float32)
    upper = jnp.asarray(singular_max, dtype=jnp.float32)
    return lower + (upper - lower) * jax.nn.sigmoid(singular_raw)


def subunitary_matrix(
    left_raw: jnp.ndarray,
    right_raw: jnp.ndarray,
    singular_raw: jnp.ndarray,
    singular_min: float | jnp.ndarray,
    singular_max: float | jnp.ndarray,
    output_features: int,
    input_features: int,
) -> jnp.ndarray:
    rank = min(output_features, input_features)
    left = unitary_matrix(left_raw)[:, :rank]
    right = unitary_matrix(right_raw)[:, :rank]
    singular_values = bounded_singular_values(singular_raw, singular_min, singular_max).astype(jnp.complex64)
    return left @ jnp.diag(singular_values) @ jnp.conj(right).T


def subunitary_linear(values: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    return values @ matrix.T
