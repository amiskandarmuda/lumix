from collections.abc import Sequence

import jax
import jax.numpy as jnp

from lumix.functional.unitary import unitary_matrix


LossBound = tuple[float | None, float | None]
LossSpec = float | LossBound


def insertion_loss_amplitude(loss_db: jnp.ndarray) -> jnp.ndarray:
    return jnp.power(10.0, -loss_db / 20.0)


def resolve_insertion_loss(raw: jnp.ndarray | None, insertion_loss_db: LossSpec) -> jnp.ndarray:
    if isinstance(insertion_loss_db, Sequence) and not isinstance(insertion_loss_db, (str, bytes)):
        if len(insertion_loss_db) != 2:
            raise ValueError("insertion_loss_db bounds must contain exactly two values")
        minimum_loss_db, maximum_loss_db = insertion_loss_db
    else:
        return jnp.asarray(insertion_loss_db, dtype=jnp.float32)

    if minimum_loss_db is None and maximum_loss_db is None:
        raise ValueError("at least one insertion-loss bound must be set")

    if raw is None:
        raise ValueError("raw insertion-loss parameter is required for ranged loss specifications")

    raw = jnp.asarray(raw, dtype=jnp.float32)

    if minimum_loss_db is not None and maximum_loss_db is not None:
        minimum = jnp.asarray(minimum_loss_db, dtype=jnp.float32)
        maximum = jnp.asarray(maximum_loss_db, dtype=jnp.float32)
        if float(maximum) < float(minimum):
            raise ValueError("maximum insertion loss must be greater than or equal to minimum insertion loss")
        if float(maximum) == float(minimum):
            return minimum
        return minimum + (maximum - minimum) * jax.nn.sigmoid(raw)

    if minimum_loss_db is not None:
        minimum = jnp.asarray(minimum_loss_db, dtype=jnp.float32)
        return minimum + jax.nn.softplus(raw)

    maximum = jnp.asarray(maximum_loss_db, dtype=jnp.float32)
    if float(maximum) < 0.0:
        raise ValueError("maximum insertion loss must be non-negative")
    return maximum * jax.nn.sigmoid(raw)


def subunitary_matrix(
    raw: jnp.ndarray,
    loss_db: jnp.ndarray,
) -> jnp.ndarray:
    return insertion_loss_amplitude(loss_db) * unitary_matrix(raw)


def subunitary_linear(
    values: jnp.ndarray,
    raw: jnp.ndarray,
    loss_db: jnp.ndarray,
) -> jnp.ndarray:
    return values @ subunitary_matrix(raw, loss_db)
