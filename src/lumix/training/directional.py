"""FFzero-style directional-derivative optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


@dataclass(frozen=True)
class DirectionalDerivativeResult:
    params: object
    gradient: object
    loss: jnp.ndarray
    key: jax.Array


def _normalized_random_direction(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
    direction = jax.random.normal(key, shape, dtype=dtype)
    return direction / (jnp.linalg.norm(direction) + jnp.asarray(1e-12, dtype=dtype))


def _validate_real_flat_params(flat_params: jnp.ndarray) -> None:
    if jnp.iscomplexobj(flat_params):
        raise ValueError("FFzero directional derivatives require real trainable parameters")


def directional_derivative_gradient(
    params,
    loss_fn: Callable[[object], jnp.ndarray],
    *,
    key: jax.Array,
    eps: float,
    num_directions: int,
    directions: jnp.ndarray | None = None,
) -> DirectionalDerivativeResult:
    """Estimate the FFzero central directional derivative gradient.

    The update gradient is exactly the reference rule:
    ``dim * mean(((L(p + eps*d) - L(p - eps*d)) / (2*eps)) * d)``.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if num_directions <= 0:
        raise ValueError("num_directions must be positive")

    flat_params, unravel = ravel_pytree(params)
    _validate_real_flat_params(flat_params)
    dim = flat_params.size
    if dim == 0:
        raise ValueError("params must contain at least one scalar")

    if directions is not None:
        directions = jnp.asarray(directions, dtype=flat_params.dtype)
        if directions.shape != (num_directions, dim):
            raise ValueError("directions must have shape (num_directions, flattened_param_size)")

    grad_acc = jnp.zeros_like(flat_params)
    next_key = key
    eps_value = jnp.asarray(eps, dtype=flat_params.dtype)

    for index in range(num_directions):
        if directions is None:
            next_key, direction_key = jax.random.split(next_key)
            direction = _normalized_random_direction(direction_key, flat_params.shape, flat_params.dtype)
        else:
            direction = directions[index]

        loss_pos = loss_fn(unravel(flat_params + eps_value * direction))
        loss_neg = loss_fn(unravel(flat_params - eps_value * direction))
        directional_slope = (loss_pos - loss_neg) / (2.0 * eps_value)
        grad_acc = grad_acc + directional_slope.astype(flat_params.dtype) * direction

    flat_gradient = grad_acc * (jnp.asarray(dim, dtype=flat_params.dtype) / jnp.asarray(num_directions, dtype=flat_params.dtype))
    return DirectionalDerivativeResult(
        params=params,
        gradient=unravel(flat_gradient),
        loss=loss_fn(params),
        key=next_key,
    )


def bp_dd_step(
    params,
    loss_fn: Callable[[object], jnp.ndarray],
    *,
    key: jax.Array,
    eps: float,
    learning_rate: float,
    num_directions: int,
    directions: jnp.ndarray | None = None,
) -> DirectionalDerivativeResult:
    """Apply one FFzero BP+DD update to a full parameter PyTree."""

    result = directional_derivative_gradient(
        params,
        loss_fn,
        key=key,
        eps=eps,
        num_directions=num_directions,
        directions=directions,
    )
    updated = jax.tree_util.tree_map(
        lambda param, grad: param - jnp.asarray(learning_rate, dtype=param.dtype) * grad,
        params,
        result.gradient,
    )
    return DirectionalDerivativeResult(
        params=updated,
        gradient=result.gradient,
        loss=result.loss,
        key=result.key,
    )
