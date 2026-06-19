"""Routing-locality utilities for optical transfer matrices."""

from collections.abc import Sequence

import jax.numpy as jnp


RoutingLimit = int | tuple[int, int]


def _routing_bounds(routing_limit: RoutingLimit) -> tuple[int, int]:
    if isinstance(routing_limit, Sequence) and not isinstance(routing_limit, (str, bytes)):
        if len(routing_limit) != 2:
            raise ValueError("routing_limit bounds must contain exactly two values")
        left, right = int(routing_limit[0]), int(routing_limit[1])
    else:
        left = right = int(routing_limit)

    if left < 0 or right < 0:
        raise ValueError("routing_limit values must be non-negative")
    return left, right


def routing_mask(
    output_features: int,
    input_features: int,
    routing_limit: RoutingLimit,
) -> jnp.ndarray:
    """Return the boolean local-routing mask for an output-by-input matrix.

    `routing_limit=n` allows input port `i` to route to output ports `j`
    satisfying `i - n <= j <= i + n`. A tuple `(left, right)` allows
    asymmetric routing with `i - left <= j <= i + right`.
    """

    left, right = _routing_bounds(routing_limit)
    output_ports = jnp.arange(output_features)[:, None]
    input_ports = jnp.arange(input_features)[None, :]
    return (output_ports >= input_ports - left) & (output_ports <= input_ports + right)


def routing_leakage(
    matrix: jnp.ndarray,
    routing_limit: RoutingLimit,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """Return fractional transmitted power outside the allowed routing band.

    The matrix is interpreted as `matrix[output_port, input_port]`. For each
    input port, the function computes outside-band power divided by total
    transmitted power, then averages over input ports. This fractional form is
    appropriate for subunitary layers because it cannot be reduced by adding
    uniform insertion loss.
    """

    allowed = routing_mask(matrix.shape[0], matrix.shape[1], routing_limit)
    outside = 1.0 - allowed.astype(jnp.float32)
    power = jnp.square(jnp.abs(matrix)).astype(jnp.float32)
    outside_power = jnp.sum(outside * power, axis=0)
    total_power = jnp.sum(power, axis=0)
    return jnp.mean(outside_power / jnp.maximum(total_power, epsilon))
