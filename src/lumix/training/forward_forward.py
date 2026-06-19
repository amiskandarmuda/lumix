"""FFzero Forward-Forward losses and train steps."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from lumix.training.directional import DirectionalDerivativeResult, bp_dd_step


LossMode = str


def _l2_normalize_reference(values: jnp.ndarray) -> jnp.ndarray:
    return values / jnp.linalg.norm(values, axis=1, keepdims=True)


def ffzero_margin_loss(
    activations: jnp.ndarray,
    labels: jnp.ndarray,
    references: jnp.ndarray,
    *,
    margin: float = 0.3,
) -> jnp.ndarray:
    """FFzero MLP/CNN local multi-class margin loss."""

    normalized = _l2_normalize_reference(activations)
    similarities = normalized @ references.T.astype(normalized.dtype)
    true_sim = similarities[jnp.arange(labels.shape[0]), labels]
    return jnp.maximum(jnp.asarray(margin, dtype=similarities.dtype) + similarities - true_sim[:, None], 0.0).sum()


def ffzero_onn_simplex_loss(
    activations: jnp.ndarray,
    labels: jnp.ndarray,
    references: jnp.ndarray,
) -> jnp.ndarray:
    """FFzero ONN local simplex loss: ``mean(1 - true_sim)``."""

    normalized = _l2_normalize_reference(activations)
    similarities = normalized @ references.T.astype(normalized.dtype)
    true_sim = jnp.sum(similarities * labels, axis=1)
    return jnp.mean(1.0 - true_sim)


def _local_loss(mode: LossMode, activations: jnp.ndarray, labels: jnp.ndarray, references: jnp.ndarray) -> jnp.ndarray:
    if mode == "ffzero_margin":
        return ffzero_margin_loss(activations, labels, references)
    if mode == "ffzero_onn_simplex":
        return ffzero_onn_simplex_loss(activations, labels, references)
    raise ValueError("loss_mode must be 'ffzero_margin' or 'ffzero_onn_simplex'")


def ff_ad_step(
    params,
    apply_fn: Callable[[object, jnp.ndarray], jnp.ndarray],
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    references: jnp.ndarray,
    *,
    learning_rate: float,
    loss_mode: LossMode,
):
    """Apply one FFzero FF+AD local update."""

    def loss_fn(current):
        return _local_loss(loss_mode, apply_fn(current, batch_x), batch_y, references)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(
        lambda param, grad: param - jnp.asarray(learning_rate, dtype=param.dtype) * grad,
        params,
        grads,
    )
    return updated, loss


def ff_dd_step(
    params,
    apply_fn: Callable[[object, jnp.ndarray], jnp.ndarray],
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    references: jnp.ndarray,
    *,
    key: jax.Array,
    eps: float,
    learning_rate: float,
    num_directions: int,
    loss_mode: LossMode,
    directions: jnp.ndarray | None = None,
) -> DirectionalDerivativeResult:
    """Apply one FFzero FF+DD local update."""

    def loss_fn(current):
        return _local_loss(loss_mode, apply_fn(current, batch_x), batch_y, references)

    return bp_dd_step(
        params,
        loss_fn,
        key=key,
        eps=eps,
        learning_rate=learning_rate,
        num_directions=num_directions,
        directions=directions,
    )
