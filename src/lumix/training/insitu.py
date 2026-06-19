"""Physical Clements phase training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from lumix.functional.clements import build_clements_spec, clements_pair


@dataclass(frozen=True)
class InSituStepResult:
    params: dict[str, jnp.ndarray]
    gradients: dict[str, jnp.ndarray]
    loss: jnp.ndarray


def _clements_outputs(params: dict[str, jnp.ndarray], inputs: jnp.ndarray) -> jnp.ndarray:
    width = inputs.shape[-1]
    depth = params["theta"].shape[0]
    return clements_pair(
        inputs,
        params["theta"],
        params["phi"],
        params["gamma"],
        spec=build_clements_spec(width, depth),
    )


def clements_square_law_logits(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    classes: int | None = None,
) -> jnp.ndarray:
    """Return square-law detected Clements outputs as real logits."""

    outputs = _clements_outputs(params, inputs)
    logits = jnp.square(jnp.abs(outputs)).astype(jnp.float32)
    return logits if classes is None else logits[:, :classes]


def insitu_mse_gradients(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Return exact Clements phase gradients for an MSE optical-field loss."""

    def loss_fn(current):
        outputs = _clements_outputs(current, inputs)
        return jnp.mean(jnp.square(jnp.abs(outputs - targets)))

    return jax.grad(loss_fn)(params)


def insitu_classification_step(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    learning_rate: float,
) -> InSituStepResult:
    """Apply one physical Clements phase update for square-law classification."""

    classes = labels.shape[-1]

    def loss_fn(current):
        logits = clements_square_law_logits(current, inputs, classes=classes)
        return optax.softmax_cross_entropy(logits, labels).mean()

    loss, gradients = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(
        lambda param, grad: param - jnp.asarray(learning_rate, dtype=param.dtype) * grad,
        params,
        gradients,
    )
    return InSituStepResult(params=updated, gradients=gradients, loss=loss)
