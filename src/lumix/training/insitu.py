"""Physical Clements phase training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import optax

from lumix.functional.clements import (
    build_clements_spec,
    clements_pair,
    differential_layout,
    mask_phases,
    stripe_layout,
)


@dataclass(frozen=True)
class InSituStepResult:
    params: dict[str, jnp.ndarray]
    gradients: dict[str, jnp.ndarray]
    loss: jnp.ndarray


@dataclass(frozen=True)
class _ClementsFieldTrace:
    outputs: jnp.ndarray
    gamma_after: jnp.ndarray
    internal_after: tuple[jnp.ndarray, ...]
    output_after: tuple[jnp.ndarray, ...]
    internal: jnp.ndarray
    output: jnp.ndarray
    perm: jnp.ndarray
    mask: jnp.ndarray


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


def _pair_coupler(values: jnp.ndarray) -> jnp.ndarray:
    pair_width = 2 * (values.shape[-1] // 2)
    even = values[..., :pair_width:2]
    odd = values[..., 1:pair_width:2]
    scale = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=values.real.dtype)
    next_even = scale * (even + 1j * odd)
    next_odd = scale * (1j * even + odd)
    next_values = values
    next_values = next_values.at[..., :pair_width:2].set(next_even)
    next_values = next_values.at[..., 1:pair_width:2].set(next_odd)
    return next_values


def _transpose_permutation(values: jnp.ndarray, permutation: jnp.ndarray) -> jnp.ndarray:
    next_values = jnp.empty_like(values)
    return next_values.at[..., permutation].set(values)


def _phase_gradient(forward_after_phase: jnp.ndarray, adjoint_after_phase: jnp.ndarray) -> jnp.ndarray:
    interference = -jnp.imag(forward_after_phase * adjoint_after_phase)
    leading_axes = tuple(range(interference.ndim - 1))
    return jnp.sum(interference, axis=leading_axes)


def _clements_field_trace(params: dict[str, jnp.ndarray], inputs: jnp.ndarray) -> _ClementsFieldTrace:
    width = inputs.shape[-1]
    depth = params["theta"].shape[0]
    spec = build_clements_spec(width, depth)
    theta_masked, phi_masked = mask_phases(params["theta"], params["phi"], spec.mask)
    internal = differential_layout(theta_masked, width)
    output = stripe_layout(phi_masked, width)

    gamma_after = inputs * jnp.exp(1j * params["gamma"])
    values = gamma_after[..., spec.perm[0]]
    internal_after = []
    output_after = []

    for layer_index in range(depth):
        values = _pair_coupler(values)
        values = values * jnp.exp(1j * internal[:, layer_index])
        internal_after.append(values)
        values = _pair_coupler(values)
        values = values * jnp.exp(1j * output[:, layer_index])
        output_after.append(values)
        values = values[..., spec.perm[layer_index + 1]]

    return _ClementsFieldTrace(
        outputs=values,
        gamma_after=gamma_after,
        internal_after=tuple(internal_after),
        output_after=tuple(output_after),
        internal=internal,
        output=output,
        perm=spec.perm,
        mask=spec.mask,
    )


def _clements_phase_gradients_from_adjoint(
    trace: _ClementsFieldTrace,
    adjoint_output: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    depth = trace.internal.shape[1]
    width = trace.internal.shape[0]
    adjoint = adjoint_output
    internal_gradient_columns = []
    output_gradient_columns = []

    for layer_index in reversed(range(depth)):
        adjoint = _transpose_permutation(adjoint, trace.perm[layer_index + 1])

        output_gradient_columns.append(_phase_gradient(trace.output_after[layer_index], adjoint))
        adjoint = adjoint * jnp.exp(1j * trace.output[:, layer_index])
        adjoint = _pair_coupler(adjoint)

        internal_gradient_columns.append(_phase_gradient(trace.internal_after[layer_index], adjoint))
        adjoint = adjoint * jnp.exp(1j * trace.internal[:, layer_index])
        adjoint = _pair_coupler(adjoint)

    adjoint = _transpose_permutation(adjoint, trace.perm[0])
    gamma_grad = _phase_gradient(trace.gamma_after, adjoint)[None, :]
    internal_eff_grad = jnp.stack(internal_gradient_columns[::-1], axis=1)
    output_eff_grad = jnp.stack(output_gradient_columns[::-1], axis=1)
    theta_stripe_grad = 0.5 * (internal_eff_grad - jnp.roll(internal_eff_grad, -1, axis=0))
    theta_grad = theta_stripe_grad[: width - 1 : 2, :].T * trace.mask
    phi_grad = output_eff_grad[: width - 1 : 2, :].T * trace.mask
    return {
        "theta": theta_grad.astype(jnp.float32),
        "phi": phi_grad.astype(jnp.float32),
        "gamma": gamma_grad.astype(jnp.float32),
    }


def _stable_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=-1, keepdims=True)


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

    trace = _clements_field_trace(params, inputs)
    cotangent = 2.0 * (trace.outputs - targets) / trace.outputs.size
    return _clements_phase_gradients_from_adjoint(trace, jnp.conj(cotangent))


def insitu_classification_step(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    learning_rate: float,
) -> InSituStepResult:
    """Apply one physical Clements phase update for square-law classification."""

    classes = labels.shape[-1]
    trace = _clements_field_trace(params, inputs)
    logits = jnp.square(jnp.abs(trace.outputs)).astype(jnp.float32)[:, :classes]
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    dlogits = (_stable_softmax(logits) - labels) / labels.shape[0]
    cotangent = jnp.zeros_like(trace.outputs)
    cotangent = cotangent.at[:, :classes].set(2.0 * trace.outputs[:, :classes] * dlogits)
    gradients = _clements_phase_gradients_from_adjoint(trace, jnp.conj(cotangent))
    updated = {
        name: param - jnp.asarray(learning_rate, dtype=param.dtype) * gradients[name]
        for name, param in params.items()
    }
    return InSituStepResult(params=updated, gradients=gradients, loss=loss)
