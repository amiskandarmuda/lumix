from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from flax.training.train_state import TrainState

from lumix.batching import iterate_batches
from lumix.losses import cross_entropy_logits
from lumix.metrics import accuracy
from lumix.params import freeze_params
from lumix.state import apply_gradients
from experiments.common.models import surrogate_routing_leakage_from_params
from experiments.common.robustness import noisy_surrogate_forward


class RoutingLossParts(NamedTuple):
    cross_entropy: jnp.ndarray
    routing_leakage: jnp.ndarray
    routing_excess: jnp.ndarray
    mean_insertion_loss_db: jnp.ndarray
    loss_excess: jnp.ndarray
    mean_output_power: jnp.ndarray
    gamma: jnp.ndarray


class PrefixRoutingLossParts(NamedTuple):
    cross_entropy: jnp.ndarray
    routing_leakage: jnp.ndarray
    routing_excess: jnp.ndarray
    mean_insertion_loss_db: jnp.ndarray
    loss_excess: jnp.ndarray
    mean_output_power: jnp.ndarray
    gamma: jnp.ndarray


def power_loss_db(intensities: jnp.ndarray) -> jnp.ndarray:
    total_power = jnp.sum(intensities, axis=-1)
    return -10.0 * jnp.log10(jnp.clip(total_power, 1e-12, None))


def mean_output_power(intensities: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sum(intensities, axis=-1))


def shared_prefix_readout_name(depth: int) -> str:
    return f"readout_depth_{int(depth)}"


def normalize_prefix_weights(
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...]:
    depths = tuple(int(depth) for depth in prefix_depths)
    if not depths:
        raise ValueError("prefix_depths must contain at least one depth")
    weights = tuple(1.0 for _ in depths) if prefix_weights is None else tuple(float(weight) for weight in prefix_weights)
    if len(weights) != len(depths):
        raise ValueError("prefix_weights must match prefix_depths")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("prefix_weights must sum to a positive value")
    return tuple(weight / total for weight in weights)


def normalize_prefix_targets(
    prefix_depths: tuple[int, ...] | list[int],
    target_accuracies: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...] | None:
    if target_accuracies is None:
        return None
    depths = tuple(int(depth) for depth in prefix_depths)
    targets = tuple(float(target) for target in target_accuracies)
    if len(targets) != len(depths):
        raise ValueError("target_accuracies must match prefix_depths")
    if any(target < 0.0 or target > 1.0 for target in targets):
        raise ValueError("target_accuracies must be probabilities in [0, 1]")
    return targets


def distillation_kl_logits(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    temperature_value = jnp.asarray(temperature, dtype=student_logits.dtype)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature_value, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature_value, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    per_example_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    return jnp.square(temperature_value) * jnp.mean(per_example_kl)


def class_margin_loss(
    labels: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    target_margin: float,
) -> jnp.ndarray:
    true_logits = jnp.sum(labels * logits, axis=-1)
    other_logits = jnp.max(jnp.where(labels > 0.0, -jnp.inf, logits), axis=-1)
    margins = true_logits - other_logits
    deficits = jnp.maximum(jnp.asarray(target_margin, dtype=logits.dtype) - margins, 0.0)
    return jnp.mean(jnp.square(deficits))


def create_shared_prefix_state(
    model,
    rng,
    sample_x: jnp.ndarray,
    *,
    learning_rate: float,
    prefix_depths: tuple[int, ...] | list[int],
) -> TrainState:
    variables = model.init(rng, sample_x)
    params = unfreeze(variables["params"])
    constants = {name: value for name, value in variables.items() if name != "params"}

    for depth in tuple(int(depth) for depth in prefix_depths):
        readout_name = shared_prefix_readout_name(depth)
        depth_rng = jax.random.fold_in(rng, depth)
        depth_variables = model.init(
            depth_rng,
            sample_x,
            depth=depth,
            readout_name=readout_name,
        )
        params[readout_name] = unfreeze(depth_variables["params"])[readout_name]

    optimizer = optax.adam(learning_rate)

    def apply_fn(variable_dict, batch_x):
        return model.apply({**constants, **variable_dict}, batch_x)

    return TrainState.create(apply_fn=apply_fn, params=freeze_params(params), tx=optimizer)


def _routing_regularized_loss_and_logits(
    model,
    params,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    *,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    depth: int | None = None,
    readout_name: str | None = None,
):
    logits, aux = model.apply(
        {"params": params},
        batch_x,
        return_aux=True,
        depth=depth,
        readout_name=readout_name,
    )
    cross_entropy = cross_entropy_logits(batch_y, logits)
    routing_leakage = surrogate_routing_leakage_from_params(model.config, params, depth=depth)
    routing_excess = jnp.maximum(
        routing_leakage - jnp.asarray(routing_target, dtype=routing_leakage.dtype),
        jnp.asarray(0.0, dtype=routing_leakage.dtype),
    )
    mean_insertion_loss_db = jnp.mean(power_loss_db(aux["intensities"]))
    if loss_guard_db is None:
        loss_excess = jnp.asarray(0.0, dtype=cross_entropy.dtype)
    else:
        loss_excess = jnp.maximum(
            mean_insertion_loss_db - jnp.asarray(loss_guard_db, dtype=mean_insertion_loss_db.dtype),
            jnp.asarray(0.0, dtype=mean_insertion_loss_db.dtype),
        )
    total = (
        cross_entropy
        + jnp.asarray(routing_weight, dtype=cross_entropy.dtype) * routing_excess
        + jnp.asarray(loss_guard_weight, dtype=cross_entropy.dtype) * loss_excess**2
    )
    parts = RoutingLossParts(
        cross_entropy=cross_entropy,
        routing_leakage=routing_leakage,
        routing_excess=routing_excess,
        mean_insertion_loss_db=mean_insertion_loss_db,
        loss_excess=loss_excess,
        mean_output_power=mean_output_power(aux["intensities"]),
        gamma=aux["gamma"],
    )
    return total, parts, logits


def routing_regularized_loss(
    model,
    params,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    *,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    depth: int | None = None,
    readout_name: str | None = None,
):
    total, parts, _ = _routing_regularized_loss_and_logits(
        model,
        params,
        batch_x,
        batch_y,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
        depth=depth,
        readout_name=readout_name,
    )
    return total, parts


def _stack_prefix_parts(parts_by_depth: list[RoutingLossParts]) -> PrefixRoutingLossParts:
    return PrefixRoutingLossParts(
        *(jnp.stack([getattr(parts, field) for parts in parts_by_depth]) for field in RoutingLossParts._fields)
    )


def shared_prefix_routing_regularized_loss(
    model,
    params,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    *,
    prefix_depths: tuple[int, ...],
    prefix_weights: tuple[float, ...],
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    totals = []
    parts_by_depth = []
    logits_by_depth = []
    for depth in prefix_depths:
        total, parts, logits = _routing_regularized_loss_and_logits(
            model,
            params,
            batch_x,
            batch_y,
            routing_weight=routing_weight,
            routing_target=routing_target,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
            depth=depth,
            readout_name=shared_prefix_readout_name(depth),
        )
        totals.append(total)
        parts_by_depth.append(parts)
        logits_by_depth.append(logits)

    weights = jnp.asarray(prefix_weights, dtype=jnp.asarray(totals[0]).dtype)
    total_loss = jnp.sum(jnp.stack(totals) * weights)
    return total_loss, _stack_prefix_parts(parts_by_depth), tuple(logits_by_depth)


def shared_prefix_distilled_loss(
    model,
    params,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    teacher_logits_by_depth: jnp.ndarray,
    *,
    prefix_depths: tuple[int, ...],
    prefix_weights: tuple[float, ...],
    routing_weight: float,
    routing_target: float,
    distillation_alpha: float,
    distillation_temperature: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    totals = []
    parts_by_depth = []
    logits_by_depth = []
    distillation_by_depth = []
    alpha = jnp.asarray(distillation_alpha, dtype=jnp.float32)
    for index, depth in enumerate(prefix_depths):
        _, parts, logits = _routing_regularized_loss_and_logits(
            model,
            params,
            batch_x,
            batch_y,
            routing_weight=routing_weight,
            routing_target=routing_target,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
            depth=depth,
            readout_name=shared_prefix_readout_name(depth),
        )
        distillation_kl = distillation_kl_logits(
            logits,
            teacher_logits_by_depth[index],
            temperature=distillation_temperature,
        )
        classification = (1.0 - alpha) * parts.cross_entropy + alpha * distillation_kl
        total = (
            classification
            + jnp.asarray(routing_weight, dtype=classification.dtype) * parts.routing_excess
            + jnp.asarray(loss_guard_weight, dtype=classification.dtype) * parts.loss_excess**2
        )
        totals.append(total)
        parts_by_depth.append(parts)
        logits_by_depth.append(logits)
        distillation_by_depth.append(distillation_kl)

    weights = jnp.asarray(prefix_weights, dtype=jnp.asarray(totals[0]).dtype)
    total_loss = jnp.sum(jnp.stack(totals) * weights)
    return (
        total_loss,
        _stack_prefix_parts(parts_by_depth),
        tuple(logits_by_depth),
        jnp.stack(distillation_by_depth),
    )


def make_routing_regularized_train_step(
    model,
    *,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    @jax.jit
    def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        def loss_fn(params):
            total, parts, logits = _routing_regularized_loss_and_logits(
                model,
                params,
                batch_x,
                batch_y,
                routing_weight=routing_weight,
                routing_target=routing_target,
                loss_guard_db=loss_guard_db,
                loss_guard_weight=loss_guard_weight,
            )
            return total, (parts, logits)

        (loss_value, (parts, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        next_state = apply_gradients(state, grads)
        score = accuracy(batch_y, logits)
        return next_state, loss_value, score, parts

    return train_step


def make_matrix_noise_regularized_train_step(
    model,
    *,
    routing_weight: float,
    routing_target: float,
    relative_error: float,
    noise_samples: int = 1,
    noisy_weight: float = 1.0,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    rescale_to_passive: bool = False,
):
    if relative_error < 0.0:
        raise ValueError("relative_error must be non-negative")
    if noise_samples < 1:
        raise ValueError("noise_samples must be at least 1")
    if noisy_weight < 0.0:
        raise ValueError("noisy_weight must be non-negative")

    @jax.jit
    def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, rng):
        def loss_fn(params):
            clean_total, parts, clean_logits = _routing_regularized_loss_and_logits(
                model,
                params,
                batch_x,
                batch_y,
                routing_weight=routing_weight,
                routing_target=routing_target,
                loss_guard_db=loss_guard_db,
                loss_guard_weight=loss_guard_weight,
            )

            noise_keys = jax.random.split(rng, noise_samples)

            def noisy_cross_entropy_for_key(key):
                noisy_logits = noisy_surrogate_forward(
                    model,
                    params,
                    batch_x,
                    key,
                    relative_error=relative_error,
                    rescale_to_passive=rescale_to_passive,
                )
                return cross_entropy_logits(batch_y, noisy_logits)

            noisy_cross_entropy = jnp.mean(jax.vmap(noisy_cross_entropy_for_key)(noise_keys))
            total = clean_total + jnp.asarray(noisy_weight, dtype=clean_total.dtype) * noisy_cross_entropy
            return total, (parts, clean_logits, noisy_cross_entropy)

        (loss_value, (parts, clean_logits, noisy_cross_entropy)), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        next_state = apply_gradients(state, grads)
        score = accuracy(batch_y, clean_logits)
        return next_state, loss_value, score, parts, noisy_cross_entropy

    return train_step


def make_matrix_noise_distilled_train_step(
    model,
    *,
    routing_weight: float,
    routing_target: float,
    relative_error: float,
    noise_samples: int = 1,
    noisy_weight: float = 1.0,
    distillation_weight: float = 1.0,
    distillation_temperature: float = 2.0,
    margin_weight: float = 0.0,
    margin_target: float = 0.0,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    rescale_to_passive: bool = False,
):
    if relative_error < 0.0:
        raise ValueError("relative_error must be non-negative")
    if noise_samples < 1:
        raise ValueError("noise_samples must be at least 1")
    if noisy_weight < 0.0:
        raise ValueError("noisy_weight must be non-negative")
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight must be non-negative")
    if margin_weight < 0.0:
        raise ValueError("margin_weight must be non-negative")
    if margin_target < 0.0:
        raise ValueError("margin_target must be non-negative")
    _validate_distillation_hyperparameters(
        distillation_alpha=0.5,
        distillation_temperature=distillation_temperature,
    )

    @jax.jit
    def train_step(
        state: TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        rng,
    ):
        def loss_fn(params):
            clean_total, parts, clean_logits = _routing_regularized_loss_and_logits(
                model,
                params,
                batch_x,
                batch_y,
                routing_weight=routing_weight,
                routing_target=routing_target,
                loss_guard_db=loss_guard_db,
                loss_guard_weight=loss_guard_weight,
            )

            noise_keys = jax.random.split(rng, noise_samples)

            def noisy_cross_entropy_for_key(key):
                noisy_logits = noisy_surrogate_forward(
                    model,
                    params,
                    batch_x,
                    key,
                    relative_error=relative_error,
                    rescale_to_passive=rescale_to_passive,
                )
                return cross_entropy_logits(batch_y, noisy_logits)

            noisy_cross_entropy = jnp.mean(jax.vmap(noisy_cross_entropy_for_key)(noise_keys))
            distillation_kl = distillation_kl_logits(
                clean_logits,
                teacher_logits,
                temperature=distillation_temperature,
            )
            raw_logits = clean_logits / jnp.maximum(parts.gamma, jnp.asarray(1e-6, dtype=clean_logits.dtype))
            margin_penalty = class_margin_loss(batch_y, raw_logits, target_margin=margin_target)
            total = (
                clean_total
                + jnp.asarray(noisy_weight, dtype=clean_total.dtype) * noisy_cross_entropy
                + jnp.asarray(distillation_weight, dtype=clean_total.dtype) * distillation_kl
                + jnp.asarray(margin_weight, dtype=clean_total.dtype) * margin_penalty
            )
            return total, (parts, clean_logits, noisy_cross_entropy, distillation_kl, margin_penalty)

        (loss_value, (parts, clean_logits, noisy_cross_entropy, distillation_kl, margin_penalty)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        )
        next_state = apply_gradients(state, grads)
        score = accuracy(batch_y, clean_logits)
        return next_state, loss_value, score, parts, noisy_cross_entropy, distillation_kl, margin_penalty

    return train_step


def _validate_distillation_hyperparameters(
    *,
    distillation_alpha: float,
    distillation_temperature: float,
) -> None:
    if distillation_alpha < 0.0 or distillation_alpha > 1.0:
        raise ValueError("distillation_alpha must be in [0, 1]")
    if distillation_temperature <= 0.0:
        raise ValueError("distillation_temperature must be positive")


def make_shared_prefix_routing_regularized_train_step(
    model,
    *,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)

    @jax.jit
    def train_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        def loss_fn(params):
            total, parts, logits_by_depth = shared_prefix_routing_regularized_loss(
                model,
                params,
                batch_x,
                batch_y,
                prefix_depths=depths,
                prefix_weights=weights,
                routing_weight=routing_weight,
                routing_target=routing_target,
                loss_guard_db=loss_guard_db,
                loss_guard_weight=loss_guard_weight,
            )
            return total, (parts, logits_by_depth)

        (loss_value, (parts, logits_by_depth)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        next_state = apply_gradients(state, grads)
        scores = jnp.stack([accuracy(batch_y, logits) for logits in logits_by_depth])
        return next_state, loss_value, scores, parts

    return train_step


def make_shared_prefix_distilled_train_step(
    model,
    *,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    distillation_alpha: float,
    distillation_temperature: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    _validate_distillation_hyperparameters(
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
    )
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)

    @jax.jit
    def train_step(
        state: TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        teacher_logits_by_depth: jnp.ndarray,
    ):
        def loss_fn(params):
            total, parts, logits_by_depth, distillation_kl = shared_prefix_distilled_loss(
                model,
                params,
                batch_x,
                batch_y,
                teacher_logits_by_depth,
                prefix_depths=depths,
                prefix_weights=weights,
                routing_weight=routing_weight,
                routing_target=routing_target,
                distillation_alpha=distillation_alpha,
                distillation_temperature=distillation_temperature,
                loss_guard_db=loss_guard_db,
                loss_guard_weight=loss_guard_weight,
            )
            return total, (parts, logits_by_depth, distillation_kl)

        (loss_value, (parts, logits_by_depth, distillation_kl)), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        next_state = apply_gradients(state, grads)
        scores = jnp.stack([accuracy(batch_y, logits) for logits in logits_by_depth])
        return next_state, loss_value, scores, parts, distillation_kl

    return train_step


def make_routing_regularized_eval_step(
    model,
    *,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    @jax.jit
    def eval_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        total, parts = routing_regularized_loss(
            model,
            state.params,
            batch_x,
            batch_y,
            routing_weight=routing_weight,
            routing_target=routing_target,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
        )
        logits = model.apply({"params": state.params}, batch_x)
        return total, accuracy(batch_y, logits), parts

    return eval_step


def make_shared_prefix_routing_regularized_eval_step(
    model,
    *,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)

    @jax.jit
    def eval_step(state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        total, parts, logits_by_depth = shared_prefix_routing_regularized_loss(
            model,
            state.params,
            batch_x,
            batch_y,
            prefix_depths=depths,
            prefix_weights=weights,
            routing_weight=routing_weight,
            routing_target=routing_target,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
        )
        scores = jnp.stack([accuracy(batch_y, logits) for logits in logits_by_depth])
        return total, scores, parts

    return eval_step


def make_shared_prefix_distilled_eval_step(
    model,
    *,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    distillation_alpha: float,
    distillation_temperature: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
):
    _validate_distillation_hyperparameters(
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
    )
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)

    @jax.jit
    def eval_step(
        state: TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        teacher_logits_by_depth: jnp.ndarray,
    ):
        total, parts, logits_by_depth, distillation_kl = shared_prefix_distilled_loss(
            model,
            state.params,
            batch_x,
            batch_y,
            teacher_logits_by_depth,
            prefix_depths=depths,
            prefix_weights=weights,
            routing_weight=routing_weight,
            routing_target=routing_target,
            distillation_alpha=distillation_alpha,
            distillation_temperature=distillation_temperature,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
        )
        scores = jnp.stack([accuracy(batch_y, logits) for logits in logits_by_depth])
        return total, scores, parts, distillation_kl

    return eval_step


def fit_routing_regularized_logits(
    model,
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    *,
    epochs: int,
    batch_size: int,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    select_best_checkpoint: bool = False,
    checkpoint_epochs: list[int] | tuple[int, ...] | None = None,
    seed: int = 0,
):
    train_step = make_routing_regularized_train_step(
        model,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    eval_step = make_routing_regularized_eval_step(
        model,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "cross_entropy": [],
        "routing_leakage": [],
        "routing_excess": [],
        "mean_insertion_loss_db": [],
        "loss_excess": [],
        "mean_output_power": [],
        "gamma": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_cross_entropy": [],
        "val_routing_leakage": [],
        "val_routing_excess": [],
        "val_mean_insertion_loss_db": [],
        "val_loss_excess": [],
        "val_mean_output_power": [],
        "val_gamma": [],
    }
    rng = jax.random.key(seed)
    selected_epochs = set(checkpoint_epochs or range(1, epochs + 1))
    selected_epochs.add(epochs)
    best_state = state
    best_record = None

    for epoch in range(1, epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        part_values = {field: [] for field in RoutingLossParts._fields}
        for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng):
            state, loss_value, score, parts = train_step(state, batch_x, batch_y)
            losses.append(loss_value)
            scores.append(score)
            for field in RoutingLossParts._fields:
                part_values[field].append(getattr(parts, field))

        val_loss, val_score, val_parts = eval_step(state, test_x, test_y)
        record = {
            "epoch": epoch,
            "loss": float(jnp.mean(jnp.stack(losses))),
            "accuracy": float(jnp.mean(jnp.stack(scores))),
            "cross_entropy": float(jnp.mean(jnp.stack(part_values["cross_entropy"]))),
            "routing_leakage": float(jnp.mean(jnp.stack(part_values["routing_leakage"]))),
            "routing_excess": float(jnp.mean(jnp.stack(part_values["routing_excess"]))),
            "mean_insertion_loss_db": float(jnp.mean(jnp.stack(part_values["mean_insertion_loss_db"]))),
            "loss_excess": float(jnp.mean(jnp.stack(part_values["loss_excess"]))),
            "mean_output_power": float(jnp.mean(jnp.stack(part_values["mean_output_power"]))),
            "gamma": float(jnp.mean(jnp.stack(part_values["gamma"]))),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_score),
            "val_cross_entropy": float(val_parts.cross_entropy),
            "val_routing_leakage": float(val_parts.routing_leakage),
            "val_routing_excess": float(val_parts.routing_excess),
            "val_mean_insertion_loss_db": float(val_parts.mean_insertion_loss_db),
            "val_loss_excess": float(val_parts.loss_excess),
            "val_mean_output_power": float(val_parts.mean_output_power),
            "val_gamma": float(val_parts.gamma),
        }
        for key, value in record.items():
            history[key].append(value)
        if select_best_checkpoint and epoch in selected_epochs:
            if best_record is None or record["val_accuracy"] > best_record["val_accuracy"]:
                best_state = state
                best_record = dict(record)

    if select_best_checkpoint:
        if best_record is None:
            best_record = {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}
            best_state = state
        history["selected_epoch"] = best_record["epoch"]
        history["selected_metrics"] = {key: value for key, value in best_record.items() if key != "epoch"}
        history["selected"] = [epoch == best_record["epoch"] for epoch in history["epoch"]]
        state = best_state

    return state, history


def _shared_prefix_history(prefix_depths: tuple[int, ...], prefix_weights: tuple[float, ...]) -> dict:
    history = {
        "metadata": {
            "prefix_depths": list(prefix_depths),
            "prefix_weights": list(prefix_weights),
        },
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "cross_entropy": [],
        "routing_leakage": [],
        "routing_excess": [],
        "mean_insertion_loss_db": [],
        "loss_excess": [],
        "mean_output_power": [],
        "gamma": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_cross_entropy": [],
        "val_routing_leakage": [],
        "val_routing_excess": [],
        "val_mean_insertion_loss_db": [],
        "val_loss_excess": [],
        "val_mean_output_power": [],
        "val_gamma": [],
    }
    for depth in prefix_depths:
        for field in ("accuracy", *RoutingLossParts._fields):
            history[f"prefix_{depth}_{field}"] = []
            history[f"prefix_{depth}_val_{field}"] = []
    return history


def _add_target_margin_history_fields(history: dict, target_accuracies: tuple[float, ...] | None) -> None:
    if target_accuracies is None:
        return
    history["metadata"]["target_accuracies"] = list(target_accuracies)
    history["min_target_margin"] = []
    history["mean_target_margin"] = []


def _add_distillation_history_fields(history: dict, prefix_depths: tuple[int, ...]) -> None:
    history["distillation_kl"] = []
    history["val_distillation_kl"] = []
    for depth in prefix_depths:
        history[f"prefix_{depth}_distillation_kl"] = []
        history[f"prefix_{depth}_val_distillation_kl"] = []


def _teacher_logits_array(
    teacher_logits_by_depth,
    prefix_depths: tuple[int, ...],
    *,
    expected_examples: int,
    expected_classes: int,
    name: str,
) -> jnp.ndarray:
    if isinstance(teacher_logits_by_depth, Mapping):
        stacked_values = []
        for depth in prefix_depths:
            if depth in teacher_logits_by_depth:
                stacked_values.append(teacher_logits_by_depth[depth])
            elif str(depth) in teacher_logits_by_depth:
                stacked_values.append(teacher_logits_by_depth[str(depth)])
            else:
                stacked_values.append(teacher_logits_by_depth[f"depth_{depth}"])
        logits = jnp.stack([jnp.asarray(value) for value in stacked_values])
    elif isinstance(teacher_logits_by_depth, tuple | list):
        logits = jnp.stack([jnp.asarray(value) for value in teacher_logits_by_depth])
    else:
        logits = jnp.asarray(teacher_logits_by_depth)

    if logits.ndim != 3:
        raise ValueError(f"{name} must have shape (depths, examples, classes)")
    if logits.shape[0] != len(prefix_depths):
        raise ValueError(f"{name} depth count must match prefix_depths")
    if logits.shape[1] != expected_examples:
        raise ValueError(f"{name} example count must match the corresponding dataset")
    if logits.shape[2] != expected_classes:
        raise ValueError(f"{name} class count must match labels")
    return logits


def _iterate_batches_with_teacher_logits(
    x: jnp.ndarray,
    y: jnp.ndarray,
    teacher_logits_by_depth: jnp.ndarray,
    batch_size: int,
    rng,
):
    indices = jax.random.permutation(rng, x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices], teacher_logits_by_depth[:, batch_indices, :]


def _weighted_value(values: jnp.ndarray, weights: tuple[float, ...]) -> float:
    return float(jnp.sum(values * jnp.asarray(weights, dtype=values.dtype)))


def _add_prefix_record(
    record: dict[str, float | int],
    *,
    prefix_depths: tuple[int, ...],
    scores: jnp.ndarray,
    parts: PrefixRoutingLossParts,
    prefix: str,
) -> None:
    infix = "" if prefix == "" else f"{prefix}_"
    for index, depth in enumerate(prefix_depths):
        key_prefix = f"prefix_{depth}_{infix}"
        record[f"{key_prefix}accuracy"] = float(scores[index])
        for field in RoutingLossParts._fields:
            record[f"{key_prefix}{field}"] = float(getattr(parts, field)[index])


def _add_prefix_array_record(
    record: dict[str, float | int],
    *,
    prefix_depths: tuple[int, ...],
    values: jnp.ndarray,
    field: str,
    prefix: str,
) -> None:
    infix = "" if prefix == "" else f"{prefix}_"
    for index, depth in enumerate(prefix_depths):
        record[f"prefix_{depth}_{infix}{field}"] = float(values[index])


def _add_target_margin_record(
    record: dict[str, float | int],
    *,
    val_scores: jnp.ndarray,
    target_accuracies: tuple[float, ...] | None,
) -> None:
    if target_accuracies is None:
        return
    targets = jnp.asarray(target_accuracies, dtype=val_scores.dtype)
    margins = val_scores - targets
    record["min_target_margin"] = float(jnp.min(margins))
    record["mean_target_margin"] = float(jnp.mean(margins))


def fit_shared_prefix_routing_regularized_logits(
    model,
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    *,
    epochs: int,
    batch_size: int,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    select_best_checkpoint: bool = False,
    checkpoint_epochs: list[int] | tuple[int, ...] | None = None,
    selection_metric: str = "val_accuracy",
    target_accuracies: tuple[float, ...] | list[float] | None = None,
    seed: int = 0,
):
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)
    targets = normalize_prefix_targets(depths, target_accuracies)
    train_step = make_shared_prefix_routing_regularized_train_step(
        model,
        prefix_depths=depths,
        prefix_weights=weights,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    eval_step = make_shared_prefix_routing_regularized_eval_step(
        model,
        prefix_depths=depths,
        prefix_weights=weights,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    history = _shared_prefix_history(depths, weights)
    history["metadata"]["selection_metric"] = selection_metric
    _add_target_margin_history_fields(history, targets)
    rng = jax.random.key(seed)
    selected_epochs = set(checkpoint_epochs or range(1, epochs + 1))
    selected_epochs.add(epochs)
    best_state = state
    best_record = None

    for epoch in range(1, epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        part_values = {field: [] for field in RoutingLossParts._fields}
        for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng):
            state, loss_value, batch_scores, parts = train_step(state, batch_x, batch_y)
            losses.append(loss_value)
            scores.append(batch_scores)
            for field in RoutingLossParts._fields:
                part_values[field].append(getattr(parts, field))

        train_scores = jnp.mean(jnp.stack(scores), axis=0)
        train_parts = PrefixRoutingLossParts(
            *(jnp.mean(jnp.stack(part_values[field]), axis=0) for field in RoutingLossParts._fields)
        )
        val_loss, val_scores, val_parts = eval_step(state, test_x, test_y)
        record = {
            "epoch": epoch,
            "loss": float(jnp.mean(jnp.stack(losses))),
            "accuracy": _weighted_value(train_scores, weights),
            "cross_entropy": _weighted_value(train_parts.cross_entropy, weights),
            "routing_leakage": _weighted_value(train_parts.routing_leakage, weights),
            "routing_excess": _weighted_value(train_parts.routing_excess, weights),
            "mean_insertion_loss_db": _weighted_value(train_parts.mean_insertion_loss_db, weights),
            "loss_excess": _weighted_value(train_parts.loss_excess, weights),
            "mean_output_power": _weighted_value(train_parts.mean_output_power, weights),
            "gamma": _weighted_value(train_parts.gamma, weights),
            "val_loss": float(val_loss),
            "val_accuracy": _weighted_value(val_scores, weights),
            "val_cross_entropy": _weighted_value(val_parts.cross_entropy, weights),
            "val_routing_leakage": _weighted_value(val_parts.routing_leakage, weights),
            "val_routing_excess": _weighted_value(val_parts.routing_excess, weights),
            "val_mean_insertion_loss_db": _weighted_value(val_parts.mean_insertion_loss_db, weights),
            "val_loss_excess": _weighted_value(val_parts.loss_excess, weights),
            "val_mean_output_power": _weighted_value(val_parts.mean_output_power, weights),
            "val_gamma": _weighted_value(val_parts.gamma, weights),
        }
        _add_prefix_record(record, prefix_depths=depths, scores=train_scores, parts=train_parts, prefix="")
        _add_prefix_record(record, prefix_depths=depths, scores=val_scores, parts=val_parts, prefix="val")
        _add_target_margin_record(record, val_scores=val_scores, target_accuracies=targets)
        for key, value in record.items():
            history[key].append(value)
        if select_best_checkpoint and epoch in selected_epochs:
            if selection_metric not in record:
                raise ValueError(f"selection_metric {selection_metric!r} is not recorded")
            if best_record is None or record[selection_metric] > best_record[selection_metric]:
                best_state = state
                best_record = dict(record)

    if select_best_checkpoint:
        if best_record is None:
            best_record = {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}
            best_state = state
        history["selected_epoch"] = best_record["epoch"]
        history["selected_metrics"] = {key: value for key, value in best_record.items() if key != "epoch"}
        history["selected"] = [epoch == best_record["epoch"] for epoch in history["epoch"]]
        state = best_state

    return state, history


def fit_shared_prefix_distilled_logits(
    model,
    state: TrainState,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    train_teacher_logits_by_depth,
    test_teacher_logits_by_depth,
    *,
    epochs: int,
    batch_size: int,
    prefix_depths: tuple[int, ...] | list[int],
    prefix_weights: tuple[float, ...] | list[float] | None = None,
    routing_weight: float,
    routing_target: float,
    distillation_alpha: float,
    distillation_temperature: float,
    loss_guard_db: float | None = None,
    loss_guard_weight: float = 0.0,
    select_best_checkpoint: bool = False,
    checkpoint_epochs: list[int] | tuple[int, ...] | None = None,
    selection_metric: str = "val_accuracy",
    seed: int = 0,
):
    depths = tuple(int(depth) for depth in prefix_depths)
    weights = normalize_prefix_weights(depths, prefix_weights)
    train_teacher_logits = _teacher_logits_array(
        train_teacher_logits_by_depth,
        depths,
        expected_examples=train_x.shape[0],
        expected_classes=train_y.shape[-1],
        name="train_teacher_logits_by_depth",
    )
    test_teacher_logits = _teacher_logits_array(
        test_teacher_logits_by_depth,
        depths,
        expected_examples=test_x.shape[0],
        expected_classes=test_y.shape[-1],
        name="test_teacher_logits_by_depth",
    )
    train_step = make_shared_prefix_distilled_train_step(
        model,
        prefix_depths=depths,
        prefix_weights=weights,
        routing_weight=routing_weight,
        routing_target=routing_target,
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    eval_step = make_shared_prefix_distilled_eval_step(
        model,
        prefix_depths=depths,
        prefix_weights=weights,
        routing_weight=routing_weight,
        routing_target=routing_target,
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    history = _shared_prefix_history(depths, weights)
    _add_distillation_history_fields(history, depths)
    history["metadata"]["selection_metric"] = selection_metric
    history["metadata"]["distillation_alpha"] = float(distillation_alpha)
    history["metadata"]["distillation_temperature"] = float(distillation_temperature)
    rng = jax.random.key(seed)
    selected_epochs = set(checkpoint_epochs or range(1, epochs + 1))
    selected_epochs.add(epochs)
    best_state = state
    best_record = None

    for epoch in range(1, epochs + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        distillation_values = []
        part_values = {field: [] for field in RoutingLossParts._fields}
        for batch_x, batch_y, batch_teacher_logits in _iterate_batches_with_teacher_logits(
            train_x,
            train_y,
            train_teacher_logits,
            batch_size,
            epoch_rng,
        ):
            state, loss_value, batch_scores, parts, distillation_kl = train_step(
                state,
                batch_x,
                batch_y,
                batch_teacher_logits,
            )
            losses.append(loss_value)
            scores.append(batch_scores)
            distillation_values.append(distillation_kl)
            for field in RoutingLossParts._fields:
                part_values[field].append(getattr(parts, field))

        train_scores = jnp.mean(jnp.stack(scores), axis=0)
        train_distillation_kl = jnp.mean(jnp.stack(distillation_values), axis=0)
        train_parts = PrefixRoutingLossParts(
            *(jnp.mean(jnp.stack(part_values[field]), axis=0) for field in RoutingLossParts._fields)
        )
        val_loss, val_scores, val_parts, val_distillation_kl = eval_step(
            state,
            test_x,
            test_y,
            test_teacher_logits,
        )
        record = {
            "epoch": epoch,
            "loss": float(jnp.mean(jnp.stack(losses))),
            "accuracy": _weighted_value(train_scores, weights),
            "cross_entropy": _weighted_value(train_parts.cross_entropy, weights),
            "routing_leakage": _weighted_value(train_parts.routing_leakage, weights),
            "routing_excess": _weighted_value(train_parts.routing_excess, weights),
            "mean_insertion_loss_db": _weighted_value(train_parts.mean_insertion_loss_db, weights),
            "loss_excess": _weighted_value(train_parts.loss_excess, weights),
            "mean_output_power": _weighted_value(train_parts.mean_output_power, weights),
            "gamma": _weighted_value(train_parts.gamma, weights),
            "distillation_kl": _weighted_value(train_distillation_kl, weights),
            "val_loss": float(val_loss),
            "val_accuracy": _weighted_value(val_scores, weights),
            "val_cross_entropy": _weighted_value(val_parts.cross_entropy, weights),
            "val_routing_leakage": _weighted_value(val_parts.routing_leakage, weights),
            "val_routing_excess": _weighted_value(val_parts.routing_excess, weights),
            "val_mean_insertion_loss_db": _weighted_value(val_parts.mean_insertion_loss_db, weights),
            "val_loss_excess": _weighted_value(val_parts.loss_excess, weights),
            "val_mean_output_power": _weighted_value(val_parts.mean_output_power, weights),
            "val_gamma": _weighted_value(val_parts.gamma, weights),
            "val_distillation_kl": _weighted_value(val_distillation_kl, weights),
        }
        _add_prefix_record(record, prefix_depths=depths, scores=train_scores, parts=train_parts, prefix="")
        _add_prefix_record(record, prefix_depths=depths, scores=val_scores, parts=val_parts, prefix="val")
        _add_prefix_array_record(
            record,
            prefix_depths=depths,
            values=train_distillation_kl,
            field="distillation_kl",
            prefix="",
        )
        _add_prefix_array_record(
            record,
            prefix_depths=depths,
            values=val_distillation_kl,
            field="distillation_kl",
            prefix="val",
        )
        for key, value in record.items():
            history[key].append(value)
        if select_best_checkpoint and epoch in selected_epochs:
            if selection_metric not in record:
                raise ValueError(f"selection_metric {selection_metric!r} is not recorded")
            if best_record is None or record[selection_metric] > best_record[selection_metric]:
                best_state = state
                best_record = dict(record)

    if select_best_checkpoint:
        if best_record is None:
            best_record = {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}
            best_state = state
        history["selected_epoch"] = best_record["epoch"]
        history["selected_metrics"] = {key: value for key, value in best_record.items() if key != "epoch"}
        history["selected"] = [epoch == best_record["epoch"] for epoch in history["epoch"]]
        state = best_state

    return state, history
