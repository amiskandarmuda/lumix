from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.mnist_pca import load_pca_dataset
from experiments.common.models import build_subunitary_surrogate, subunitary_surrogate_config_from_mapping
from experiments.common.robustness import relative_frobenius_perturbation, surrogate_layer_matrix_from_params
from experiments.scripts.train_shared_prefix_surrogate import _write_csv
from experiments.scripts.train_surrogate import _ensure_dataset
from lumix.batching import iterate_batches
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear
from lumix.losses import cross_entropy_logits
from lumix.metrics import accuracy
from lumix.state import create_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate fixed repeated-encoding phase offsets under matrix error.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-run-template", required=True, type=str)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--relative-error", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--phase-bias-scale", type=float, default=math.pi)
    parser.add_argument("--shared-bias", action="store_true")
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def phase_bias_multiplier(phase_bias: jnp.ndarray, *, phase_bias_scale: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(1j * jnp.asarray(phase_bias_scale, dtype=jnp.float32) * phase_bias).astype(jnp.complex64)


def _layer_phase_bias(phase_bias: jnp.ndarray, layer_index: int, *, per_layer: bool) -> jnp.ndarray:
    return phase_bias[layer_index] if per_layer else phase_bias


def _perturbed_layer_matrix(
    model,
    params,
    layer_index: int,
    rng,
    *,
    relative_error: float,
    rescale_to_passive: bool = False,
) -> jnp.ndarray:
    _, singular_max = insertion_loss_bounds(model.config.loss_db)
    matrix = surrogate_layer_matrix_from_params(model.config, params, layer_index)
    return relative_frobenius_perturbation(
        matrix,
        jax.random.fold_in(rng, layer_index),
        relative_error=relative_error,
        rescale_to_passive=rescale_to_passive,
        singular_max=singular_max,
    )


def calibrated_noisy_forward(
    model,
    params,
    values: jnp.ndarray,
    phase_bias: jnp.ndarray,
    rng,
    *,
    relative_error: float,
    phase_bias_scale: float,
    per_layer: bool,
    return_aux: bool = False,
):
    fields = model.input_fields(values)
    for layer_index in range(model.config.layers):
        matrix = _perturbed_layer_matrix(
            model,
            params,
            layer_index,
            rng,
            relative_error=relative_error,
        )
        fixed_phase = phase_bias_multiplier(
            _layer_phase_bias(phase_bias, layer_index, per_layer=per_layer),
            phase_bias_scale=phase_bias_scale,
        )
        phase_mask = model.phase_mask_for_layer(values, layer_index) * fixed_phase
        fields = subunitary_linear(fields * phase_mask, matrix)
    return model.apply(
        {"params": params},
        fields,
        return_aux=return_aux,
        method=type(model).readout_fields,
    )


def _make_phase_bias(layers: int, width: int, *, per_layer: bool) -> jnp.ndarray:
    shape = (layers, width) if per_layer else (width,)
    return jnp.zeros(shape, dtype=jnp.float32)


def _evaluate(model, params, x, y, phase_bias, rng, *, relative_error: float, phase_bias_scale: float, per_layer: bool):
    logits = calibrated_noisy_forward(
        model,
        params,
        x,
        phase_bias,
        rng,
        relative_error=relative_error,
        phase_bias_scale=phase_bias_scale,
        per_layer=per_layer,
    )
    return accuracy(y, logits), cross_entropy_logits(y, logits)


def calibrate_phase_bias(
    model,
    params,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    *,
    relative_error: float,
    perturbation_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phase_bias_scale: float,
    per_layer: bool,
    seed: int,
) -> tuple[jnp.ndarray, dict[str, float]]:
    rng = jax.random.key(seed)
    perturbation_rng = jax.random.key(perturbation_seed)
    phase_bias = _make_phase_bias(model.config.layers, model.config.width, per_layer=per_layer)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(phase_bias)

    def loss_fn(candidate_bias, batch_x, batch_y):
        logits = calibrated_noisy_forward(
            model,
            params,
            batch_x,
            candidate_bias,
            perturbation_rng,
            relative_error=relative_error,
            phase_bias_scale=phase_bias_scale,
            per_layer=per_layer,
        )
        return cross_entropy_logits(batch_y, logits)

    train_step = jax.jit(lambda bias, state, bx, by: _calibration_step(loss_fn, optimizer, bias, state, bx, by))

    clean_score, clean_loss = _evaluate(
        model,
        params,
        test_x,
        test_y,
        phase_bias,
        perturbation_rng,
        relative_error=0.0,
        phase_bias_scale=phase_bias_scale,
        per_layer=per_layer,
    )
    before_score, before_loss = _evaluate(
        model,
        params,
        test_x,
        test_y,
        phase_bias,
        perturbation_rng,
        relative_error=relative_error,
        phase_bias_scale=phase_bias_scale,
        per_layer=per_layer,
    )

    for _ in range(epochs):
        rng, epoch_rng = jax.random.split(rng)
        for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng):
            phase_bias, opt_state, _ = train_step(phase_bias, opt_state, batch_x, batch_y)

    after_score, after_loss = _evaluate(
        model,
        params,
        test_x,
        test_y,
        phase_bias,
        perturbation_rng,
        relative_error=relative_error,
        phase_bias_scale=phase_bias_scale,
        per_layer=per_layer,
    )
    metrics = {
        "clean_accuracy": float(clean_score),
        "before_accuracy": float(before_score),
        "after_accuracy": float(after_score),
        "clean_cross_entropy": float(clean_loss),
        "before_cross_entropy": float(before_loss),
        "after_cross_entropy": float(after_loss),
        "accuracy_recovery_pp": 100.0 * float(after_score - before_score),
        "remaining_drop_pp": 100.0 * float(after_score - clean_score),
        "phase_bias_l2": float(jnp.linalg.norm(phase_bias)),
    }
    return phase_bias, metrics


def _calibration_step(loss_fn, optimizer, phase_bias, opt_state, batch_x, batch_y):
    loss_value, grads = jax.value_and_grad(loss_fn)(phase_bias, batch_x, batch_y)
    updates, opt_state = optimizer.update(grads, opt_state, phase_bias)
    phase_bias = optax.apply_updates(phase_bias, updates)
    return phase_bias, opt_state, loss_value


def _load_source_state(config_path: Path, source_template: str, depth: int, dataset, training_config: dict):
    source_dir = resolve_config_path(config_path, source_template.format(depth=depth))
    config = load_json_config(source_dir / "config.resolved.json")
    model_config = subunitary_surrogate_config_from_mapping(config["model"])
    model = build_subunitary_surrogate(model_config)
    state = create_state(
        model,
        jax.random.key(int(training_config["seed"])),
        jnp.asarray(dataset.x_train[:1]),
        learning_rate=float(training_config["learning_rate"]),
    )
    params = serialization.from_bytes(state.params, (source_dir / "params.msgpack").read_bytes())
    return model, params, source_dir


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    dataset_path = _ensure_dataset(config_path, config)
    dataset = load_pca_dataset(dataset_path)
    train_x = jnp.asarray(dataset.x_train)
    train_y = jnp.asarray(dataset.y_train)
    if args.train_examples is not None:
        train_x = train_x[: args.train_examples]
        train_y = train_y[: args.train_examples]
    test_x = jnp.asarray(dataset.x_test)
    test_y = jnp.asarray(dataset.y_test)
    training_config = config["training"]
    batch_size = int(args.batch_size or training_config["batch_size"])
    root_run_dir = resolve_config_path(config_path, args.run_dir)
    root_run_dir.mkdir(parents=True, exist_ok=True)
    per_layer = not args.shared_bias

    records = []
    for depth in range(args.min_depth, args.max_depth + 1):
        model, params, source_dir = _load_source_state(config_path, args.source_run_template, depth, dataset, training_config)
        if model.config.width != 16 or model.config.bias_ports != 0:
            raise ValueError("Phase-bias calibration run expects strict width=16 and bias_ports=0 source models.")
        depth_dir = root_run_dir / f"depth_{depth}"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for seed_index in range(args.seeds):
            perturbation_seed = int(args.seed + seed_index)
            phase_bias, metrics = calibrate_phase_bias(
                model,
                params,
                train_x,
                train_y,
                test_x,
                test_y,
                relative_error=float(args.relative_error),
                perturbation_seed=perturbation_seed,
                epochs=int(args.epochs),
                batch_size=batch_size,
                learning_rate=float(args.learning_rate),
                phase_bias_scale=float(args.phase_bias_scale),
                per_layer=per_layer,
                seed=int(args.seed + 10_000 + seed_index),
            )
            record = {
                "depth": depth,
                "seed": perturbation_seed,
                "source_run_dir": str(source_dir),
                "per_layer": per_layer,
                "relative_error": float(args.relative_error),
                **metrics,
            }
            records.append(record)
            np.save(depth_dir / f"phase_bias_seed_{perturbation_seed}.npy", np.asarray(jax.device_get(phase_bias)))
            print(
                f"depth {depth} seed {perturbation_seed}: "
                f"before={metrics['before_accuracy']:.6f} "
                f"after={metrics['after_accuracy']:.6f} "
                f"recovery_pp={metrics['accuracy_recovery_pp']:.3f}"
            )

    (root_run_dir / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    _write_csv(root_run_dir / "summary.csv", records)
    aggregate = []
    for depth in range(args.min_depth, args.max_depth + 1):
        depth_records = [record for record in records if record["depth"] == depth]
        aggregate.append(
            {
                "depth": depth,
                "seeds": len(depth_records),
                "mean_clean_accuracy": sum(record["clean_accuracy"] for record in depth_records) / len(depth_records),
                "mean_before_accuracy": sum(record["before_accuracy"] for record in depth_records) / len(depth_records),
                "mean_after_accuracy": sum(record["after_accuracy"] for record in depth_records) / len(depth_records),
                "mean_recovery_pp": sum(record["accuracy_recovery_pp"] for record in depth_records) / len(depth_records),
                "mean_remaining_drop_pp": sum(record["remaining_drop_pp"] for record in depth_records) / len(depth_records),
            }
        )
    (root_run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    _write_csv(root_run_dir / "aggregate.csv", aggregate)
    print(f"saved phase-bias calibration results to {root_run_dir}")


if __name__ == "__main__":
    main()
