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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.mnist_pca import load_pca_dataset
from experiments.common.models import (
    build_subunitary_surrogate,
    subunitary_surrogate_config_from_mapping,
)
from experiments.common.robustness import noisy_surrogate_forward
from experiments.common.training import (
    RoutingLossParts,
    make_matrix_noise_distilled_train_step,
    make_matrix_noise_regularized_train_step,
    make_routing_regularized_eval_step,
)
from experiments.scripts.sweep_independent_surrogate import _metrics_from_history, _summary_record
from experiments.scripts.train_shared_prefix_surrogate import _save_layer_diagnostics, _write_csv, apply_cli_overrides
from experiments.scripts.train_surrogate import _ensure_dataset, _save_matrices
from lumix.batching import iterate_batches
from lumix.losses import cross_entropy_logits
from lumix.metrics import accuracy
from lumix.state import create_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent surrogates with relative matrix-noise augmentation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--run-name-template", type=str, default="robust_layers_{depth}")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--bias-ports", type=int, default=None)
    parser.add_argument("--routing-limit", type=int, default=None)
    parser.add_argument("--routing-weight", type=float, default=None)
    parser.add_argument("--routing-target", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--relative-error", type=float, default=0.2)
    parser.add_argument("--eval-relative-error", type=float, default=None)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--noisy-weight", type=float, default=1.0)
    parser.add_argument("--selection-eval-seeds", type=int, default=5)
    parser.add_argument("--robustness-eval-seeds", type=int, default=20)
    parser.add_argument("--selection-metric", choices=("val_accuracy", "val_noisy_accuracy"), default="val_noisy_accuracy")
    parser.add_argument("--clean-accuracy-floor", type=float, default=None)
    parser.add_argument("--initial-run-template", type=str, default=None)
    parser.add_argument("--teacher-run-template", type=str, default=None)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin-target", type=float, default=0.0)
    parser.add_argument("--rescale-to-passive", action="store_true")
    return parser.parse_args()


def checkpoint_selection_value(
    record: dict,
    selection_metric: str,
    *,
    clean_accuracy_floor: float | None = None,
) -> float:
    if clean_accuracy_floor is not None and float(record["val_accuracy"]) < float(clean_accuracy_floor):
        return float("-inf")
    return float(record[selection_metric])


def load_params_into_state(state, params_path: Path):
    params = serialization.from_bytes(state.params, params_path.read_bytes())
    return state.replace(params=params)


def _run_dir_from_template(config_path: Path, template: str, *, depth: int) -> Path:
    return resolve_config_path(config_path, template.format(depth=depth))


def _iterate_batches_with_teacher_logits(
    x: jnp.ndarray,
    y: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    batch_size: int,
    rng,
):
    indices = jax.random.permutation(rng, x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices], teacher_logits[batch_indices]


def _mean_noisy_eval(
    model,
    params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    relative_error: float,
    seed: int,
    samples: int,
    rescale_to_passive: bool,
) -> dict[str, float]:
    if samples < 1:
        return {
            "accuracy": math.nan,
            "std_accuracy": math.nan,
            "cross_entropy": math.nan,
            "std_cross_entropy": math.nan,
        }

    @jax.jit
    def eval_one(key):
        logits = noisy_surrogate_forward(
            model,
            params,
            x,
            key,
            relative_error=relative_error,
            rescale_to_passive=rescale_to_passive,
        )
        return accuracy(y, logits), cross_entropy_logits(y, logits)

    keys = jax.random.split(jax.random.key(seed), samples)
    scores, losses = jax.vmap(eval_one)(keys)
    return {
        "accuracy": float(jnp.mean(scores)),
        "std_accuracy": float(jnp.std(scores)),
        "cross_entropy": float(jnp.mean(losses)),
        "std_cross_entropy": float(jnp.std(losses)),
    }


def _fit_matrix_noise_regularized(
    model,
    state,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    test_x: jnp.ndarray,
    test_y: jnp.ndarray,
    train_teacher_logits: jnp.ndarray | None = None,
    *,
    epochs: int,
    batch_size: int,
    routing_weight: float,
    routing_target: float,
    relative_error: float,
    eval_relative_error: float,
    noise_samples: int,
    noisy_weight: float,
    loss_guard_db: float | None,
    loss_guard_weight: float,
    select_best_checkpoint: bool,
    checkpoint_epochs: list[int] | tuple[int, ...] | None,
    selection_metric: str,
    clean_accuracy_floor: float | None,
    selection_eval_seeds: int,
    distillation_weight: float,
    distillation_temperature: float,
    margin_weight: float,
    margin_target: float,
    rescale_to_passive: bool,
    seed: int,
):
    use_distillation = train_teacher_logits is not None and distillation_weight > 0.0
    if use_distillation:
        train_step = make_matrix_noise_distilled_train_step(
            model,
            routing_weight=routing_weight,
            routing_target=routing_target,
            relative_error=relative_error,
            noise_samples=noise_samples,
            noisy_weight=noisy_weight,
            distillation_weight=distillation_weight,
            distillation_temperature=distillation_temperature,
            margin_weight=margin_weight,
            margin_target=margin_target,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
            rescale_to_passive=rescale_to_passive,
        )
    else:
        train_step = make_matrix_noise_regularized_train_step(
            model,
            routing_weight=routing_weight,
            routing_target=routing_target,
            relative_error=relative_error,
            noise_samples=noise_samples,
            noisy_weight=noisy_weight,
            loss_guard_db=loss_guard_db,
            loss_guard_weight=loss_guard_weight,
            rescale_to_passive=rescale_to_passive,
        )
    eval_step = make_routing_regularized_eval_step(
        model,
        routing_weight=routing_weight,
        routing_target=routing_target,
        loss_guard_db=loss_guard_db,
        loss_guard_weight=loss_guard_weight,
    )
    history = {
        "metadata": {
            "relative_error": relative_error,
            "eval_relative_error": eval_relative_error,
            "noise_samples": noise_samples,
            "noisy_weight": noisy_weight,
            "selection_metric": selection_metric,
            "clean_accuracy_floor": clean_accuracy_floor,
            "selection_eval_seeds": selection_eval_seeds,
            "distillation_weight": distillation_weight,
            "distillation_temperature": distillation_temperature,
            "margin_weight": margin_weight,
            "margin_target": margin_target,
            "rescale_to_passive": rescale_to_passive,
        },
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "noisy_cross_entropy": [],
        "distillation_kl": [],
        "margin_loss": [],
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
        "val_noisy_accuracy": [],
        "val_noisy_accuracy_std": [],
        "val_noisy_cross_entropy": [],
        "val_noisy_cross_entropy_std": [],
        "val_noisy_delta_pp": [],
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
        noisy_losses = []
        distillation_values = []
        margin_values = []
        part_values = {field: [] for field in RoutingLossParts._fields}
        if use_distillation:
            batch_iterator = _iterate_batches_with_teacher_logits(
                train_x,
                train_y,
                train_teacher_logits,
                batch_size,
                epoch_rng,
            )
        else:
            batch_iterator = ((batch_x, batch_y, None) for batch_x, batch_y in iterate_batches(train_x, train_y, batch_size, epoch_rng))
        for batch_x, batch_y, batch_teacher_logits in batch_iterator:
            rng, batch_rng = jax.random.split(rng)
            if use_distillation:
                (
                    state,
                    loss_value,
                    score,
                    parts,
                    noisy_cross_entropy,
                    distillation_kl,
                    margin_loss,
                ) = train_step(state, batch_x, batch_y, batch_teacher_logits, batch_rng)
            else:
                state, loss_value, score, parts, noisy_cross_entropy = train_step(state, batch_x, batch_y, batch_rng)
                distillation_kl = jnp.asarray(0.0, dtype=noisy_cross_entropy.dtype)
                margin_loss = jnp.asarray(0.0, dtype=noisy_cross_entropy.dtype)
            losses.append(loss_value)
            scores.append(score)
            noisy_losses.append(noisy_cross_entropy)
            distillation_values.append(distillation_kl)
            margin_values.append(margin_loss)
            for field in RoutingLossParts._fields:
                part_values[field].append(getattr(parts, field))

        val_loss, val_score, val_parts = eval_step(state, test_x, test_y)
        if epoch in selected_epochs:
            noisy_eval = _mean_noisy_eval(
                model,
                state.params,
                test_x,
                test_y,
                relative_error=eval_relative_error,
                seed=seed + 10_000 + epoch,
                samples=selection_eval_seeds,
                rescale_to_passive=rescale_to_passive,
            )
        else:
            noisy_eval = _mean_noisy_eval(
                model,
                state.params,
                test_x,
                test_y,
                relative_error=eval_relative_error,
                seed=seed,
                samples=0,
                rescale_to_passive=rescale_to_passive,
            )

        record = {
            "epoch": epoch,
            "loss": float(jnp.mean(jnp.stack(losses))),
            "accuracy": float(jnp.mean(jnp.stack(scores))),
            "noisy_cross_entropy": float(jnp.mean(jnp.stack(noisy_losses))),
            "distillation_kl": float(jnp.mean(jnp.stack(distillation_values))),
            "margin_loss": float(jnp.mean(jnp.stack(margin_values))),
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
            "val_noisy_accuracy": noisy_eval["accuracy"],
            "val_noisy_accuracy_std": noisy_eval["std_accuracy"],
            "val_noisy_cross_entropy": noisy_eval["cross_entropy"],
            "val_noisy_cross_entropy_std": noisy_eval["std_cross_entropy"],
            "val_noisy_delta_pp": 100.0 * (noisy_eval["accuracy"] - float(val_score)),
        }
        for key, value in record.items():
            history[key].append(value)
        if select_best_checkpoint and epoch in selected_epochs:
            candidate_value = checkpoint_selection_value(
                record,
                selection_metric,
                clean_accuracy_floor=clean_accuracy_floor,
            )
            best_value = (
                float("-inf")
                if best_record is None
                else checkpoint_selection_value(
                    best_record,
                    selection_metric,
                    clean_accuracy_floor=clean_accuracy_floor,
                )
            )
            if candidate_value > best_value:
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


def main() -> None:
    args = parse_args()
    if args.min_depth < 1:
        raise ValueError("--min-depth must be at least 1")
    if args.max_depth < args.min_depth:
        raise ValueError("--max-depth must be greater than or equal to --min-depth")

    config_path = args.config.resolve()
    config = load_json_config(config_path)
    apply_cli_overrides(
        config,
        routing_limit=args.routing_limit,
        routing_weight=args.routing_weight,
        routing_target=args.routing_target,
        learning_rate=args.learning_rate,
    )
    if args.width is not None:
        config["model"]["width"] = int(args.width)
    if args.bias_ports is not None:
        config["model"]["bias_ports"] = int(args.bias_ports)

    dataset_path = _ensure_dataset(config_path, config)
    dataset = load_pca_dataset(dataset_path)
    training_config = config["training"]
    epochs = int(args.epochs or training_config["epochs"])
    eval_relative_error = float(args.relative_error if args.eval_relative_error is None else args.eval_relative_error)
    root_run_dir = resolve_config_path(config_path, args.run_dir)
    root_run_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    robustness_records = []
    for depth in range(args.min_depth, args.max_depth + 1):
        model_mapping = dict(config["model"])
        model_mapping["layers"] = depth
        run_dir = root_run_dir / args.run_name_template.format(depth=depth)
        run_dir.mkdir(parents=True, exist_ok=True)

        model_config = subunitary_surrogate_config_from_mapping(model_mapping)
        model = build_subunitary_surrogate(model_config)
        state = create_state(
            model,
            jax.random.key(int(training_config["seed"])),
            jnp.asarray(dataset.x_train[:1]),
            learning_rate=float(training_config["learning_rate"]),
        )
        initial_run_dir = None
        if args.initial_run_template is not None:
            initial_run_dir = _run_dir_from_template(config_path, args.initial_run_template, depth=depth)
            state = load_params_into_state(state, initial_run_dir / "params.msgpack")

        teacher_template = args.teacher_run_template or args.initial_run_template
        if args.distillation_weight > 0.0 and teacher_template is None:
            raise ValueError("--distillation-weight requires --teacher-run-template or --initial-run-template")
        teacher_run_dir = None
        train_teacher_logits = None
        if args.distillation_weight > 0.0:
            teacher_run_dir = _run_dir_from_template(config_path, teacher_template, depth=depth)
            teacher_params = serialization.from_bytes(state.params, (teacher_run_dir / "params.msgpack").read_bytes())
            train_teacher_logits = model.apply({"params": teacher_params}, jnp.asarray(dataset.x_train))

        loss_guard_db = training_config.get("loss_guard_db")
        state, history = _fit_matrix_noise_regularized(
            model,
            state,
            jnp.asarray(dataset.x_train),
            jnp.asarray(dataset.y_train),
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            train_teacher_logits,
            epochs=epochs,
            batch_size=int(training_config["batch_size"]),
            routing_weight=float(training_config.get("routing_penalty_weight", 0.0)),
            routing_target=float(training_config.get("routing_leakage_target", 0.0)),
            relative_error=float(args.relative_error),
            eval_relative_error=eval_relative_error,
            noise_samples=int(args.noise_samples),
            noisy_weight=float(args.noisy_weight),
            loss_guard_db=None if loss_guard_db is None else float(loss_guard_db),
            loss_guard_weight=float(training_config.get("loss_guard_weight", 0.0)),
            select_best_checkpoint=bool(training_config.get("select_best_checkpoint", False)),
            checkpoint_epochs=training_config.get("checkpoint_epochs"),
            selection_metric=args.selection_metric,
            clean_accuracy_floor=args.clean_accuracy_floor,
            selection_eval_seeds=int(args.selection_eval_seeds),
            distillation_weight=float(args.distillation_weight),
            distillation_temperature=float(args.distillation_temperature),
            margin_weight=float(args.margin_weight),
            margin_target=float(args.margin_target),
            rescale_to_passive=bool(args.rescale_to_passive),
            seed=int(training_config["seed"]),
        )

        (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        metrics = _metrics_from_history(history)
        final_noisy_eval = _mean_noisy_eval(
            model,
            state.params,
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            relative_error=eval_relative_error,
            seed=int(training_config["seed"]) + 50_000 + depth,
            samples=int(args.robustness_eval_seeds),
            rescale_to_passive=bool(args.rescale_to_passive),
        )
        metrics["final_noisy_accuracy"] = final_noisy_eval["accuracy"]
        metrics["final_noisy_accuracy_std"] = final_noisy_eval["std_accuracy"]
        metrics["final_noisy_cross_entropy"] = final_noisy_eval["cross_entropy"]
        metrics["final_noisy_delta_pp"] = 100.0 * (final_noisy_eval["accuracy"] - float(metrics["val_accuracy"]))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (run_dir / "robustness.json").write_text(json.dumps(final_noisy_eval, indent=2) + "\n")

        resolved_config = dict(config)
        resolved_config["model"] = model_mapping
        resolved_config["training"] = {
            **training_config,
            "epochs": epochs,
            "matrix_noise_relative_error": float(args.relative_error),
            "matrix_noise_eval_relative_error": eval_relative_error,
            "matrix_noise_samples": int(args.noise_samples),
            "matrix_noise_weight": float(args.noisy_weight),
            "matrix_noise_clean_accuracy_floor": args.clean_accuracy_floor,
            "matrix_noise_initial_run_dir": None if initial_run_dir is None else str(initial_run_dir),
            "matrix_noise_teacher_run_dir": None if teacher_run_dir is None else str(teacher_run_dir),
            "matrix_noise_distillation_weight": float(args.distillation_weight),
            "matrix_noise_distillation_temperature": float(args.distillation_temperature),
            "matrix_noise_margin_weight": float(args.margin_weight),
            "matrix_noise_margin_target": float(args.margin_target),
            "matrix_noise_rescale_to_passive": bool(args.rescale_to_passive),
        }
        resolved_config["outputs"] = {"run_dir": str(run_dir)}
        (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2) + "\n")

        sample_width = model_config.width - model_config.bias_ports
        _save_matrices(model, state.params, sample_width, run_dir)
        _save_layer_diagnostics(model, state.params, sample_width, run_dir)
        summary = _summary_record(depth=depth, run_dir=run_dir, metrics=metrics, model_mapping=model_mapping)
        summary.update(
            {
                "final_noisy_accuracy": float(metrics["final_noisy_accuracy"]),
                "final_noisy_accuracy_std": float(metrics["final_noisy_accuracy_std"]),
                "final_noisy_delta_pp": float(metrics["final_noisy_delta_pp"]),
                "relative_error": float(args.relative_error),
                "eval_relative_error": eval_relative_error,
                "noise_samples": int(args.noise_samples),
                "noisy_weight": float(args.noisy_weight),
            }
        )
        summary_records.append(summary)
        robustness_records.append({"depth": depth, **final_noisy_eval})
        print(
            "depth "
            f"{depth}: clean={metrics['val_accuracy']:.6f} "
            f"noisy={metrics['final_noisy_accuracy']:.6f} "
            f"delta_pp={metrics['final_noisy_delta_pp']:.3f} saved to {run_dir}"
        )

    (root_run_dir / "summary.json").write_text(json.dumps(summary_records, indent=2) + "\n")
    _write_csv(root_run_dir / "summary.csv", summary_records)
    (root_run_dir / "robustness.json").write_text(json.dumps(robustness_records, indent=2) + "\n")
    _write_csv(root_run_dir / "robustness.csv", robustness_records)

    with (root_run_dir / "summary_compact.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("depth", "val_accuracy", "final_noisy_accuracy", "final_noisy_delta_pp", "selected_epoch"),
        )
        writer.writeheader()
        for record in summary_records:
            writer.writerow(
                {
                    "depth": record["depth"],
                    "val_accuracy": record["val_accuracy"],
                    "final_noisy_accuracy": record["final_noisy_accuracy"],
                    "final_noisy_delta_pp": record["final_noisy_delta_pp"],
                    "selected_epoch": record["selected_epoch"],
                }
            )

    print(f"saved matrix-noise robust sweep summary to {root_run_dir}")


if __name__ == "__main__":
    main()
