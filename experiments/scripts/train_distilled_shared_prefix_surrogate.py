from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.mnist_pca import load_pca_dataset
from experiments.common.models import (
    build_subunitary_surrogate,
    subunitary_surrogate_config_from_mapping,
)
from experiments.common.training import (
    create_shared_prefix_state,
    fit_shared_prefix_distilled_logits,
)
from experiments.scripts.train_shared_prefix_surrogate import (
    _save_layer_diagnostics,
    _write_csv,
    apply_cli_overrides,
    parse_prefix_weights,
)
from experiments.scripts.train_surrogate import _ensure_dataset, _save_matrices


PREFIX_METRIC_FIELDS = (
    "accuracy",
    "cross_entropy",
    "distillation_kl",
    "routing_leakage",
    "routing_excess",
    "mean_insertion_loss_db",
    "loss_excess",
    "mean_output_power",
    "gamma",
    "val_accuracy",
    "val_cross_entropy",
    "val_distillation_kl",
    "val_routing_leakage",
    "val_routing_excess",
    "val_mean_insertion_loss_db",
    "val_loss_excess",
    "val_mean_output_power",
    "val_gamma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared-prefix surrogate with independent-depth teachers.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--teacher-run-dir", required=True, type=Path)
    parser.add_argument("--teacher-template", type=str, default="repeated_layers_{depth}")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--prefix-weights", type=str, default=None)
    parser.add_argument("--selection-metric", type=str, default=None)
    parser.add_argument("--routing-limit", type=int, default=None)
    parser.add_argument("--routing-weight", type=float, default=None)
    parser.add_argument("--routing-target", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--distillation-alpha", type=float, default=0.7)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-logits-cache", type=Path, default=None)
    parser.add_argument("--reuse-teacher-logits", action="store_true")
    parser.add_argument("--initial-params", type=Path, default=None)
    return parser.parse_args()


def _predict_logits(model, params, values: np.ndarray, *, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, values.shape[0], batch_size):
        batch = jnp.asarray(values[start : start + batch_size])
        logits = model.apply({"params": params}, batch)
        chunks.append(np.asarray(jax.device_get(logits), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _teacher_run_path(teacher_root: Path, template: str, depth: int) -> Path:
    return teacher_root / template.format(depth=depth)


def _load_teacher(teacher_run_dir: Path, sample_x: jnp.ndarray):
    teacher_config = load_json_config(teacher_run_dir / "config.resolved.json")
    model_config = subunitary_surrogate_config_from_mapping(teacher_config["model"])
    model = build_subunitary_surrogate(model_config)
    template_params = model.init(jax.random.key(0), sample_x)["params"]
    params = serialization.from_bytes(template_params, (teacher_run_dir / "params.msgpack").read_bytes())
    return model, params


def _compute_teacher_logits(
    *,
    teacher_root: Path,
    teacher_template: str,
    prefix_depths: tuple[int, ...],
    train_x: np.ndarray,
    test_x: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_logits = []
    test_logits = []
    sample_x = jnp.asarray(train_x[:1])
    for depth in prefix_depths:
        teacher_run_dir = _teacher_run_path(teacher_root, teacher_template, depth)
        model, params = _load_teacher(teacher_run_dir, sample_x)
        train_logits.append(_predict_logits(model, params, train_x, batch_size=batch_size))
        test_logits.append(_predict_logits(model, params, test_x, batch_size=batch_size))
    return np.stack(train_logits).astype(np.float32), np.stack(test_logits).astype(np.float32)


def _load_or_compute_teacher_logits(
    *,
    cache_path: Path,
    reuse_cache: bool,
    teacher_root: Path,
    teacher_template: str,
    prefix_depths: tuple[int, ...],
    train_x: np.ndarray,
    test_x: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if reuse_cache and cache_path.exists():
        cached = np.load(cache_path)
        return cached["train_logits"], cached["test_logits"]

    train_logits, test_logits = _compute_teacher_logits(
        teacher_root=teacher_root,
        teacher_template=teacher_template,
        prefix_depths=prefix_depths,
        train_x=train_x,
        test_x=test_x,
        batch_size=batch_size,
    )
    np.savez_compressed(
        cache_path,
        train_logits=train_logits,
        test_logits=test_logits,
        prefix_depths=np.asarray(prefix_depths, dtype=np.int32),
    )
    return train_logits, test_logits


def _prefix_metric_records(metrics: dict, prefix_depths: tuple[int, ...]) -> list[dict[str, float | int]]:
    records = []
    for depth in prefix_depths:
        record: dict[str, float | int] = {"depth": depth}
        for field in PREFIX_METRIC_FIELDS:
            record[field] = float(metrics[f"prefix_{depth}_{field}"])
        records.append(record)
    return records


def _teacher_metric_records(
    *,
    teacher_root: Path,
    teacher_template: str,
    prefix_depths: tuple[int, ...],
) -> list[dict[str, float | int | str]]:
    records = []
    for depth in prefix_depths:
        teacher_run_dir = _teacher_run_path(teacher_root, teacher_template, depth)
        metrics = load_json_config(teacher_run_dir / "metrics.json")
        records.append(
            {
                "depth": depth,
                "teacher_run_dir": str(teacher_run_dir),
                "teacher_val_accuracy": float(metrics["val_accuracy"]),
                "teacher_selected_epoch": int(metrics.get("selected_epoch", 0)),
            }
        )
    return records


def _write_teacher_comparison(run_dir: Path, prefix_records: list[dict], teacher_records: list[dict]) -> None:
    teacher_by_depth = {int(record["depth"]): record for record in teacher_records}
    records = []
    for record in prefix_records:
        teacher = teacher_by_depth[int(record["depth"])]
        records.append(
            {
                "depth": record["depth"],
                "shared_val_accuracy": record["val_accuracy"],
                "teacher_val_accuracy": teacher["teacher_val_accuracy"],
                "accuracy_gap": float(record["val_accuracy"]) - float(teacher["teacher_val_accuracy"]),
            }
        )
    (run_dir / "teacher_comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    _write_csv(run_dir / "teacher_comparison.csv", records)


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
    dataset_path = _ensure_dataset(config_path, config)
    dataset = load_pca_dataset(dataset_path)
    training_config = config["training"]
    epochs = int(args.epochs or training_config["epochs"])
    prefix_depths = tuple(range(args.min_depth, args.max_depth + 1))
    prefix_weights = parse_prefix_weights(args.prefix_weights, prefix_depths)

    run_dir = (
        resolve_config_path(config_path, args.run_dir)
        if args.run_dir is not None
        else config_path.parent / f"runs/shared_prefix_distilled_depth_{args.min_depth}_{args.max_depth}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_cache_path = args.teacher_logits_cache or (run_dir / "teacher_logits.npz")

    train_teacher_logits, test_teacher_logits = _load_or_compute_teacher_logits(
        cache_path=teacher_cache_path,
        reuse_cache=args.reuse_teacher_logits,
        teacher_root=args.teacher_run_dir.resolve(),
        teacher_template=args.teacher_template,
        prefix_depths=prefix_depths,
        train_x=dataset.x_train,
        test_x=dataset.x_test,
        batch_size=int(training_config["batch_size"]),
    )

    model_mapping = dict(config["model"])
    model_mapping["layers"] = args.max_depth
    model_config = subunitary_surrogate_config_from_mapping(model_mapping)
    model = build_subunitary_surrogate(model_config)
    state = create_shared_prefix_state(
        model,
        jax.random.key(int(training_config["seed"])),
        jnp.asarray(dataset.x_train[:1]),
        learning_rate=float(training_config["learning_rate"]),
        prefix_depths=prefix_depths,
    )
    if args.initial_params is not None:
        state = state.replace(params=serialization.from_bytes(state.params, args.initial_params.read_bytes()))

    loss_guard_db = training_config.get("loss_guard_db")
    state, history = fit_shared_prefix_distilled_logits(
        model,
        state,
        jnp.asarray(dataset.x_train),
        jnp.asarray(dataset.y_train),
        jnp.asarray(dataset.x_test),
        jnp.asarray(dataset.y_test),
        jnp.asarray(train_teacher_logits),
        jnp.asarray(test_teacher_logits),
        epochs=epochs,
        batch_size=int(training_config["batch_size"]),
        prefix_depths=prefix_depths,
        prefix_weights=prefix_weights,
        routing_weight=float(training_config.get("routing_penalty_weight", 0.0)),
        routing_target=float(training_config.get("routing_leakage_target", 0.0)),
        distillation_alpha=float(args.distillation_alpha),
        distillation_temperature=float(args.distillation_temperature),
        loss_guard_db=None if loss_guard_db is None else float(loss_guard_db),
        loss_guard_weight=float(training_config.get("loss_guard_weight", 0.0)),
        select_best_checkpoint=bool(training_config.get("select_best_checkpoint", False)),
        checkpoint_epochs=training_config.get("checkpoint_epochs"),
        selection_metric=args.selection_metric or "val_accuracy",
        seed=int(training_config["seed"]),
    )

    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if isinstance(history.get("selected_metrics"), dict):
        metrics = dict(history["selected_metrics"])
        metrics["selected_epoch"] = history["selected_epoch"]
    else:
        metrics = {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}
    metrics["prefix_depths"] = list(prefix_depths)
    metrics["max_depth"] = args.max_depth
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    prefix_records = _prefix_metric_records(metrics, prefix_depths)
    (run_dir / "prefix_metrics.json").write_text(json.dumps(prefix_records, indent=2) + "\n")
    _write_csv(run_dir / "prefix_metrics.csv", prefix_records)

    teacher_records = _teacher_metric_records(
        teacher_root=args.teacher_run_dir.resolve(),
        teacher_template=args.teacher_template,
        prefix_depths=prefix_depths,
    )
    (run_dir / "teacher_metrics.json").write_text(json.dumps(teacher_records, indent=2) + "\n")
    _write_teacher_comparison(run_dir, prefix_records, teacher_records)

    resolved_config = dict(config)
    resolved_config["model"] = model_mapping
    resolved_config["outputs"] = {
        "run_dir": str(run_dir),
        "teacher_logits_cache": str(teacher_cache_path),
    }
    resolved_config["training"] = {
        **training_config,
        "epochs": epochs,
        "prefix_depths": list(prefix_depths),
        "prefix_weights": None if prefix_weights is None else list(prefix_weights),
        "selection_metric": args.selection_metric or "val_accuracy",
        "distillation_alpha": float(args.distillation_alpha),
        "distillation_temperature": float(args.distillation_temperature),
        "initial_params": None if args.initial_params is None else str(args.initial_params),
    }
    resolved_config["teachers"] = {
        "run_dir": str(args.teacher_run_dir.resolve()),
        "template": args.teacher_template,
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2) + "\n")

    _save_matrices(model, state.params, model_config.width, run_dir)
    _save_layer_diagnostics(model, state.params, model_config.width, run_dir)
    print(f"saved distilled shared-prefix surrogate run to {run_dir}")


if __name__ == "__main__":
    main()
