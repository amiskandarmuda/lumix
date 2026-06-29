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
    fit_shared_prefix_routing_regularized_logits,
)
from experiments.scripts.train_shared_prefix_surrogate import (
    _prefix_metric_records,
    _save_layer_diagnostics,
    _write_csv,
    apply_cli_overrides,
)
from experiments.scripts.train_surrogate import _ensure_dataset, _save_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressively train an optical-only shared-prefix surrogate.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="Alias for --stage-epochs.")
    parser.add_argument("--stage-epochs", type=int, default=80)
    parser.add_argument("--adaptive-rounds", type=int, default=3)
    parser.add_argument("--adaptive-epochs", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--skip-growth-stages", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--target-accuracies", type=str, default="0.80,0.86,0.90,0.92,0.93,0.936,0.94,0.943")
    parser.add_argument("--base-weight", type=float, default=0.1)
    parser.add_argument("--deficit-scale", type=float, default=20.0)
    parser.add_argument("--deficit-power", type=float, default=1.0)
    parser.add_argument("--new-depth-boost", type=float, default=1.5)
    parser.add_argument("--routing-limit", type=int, default=None)
    parser.add_argument("--routing-weight", type=float, default=None)
    parser.add_argument("--routing-target", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--initial-params", type=Path, default=None)
    return parser.parse_args()


def parse_float_list(value: str, *, expected_count: int, label: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(values) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} comma-separated values")
    return values


def adaptive_prefix_weights(
    *,
    prefix_depths: tuple[int, ...],
    target_accuracies: tuple[float, ...],
    previous_metrics: dict[str, float] | None,
    base_weight: float,
    deficit_scale: float,
    deficit_power: float,
    new_depth_boost: float,
) -> tuple[float, ...]:
    if len(prefix_depths) != len(target_accuracies):
        raise ValueError("target_accuracies must match prefix_depths")
    if base_weight < 0.0 or deficit_scale < 0.0 or deficit_power <= 0.0 or new_depth_boost < 0.0:
        raise ValueError("weighting parameters must be non-negative and deficit_power must be positive")

    raw_weights = []
    newest_depth = max(prefix_depths)
    for depth, target in zip(prefix_depths, target_accuracies, strict=True):
        score = None if previous_metrics is None else previous_metrics.get(f"prefix_{depth}_val_accuracy")
        deficit = max(float(target) - float(score), 0.0) if score is not None else float(target)
        weight = float(base_weight) + float(deficit_scale) * deficit**float(deficit_power)
        if depth == newest_depth:
            weight += float(new_depth_boost)
        raw_weights.append(weight)

    total = sum(raw_weights)
    if total <= 0.0:
        raise ValueError("adaptive weights must sum to a positive value")
    return tuple(weight / total for weight in raw_weights)


def progressive_growth_depths(*, max_depth: int, skip_growth_stages: bool) -> tuple[int, ...]:
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if skip_growth_stages:
        return ()
    return tuple(range(1, max_depth + 1))


def _stage_record(
    *,
    stage: str,
    epoch_count: int,
    prefix_depths: tuple[int, ...],
    prefix_weights: tuple[float, ...],
    metrics: dict[str, float],
) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {
        "stage": stage,
        "epochs": epoch_count,
        "prefix_depths": ",".join(str(depth) for depth in prefix_depths),
        "prefix_weights": ",".join(f"{weight:.8g}" for weight in prefix_weights),
    }
    if "min_target_margin" in metrics:
        record["min_target_margin"] = float(metrics["min_target_margin"])
    if "val_accuracy" in metrics:
        record["val_accuracy"] = float(metrics["val_accuracy"])
    for depth in prefix_depths:
        key = f"prefix_{depth}_val_accuracy"
        if key in metrics:
            record[key] = float(metrics[key])
    return record


def _selected_or_last_metrics(history: dict) -> dict[str, float]:
    if isinstance(history.get("selected_metrics"), dict):
        metrics = dict(history["selected_metrics"])
        metrics["selected_epoch"] = history["selected_epoch"]
        return metrics
    return {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}


def _write_stage_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    if not records:
        return
    fieldnames = sorted({field for record in records for field in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.max_depth < 1:
        raise ValueError("--max-depth must be at least 1")

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
    stage_epochs = int(args.epochs or args.stage_epochs)
    target_accuracies = parse_float_list(
        args.target_accuracies,
        expected_count=args.max_depth,
        label="target accuracies",
    )

    run_dir = (
        resolve_config_path(config_path, args.run_dir)
        if args.run_dir is not None
        else config_path.parent / f"runs/shared_prefix_progressive_depth_1_{args.max_depth}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model_mapping = dict(config["model"])
    model_mapping["layers"] = args.max_depth
    model_config = subunitary_surrogate_config_from_mapping(model_mapping)
    model = build_subunitary_surrogate(model_config)
    state = create_shared_prefix_state(
        model,
        jax.random.key(int(training_config["seed"])),
        jnp.asarray(dataset.x_train[:1]),
        learning_rate=float(training_config["learning_rate"]),
        prefix_depths=tuple(range(1, args.max_depth + 1)),
    )
    if args.initial_params is not None:
        state = state.replace(params=serialization.from_bytes(state.params, args.initial_params.read_bytes()))

    loss_guard_db = training_config.get("loss_guard_db")
    previous_metrics: dict[str, float] | None = None
    stage_records: list[dict[str, float | int | str]] = []
    stage_histories: list[dict] = []

    def run_stage(stage_name: str, active_depths: tuple[int, ...], epoch_count: int, new_depth_boost: float):
        nonlocal state, previous_metrics
        active_targets = target_accuracies[: len(active_depths)]
        weights = adaptive_prefix_weights(
            prefix_depths=active_depths,
            target_accuracies=active_targets,
            previous_metrics=previous_metrics,
            base_weight=float(args.base_weight),
            deficit_scale=float(args.deficit_scale),
            deficit_power=float(args.deficit_power),
            new_depth_boost=new_depth_boost,
        )
        state, history = fit_shared_prefix_routing_regularized_logits(
            model,
            state,
            jnp.asarray(dataset.x_train),
            jnp.asarray(dataset.y_train),
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            epochs=epoch_count,
            batch_size=int(training_config["batch_size"]),
            prefix_depths=active_depths,
            prefix_weights=weights,
            routing_weight=float(training_config.get("routing_penalty_weight", 0.0)),
            routing_target=float(training_config.get("routing_leakage_target", 0.0)),
            loss_guard_db=None if loss_guard_db is None else float(loss_guard_db),
            loss_guard_weight=float(training_config.get("loss_guard_weight", 0.0)),
            select_best_checkpoint=True,
            checkpoint_epochs=None,
            selection_metric="min_target_margin",
            target_accuracies=active_targets,
            seed=int(training_config["seed"]),
        )
        previous_metrics = _selected_or_last_metrics(history)
        stage_histories.append({"stage": stage_name, "history": history})
        stage_records.append(
            _stage_record(
                stage=stage_name,
                epoch_count=epoch_count,
                prefix_depths=active_depths,
                prefix_weights=weights,
                metrics=previous_metrics,
            )
        )

    for depth in progressive_growth_depths(max_depth=args.max_depth, skip_growth_stages=bool(args.skip_growth_stages)):
        run_stage(
            f"grow_depth_{depth}",
            tuple(range(1, depth + 1)),
            stage_epochs,
            float(args.new_depth_boost),
        )

    for round_index in range(1, int(args.adaptive_rounds) + 1):
        run_stage(
            f"adaptive_round_{round_index}",
            tuple(range(1, args.max_depth + 1)),
            int(args.adaptive_epochs),
            0.0,
        )

    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    (run_dir / "stage_history.json").write_text(json.dumps(stage_histories, indent=2) + "\n")
    _write_stage_csv(run_dir / "stage_summary.csv", stage_records)
    (run_dir / "stage_summary.json").write_text(json.dumps(stage_records, indent=2) + "\n")

    final_metrics = previous_metrics or {}
    final_metrics["prefix_depths"] = list(range(1, args.max_depth + 1))
    final_metrics["max_depth"] = args.max_depth
    final_metrics["target_accuracies"] = list(target_accuracies)
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")

    prefix_records = _prefix_metric_records(final_metrics, tuple(range(1, args.max_depth + 1)))
    (run_dir / "prefix_metrics.json").write_text(json.dumps(prefix_records, indent=2) + "\n")
    _write_csv(run_dir / "prefix_metrics.csv", prefix_records)

    resolved_config = dict(config)
    resolved_config["model"] = model_mapping
    resolved_config["outputs"] = {"run_dir": str(run_dir)}
    resolved_config["training"] = {
        **training_config,
        "stage_epochs": stage_epochs,
        "adaptive_rounds": int(args.adaptive_rounds),
        "adaptive_epochs": int(args.adaptive_epochs),
        "target_accuracies": list(target_accuracies),
        "base_weight": float(args.base_weight),
        "deficit_scale": float(args.deficit_scale),
        "deficit_power": float(args.deficit_power),
        "new_depth_boost": float(args.new_depth_boost),
        "skip_growth_stages": bool(args.skip_growth_stages),
        "initial_params": None if args.initial_params is None else str(args.initial_params),
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2) + "\n")

    _save_matrices(model, state.params, model_config.width, run_dir)
    _save_layer_diagnostics(model, state.params, model_config.width, run_dir)
    print(f"saved progressive shared-prefix surrogate run to {run_dir}")


if __name__ == "__main__":
    main()
