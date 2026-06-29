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
    extract_inverse_design_matrices,
    subunitary_surrogate_config_from_mapping,
)
from experiments.common.training import (
    create_shared_prefix_state,
    fit_shared_prefix_routing_regularized_logits,
)
from experiments.scripts.train_surrogate import _ensure_dataset, _save_matrices


PREFIX_METRIC_FIELDS = (
    "accuracy",
    "cross_entropy",
    "routing_leakage",
    "routing_excess",
    "mean_insertion_loss_db",
    "loss_excess",
    "mean_output_power",
    "gamma",
    "val_accuracy",
    "val_cross_entropy",
    "val_routing_leakage",
    "val_routing_excess",
    "val_mean_insertion_loss_db",
    "val_loss_excess",
    "val_mean_output_power",
    "val_gamma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one shared optical stack and evaluate prefix depths.")
    parser.add_argument("--config", required=True, type=Path)
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
    return parser.parse_args()


def parse_prefix_weights(value: str | None, prefix_depths: tuple[int, ...]) -> tuple[float, ...] | None:
    if value is None:
        return None
    weights = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(weights) != len(prefix_depths):
        raise ValueError("prefix weight count must match prefix depth count")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("prefix weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("prefix weights must sum to a positive value")
    return weights


def apply_cli_overrides(
    config: dict,
    *,
    routing_limit: int | None = None,
    routing_weight: float | None = None,
    routing_target: float | None = None,
    learning_rate: float | None = None,
) -> None:
    if routing_limit is not None:
        config["model"]["routing_limit"] = int(routing_limit)
    if routing_weight is not None:
        config["training"]["routing_penalty_weight"] = float(routing_weight)
    if routing_target is not None:
        config["training"]["routing_leakage_target"] = float(routing_target)
    if learning_rate is not None:
        config["training"]["learning_rate"] = float(learning_rate)


def _prefix_metric_records(metrics: dict, prefix_depths: tuple[int, ...]) -> list[dict[str, float | int]]:
    records = []
    for depth in prefix_depths:
        record: dict[str, float | int] = {"depth": depth}
        for field in PREFIX_METRIC_FIELDS:
            record[field] = float(metrics[f"prefix_{depth}_{field}"])
        records.append(record)
    return records


def _write_csv(path: Path, records: list[dict[str, float | int]]) -> None:
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _layer_insertion_loss_records(matrices: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    records = []
    for index, (name, matrix) in enumerate(sorted(matrices.items())):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        spectral_norm = float(singular_values[0])
        records.append(
            {
                "layer": index,
                "name": name,
                "spectral_norm": spectral_norm,
                "insertion_loss_db": float(-20.0 * np.log10(max(spectral_norm, 1e-12))),
            }
        )
    return records


def _save_layer_diagnostics(model, params, width: int, run_dir: Path) -> None:
    sample_x = jnp.zeros((1, width), dtype=jnp.float32)
    matrices = {
        name: np.asarray(jax.device_get(matrix))
        for name, matrix in extract_inverse_design_matrices(model, params, sample_x).items()
    }
    loss_records = _layer_insertion_loss_records(matrices)
    (run_dir / "per_layer_insertion_loss.json").write_text(json.dumps(loss_records, indent=2) + "\n")
    _write_csv(run_dir / "per_layer_insertion_loss.csv", loss_records)
    _save_plasma_heatmaps(matrices, run_dir)


def _save_plasma_heatmaps(matrices: dict[str, np.ndarray], run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    visual_dir = run_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=180)
        image = ax.imshow(np.abs(matrix), cmap="plasma", aspect="auto")
        ax.set_title(f"{name} |abs|")
        ax.set_xlabel("input port")
        ax.set_ylabel("output port")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(visual_dir / f"{name}_plasma_abs.png")
        plt.close(fig)


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
        else config_path.parent / f"runs/shared_prefix_depth_{args.min_depth}_{args.max_depth}"
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
        prefix_depths=prefix_depths,
    )
    loss_guard_db = training_config.get("loss_guard_db")
    state, history = fit_shared_prefix_routing_regularized_logits(
        model,
        state,
        jnp.asarray(dataset.x_train),
        jnp.asarray(dataset.y_train),
        jnp.asarray(dataset.x_test),
        jnp.asarray(dataset.y_test),
        epochs=epochs,
        batch_size=int(training_config["batch_size"]),
        prefix_depths=prefix_depths,
        prefix_weights=prefix_weights,
        routing_weight=float(training_config.get("routing_penalty_weight", 0.0)),
        routing_target=float(training_config.get("routing_leakage_target", 0.0)),
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

    resolved_config = dict(config)
    resolved_config["model"] = model_mapping
    resolved_config["outputs"] = {"run_dir": str(run_dir)}
    resolved_config["training"] = {
        **training_config,
        "epochs": epochs,
        "prefix_depths": list(prefix_depths),
        "prefix_weights": None if prefix_weights is None else list(prefix_weights),
        "selection_metric": args.selection_metric or "val_accuracy",
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2) + "\n")

    _save_matrices(model, state.params, model_config.width, run_dir)
    _save_layer_diagnostics(model, state.params, model_config.width, run_dir)
    print(f"saved shared-prefix surrogate run to {run_dir}")


if __name__ == "__main__":
    main()
