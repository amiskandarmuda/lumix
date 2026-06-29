from __future__ import annotations

import argparse
import csv
import json
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
from experiments.common.training import fit_routing_regularized_logits
from experiments.scripts.train_shared_prefix_surrogate import _save_layer_diagnostics, _write_csv, apply_cli_overrides
from experiments.scripts.train_surrogate import _ensure_dataset, _save_matrices
from lumix.state import create_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an independent-depth surrogate sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--run-name-template", type=str, default="layers_{depth}")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--bias-ports", type=int, default=None)
    parser.add_argument("--routing-limit", type=int, default=None)
    parser.add_argument("--routing-weight", type=float, default=None)
    parser.add_argument("--routing-target", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def _metrics_from_history(history: dict) -> dict:
    if isinstance(history.get("selected_metrics"), dict):
        metrics = dict(history["selected_metrics"])
        metrics["selected_epoch"] = history["selected_epoch"]
        return metrics
    return {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}


def _summary_record(*, depth: int, run_dir: Path, metrics: dict, model_mapping: dict) -> dict:
    return {
        "depth": depth,
        "run_dir": str(run_dir),
        "width": int(model_mapping["width"]),
        "bias_ports": int(model_mapping.get("bias_ports", 0)),
        "routing_limit": model_mapping.get("routing_limit"),
        "val_accuracy": float(metrics["val_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "val_cross_entropy": float(metrics["val_cross_entropy"]),
        "val_mean_insertion_loss_db": float(metrics["val_mean_insertion_loss_db"]),
        "val_mean_output_power": float(metrics["val_mean_output_power"]),
        "val_routing_leakage": float(metrics["val_routing_leakage"]),
        "val_gamma": float(metrics["val_gamma"]),
        "selected_epoch": int(metrics.get("selected_epoch", 0)),
    }


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
    root_run_dir = resolve_config_path(config_path, args.run_dir)
    root_run_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
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
        loss_guard_db = training_config.get("loss_guard_db")
        state, history = fit_routing_regularized_logits(
            model,
            state,
            jnp.asarray(dataset.x_train),
            jnp.asarray(dataset.y_train),
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            epochs=epochs,
            batch_size=int(training_config["batch_size"]),
            routing_weight=float(training_config.get("routing_penalty_weight", 0.0)),
            routing_target=float(training_config.get("routing_leakage_target", 0.0)),
            loss_guard_db=None if loss_guard_db is None else float(loss_guard_db),
            loss_guard_weight=float(training_config.get("loss_guard_weight", 0.0)),
            select_best_checkpoint=bool(training_config.get("select_best_checkpoint", False)),
            checkpoint_epochs=training_config.get("checkpoint_epochs"),
            seed=int(training_config["seed"]),
        )

        (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        metrics = _metrics_from_history(history)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        resolved_config = dict(config)
        resolved_config["model"] = model_mapping
        resolved_config["training"] = {**training_config, "epochs": epochs}
        resolved_config["outputs"] = {"run_dir": str(run_dir)}
        (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2) + "\n")

        sample_width = model_config.width - model_config.bias_ports
        _save_matrices(model, state.params, sample_width, run_dir)
        _save_layer_diagnostics(model, state.params, sample_width, run_dir)
        summary_records.append(_summary_record(depth=depth, run_dir=run_dir, metrics=metrics, model_mapping=model_mapping))
        print(f"depth {depth}: val_accuracy={metrics['val_accuracy']:.6f} saved to {run_dir}")

    (root_run_dir / "summary.json").write_text(json.dumps(summary_records, indent=2) + "\n")
    _write_csv(root_run_dir / "summary.csv", summary_records)

    with (root_run_dir / "summary_compact.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("depth", "val_accuracy", "selected_epoch"))
        writer.writeheader()
        for record in summary_records:
            writer.writerow(
                {
                    "depth": record["depth"],
                    "val_accuracy": record["val_accuracy"],
                    "selected_epoch": record["selected_epoch"],
                }
            )

    print(f"saved independent sweep summary to {root_run_dir}")


if __name__ == "__main__":
    main()
