from __future__ import annotations

import argparse
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
from experiments.common.mnist_pca import (
    fit_pca16_dataset,
    is_current_pca_dataset,
    load_mnist,
    load_pca_dataset,
    save_pca_dataset,
)
from experiments.common.models import (
    build_subunitary_surrogate,
    extract_inverse_design_matrices,
    subunitary_surrogate_config_from_mapping,
)
from lumix.state import create_state
from lumix.train import fit_logits
from experiments.common.training import fit_routing_regularized_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST PCA-16 subunitary surrogate.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _ensure_dataset(config_path: Path, config: dict) -> Path:
    dataset_config = config["dataset"]
    processed_path = resolve_config_path(config_path, dataset_config["processed_path"])
    expected_preprocessing = str(dataset_config.get("preprocessing", "standardize_minmax_no_clip"))
    if is_current_pca_dataset(processed_path, expected_preprocessing=expected_preprocessing):
        return processed_path

    raw_dir = resolve_config_path(config_path, dataset_config["raw_dir"])
    train_images, train_labels, test_images, test_labels = load_mnist(raw_dir)
    dataset = fit_pca16_dataset(
        train_images,
        train_labels,
        test_images,
        test_labels,
        components=int(dataset_config["components"]),
        preprocessing=expected_preprocessing,
    )
    save_pca_dataset(dataset, processed_path)
    return processed_path


def _save_matrices(model, params, width: int, run_dir: Path) -> None:
    sample_x = jnp.zeros((1, width), dtype=jnp.float32)
    matrices = extract_inverse_design_matrices(model, params, sample_x)
    target_dir = run_dir / "inverse_design_targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale_target in target_dir.glob("layer_*.npy"):
        stale_target.unlink()
    payload = {}
    for name, matrix in matrices.items():
        array = jax.device_get(matrix)
        payload[name] = array
        with (target_dir / f"layer_{name}.npy").open("wb") as handle:
            import numpy as np

            np.save(handle, array)
    import numpy as np

    np.savez_compressed(target_dir / "targets.npz", **payload)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    dataset_path = _ensure_dataset(config_path, config)
    dataset = load_pca_dataset(dataset_path)
    training_config = config["training"]
    epochs = int(args.epochs or training_config["epochs"])
    run_dir = resolve_config_path(config_path, config["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = subunitary_surrogate_config_from_mapping(config["model"])
    model = build_subunitary_surrogate(model_config)
    state = create_state(
        model,
        jax.random.key(int(training_config["seed"])),
        jnp.asarray(dataset.x_train[:1]),
        learning_rate=float(training_config["learning_rate"]),
    )
    routing_weight = float(training_config.get("routing_penalty_weight", 0.0))
    routing_target = float(training_config.get("routing_leakage_target", 0.0))
    loss_guard_db = training_config.get("loss_guard_db")
    loss_guard_weight = float(training_config.get("loss_guard_weight", 0.0))
    if routing_weight > 0.0 or loss_guard_weight > 0.0:
        state, history = fit_routing_regularized_logits(
            model,
            state,
            jnp.asarray(dataset.x_train),
            jnp.asarray(dataset.y_train),
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            epochs=epochs,
            batch_size=int(training_config["batch_size"]),
            routing_weight=routing_weight,
            routing_target=routing_target,
            loss_guard_db=None if loss_guard_db is None else float(loss_guard_db),
            loss_guard_weight=loss_guard_weight,
            select_best_checkpoint=bool(training_config.get("select_best_checkpoint", False)),
            checkpoint_epochs=training_config.get("checkpoint_epochs"),
            seed=int(training_config["seed"]),
        )
    else:
        state, history = fit_logits(
            state,
            jnp.asarray(dataset.x_train),
            jnp.asarray(dataset.y_train),
            jnp.asarray(dataset.x_test),
            jnp.asarray(dataset.y_test),
            epochs=epochs,
            batch_size=int(training_config["batch_size"]),
            seed=int(training_config["seed"]),
        )

    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if isinstance(history.get("selected_metrics"), dict):
        metrics = dict(history["selected_metrics"])
        metrics["selected_epoch"] = history["selected_epoch"]
    else:
        metrics = {key: values[-1] for key, values in history.items() if isinstance(values, list) and values}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _save_matrices(model, state.params, model_config.width, run_dir)
    print(f"saved surrogate run to {run_dir}")


if __name__ == "__main__":
    main()
