from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from lumix.batching import iterate_batches
from lumix.functional import solve_ridge
from lumix.linen import RidgeReadout
from lumix.losses import cross_entropy
from lumix.metrics import accuracy, mean_squared_error


def load_pca_module():
    module_path = Path(__file__).resolve().with_name("benchmark_mnist_pca_phase.py")
    spec = importlib.util.spec_from_file_location("benchmark_mnist_pca_phase", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("could not load PCA benchmark module")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PCA = load_pca_module()

ARTIFACT_DIR = Path("artifacts/mnist_pca_phase_comparison")
RESULTS_PATH = ARTIFACT_DIR / "ridge_readout_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "ridge_readout_summary.md"

EPOCHS = 300
BATCH_SIZE = PCA.BATCH_SIZE
LEARNING_RATE = 5e-3
RIDGE_ALPHA = 1e-3
RANDOM_SEED = PCA.RANDOM_SEED
NUM_CLASSES = PCA.NUM_CLASSES


@dataclass(frozen=True)
class RidgeConfig:
    name: str
    kind: PCA.ModelKind
    layers: int
    phase_scale_pi: float = 1.0
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False


def standardize_from_train(train_features: jnp.ndarray, test_features: jnp.ndarray):
    mean = jnp.mean(train_features, axis=0, keepdims=True)
    std = jnp.std(train_features, axis=0, keepdims=True)
    std = jnp.where(std == 0.0, 1.0, std)
    return (train_features - mean) / std, (test_features - mean) / std


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def run_config(config: RidgeConfig, data: PCA.DataSplit) -> dict[str, Any]:
    train_x = jnp.asarray(data.x_train)
    train_y = jnp.asarray(data.y_train)
    test_x = jnp.asarray(data.x_test)
    test_y = jnp.asarray(data.y_test)

    model = PCA.PCAPhaseOpticalModel(
        kind=config.kind,
        layers=config.layers,
        phase_scale=config.phase_scale_pi * float(jnp.pi),
        train_williamson_gain=config.train_williamson_gain,
        train_williamson_bias=config.train_williamson_bias,
    )
    params = model.init(jax.random.key(RANDOM_SEED), train_x[:8])["params"]
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    def loss_fn(optical_params, batch_x, batch_y):
        probs = model.apply({"params": optical_params}, batch_x)
        return cross_entropy(batch_y, probs), probs

    @jax.jit
    def train_step(optical_params, state, batch_x, batch_y):
        (loss_value, probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(optical_params, batch_x, batch_y)
        updates, state = optimizer.update(grads, state, optical_params)
        return optax.apply_updates(optical_params, updates), state, loss_value, accuracy(batch_y, probs)

    @jax.jit
    def prob_eval(optical_params, eval_x, eval_y):
        loss_value, probs = loss_fn(optical_params, eval_x, eval_y)
        return loss_value, accuracy(eval_y, probs)

    rng = jax.random.key(RANDOM_SEED)
    for _epoch in range(EPOCHS):
        rng, epoch_rng = jax.random.split(rng)
        for batch_x, batch_y in iterate_batches(train_x, train_y, BATCH_SIZE, epoch_rng):
            params, opt_state, _loss_value, _score = train_step(params, opt_state, batch_x, batch_y)

    prob_val_loss, prob_val_accuracy = prob_eval(params, test_x, test_y)

    feature_model = PCA.PCAPhaseOpticalModel(
        kind=config.kind,
        layers=config.layers,
        phase_scale=config.phase_scale_pi * float(jnp.pi),
        train_williamson_gain=config.train_williamson_gain,
        train_williamson_bias=config.train_williamson_bias,
        return_intensity=True,
    )
    train_features_raw = feature_model.apply({"params": params}, train_x)
    test_features_raw = feature_model.apply({"params": params}, test_x)
    train_features, test_features = standardize_from_train(train_features_raw, test_features_raw)

    ridge_params = solve_ridge(train_features, train_y, alpha=RIDGE_ALPHA, use_bias=True)
    ridge = RidgeReadout(features=NUM_CLASSES)
    train_logits = ridge.apply({"params": ridge_params}, train_features)
    test_logits = ridge.apply({"params": ridge_params}, test_features)

    return {
        "name": config.name,
        "kind": config.kind,
        "layers": config.layers,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "ridge_alpha": RIDGE_ALPHA,
        "phase_scale_pi": config.phase_scale_pi,
        "stored_optical_param_count": count_params(params),
        "ridge_param_count": int(train_features.shape[-1] * NUM_CLASSES + NUM_CLASSES),
        "feature_count": int(train_features.shape[-1]),
        "prob_val_accuracy": float(prob_val_accuracy),
        "prob_val_loss": float(prob_val_loss),
        "ridge_train_accuracy": float(accuracy(train_y, train_logits)),
        "ridge_val_accuracy": float(accuracy(test_y, test_logits)),
        "ridge_train_mse": float(mean_squared_error(train_y, train_logits)),
        "ridge_val_mse": float(mean_squared_error(test_y, test_logits)),
    }


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MNIST PCA Ridge Readout Comparison",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset/encoding: no-clipping PCA phase mapping. Optical stacks are first trained with normalized-intensity cross-entropy, then raw output intensities are standardized and fed to a closed-form ridge readout.",
        "",
        f"Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, Adam lr={LEARNING_RATE}. Ridge alpha={RIDGE_ALPHA}.",
        "",
        "| Model | Kind | Depth | Optical Params | Ridge Params | Prob Acc | Prob CE | Ridge Acc | Ridge MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['layers']} | "
            f"{result['stored_optical_param_count']} | {result['ridge_param_count']} | "
            f"{result['prob_val_accuracy']:.4f} | {result['prob_val_loss']:.4f} | "
            f"{result['ridge_val_accuracy']:.4f} | {result['ridge_val_mse']:.4f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = PCA.load_or_create_no_clip_cache()
    configs = [
        RidgeConfig("repeated-unitary-depth6", "repeated_phase", 6),
        RidgeConfig("repeated-unitary-depth7", "repeated_phase", 7),
        RidgeConfig("repeated-subunitary-depth6", "repeated_subunitary", 6),
        RidgeConfig("repeated-final-subunitary-trainable-phase-depth7", "repeated_final_subunitary_trainable_phase", 7),
        RidgeConfig("williamson-depth6-train-gain-bias", "williamson", 6, train_williamson_gain=True, train_williamson_bias=True),
        RidgeConfig("williamson-depth7-train-gain-bias", "williamson", 7, train_williamson_gain=True, train_williamson_bias=True),
    ]
    results = [run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
