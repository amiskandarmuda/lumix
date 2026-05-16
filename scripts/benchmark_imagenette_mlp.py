from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn


def load_runner_module():
    module_path = Path(__file__).resolve().with_name("run_imagenette_adaptive_iteration.py")
    spec = importlib.util.spec_from_file_location("run_imagenette_adaptive_iteration", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("could not load adaptive runner")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner_module()
CACHE_PATH = RUNNER.CACHE_PATH
NUM_CLASSES = RUNNER.NUM_CLASSES
load_or_create_cache = RUNNER.load_or_create_cache


ARTIFACT_DIR = Path("artifacts/imagenette_mlp_benchmark")
TARGET_PARAMS = 4618
EPOCHS = 500
LEARNING_RATE = 1e-3
SEEDS = (7, 11, 17)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    input_kind: str
    input_dim: int
    hidden_dim: int

    @property
    def parameter_count(self) -> int:
        return self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim * NUM_CLASSES + NUM_CLASSES


class SmallMLP(nn.Module):
    hidden_dim: int
    outputs: int = NUM_CLASSES

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        values = nn.Dense(self.hidden_dim)(values)
        values = nn.gelu(values)
        return nn.Dense(self.outputs)(values)


def average_pool_images(images: np.ndarray, pool: int) -> np.ndarray:
    samples, height, width = images.shape
    if height % pool != 0 or width % pool != 0:
        raise ValueError("image dimensions must be divisible by pool")
    pooled = images.reshape(samples, height // pool, pool, width // pool, pool).mean(axis=(2, 4))
    return pooled.reshape(samples, -1)


def build_inputs(images: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw64":
        return images.reshape(images.shape[0], -1)
    if kind == "pooled16":
        return average_pool_images(images, pool=4)
    raise ValueError(f"unknown input kind: {kind}")


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return jnp.asarray((train - mean) / std, dtype=jnp.float32), jnp.asarray((val - mean) / std, dtype=jnp.float32)


def one_hot(labels: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.one_hot(labels, NUM_CLASSES, dtype=jnp.float32)


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def count_params(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))


def train_once(
    config: BenchmarkConfig,
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    val_x: jnp.ndarray,
    val_y: jnp.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    model = SmallMLP(hidden_dim=config.hidden_dim)
    params = model.init(jax.random.key(seed), train_x[:8])["params"]
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    targets = one_hot(train_y)

    def loss_fn(current_params):
        logits = model.apply({"params": current_params}, train_x)
        return optax.softmax_cross_entropy(logits, targets).mean()

    @jax.jit
    def train_step(current_params, state):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, state = optimizer.update(grads, state, current_params)
        return optax.apply_updates(current_params, updates), state, loss

    @jax.jit
    def eval_metrics(current_params):
        train_logits = model.apply({"params": current_params}, train_x)
        val_logits = model.apply({"params": current_params}, val_x)
        return {
            "train_accuracy": accuracy(train_logits, train_y),
            "val_accuracy": accuracy(val_logits, val_y),
            "train_loss": optax.softmax_cross_entropy(train_logits, targets).mean(),
        }

    history = []
    best = None
    for epoch in range(EPOCHS + 1):
        metrics = eval_metrics(params)
        record = {
            "epoch": epoch,
            "train_accuracy": float(metrics["train_accuracy"]),
            "val_accuracy": float(metrics["val_accuracy"]),
            "train_loss": float(metrics["train_loss"]),
        }
        history.append(record)
        if best is None or record["val_accuracy"] > best["val_accuracy"]:
            best = record
        if epoch == EPOCHS:
            break
        params, opt_state, _loss = train_step(params, opt_state)

    final = history[-1]
    return {
        "seed": seed,
        "parameter_count": count_params(params),
        "final": final,
        "best": best,
    }


def run_config(config: BenchmarkConfig, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    train_inputs = build_inputs(x_train, config.input_kind)
    val_inputs = build_inputs(x_val, config.input_kind)
    train_x, val_x = standardize(train_inputs, val_inputs)
    train_y = jnp.asarray(y_train, dtype=jnp.int32)
    val_y = jnp.asarray(y_val, dtype=jnp.int32)

    runs = [
        train_once(config, train_x, train_y, val_x, val_y, seed=seed)
        for seed in SEEDS
    ]
    best_run = max(runs, key=lambda item: item["best"]["val_accuracy"])
    return {
        "name": config.name,
        "input_kind": config.input_kind,
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "parameter_count": config.parameter_count,
        "target_parameter_count": TARGET_PARAMS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "runs": runs,
        "best_run": best_run,
    }


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Imagenette MLP Parameter-Matched Benchmark",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Optical reference: iteration 07, 4,618 learned scalars including optical parameters plus ridge readout, validation accuracy 0.5270.",
        "",
        "| Model | Input | Hidden | Params | Best Seed | Best Epoch | Train Acc | Val Acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        best = item["best_run"]["best"]
        lines.append(
            f"| {item['name']} | {item['input_kind']} | {item['hidden_dim']} | "
            f"{item['parameter_count']} | {item['best_run']['seed']} | {best['epoch']} | "
            f"{best['train_accuracy']:.4f} | {best['val_accuracy']:.4f} |"
        )
    (ARTIFACT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x_train, y_train, x_val, y_val = load_or_create_cache(CACHE_PATH)
    configs = [
        BenchmarkConfig("mlp-raw64-param-matched", "raw64", 4096, 1),
        BenchmarkConfig("mlp-pooled16-param-matched", "pooled16", 256, 17),
    ]
    results = [run_config(config, x_train, y_train, x_val, y_val) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
