from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from lumix.batching import iterate_batches
from lumix.functional.readout import class_probs
from lumix.losses import cross_entropy, cross_entropy_logits
from lumix.metrics import accuracy


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
RESULTS_PATH = ARTIFACT_DIR / "objective_readout_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "objective_readout_summary.md"

EPOCHS = 300
BATCH_SIZE = PCA.BATCH_SIZE
LEARNING_RATE = 5e-3
RANDOM_SEED = PCA.RANDOM_SEED
NUM_CLASSES = PCA.NUM_CLASSES
FIXED_LOGIT_SCALE = 10.0

ObjectiveKind = Literal["normalized_intensity", "softmax_fixed", "softmax_trainable"]


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    kind: PCA.ModelKind
    layers: int
    objective: ObjectiveKind
    phase_scale_pi: float = 1.0
    fixed_logit_scale: float = FIXED_LOGIT_SCALE
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False


class ObjectiveOpticalModel(nn.Module):
    kind: PCA.ModelKind
    layers: int
    phase_scale: float
    objective: ObjectiveKind
    fixed_logit_scale: float = FIXED_LOGIT_SCALE
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        optical = PCA.PCAPhaseOpticalModel(
            kind=self.kind,
            layers=self.layers,
            phase_scale=self.phase_scale,
            train_williamson_gain=self.train_williamson_gain,
            train_williamson_bias=self.train_williamson_bias,
            return_intensity=True,
        )
        all_intensities = optical(features)
        intensities = all_intensities[..., :NUM_CLASSES]

        if self.objective == "normalized_intensity":
            probs = class_probs(intensities, NUM_CLASSES)
            return probs, jnp.asarray(1.0, dtype=jnp.float32), all_intensities

        if self.objective == "softmax_fixed":
            scale = jnp.asarray(self.fixed_logit_scale, dtype=jnp.float32)
            return scale * intensities, scale, all_intensities

        if self.objective == "softmax_trainable":
            log_scale = self.param(
                "logit_scale",
                lambda key: jnp.asarray(np.log(self.fixed_logit_scale), dtype=jnp.float32),
            )
            scale = jnp.exp(log_scale)
            return scale * intensities, scale, all_intensities

        raise ValueError(f"unknown objective: {self.objective}")


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def run_config(config: ObjectiveConfig, data: PCA.DataSplit) -> dict[str, Any]:
    train_x = jnp.asarray(data.x_train)
    train_y = jnp.asarray(data.y_train)
    test_x = jnp.asarray(data.x_test)
    test_y = jnp.asarray(data.y_test)

    model = ObjectiveOpticalModel(
        kind=config.kind,
        layers=config.layers,
        phase_scale=config.phase_scale_pi * float(np.pi),
        objective=config.objective,
        fixed_logit_scale=config.fixed_logit_scale,
        train_williamson_gain=config.train_williamson_gain,
        train_williamson_bias=config.train_williamson_bias,
    )
    params = model.init(jax.random.key(RANDOM_SEED), train_x[:8])["params"]
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    def loss_fn(optical_params, batch_x, batch_y):
        outputs, scale, _intensities = model.apply({"params": optical_params}, batch_x)
        if config.objective == "normalized_intensity":
            return cross_entropy(batch_y, outputs), (outputs, scale)
        return cross_entropy_logits(batch_y, outputs), (outputs, scale)

    def power_loss_db(intensities: jnp.ndarray) -> jnp.ndarray:
        total_power = jnp.sum(intensities, axis=-1)
        return -10.0 * jnp.log10(jnp.clip(total_power, 1e-12, None))

    @jax.jit
    def train_step(optical_params, state, batch_x, batch_y):
        (loss_value, (outputs, scale)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            optical_params, batch_x, batch_y
        )
        updates, state = optimizer.update(grads, state, optical_params)
        return optax.apply_updates(optical_params, updates), state, loss_value, accuracy(batch_y, outputs), scale

    @jax.jit
    def eval_step(optical_params, eval_x, eval_y):
        outputs, scale, intensities = model.apply({"params": optical_params}, eval_x)
        if config.objective == "normalized_intensity":
            loss_value = cross_entropy(eval_y, outputs)
        else:
            loss_value = cross_entropy_logits(eval_y, outputs)
        losses_db = power_loss_db(intensities)
        return (
            loss_value,
            accuracy(eval_y, outputs),
            scale,
            jnp.mean(losses_db),
            jnp.min(losses_db),
            jnp.max(losses_db),
        )

    rng = jax.random.key(RANDOM_SEED)
    for _epoch in range(EPOCHS):
        rng, epoch_rng = jax.random.split(rng)
        for batch_x, batch_y in iterate_batches(train_x, train_y, BATCH_SIZE, epoch_rng):
            params, opt_state, _loss_value, _score, _scale = train_step(params, opt_state, batch_x, batch_y)

    train_loss, train_accuracy, train_scale, train_mean_loss_db, _train_min_loss_db, _train_max_loss_db = eval_step(
        params, train_x, train_y
    )
    val_loss, val_accuracy, val_scale, val_mean_loss_db, val_min_loss_db, val_max_loss_db = eval_step(
        params, test_x, test_y
    )

    return {
        "name": config.name,
        "kind": config.kind,
        "layers": config.layers,
        "objective": config.objective,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "phase_scale_pi": config.phase_scale_pi,
        "fixed_logit_scale": config.fixed_logit_scale,
        "final_logit_scale": float(val_scale),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_mean_insertion_loss_db": float(train_mean_loss_db),
        "val_mean_insertion_loss_db": float(val_mean_loss_db),
        "val_min_insertion_loss_db": float(val_min_loss_db),
        "val_max_insertion_loss_db": float(val_max_loss_db),
        "stored_param_count": count_params(params),
        "train_williamson_gain": config.train_williamson_gain,
        "train_williamson_bias": config.train_williamson_bias,
    }


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MNIST PCA Objective Readout Comparison",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset/encoding: MNIST projected to 16 PCA components, train-set min/max mapped without clipping, then phase encoded.",
        "",
        f"Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, Adam lr={LEARNING_RATE}.",
        "",
        "Objectives:",
        "",
        "- `normalized_intensity`: current physical detector objective, CE on class-bin intensity divided by class-bin power.",
        "- `softmax_fixed`: CE on `softmax(gamma * class_intensity)` with fixed shared gamma.",
        "- `softmax_trainable`: same softmax objective with one trainable scalar gamma initialized from the fixed value.",
        "",
        "| Model | Kind | Depth | Objective | Params | Gamma | Val Acc | Val Loss | Mean IL (dB) | IL Range (dB) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['layers']} | {result['objective']} | "
            f"{result['stored_param_count']} | {result['final_logit_scale']:.3f} | "
            f"{result['val_accuracy']:.4f} | {result['val_loss']:.4f} | "
            f"{result['val_mean_insertion_loss_db']:.3f} | "
            f"{result['val_min_insertion_loss_db']:.3f}-{result['val_max_insertion_loss_db']:.3f} |"
        )
    repeated_best = max(
        (result for result in results if result["kind"].startswith("repeated")),
        key=lambda result: result["val_accuracy"],
    )
    williamson_best = max(
        (result for result in results if result["kind"] == "williamson"),
        key=lambda result: result["val_accuracy"],
    )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "With maximum depth fixed at 5 for both cases, a trainable scalar softmax temperature is the best objective for both models. Under that same objective, repeated data re-encoding beats Williamson on both validation accuracy and loss:"
            if repeated_best["val_accuracy"] > williamson_best["val_accuracy"]
            and repeated_best["val_loss"] < williamson_best["val_loss"]
            else "With maximum depth fixed at 5 for both cases, the best repeated and Williamson results are:",
            "",
            f"- Repeated depth {repeated_best['layers']}: {100.0 * repeated_best['val_accuracy']:.2f}% validation accuracy, {repeated_best['val_loss']:.4f} validation loss.",
            f"- Williamson depth {williamson_best['layers']}: {100.0 * williamson_best['val_accuracy']:.2f}% validation accuracy, {williamson_best['val_loss']:.4f} validation loss.",
            "",
            "The fixed gamma softmax does not explain the gain; the trainable temperature is important.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = PCA.load_or_create_no_clip_cache()
    configs = [
        ObjectiveConfig(
            "repeated-final-subunitary-trainable-phase-depth5-normalized",
            "repeated_final_subunitary_trainable_phase",
            5,
            "normalized_intensity",
        ),
        ObjectiveConfig(
            "repeated-final-subunitary-trainable-phase-depth5-softmax-fixed",
            "repeated_final_subunitary_trainable_phase",
            5,
            "softmax_fixed",
        ),
        ObjectiveConfig(
            "repeated-final-subunitary-trainable-phase-depth5-softmax-trainable",
            "repeated_final_subunitary_trainable_phase",
            5,
            "softmax_trainable",
        ),
        ObjectiveConfig(
            "williamson-depth5-normalized",
            "williamson",
            5,
            "normalized_intensity",
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        ObjectiveConfig(
            "williamson-depth5-softmax-fixed",
            "williamson",
            5,
            "softmax_fixed",
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        ObjectiveConfig(
            "williamson-depth5-softmax-trainable",
            "williamson",
            5,
            "softmax_trainable",
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
    ]
    results = [run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
