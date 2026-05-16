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
from lumix.functional.readout import intensity
from lumix.functional.williamson import williamson_response
from lumix.linen import InformationEncoder, SubUnitaryLinear, UnitaryLinear
from lumix.losses import cross_entropy_logits
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
RESULTS_PATH = ARTIFACT_DIR / "tied_three_way_depth_curve_trainable_softmax_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "tied_three_way_depth_curve_trainable_softmax_summary.md"

EPOCHS = 300
BATCH_SIZE = PCA.BATCH_SIZE
LEARNING_RATE = 5e-3
RANDOM_SEED = PCA.RANDOM_SEED
NUM_CLASSES = PCA.NUM_CLASSES
WIDTH = PCA.WIDTH
FIXED_LOGIT_SCALE = 10.0

Variant = Literal["tied_unitary_repeated", "tied_subunitary_repeated", "tied_williamson"]


@dataclass(frozen=True)
class TiedConfig:
    name: str
    variant: Variant
    layers: int
    phase_scale_pi: float = 1.0
    fixed_logit_scale: float = FIXED_LOGIT_SCALE


class TiedOpticalModel(nn.Module):
    variant: Variant
    layers: int
    phase_scale: float
    fixed_logit_scale: float = FIXED_LOGIT_SCALE
    williamson_tap: float = PCA.WILLIAMSON_TAP
    williamson_gain: float = PCA.WILLIAMSON_GAIN
    williamson_bias: float = 0.50 * float(np.pi)

    def _input_fields(self, features: jnp.ndarray) -> jnp.ndarray:
        amplitude = jnp.sqrt(jnp.asarray(1.0 / WIDTH, dtype=jnp.float32))
        return jnp.full((*features.shape[:-1], WIDTH), amplitude, dtype=jnp.complex64)

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        encoder = InformationEncoder(mode="phase", normalize=False)
        phase_mask = encoder(self.phase_scale * features)
        fields = self._input_fields(features)

        if self.variant == "tied_unitary_repeated":
            unitary = UnitaryLinear(width=WIDTH, name="shared_unitary")
            for _layer_index in range(self.layers):
                fields = unitary(fields * phase_mask)

        elif self.variant == "tied_subunitary_repeated":
            subunitary = SubUnitaryLinear(width=WIDTH, insertion_loss_db=(0.0, 1.5), name="shared_subunitary")
            for _layer_index in range(self.layers):
                fields = subunitary(fields * phase_mask)

        elif self.variant == "tied_williamson":
            unitary = UnitaryLinear(width=WIDTH, name="shared_unitary")
            gain = self.param("williamson_gain", lambda key: jnp.asarray(self.williamson_gain, dtype=jnp.float32))
            bias = self.param("williamson_bias", lambda key: jnp.asarray(self.williamson_bias, dtype=jnp.float32))
            fields = fields * phase_mask
            for _layer_index in range(self.layers):
                fields = unitary(fields)
                fields = williamson_response(fields, gain, bias, self.williamson_tap)

        else:
            raise ValueError(f"unknown tied variant: {self.variant}")

        all_intensities = intensity(fields)
        class_intensities = all_intensities[..., :NUM_CLASSES]
        log_scale = self.param(
            "logit_scale",
            lambda key: jnp.asarray(np.log(self.fixed_logit_scale), dtype=jnp.float32),
        )
        scale = jnp.exp(log_scale)
        return scale * class_intensities, scale, all_intensities


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def power_loss_db(intensities: jnp.ndarray) -> jnp.ndarray:
    total_power = jnp.sum(intensities, axis=-1)
    return -10.0 * jnp.log10(jnp.clip(total_power, 1e-12, None))


def run_config(config: TiedConfig, data: PCA.DataSplit) -> dict[str, Any]:
    train_x = jnp.asarray(data.x_train)
    train_y = jnp.asarray(data.y_train)
    test_x = jnp.asarray(data.x_test)
    test_y = jnp.asarray(data.y_test)

    model = TiedOpticalModel(
        variant=config.variant,
        layers=config.layers,
        phase_scale=config.phase_scale_pi * float(np.pi),
        fixed_logit_scale=config.fixed_logit_scale,
    )
    params = model.init(jax.random.key(RANDOM_SEED), train_x[:8])["params"]
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    def loss_fn(optical_params, batch_x, batch_y):
        logits, scale, _intensities = model.apply({"params": optical_params}, batch_x)
        return cross_entropy_logits(batch_y, logits), (logits, scale)

    @jax.jit
    def train_step(optical_params, state, batch_x, batch_y):
        (loss_value, (logits, scale)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            optical_params, batch_x, batch_y
        )
        updates, state = optimizer.update(grads, state, optical_params)
        return optax.apply_updates(optical_params, updates), state, loss_value, accuracy(batch_y, logits), scale

    @jax.jit
    def eval_step(optical_params, eval_x, eval_y):
        logits, scale, intensities = model.apply({"params": optical_params}, eval_x)
        losses_db = power_loss_db(intensities)
        return (
            cross_entropy_logits(eval_y, logits),
            accuracy(eval_y, logits),
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

    train_loss, train_accuracy, _train_scale, train_mean_loss_db, _train_min_loss_db, _train_max_loss_db = eval_step(
        params, train_x, train_y
    )
    val_loss, val_accuracy, val_scale, val_mean_loss_db, val_min_loss_db, val_max_loss_db = eval_step(
        params, test_x, test_y
    )

    return {
        "name": config.name,
        "variant": config.variant,
        "layers": config.layers,
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
    }


VARIANT_LABELS = {
    "tied_unitary_repeated": "Tied unitary repeated",
    "tied_subunitary_repeated": "Tied subunitary repeated",
    "tied_williamson": "Tied Williamson",
}


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MNIST PCA Tied Three-Way Depth Curve: Trainable-Temperature Softmax",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset/encoding: MNIST projected to 16 PCA components, train-set min/max mapped without clipping, then phase encoded.",
        "",
        f"Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, Adam lr={LEARNING_RATE}.",
        "",
        "Objective: CE on `softmax(gamma * class_intensity)` with one trainable scalar gamma initialized at 10.",
        "",
        "Tying convention: one shared unitary/subunitary module is reused at every layer. Tied Williamson also reuses one shared trainable gain and one shared trainable bias at every layer.",
        "",
        "| Depth | Variant | Val Acc | Val Loss | Gamma | Mean IL (dB) | IL Range (dB) | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in range(1, 6):
        for variant, label in VARIANT_LABELS.items():
            result = next(item for item in results if item["variant"] == variant and item["layers"] == depth)
            lines.append(
                f"| {depth} | {label} | {result['val_accuracy']:.4f} | {result['val_loss']:.4f} | "
                f"{result['final_logit_scale']:.3f} | {result['val_mean_insertion_loss_db']:.3f} | "
                f"{result['val_min_insertion_loss_db']:.3f}-{result['val_max_insertion_loss_db']:.3f} | "
                f"{result['stored_param_count']} |"
            )

    lines.extend(["", "## Best Points", ""])
    for variant, label in VARIANT_LABELS.items():
        best = max((result for result in results if result["variant"] == variant), key=lambda result: result["val_accuracy"])
        lines.append(
            f"- {label}: depth {best['layers']}, {100.0 * best['val_accuracy']:.2f}% accuracy, "
            f"{best['val_loss']:.4f} loss, gamma {best['final_logit_scale']:.3f}, "
            f"mean IL {best['val_mean_insertion_loss_db']:.3f} dB, params {best['stored_param_count']}."
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = PCA.load_or_create_no_clip_cache()
    configs = []
    for depth in range(1, 6):
        configs.extend(
            [
                TiedConfig(f"tied-unitary-repeated-depth{depth}", "tied_unitary_repeated", depth),
                TiedConfig(f"tied-subunitary-repeated-depth{depth}", "tied_subunitary_repeated", depth),
                TiedConfig(f"tied-williamson-depth{depth}", "tied_williamson", depth),
            ]
        )

    results = [run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
