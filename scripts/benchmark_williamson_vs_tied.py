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

from lumix.functional import solve_ridge
from lumix.functional.readout import intensity
from lumix.functional.williamson import williamson_response
from lumix.linen import InformationEncoder, RidgeReadout, UnitaryLinear


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

ARTIFACT_DIR = Path("artifacts/imagenette_williamson_vs_tied")
NUM_CLASSES = RUNNER.NUM_CLASSES
RANDOM_SEED = RUNNER.RANDOM_SEED
OPTICAL_STEPS = RUNNER.OPTICAL_STEPS
OPTICAL_LR = RUNNER.OPTICAL_LR
RIDGE_ALPHA = RUNNER.RIDGE_ALPHA
PAPER_WILLIAMSON_TAP = 0.1
PAPER_WILLIAMSON_GAIN = 0.05 * float(np.pi)
PAPER_WILLIAMSON_BIAS = 1.0 * float(np.pi)


ModelKind = Literal["tied_unitary", "single_williamson", "williamson_stack"]
UnitarySharing = Literal["tied", "untied"]


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    kind: ModelKind
    layers: int
    unitary_sharing: UnitarySharing = "tied"
    williamson_tap: float = PAPER_WILLIAMSON_TAP
    williamson_gain: float = PAPER_WILLIAMSON_GAIN
    williamson_bias: float = PAPER_WILLIAMSON_BIAS
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False


class OpticalComparisonModel(nn.Module):
    kind: ModelKind
    layers: int
    unitary_sharing: UnitarySharing = "tied"
    channels: int = 16
    phase_scale: float = float(np.pi)
    williamson_tap: float = PAPER_WILLIAMSON_TAP
    williamson_gain: float = PAPER_WILLIAMSON_GAIN
    williamson_bias: float = PAPER_WILLIAMSON_BIAS
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False

    def _williamson_params(self, shape: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
        gain_init = jnp.full(shape, self.williamson_gain, dtype=jnp.float32)
        bias_init = jnp.full(shape, self.williamson_bias, dtype=jnp.float32)
        gain = self.param("williamson_gain", lambda key: gain_init)
        bias = self.param("williamson_bias", lambda key: bias_init)
        if not self.train_williamson_gain:
            gain = jax.lax.stop_gradient(gain)
        if not self.train_williamson_bias:
            bias = jax.lax.stop_gradient(bias)
        return gain, bias

    def _unitary(self, layer_index: int):
        name = "shared_unitary" if self.unitary_sharing == "tied" else f"unitary_{layer_index}"
        return UnitaryLinear(width=self.channels, name=name)

    @nn.compact
    def __call__(self, patch_values: jnp.ndarray) -> jnp.ndarray:
        encoder = InformationEncoder(mode="phase", normalize=False)
        shared_unitary = self._unitary(0) if self.unitary_sharing == "tied" else None
        amplitude = jnp.sqrt(jnp.asarray(1.0 / self.channels, dtype=jnp.float32))
        fields = jnp.full((*patch_values.shape[:-1], self.channels), amplitude, dtype=jnp.complex64)
        encoded = encoder(self.phase_scale * patch_values)

        if self.kind == "tied_unitary":
            for layer_index in range(self.layers):
                unitary = shared_unitary if self.unitary_sharing == "tied" else self._unitary(layer_index)
                fields = unitary(fields * encoded)
            return intensity(fields)

        if self.kind == "single_williamson":
            unitary = shared_unitary if self.unitary_sharing == "tied" else self._unitary(0)
            fields = unitary(fields * encoded)
            gain, bias = self._williamson_params(())
            fields = williamson_response(
                fields,
                gain,
                bias,
                self.williamson_tap,
            )
            return intensity(fields)

        if self.kind == "williamson_stack":
            gain, bias = self._williamson_params((self.layers,))
            fields = fields * encoded
            for layer_index in range(self.layers):
                unitary = shared_unitary if self.unitary_sharing == "tied" else self._unitary(layer_index)
                fields = unitary(fields)
                fields = williamson_response(
                    fields,
                    gain[layer_index],
                    bias[layer_index],
                    self.williamson_tap,
                )
            return intensity(fields)

        raise ValueError(f"unknown model kind: {self.kind}")


def one_hot(labels: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.one_hot(labels, NUM_CLASSES, dtype=jnp.float32)


def standardize_from_train(train_features: jnp.ndarray, val_features: jnp.ndarray):
    mean = jnp.mean(train_features, axis=0, keepdims=True)
    std = jnp.std(train_features, axis=0, keepdims=True)
    std = jnp.where(std == 0.0, 1.0, std)
    return (train_features - mean) / std, (val_features - mean) / std


def ridge_logits(train_features: jnp.ndarray, labels: jnp.ndarray, features: jnp.ndarray):
    ridge_params = solve_ridge(train_features, one_hot(labels), alpha=RIDGE_ALPHA, use_bias=True)
    return RidgeReadout(features=NUM_CLASSES).apply({"params": ridge_params}, features), ridge_params


def accuracy_from_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def williamson_trainable_param_count(config: ComparisonConfig) -> int:
    per_layer = int(config.train_williamson_gain) + int(config.train_williamson_bias)
    if config.kind == "single_williamson":
        return per_layer
    if config.kind == "williamson_stack":
        return per_layer * config.layers
    return 0


def learnable_unitary_param_count(config: ComparisonConfig) -> int:
    unitary_count = config.layers if config.unitary_sharing == "untied" else 1
    return 1024 * unitary_count


def run_config(config: ComparisonConfig) -> dict[str, Any]:
    x_train, y_train_np, x_val, y_val_np = RUNNER.load_or_create_cache(RUNNER.CACHE_PATH)
    architecture = RUNNER.Architecture(
        name=config.name,
        decision="Williamson/tied-unitary comparison",
        patch_size=4,
        patch_stride=4,
        layers=max(config.layers, 1),
        channels=16,
        pool_grid=4,
    )
    train_patches = jnp.asarray(RUNNER.image_patch_matrix(x_train, patch_size=4, stride=4), dtype=jnp.float32)
    val_patches = jnp.asarray(RUNNER.image_patch_matrix(x_val, patch_size=4, stride=4), dtype=jnp.float32)
    y_train = jnp.asarray(y_train_np, dtype=jnp.int32)
    y_val = jnp.asarray(y_val_np, dtype=jnp.int32)

    model = OpticalComparisonModel(
        kind=config.kind,
        layers=config.layers,
        unitary_sharing=config.unitary_sharing,
        williamson_tap=config.williamson_tap,
        williamson_gain=config.williamson_gain,
        williamson_bias=config.williamson_bias,
        train_williamson_gain=config.train_williamson_gain,
        train_williamson_bias=config.train_williamson_bias,
    )
    params = model.init(jax.random.key(RANDOM_SEED), train_patches[:8])["params"]
    optimizer = optax.adam(OPTICAL_LR)
    opt_state = optimizer.init(params)

    def optical_features(optical_params, patches: jnp.ndarray) -> jnp.ndarray:
        patch_intensities = model.apply({"params": optical_params}, patches)
        return RUNNER.pool_patch_intensities(patch_intensities, architecture)

    def training_objective(optical_params) -> jnp.ndarray:
        features = optical_features(optical_params, train_patches)
        features_std, _ = standardize_from_train(features, features)
        logits, _ridge_params = ridge_logits(features_std, y_train, features_std)
        return jnp.mean(jnp.square(one_hot(y_train) - logits))

    @jax.jit
    def train_step(optical_params, state):
        loss, grads = jax.value_and_grad(training_objective)(optical_params)
        updates, state = optimizer.update(grads, state, optical_params)
        return optax.apply_updates(optical_params, updates), state, loss

    loss_history = []
    for step in range(OPTICAL_STEPS + 1):
        loss = training_objective(params)
        loss_history.append(float(loss))
        if step == OPTICAL_STEPS:
            break
        params, opt_state, _ = train_step(params, opt_state)

    train_features_raw = optical_features(params, train_patches)
    val_features_raw = optical_features(params, val_patches)
    train_features, val_features = standardize_from_train(train_features_raw, val_features_raw)
    train_logits, ridge_params = ridge_logits(train_features, y_train, train_features)
    val_logits = RidgeReadout(features=NUM_CLASSES).apply({"params": ridge_params}, val_features)

    stored_optical_param_count = count_params(params)
    williamson_param_count = williamson_trainable_param_count(config)
    unitary_param_count = learnable_unitary_param_count(config)
    learnable_optical_param_count = unitary_param_count + williamson_param_count
    readout_param_count = int(train_features.shape[-1] * NUM_CLASSES + NUM_CLASSES)
    return {
        "name": config.name,
        "kind": config.kind,
        "layers": config.layers,
        "unitary_sharing": config.unitary_sharing,
        "train_accuracy": float(accuracy_from_logits(train_logits, y_train)),
        "val_accuracy": float(accuracy_from_logits(val_logits, y_val)),
        "initial_ridge_mse": loss_history[0],
        "final_ridge_mse": loss_history[-1],
        "feature_count": int(train_features.shape[-1]),
        "time_steps": int(architecture.time_steps),
        "learnable_unitary_param_count": unitary_param_count,
        "williamson_trainable_params": williamson_param_count,
        "learnable_optical_param_count": learnable_optical_param_count,
        "stored_optical_param_count": stored_optical_param_count,
        "readout_param_count": readout_param_count,
        "total_learnable_param_count": learnable_optical_param_count + readout_param_count,
        "total_stored_param_count": stored_optical_param_count + readout_param_count,
        "williamson": {
            "tap": config.williamson_tap,
            "gain": config.williamson_gain,
            "bias": config.williamson_bias,
            "train_gain": config.train_williamson_gain,
            "train_bias": config.train_williamson_bias,
        }
        if config.kind in {"single_williamson", "williamson_stack"}
        else None,
    }


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Williamson vs Tied Unitary Imagenette Benchmark",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "All models use 4x4 non-overlapping patches, width 16, 256 time steps, 256 pooled features, ridge readout, and 50 optical optimization steps.",
        "",
        f"Williamson rows use alpha/tap={PAPER_WILLIAMSON_TAP}. The paper MNIST setting is g_phi={PAPER_WILLIAMSON_GAIN / float(np.pi):.2f}pi, phi_b={PAPER_WILLIAMSON_BIAS / float(np.pi):.2f}pi; retuned rows use the representative paper biases from Fig. 2.",
        "",
        "Tied/untied unitary rows use repeated data phase encoding. Williamson rows encode data once at the input, then apply unitary + Williamson per layer.",
        "",
        "Untied rows learn an independent UnitaryLinear per layer. Williamson gain/bias are fixed unless marked trainable.",
        "",
        "| Model | Kind | Layers | Unitary Sharing | g_phi/pi | phi_b/pi | Train g_phi | Train phi_b | Learnable Unitary Params | Learnable Optical Params | Total Learnable Params | Train Acc | Val Acc |",
        "|---|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['layers']} | "
            f"{result['unitary_sharing']} | "
            f"{result['williamson']['gain'] / float(np.pi):.2f} | {result['williamson']['bias'] / float(np.pi):.2f} | "
            f"{bool(result['williamson']['train_gain'])} | {bool(result['williamson']['train_bias'])} | "
            if result["williamson"]
            else f"| {result['name']} | {result['kind']} | {result['layers']} | {result['unitary_sharing']} |  |  | False | False | "
        )
        lines[-1] += (
            f"{result['learnable_unitary_param_count']} | {result['learnable_optical_param_count']} | "
            f"{result['total_learnable_param_count']} | "
            f"{result['train_accuracy']:.4f} | {result['val_accuracy']:.4f} |"
        )
    (ARTIFACT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    unitary_depths = (2, 3, 4, 5)
    configs = [
        *[
            ComparisonConfig(f"tied-unitary-depth{depth}", "tied_unitary", depth)
            for depth in unitary_depths
        ],
        *[
            ComparisonConfig(f"untied-unitary-depth{depth}", "tied_unitary", depth, unitary_sharing="untied")
            for depth in unitary_depths
        ],
        ComparisonConfig("single-williamson-fixed", "single_williamson", 1),
        ComparisonConfig("williamson-depth2-paper-fixed", "williamson_stack", 2, unitary_sharing="untied"),
        ComparisonConfig(
            "williamson-depth2-bias085-fixed",
            "williamson_stack",
            2,
            unitary_sharing="untied",
            williamson_bias=0.85 * float(np.pi),
        ),
        ComparisonConfig(
            "williamson-depth2-bias050-fixed",
            "williamson_stack",
            2,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
        ),
        ComparisonConfig(
            "williamson-depth2-bias050-train-gain-bias",
            "williamson_stack",
            2,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        ComparisonConfig(
            "williamson-depth2-bias000-fixed",
            "williamson_stack",
            2,
            unitary_sharing="untied",
            williamson_bias=0.0,
        ),
        ComparisonConfig(
            "williamson-depth2-bias085-train-gain",
            "williamson_stack",
            2,
            unitary_sharing="untied",
            williamson_bias=0.85 * float(np.pi),
            train_williamson_gain=True,
        ),
        ComparisonConfig("williamson-depth3-untied-unitary-fixed", "williamson_stack", 3, unitary_sharing="untied"),
        ComparisonConfig(
            "williamson-depth3-bias050-fixed",
            "williamson_stack",
            3,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
        ),
        ComparisonConfig(
            "williamson-depth3-bias050-train-gain-bias",
            "williamson_stack",
            3,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        ComparisonConfig(
            "williamson-depth3-bias085-train-gain",
            "williamson_stack",
            3,
            unitary_sharing="untied",
            williamson_bias=0.85 * float(np.pi),
            train_williamson_gain=True,
        ),
        ComparisonConfig("williamson-depth4-untied-unitary-fixed", "williamson_stack", 4, unitary_sharing="untied"),
        ComparisonConfig(
            "williamson-depth4-bias050-fixed",
            "williamson_stack",
            4,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
        ),
        ComparisonConfig(
            "williamson-depth4-bias050-train-gain-bias",
            "williamson_stack",
            4,
            unitary_sharing="untied",
            williamson_bias=0.50 * float(np.pi),
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
    ]
    results = [run_config(config) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
