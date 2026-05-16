from __future__ import annotations

import json
import urllib.request
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
from lumix.functional.readout import class_probs, intensity
from lumix.functional.williamson import williamson_response
from lumix.losses import cross_entropy
from lumix.metrics import accuracy
from lumix.linen import InformationEncoder, SubUnitaryLinear, UnitaryLinear


ARTIFACT_DIR = Path("artifacts/mnist_pca_phase_comparison")
CACHE_PATH = ARTIFACT_DIR / "cache" / "mnist_pca16_phase.npz"
NO_CLIP_CACHE_PATH = ARTIFACT_DIR / "cache" / "mnist_pca16_phase_no_clip_minmax.npz"
MNIST_RAW_PATH = ARTIFACT_DIR / "cache" / "mnist_raw.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

NUM_CLASSES = 10
WIDTH = 16
EPOCHS = 200
BATCH_SIZE = 500
LEARNING_RATE = 1e-2
RANDOM_SEED = 7
WILLIAMSON_TAP = 0.1
WILLIAMSON_GAIN = 0.05 * float(np.pi)

ModelKind = Literal[
    "linear",
    "repeated_phase",
    "repeated_subunitary",
    "repeated_final_subunitary",
    "repeated_trainable_phase",
    "repeated_final_subunitary_trainable_phase",
    "repeated_harmonic",
    "williamson",
]


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class Config:
    name: str
    kind: ModelKind
    layers: int
    phase_scale_pi: float = 2.0
    williamson_bias: float = 0.50 * float(np.pi)
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False


def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(NUM_CLASSES, dtype=np.float32)[labels]


def download_mnist() -> None:
    MNIST_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MNIST_RAW_PATH.exists():
        return
    urllib.request.urlretrieve(MNIST_URL, MNIST_RAW_PATH)


def fit_pca_phase_features(train_images: np.ndarray, test_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_flat = train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0
    test_flat = test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0
    mean = np.mean(train_flat, axis=0, keepdims=True)
    train_centered = train_flat - mean
    test_centered = test_flat - mean

    covariance = (train_centered.T @ train_centered) / train_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:WIDTH]].astype(np.float32)

    train_scores = train_centered @ components
    test_scores = test_centered @ components
    score_mean = np.mean(train_scores, axis=0, keepdims=True)
    score_std = np.std(train_scores, axis=0, keepdims=True)
    score_std = np.where(score_std == 0.0, 1.0, score_std)

    train_scaled = np.clip((train_scores - score_mean) / score_std, -3.0, 3.0)
    test_scaled = np.clip((test_scores - score_mean) / score_std, -3.0, 3.0)
    return ((train_scaled + 3.0) / 6.0).astype(np.float32), ((test_scaled + 3.0) / 6.0).astype(np.float32)


def fit_pca_phase_features_no_clip(train_images: np.ndarray, test_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_flat = train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0
    test_flat = test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0
    mean = np.mean(train_flat, axis=0, keepdims=True)
    train_centered = train_flat - mean
    test_centered = test_flat - mean

    covariance = (train_centered.T @ train_centered) / train_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:WIDTH]].astype(np.float32)

    train_scores = train_centered @ components
    test_scores = test_centered @ components
    score_mean = np.mean(train_scores, axis=0, keepdims=True)
    score_std = np.std(train_scores, axis=0, keepdims=True)
    score_std = np.where(score_std == 0.0, 1.0, score_std)

    train_standardized = (train_scores - score_mean) / score_std
    test_standardized = (test_scores - score_mean) / score_std
    train_min = np.min(train_standardized, axis=0, keepdims=True)
    train_max = np.max(train_standardized, axis=0, keepdims=True)
    train_range = np.where(train_max == train_min, 1.0, train_max - train_min)
    return ((train_standardized - train_min) / train_range).astype(np.float32), (
        (test_standardized - train_min) / train_range
    ).astype(np.float32)


def load_or_create_cache() -> DataSplit:
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        return DataSplit(data["x_train"], data["y_train"], data["x_test"], data["y_test"])

    download_mnist()
    raw = np.load(MNIST_RAW_PATH)
    x_train, x_test = fit_pca_phase_features(raw["x_train"], raw["x_test"])
    y_train = one_hot(raw["y_train"])
    y_test = one_hot(raw["y_test"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return DataSplit(x_train, y_train, x_test, y_test)


def load_or_create_no_clip_cache() -> DataSplit:
    if NO_CLIP_CACHE_PATH.exists():
        data = np.load(NO_CLIP_CACHE_PATH)
        return DataSplit(data["x_train"], data["y_train"], data["x_test"], data["y_test"])

    download_mnist()
    raw = np.load(MNIST_RAW_PATH)
    x_train, x_test = fit_pca_phase_features_no_clip(raw["x_train"], raw["x_test"])
    y_train = one_hot(raw["y_train"])
    y_test = one_hot(raw["y_test"])
    NO_CLIP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NO_CLIP_CACHE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return DataSplit(x_train, y_train, x_test, y_test)


class PCAPhaseOpticalModel(nn.Module):
    kind: ModelKind
    layers: int
    phase_scale: float
    williamson_bias: float = 0.50 * float(np.pi)
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False
    width: int = WIDTH
    williamson_tap: float = WILLIAMSON_TAP
    williamson_gain: float = WILLIAMSON_GAIN
    return_intensity: bool = False

    def _readout(self, fields: jnp.ndarray) -> jnp.ndarray:
        features = intensity(fields)
        if self.return_intensity:
            return features
        return class_probs(features, NUM_CLASSES)

    def _phase_scale_params(self) -> jnp.ndarray:
        return self.param("phase_scale", lambda key: jnp.full((self.layers,), self.phase_scale, dtype=jnp.float32))

    def _unitary(self, layer_index: int):
        return UnitaryLinear(width=self.width, name=f"unitary_{layer_index}")

    def _subunitary(self, layer_index: int):
        return SubUnitaryLinear(width=self.width, insertion_loss_db=(0.0, 1.5), name=f"subunitary_{layer_index}")

    def _williamson_params(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        gain_init = jnp.full((self.layers,), self.williamson_gain, dtype=jnp.float32)
        bias_init = jnp.full((self.layers,), self.williamson_bias, dtype=jnp.float32)
        gain = self.param("williamson_gain", lambda key: gain_init)
        bias = self.param("williamson_bias", lambda key: bias_init)
        if not self.train_williamson_gain:
            gain = jax.lax.stop_gradient(gain)
        if not self.train_williamson_bias:
            bias = jax.lax.stop_gradient(bias)
        return gain, bias

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        encoder = InformationEncoder(mode="phase", normalize=False)
        phase_mask = encoder(self.phase_scale * features)
        amplitude = jnp.sqrt(jnp.asarray(1.0 / self.width, dtype=jnp.float32))
        fields = jnp.full((*features.shape[:-1], self.width), amplitude, dtype=jnp.complex64)

        if self.kind == "linear":
            fields = fields * phase_mask
            for layer_index in range(self.layers):
                fields = self._unitary(layer_index)(fields)
            return self._readout(fields)

        if self.kind == "repeated_phase":
            for layer_index in range(self.layers):
                fields = self._unitary(layer_index)(fields * phase_mask)
            return self._readout(fields)

        if self.kind == "repeated_subunitary":
            for layer_index in range(self.layers):
                fields = self._subunitary(layer_index)(fields * phase_mask)
            return self._readout(fields)

        if self.kind == "repeated_final_subunitary":
            for layer_index in range(self.layers - 1):
                fields = self._unitary(layer_index)(fields * phase_mask)
            fields = self._subunitary(self.layers - 1)(fields * phase_mask)
            return self._readout(fields)

        if self.kind == "repeated_trainable_phase":
            phase_scales = self._phase_scale_params()
            for layer_index in range(self.layers):
                learned_mask = encoder(phase_scales[layer_index] * features)
                fields = self._unitary(layer_index)(fields * learned_mask)
            return self._readout(fields)

        if self.kind == "repeated_final_subunitary_trainable_phase":
            phase_scales = self._phase_scale_params()
            for layer_index in range(self.layers - 1):
                learned_mask = encoder(phase_scales[layer_index] * features)
                fields = self._unitary(layer_index)(fields * learned_mask)
            learned_mask = encoder(phase_scales[self.layers - 1] * features)
            fields = self._subunitary(self.layers - 1)(fields * learned_mask)
            return self._readout(fields)

        if self.kind == "repeated_harmonic":
            multipliers = (1.0, 2.0, 0.5)
            for layer_index in range(self.layers):
                harmonic_mask = encoder(self.phase_scale * multipliers[layer_index % len(multipliers)] * features)
                fields = self._unitary(layer_index)(fields * harmonic_mask)
            return self._readout(fields)

        if self.kind == "williamson":
            gain, bias = self._williamson_params()
            fields = fields * phase_mask
            for layer_index in range(self.layers):
                fields = self._unitary(layer_index)(fields)
                fields = williamson_response(fields, gain[layer_index], bias[layer_index], self.williamson_tap)
            return self._readout(fields)

        raise ValueError(f"unknown model kind: {self.kind}")


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def trainable_williamson_params(config: Config) -> int:
    if config.kind != "williamson":
        return 0
    return (int(config.train_williamson_gain) + int(config.train_williamson_bias)) * config.layers


def run_config(config: Config, data: DataSplit) -> dict[str, Any]:
    train_x = jnp.asarray(data.x_train)
    train_y = jnp.asarray(data.y_train)
    test_x = jnp.asarray(data.x_test)
    test_y = jnp.asarray(data.y_test)

    model = PCAPhaseOpticalModel(
        kind=config.kind,
        layers=config.layers,
        phase_scale=config.phase_scale_pi * float(np.pi),
        williamson_bias=config.williamson_bias,
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
    def eval_step(optical_params, eval_x, eval_y):
        loss_value, probs = loss_fn(optical_params, eval_x, eval_y)
        return loss_value, accuracy(eval_y, probs)

    rng = jax.random.key(RANDOM_SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        rng, epoch_rng = jax.random.split(rng)
        losses = []
        scores = []
        for batch_x, batch_y in iterate_batches(train_x, train_y, BATCH_SIZE, epoch_rng):
            params, opt_state, loss_value, score = train_step(params, opt_state, batch_x, batch_y)
            losses.append(loss_value)
            scores.append(score)
        val_loss, val_accuracy = eval_step(params, test_x, test_y)
        history.append(
            {
                "epoch": epoch,
                "loss": float(jnp.mean(jnp.stack(losses))),
                "accuracy": float(jnp.mean(jnp.stack(scores))),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
            }
        )

    unitary_params = 1024 * config.layers
    williamson_params = trainable_williamson_params(config)
    return {
        "name": config.name,
        "kind": config.kind,
        "layers": config.layers,
        "feature_count": WIDTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "phase_scale_pi": config.phase_scale_pi,
        "train_accuracy": history[-1]["accuracy"],
        "val_accuracy": history[-1]["val_accuracy"],
        "train_loss": history[-1]["loss"],
        "val_loss": history[-1]["val_loss"],
        "learnable_unitary_param_count": unitary_params,
        "williamson_trainable_params": williamson_params,
        "learnable_optical_param_count": unitary_params + williamson_params,
        "stored_optical_param_count": count_params(params),
        "williamson": {
            "tap": WILLIAMSON_TAP,
            "gain": WILLIAMSON_GAIN,
            "bias": config.williamson_bias,
            "train_gain": config.train_williamson_gain,
            "train_bias": config.train_williamson_bias,
        }
        if config.kind == "williamson"
        else None,
        "history": history,
    }


def write_report(results: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MNIST PCA Phase Comparison",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset: MNIST projected to 16 PCA components. Components are train-standardized, clipped to [-3, 3], mapped to [0, 1], and phase-encoded.",
        "",
        f"Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, Adam lr={LEARNING_RATE}. Loss is cross-entropy on normalized output intensities.",
        "",
        "Both Williamson and repeated encoding receive the same PCA phase mask. Repeated encoding reuses that same mask at every layer.",
        "",
        "| Model | Kind | Layers | Phase Scale | Train g_phi | Train phi_b | Learnable Optical Params | Stored Optical Params | Train Acc | Val Acc | Val Loss |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        train_gain = bool(result["williamson"]["train_gain"]) if result["williamson"] else False
        train_bias = bool(result["williamson"]["train_bias"]) if result["williamson"] else False
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['layers']} | {result['phase_scale_pi']:.2f}π | "
            f"{train_gain} | {train_bias} | {result['learnable_optical_param_count']} | "
            f"{result['stored_optical_param_count']} | {result['train_accuracy']:.4f} | "
            f"{result['val_accuracy']:.4f} | {result['val_loss']:.4f} |"
        )
    (ARTIFACT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_or_create_cache()
    configs = [
        Config("linear-depth3-scale2", "linear", 3, phase_scale_pi=2.0),
        Config("linear-depth4-scale2", "linear", 4, phase_scale_pi=2.0),
        Config("repeated-depth3-scale1", "repeated_phase", 3, phase_scale_pi=1.0),
        Config("repeated-depth4-scale1", "repeated_phase", 4, phase_scale_pi=1.0),
        Config("repeated-depth3-scale2", "repeated_phase", 3, phase_scale_pi=2.0),
        Config("repeated-depth4-scale2", "repeated_phase", 4, phase_scale_pi=2.0),
        Config("repeated-depth5-scale2", "repeated_phase", 5, phase_scale_pi=2.0),
        Config("williamson-depth3-scale2-fixed", "williamson", 3, phase_scale_pi=2.0),
        Config("williamson-depth4-scale2-fixed", "williamson", 4, phase_scale_pi=2.0),
        Config(
            "williamson-depth3-scale2-train-gain-bias",
            "williamson",
            3,
            phase_scale_pi=2.0,
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        Config(
            "williamson-depth4-scale2-train-gain-bias",
            "williamson",
            4,
            phase_scale_pi=2.0,
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
    ]
    results = [run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps([{key: value for key, value in result.items() if key != "history"} for result in results], indent=2))


if __name__ == "__main__":
    main()
