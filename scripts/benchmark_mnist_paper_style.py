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
from lumix.linen import UnitaryLinear


ARTIFACT_DIR = Path("artifacts/mnist_paper_style_comparison")
CACHE_PATH = ARTIFACT_DIR / "cache" / "mnist_low_k_complex_16.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_RAW_PATH = ARTIFACT_DIR / "cache" / "mnist_raw.npz"

NUM_CLASSES = 10
WIDTH = 16
EPOCHS = 200
BATCH_SIZE = 500
LEARNING_RATE = 1e-2
RANDOM_SEED = 7
WILLIAMSON_TAP = 0.1
WILLIAMSON_GAIN = 0.05 * float(np.pi)
WILLIAMSON_BIAS = 1.0 * float(np.pi)

ModelKind = Literal[
    "linear",
    "williamson",
    "repeated_phase_angle",
    "repeated_phase_magnitude",
    "repeated_phase_mixed",
    "repeated_phase_trainable_mix",
    "repeated_phase_channel_mix",
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
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False


def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(NUM_CLASSES, dtype=np.float32)[labels]


def download_mnist() -> None:
    MNIST_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MNIST_RAW_PATH.exists():
        return
    urllib.request.urlretrieve(MNIST_URL, MNIST_RAW_PATH)


def low_k_indices(size: int, count: int) -> list[tuple[int, int]]:
    center = size // 2
    coordinates = []
    for row in range(size):
        for col in range(size):
            kx = row - center
            ky = col - center
            coordinates.append((kx * kx + ky * ky, abs(kx) + abs(ky), row, col))
    coordinates.sort()
    return [(row, col) for *_unused, row, col in coordinates[:count]]


def low_k_complex_features(images: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(images.astype(np.float32) / 255.0), axes=(1, 2))
    indices = low_k_indices(spectrum.shape[-1], WIDTH)
    values = np.stack([spectrum[:, row, col] for row, col in indices], axis=-1)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (values / norm).astype(np.complex64)


def load_or_create_cache() -> DataSplit:
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        return DataSplit(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
        )

    download_mnist()
    raw = np.load(MNIST_RAW_PATH)
    x_train = low_k_complex_features(raw["x_train"])
    x_test = low_k_complex_features(raw["x_test"])
    y_train = one_hot(raw["y_train"])
    y_test = one_hot(raw["y_test"])

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return DataSplit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


class PaperStyleOpticalModel(nn.Module):
    kind: ModelKind
    layers: int
    train_williamson_gain: bool = False
    train_williamson_bias: bool = False
    width: int = WIDTH
    williamson_tap: float = WILLIAMSON_TAP
    williamson_gain: float = WILLIAMSON_GAIN
    williamson_bias: float = WILLIAMSON_BIAS

    def _unitary(self, layer_index: int):
        return UnitaryLinear(width=self.width, name=f"unitary_{layer_index}")

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

    def _trainable_repeat_mix(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        angle_gain = self.param("repeat_angle_gain", lambda key: jnp.ones((self.layers,), dtype=jnp.float32))
        magnitude_gain = self.param("repeat_magnitude_gain", lambda key: jnp.ones((self.layers,), dtype=jnp.float32))
        real_gain = self.param("repeat_real_gain", lambda key: jnp.zeros((self.layers,), dtype=jnp.float32))
        imag_gain = self.param("repeat_imag_gain", lambda key: jnp.zeros((self.layers,), dtype=jnp.float32))
        return angle_gain, magnitude_gain, real_gain, imag_gain

    def _trainable_repeat_channel_mix(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        shape = (self.layers, self.width)
        angle_gain = self.param("repeat_channel_angle_gain", lambda key: jnp.zeros(shape, dtype=jnp.float32))
        magnitude_gain = self.param("repeat_channel_magnitude_gain", lambda key: jnp.ones(shape, dtype=jnp.float32))
        real_gain = self.param("repeat_channel_real_gain", lambda key: jnp.zeros(shape, dtype=jnp.float32))
        imag_gain = self.param("repeat_channel_imag_gain", lambda key: jnp.zeros(shape, dtype=jnp.float32))
        bias = self.param("repeat_channel_bias", lambda key: jnp.zeros(shape, dtype=jnp.float32))
        return angle_gain, magnitude_gain, real_gain, imag_gain, bias

    def _repeat_phase_terms(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        angle = jnp.angle(inputs)
        magnitude = jnp.abs(inputs)
        max_magnitude = jnp.maximum(jnp.max(magnitude, axis=-1, keepdims=True), 1e-7)
        magnitude_phase = jnp.pi * magnitude / max_magnitude
        real_phase = jnp.pi * jnp.real(inputs) / max_magnitude
        imag_phase = jnp.pi * jnp.imag(inputs) / max_magnitude
        return angle, magnitude_phase, real_phase, imag_phase

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        fields = inputs.astype(jnp.complex64)

        if self.kind == "linear":
            for layer_index in range(self.layers):
                fields = self._unitary(layer_index)(fields)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "williamson":
            gain, bias = self._williamson_params()
            for layer_index in range(self.layers):
                fields = self._unitary(layer_index)(fields)
                fields = williamson_response(fields, gain[layer_index], bias[layer_index], self.williamson_tap)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "repeated_phase_angle":
            phase, _magnitude_phase, _real_phase, _imag_phase = self._repeat_phase_terms(inputs)
            for layer_index in range(self.layers):
                phase_mask = jnp.exp(1j * phase).astype(jnp.complex64)
                fields = self._unitary(layer_index)(fields * phase_mask)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "repeated_phase_magnitude":
            _angle, phase, _real_phase, _imag_phase = self._repeat_phase_terms(inputs)
            for layer_index in range(self.layers):
                phase_mask = jnp.exp(1j * phase).astype(jnp.complex64)
                fields = self._unitary(layer_index)(fields * phase_mask)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "repeated_phase_mixed":
            angle, magnitude_phase, real_phase, imag_phase = self._repeat_phase_terms(inputs)
            phase = angle + magnitude_phase + 0.5 * real_phase + 0.5 * imag_phase
            for layer_index in range(self.layers):
                phase_mask = jnp.exp(1j * phase).astype(jnp.complex64)
                fields = self._unitary(layer_index)(fields * phase_mask)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "repeated_phase_trainable_mix":
            angle, magnitude_phase, real_phase, imag_phase = self._repeat_phase_terms(inputs)
            angle_gain, magnitude_gain, real_gain, imag_gain = self._trainable_repeat_mix()
            for layer_index in range(self.layers):
                phase = (
                    angle_gain[layer_index] * angle
                    + magnitude_gain[layer_index] * magnitude_phase
                    + real_gain[layer_index] * real_phase
                    + imag_gain[layer_index] * imag_phase
                )
                phase_mask = jnp.exp(1j * phase).astype(jnp.complex64)
                fields = self._unitary(layer_index)(fields * phase_mask)
            return class_probs(intensity(fields), NUM_CLASSES)

        if self.kind == "repeated_phase_channel_mix":
            angle, magnitude_phase, real_phase, imag_phase = self._repeat_phase_terms(inputs)
            angle_gain, magnitude_gain, real_gain, imag_gain, bias = self._trainable_repeat_channel_mix()
            for layer_index in range(self.layers):
                phase = (
                    angle_gain[layer_index] * angle
                    + magnitude_gain[layer_index] * magnitude_phase
                    + real_gain[layer_index] * real_phase
                    + imag_gain[layer_index] * imag_phase
                    + bias[layer_index]
                )
                phase_mask = jnp.exp(1j * phase).astype(jnp.complex64)
                fields = self._unitary(layer_index)(fields * phase_mask)
            return class_probs(intensity(fields), NUM_CLASSES)

        raise ValueError(f"unknown model kind: {self.kind}")


def count_params(params: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def trainable_williamson_params(config: Config) -> int:
    if config.kind != "williamson":
        return 0
    return (int(config.train_williamson_gain) + int(config.train_williamson_bias)) * config.layers


def trainable_repeat_params(config: Config) -> int:
    if config.kind == "repeated_phase_trainable_mix":
        return 4 * config.layers
    if config.kind == "repeated_phase_channel_mix":
        return 5 * WIDTH * config.layers
    return 0


def run_config(config: Config, data: DataSplit) -> dict[str, Any]:
    train_x = jnp.asarray(data.x_train)
    train_y = jnp.asarray(data.y_train)
    test_x = jnp.asarray(data.x_test)
    test_y = jnp.asarray(data.y_test)

    model = PaperStyleOpticalModel(
        kind=config.kind,
        layers=config.layers,
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
        batch_losses = []
        batch_scores = []
        for batch_x, batch_y in iterate_batches(train_x, train_y, BATCH_SIZE, epoch_rng):
            params, opt_state, loss_value, score = train_step(params, opt_state, batch_x, batch_y)
            batch_losses.append(loss_value)
            batch_scores.append(score)
        val_loss, val_accuracy = eval_step(params, test_x, test_y)
        history.append(
            {
                "epoch": epoch,
                "loss": float(jnp.mean(jnp.stack(batch_losses))),
                "accuracy": float(jnp.mean(jnp.stack(batch_scores))),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
            }
        )

    unitary_params = 1024 * config.layers
    williamson_params = trainable_williamson_params(config)
    repeat_params = trainable_repeat_params(config)
    return {
        "name": config.name,
        "kind": config.kind,
        "layers": config.layers,
        "feature_count": WIDTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_accuracy": history[-1]["accuracy"],
        "val_accuracy": history[-1]["val_accuracy"],
        "train_loss": history[-1]["loss"],
        "val_loss": history[-1]["val_loss"],
        "learnable_unitary_param_count": unitary_params,
        "williamson_trainable_params": williamson_params,
        "repetition_trainable_params": repeat_params,
        "learnable_optical_param_count": unitary_params + williamson_params + repeat_params,
        "stored_optical_param_count": count_params(params),
        "williamson": {
            "tap": WILLIAMSON_TAP,
            "gain": WILLIAMSON_GAIN,
            "bias": WILLIAMSON_BIAS,
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
        "# MNIST Paper-Style Optical Comparison",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset: MNIST transformed to the 16 lowest-|k| complex Fourier coefficients, with each complex vector L2-normalized to unit norm.",
        "",
        "All models start from the same complex optical input field. Repeated rows re-inject phase-only masks derived from the same input at every layer; trainable-mix rows add four scalar phase-mix parameters per layer.",
        "",
        f"Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, Adam lr={LEARNING_RATE}. Loss is cross-entropy on normalized output intensities.",
        "",
        f"Williamson setting: alpha/tap={WILLIAMSON_TAP}, initial g_phi={WILLIAMSON_GAIN / float(np.pi):.2f}pi, phi_b={WILLIAMSON_BIAS / float(np.pi):.2f}pi.",
        "",
        "| Model | Kind | Layers | Train g_phi | Train phi_b | Learnable Optical Params | Stored Optical Params | Train Acc | Val Acc | Val Loss |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        train_gain = bool(result["williamson"]["train_gain"]) if result["williamson"] else False
        train_bias = bool(result["williamson"]["train_bias"]) if result["williamson"] else False
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['layers']} | {train_gain} | {train_bias} | "
            f"{result['learnable_optical_param_count']} | {result['stored_optical_param_count']} | "
            f"{result['train_accuracy']:.4f} | {result['val_accuracy']:.4f} | {result['val_loss']:.4f} |"
        )
    (ARTIFACT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_or_create_cache()
    configs = [
        Config("linear-depth2", "linear", 2),
        Config("linear-depth3", "linear", 3),
        Config("williamson-depth2-fixed", "williamson", 2),
        Config("williamson-depth3-fixed", "williamson", 3),
        Config("williamson-depth2-train-gain", "williamson", 2, train_williamson_gain=True),
        Config("williamson-depth3-train-gain", "williamson", 3, train_williamson_gain=True),
        Config(
            "williamson-depth2-train-gain-bias",
            "williamson",
            2,
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        Config(
            "williamson-depth3-train-gain-bias",
            "williamson",
            3,
            train_williamson_gain=True,
            train_williamson_bias=True,
        ),
        Config("repeated-angle-depth2", "repeated_phase_angle", 2),
        Config("repeated-angle-depth3", "repeated_phase_angle", 3),
        Config("repeated-magnitude-depth2", "repeated_phase_magnitude", 2),
        Config("repeated-magnitude-depth3", "repeated_phase_magnitude", 3),
        Config("repeated-mixed-depth2", "repeated_phase_mixed", 2),
        Config("repeated-mixed-depth3", "repeated_phase_mixed", 3),
        Config("repeated-trainable-mix-depth2", "repeated_phase_trainable_mix", 2),
        Config("repeated-trainable-mix-depth3", "repeated_phase_trainable_mix", 3),
        Config("repeated-channel-mix-depth2", "repeated_phase_channel_mix", 2),
        Config("repeated-channel-mix-depth3", "repeated_phase_channel_mix", 3),
        Config("repeated-channel-mix-depth4", "repeated_phase_channel_mix", 4),
        Config("repeated-channel-mix-depth5", "repeated_phase_channel_mix", 5),
    ]
    results = [run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps([{key: value for key, value in result.items() if key != "history"} for result in results], indent=2))


if __name__ == "__main__":
    main()
