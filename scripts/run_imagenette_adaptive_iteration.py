from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from lumix.functional import solve_ridge
from lumix.functional.readout import intensity
from lumix.linen import (
    BlockParallelUnitary,
    ClementsLinear,
    InformationEncoder,
    RidgeReadout,
    SubUnitaryLinear,
    UnitaryLinear,
)


DATA_ROOT = Path("data")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIR = DATA_ROOT / "imagenette2-160"
ARTIFACT_DIR = Path("artifacts/imagenette_lumix_architecture_iterations")
CACHE_PATH = ARTIFACT_DIR / "cache" / "imagenette_gray_5000_1000_64.npz"
ITERATIONS_DIR = ARTIFACT_DIR / "iterations"
RESULTS_PATH = ARTIFACT_DIR / "results.json"
SUMMARY_PATH = ARTIFACT_DIR / "summary.md"

MAX_TRAIN = 5000
MAX_VAL = 1000
IMAGE_SIZE = 160
POOLED_IMAGE_SIZE = 64
NUM_CLASSES = 10
OPTICAL_STEPS = 50
OPTICAL_LR = 1e-2
RIDGE_ALPHA = 1e-3
RANDOM_SEED = 7
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


InsertionLossSpec = float | tuple[float, float]


def _is_insertion_loss_range(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def normalize_insertion_loss_db(value: object) -> InsertionLossSpec:
    if _is_insertion_loss_range(value):
        if len(value) != 2:
            raise ValueError("insertion_loss_db range must contain exactly two values")
        lower, upper = value
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ValueError("insertion_loss_db range must contain two numbers")
        return float(lower), float(upper)

    if not isinstance(value, (int, float)):
        raise ValueError("insertion_loss_db must be a number or two-number range")
    return float(value)


@dataclass(frozen=True)
class Architecture:
    name: str
    decision: str
    patch_size: int
    patch_stride: int
    layers: int
    channels: int
    pool_grid: int
    phase_scales_pi: tuple[float, ...] = (1.0,)
    phase_offset: float = 0.0
    encoding_mode: str = "phase"
    amplitude_range: tuple[float, float] = (0.0, 1.0)
    block: str = "unitary"
    sharing: str = "tied"
    post_encode: bool = False
    insertion_loss_db: InsertionLossSpec = 0.0
    clements_depth: int | None = None
    clements_hadamard: bool = False
    block_count: int | None = None

    @classmethod
    def from_json(cls, raw: str) -> "Architecture":
        values = json.loads(raw)
        values["phase_scales_pi"] = tuple(values.get("phase_scales_pi", (1.0,)))
        values["amplitude_range"] = tuple(values.get("amplitude_range", (0.0, 1.0)))
        values["insertion_loss_db"] = normalize_insertion_loss_db(values.get("insertion_loss_db", 0.0))
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision": self.decision,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "layers": self.layers,
            "channels": self.channels,
            "pool_grid": self.pool_grid,
            "phase_scales_pi": list(self.phase_scales_pi),
            "phase_offset": self.phase_offset,
            "encoding_mode": self.encoding_mode,
            "amplitude_range": list(self.amplitude_range),
            "block": self.block,
            "sharing": self.sharing,
            "post_encode": self.post_encode,
            "insertion_loss_db": self.insertion_loss_db,
            "clements_depth": self.clements_depth,
            "clements_hadamard": self.clements_hadamard,
            "block_count": self.block_count,
        }

    @property
    def patch_grid(self) -> int:
        return ((POOLED_IMAGE_SIZE - self.patch_size) // self.patch_stride) + 1

    @property
    def feature_count(self) -> int:
        return self.pool_grid * self.pool_grid * self.channels

    @property
    def pool_window(self) -> int:
        return self.patch_grid // self.pool_grid

    @property
    def time_steps(self) -> int:
        if self.patch_size == 16:
            return self.patch_grid * self.patch_grid * self.patch_size
        return self.patch_grid * self.patch_grid


class RepeatedEncodingPatchEncoder(nn.Module):
    channels: int
    layers: int
    phase_scales: tuple[float, ...]
    block: str
    sharing: str
    post_encode: bool
    insertion_loss_db: InsertionLossSpec
    clements_depth: int | None
    clements_hadamard: bool
    block_count: int | None
    encoding_mode: str = "phase"
    phase_offset: float = 0.0
    amplitude_range: tuple[float, float] = (0.0, 1.0)

    def _linear(self, name: str):
        if self.block == "unitary":
            return UnitaryLinear(width=self.channels, name=name)
        if self.block == "block_unitary":
            if self.block_count is None:
                raise ValueError("block_count must be set for block_unitary")
            return BlockParallelUnitary(
                num_blocks=self.block_count,
                block_in_features=self.channels // self.block_count,
                name=name,
            )
        if self.block == "subunitary":
            return SubUnitaryLinear(
                width=self.channels,
                insertion_loss_db=self.insertion_loss_db,
                name=name,
            )
        if self.block == "clements":
            return ClementsLinear(
                width=self.channels,
                depth=self.clements_depth,
                hadamard=self.clements_hadamard,
                name=name,
            )
        raise ValueError(f"unsupported block: {self.block}")

    @nn.compact
    def __call__(self, patch_values: jnp.ndarray) -> jnp.ndarray:
        phase_encoder = InformationEncoder(mode="phase", normalize=False)
        amplitude_encoder = InformationEncoder(
            mode="amplitude",
            normalize=True,
            amplitude_range=self.amplitude_range,
            input_range=(0.0, 1.0),
            clip=True,
        )
        shared_unitary = self._linear("shared_unitary") if self.sharing == "tied" else None
        alt_a = self._linear("alt_a") if self.sharing == "alternating" else None
        alt_b = self._linear("alt_b") if self.sharing == "alternating" else None
        palindrome_a = self._linear("palindrome_a") if self.sharing == "palindrome" else None
        palindrome_b = self._linear("palindrome_b") if self.sharing == "palindrome" else None
        aa_bb_a = self._linear("aa_bb_a") if self.sharing == "aa_bb" else None
        aa_bb_b = self._linear("aa_bb_b") if self.sharing == "aa_bb" else None
        aa_bb_modules = (aa_bb_a, aa_bb_a, aa_bb_b, aa_bb_b) if self.sharing == "aa_bb" else ()
        ab_aa_a = self._linear("ab_aa_a") if self.sharing == "ab_aa" else None
        ab_aa_b = self._linear("ab_aa_b") if self.sharing == "ab_aa" else None
        ab_aa_modules = (ab_aa_a, ab_aa_b, ab_aa_a, ab_aa_a) if self.sharing == "ab_aa" else ()
        abc_a = self._linear("abc_a") if self.sharing == "abc_a" else None
        abc_b = self._linear("abc_b") if self.sharing == "abc_a" else None
        abc_c = self._linear("abc_c") if self.sharing == "abc_a" else None
        abc_modules = (abc_a, abc_b, abc_c, abc_a) if self.sharing == "abc_a" else ()
        amplitude = jnp.sqrt(jnp.asarray(1.0 / self.channels, dtype=jnp.float32))
        fields = jnp.full((*patch_values.shape[:-1], self.channels), amplitude, dtype=jnp.complex64)
        for layer_index in range(self.layers):
            scale = self.phase_scales[layer_index % len(self.phase_scales)]
            encoded = phase_encoder(scale * (patch_values + self.phase_offset))
            if self.encoding_mode == "phase_amplitude":
                encoded = encoded * amplitude_encoder(patch_values)
            elif self.encoding_mode != "phase":
                raise ValueError(f"unsupported encoding_mode: {self.encoding_mode}")
            fields = fields * encoded
            if self.sharing == "tied":
                fields = shared_unitary(fields)
            elif self.sharing == "untied":
                fields = self._linear(f"unitary_{layer_index}")(fields)
            elif self.sharing == "alternating":
                fields = (alt_a if layer_index % 2 == 0 else alt_b)(fields)
            elif self.sharing == "palindrome":
                fields = (palindrome_a if layer_index in {0, self.layers - 1} else palindrome_b)(fields)
            elif self.sharing == "aa_bb":
                fields = aa_bb_modules[layer_index](fields)
            elif self.sharing == "ab_aa":
                fields = ab_aa_modules[layer_index](fields)
            elif self.sharing == "abc_a":
                fields = abc_modules[layer_index](fields)
            else:
                raise ValueError(f"unsupported sharing: {self.sharing}")
            if self.post_encode:
                fields = fields * encoded
        return intensity(fields)


def validate_architecture(architecture: Architecture) -> None:
    insertion_loss_db = normalize_insertion_loss_db(architecture.insertion_loss_db)
    insertion_loss_is_range = _is_insertion_loss_range(insertion_loss_db)
    if architecture.encoding_mode not in {"phase", "phase_amplitude"}:
        raise ValueError("encoding_mode must be phase or phase_amplitude")
    if len(architecture.amplitude_range) != 2:
        raise ValueError("amplitude_range must have exactly two values")
    amplitude_min, amplitude_max = architecture.amplitude_range
    if amplitude_min < 0.0 or amplitude_max > 1.0 or amplitude_min > amplitude_max:
        raise ValueError("amplitude_range must be ordered within [0.0, 1.0]")
    if architecture.block not in {"unitary", "block_unitary", "subunitary", "clements"}:
        raise ValueError("block must be unitary, block_unitary, subunitary, or clements")
    if architecture.sharing not in {"tied", "untied", "alternating", "palindrome", "aa_bb", "ab_aa", "abc_a"}:
        raise ValueError("sharing must be tied, untied, alternating, palindrome, aa_bb, ab_aa, or abc_a")
    if architecture.sharing == "aa_bb" and architecture.layers != 4:
        raise ValueError("aa_bb sharing requires exactly 4 layers")
    if architecture.sharing == "ab_aa" and architecture.layers != 4:
        raise ValueError("ab_aa sharing requires exactly 4 layers")
    if architecture.sharing == "abc_a" and architecture.layers != 4:
        raise ValueError("abc_a sharing requires exactly 4 layers")
    if insertion_loss_is_range and architecture.block != "subunitary":
        raise ValueError("insertion_loss_db ranges are only supported for subunitary blocks")
    if architecture.block in {"unitary", "block_unitary"} and insertion_loss_db != 0.0:
        raise ValueError("unitary insertion_loss_db must remain 0.0")
    if architecture.block == "subunitary":
        if insertion_loss_is_range:
            lower_loss_db, upper_loss_db = insertion_loss_db
            if lower_loss_db < 0.0 or upper_loss_db < 0.0 or lower_loss_db > upper_loss_db:
                raise ValueError("subunitary insertion_loss_db range must be ordered and nonnegative")
        elif insertion_loss_db <= 0.0:
            raise ValueError("subunitary insertion_loss_db must be positive")
    if architecture.block == "clements":
        if insertion_loss_db != 0.0:
            raise ValueError("clements insertion_loss_db must remain 0.0")
        if architecture.clements_depth is None or architecture.clements_depth < 1:
            raise ValueError("clements_depth must be a positive integer for clements blocks")
        if architecture.clements_hadamard:
            raise ValueError("clements_hadamard must remain false for these experiments")
    elif architecture.clements_depth is not None or architecture.clements_hadamard:
        raise ValueError("Clements config is only supported when block is clements")
    if architecture.block == "block_unitary":
        if architecture.block_count is None or architecture.block_count < 1:
            raise ValueError("block_count must be a positive integer for block_unitary")
        if architecture.channels % architecture.block_count != 0:
            raise ValueError("block_count must divide channels for block_unitary")
    elif architecture.block_count is not None:
        raise ValueError("block_count is only supported when block is block_unitary")
    if (architecture.patch_size, architecture.patch_stride) not in {(4, 4), (16, 16)}:
        raise ValueError("patch strategy must be either 4x4/stride4 or 16x16/stride16")
    if (POOLED_IMAGE_SIZE - architecture.patch_size) % architecture.patch_stride != 0:
        raise ValueError("patch geometry must evenly tile the 64x64 cached images")
    if architecture.channels != 16:
        raise ValueError("optical width must be exactly 16 for the constrained experiments")
    if architecture.pool_grid <= 0 or architecture.patch_grid % architecture.pool_grid != 0:
        raise ValueError("pool_grid must be positive and divide the patch grid")
    if architecture.patch_size == 16 and architecture.pool_grid != architecture.patch_grid:
        raise ValueError("16x16 row-vector mode must keep one pooled output per spatial patch")
    if architecture.time_steps > 256:
        raise ValueError(f"time multiplexing must be <= 256, got {architecture.time_steps}")
    if architecture.feature_count > 256:
        raise ValueError(f"feature_count must be <= 256, got {architecture.feature_count}")
    if architecture.layers < 1:
        raise ValueError("layers must be at least 1")
    if not architecture.phase_scales_pi:
        raise ValueError("phase_scales_pi must not be empty")


def safe_extract_tar(archive: tarfile.TarFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in archive.getmembers():
        member_path = (target_dir / member.name).resolve()
        if target_dir not in member_path.parents and member_path != target_dir:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    archive.extractall(target_dir)


def download_imagenette() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if IMAGENETTE_DIR.exists():
        return IMAGENETTE_DIR

    archive_path = DATA_ROOT / Path(IMAGENETTE_URL).name
    if not archive_path.exists():
        print(f"Downloading {IMAGENETTE_URL} -> {archive_path}")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract_tar(archive, DATA_ROOT)
    return IMAGENETTE_DIR


def stratified_take(paths: list[tuple[Path, int]], *, count: int, random_state: int) -> list[tuple[Path, int]]:
    if count > len(paths):
        raise RuntimeError("Requested more samples than are available")
    rng = np.random.default_rng(random_state)
    selected: list[tuple[Path, int]] = []
    labels = sorted({label for _path, label in paths})
    base = count // len(labels)
    remainder = count % len(labels)
    for label_index, label in enumerate(labels):
        candidates = [item for item in paths if item[1] == label]
        order = rng.permutation(len(candidates))
        take = min(len(candidates), base + (1 if label_index < remainder else 0))
        selected.extend(candidates[index] for index in order[:take])
    if len(selected) != count:
        raise RuntimeError("Could not draw the requested stratified sample count")
    rng.shuffle(selected)
    return selected


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        image = image.resize((POOLED_IMAGE_SIZE, POOLED_IMAGE_SIZE), Image.Resampling.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_imagefolder_split(root: Path, split: str, *, max_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    split_dir = root / split
    classes = sorted(path for path in split_dir.iterdir() if path.is_dir())
    paths: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(classes):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append((path, label))
    selected = stratified_take(paths, count=max_samples, random_state=random_state)
    images = np.stack([load_image(path) for path, _label in selected])
    labels = np.asarray([label for _path, label in selected], dtype=np.int32)
    return images, labels


def load_or_create_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as data:
            return data["x_train"], data["y_train"], data["x_val"], data["y_val"]

    root = download_imagenette()
    x_train, y_train = load_imagefolder_split(root, "train", max_samples=MAX_TRAIN, random_state=RANDOM_SEED)
    x_val, y_val = load_imagefolder_split(root, "val", max_samples=MAX_VAL, random_state=RANDOM_SEED + 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    return x_train, y_train, x_val, y_val


def image_patch_matrix(images: np.ndarray, *, patch_size: int, stride: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(images.astype(np.float32), (patch_size, patch_size), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :]
    if patch_size == 16:
        return windows.reshape(images.shape[0], -1, patch_size)
    return windows.reshape(images.shape[0], -1, patch_size * patch_size)


def pool_patch_intensities(patch_intensities: jnp.ndarray, architecture: Architecture) -> jnp.ndarray:
    if architecture.patch_size == 16:
        patch_count = architecture.patch_grid * architecture.patch_grid
        patch_rows = patch_intensities.reshape(
            patch_intensities.shape[0],
            patch_count,
            architecture.patch_size,
            architecture.channels,
        )
        return jnp.mean(patch_rows, axis=2).reshape(patch_intensities.shape[0], -1)

    feature_map = patch_intensities.reshape(
        patch_intensities.shape[0],
        architecture.patch_grid,
        architecture.patch_grid,
        architecture.channels,
    )
    pooled = nn.avg_pool(
        feature_map,
        window_shape=(architecture.pool_window, architecture.pool_window),
        strides=(architecture.pool_window, architecture.pool_window),
        padding="VALID",
    )
    return pooled.reshape(pooled.shape[0], -1)


def standardize_from_train(train_features: jnp.ndarray, val_features: jnp.ndarray):
    mean = jnp.mean(train_features, axis=0, keepdims=True)
    std = jnp.std(train_features, axis=0, keepdims=True)
    std = jnp.where(std == 0.0, 1.0, std)
    return (train_features - mean) / std, (val_features - mean) / std


def one_hot(labels: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.asarray(labels, dtype=jnp.int32), NUM_CLASSES, dtype=jnp.float32)


def ridge_logits(train_features: jnp.ndarray, labels: jnp.ndarray, features: jnp.ndarray):
    ridge_params = solve_ridge(train_features, one_hot(labels), alpha=RIDGE_ALPHA, use_bias=True)
    return RidgeReadout(features=NUM_CLASSES).apply({"params": ridge_params}, features), ridge_params


def accuracy_from_logits(logits: jnp.ndarray, labels: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.asarray(labels))


def run_experiment(architecture: Architecture) -> dict[str, Any]:
    validate_architecture(architecture)
    x_train, y_train_np, x_val, y_val_np = load_or_create_cache(CACHE_PATH)
    train_patches_np = image_patch_matrix(x_train, patch_size=architecture.patch_size, stride=architecture.patch_stride)
    val_patches_np = image_patch_matrix(x_val, patch_size=architecture.patch_size, stride=architecture.patch_stride)

    train_patches = jnp.asarray(train_patches_np, dtype=jnp.float32)
    val_patches = jnp.asarray(val_patches_np, dtype=jnp.float32)
    y_train = jnp.asarray(y_train_np, dtype=jnp.int32)
    y_val = jnp.asarray(y_val_np, dtype=jnp.int32)
    phase_scales = tuple(float(scale * np.pi) for scale in architecture.phase_scales_pi)
    model = RepeatedEncodingPatchEncoder(
        channels=architecture.channels,
        layers=architecture.layers,
        phase_scales=phase_scales,
        block=architecture.block,
        sharing=architecture.sharing,
        post_encode=architecture.post_encode,
        insertion_loss_db=architecture.insertion_loss_db,
        clements_depth=architecture.clements_depth,
        clements_hadamard=architecture.clements_hadamard,
        block_count=architecture.block_count,
        encoding_mode=architecture.encoding_mode,
        phase_offset=architecture.phase_offset,
        amplitude_range=architecture.amplitude_range,
    )
    variables = model.init(jax.random.key(RANDOM_SEED), train_patches[:8])
    params = variables["params"]
    constants = variables.get("constants")
    optimizer = optax.adam(OPTICAL_LR)
    opt_state = optimizer.init(params)

    def optical_features(optical_params, patches: jnp.ndarray) -> jnp.ndarray:
        apply_variables = {"params": optical_params}
        if constants is not None:
            apply_variables["constants"] = constants
        return pool_patch_intensities(model.apply(apply_variables, patches), architecture)

    def training_objective(optical_params, patches: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        features = optical_features(optical_params, patches)
        features_std, _ = standardize_from_train(features, features)
        logits, _ridge_params = ridge_logits(features_std, labels, features_std)
        return jnp.mean(jnp.square(one_hot(labels) - logits))

    @jax.jit
    def full_train_step(optical_params, state, patches: jnp.ndarray, labels: jnp.ndarray):
        loss, grads = jax.value_and_grad(training_objective)(optical_params, patches, labels)
        updates, state = optimizer.update(grads, state, optical_params)
        return optax.apply_updates(optical_params, updates), state, loss

    loss_history: list[float] = []
    for step in range(OPTICAL_STEPS + 1):
        loss = training_objective(params, train_patches, y_train)
        loss_history.append(float(loss))
        if step == OPTICAL_STEPS:
            break
        params, opt_state, step_loss = full_train_step(params, opt_state, train_patches, y_train)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step {step + 1:03d} pre_update_ridge_mse={float(step_loss):.6f}")

    train_features_raw = optical_features(params, train_patches)
    val_features_raw = optical_features(params, val_patches)
    train_features_std, val_features_std = standardize_from_train(train_features_raw, val_features_raw)
    train_logits, final_ridge_params = ridge_logits(train_features_std, y_train, train_features_std)
    val_logits = RidgeReadout(features=NUM_CLASSES).apply({"params": final_ridge_params}, val_features_std)
    train_accuracy = accuracy_from_logits(train_logits, y_train)
    val_accuracy = accuracy_from_logits(val_logits, y_val)

    return {
        "name": architecture.name,
        "decision": architecture.decision,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "initial_ridge_mse": float(loss_history[0]),
        "final_ridge_mse": float(loss_history[-1]),
        "feature_count": int(train_features_std.shape[-1]),
        "time_steps": int(architecture.time_steps),
        "patch_grid": int(architecture.patch_grid),
        "pool_grid": int(architecture.pool_grid),
        "channels": int(architecture.channels),
        "optical_steps": OPTICAL_STEPS,
        "optical_lr": OPTICAL_LR,
        "ridge_alpha": RIDGE_ALPHA,
        "architecture": architecture.as_dict(),
    }


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def ensure_iteration_is_new(iteration: int) -> None:
    if any(item.get("iteration") == iteration for item in load_results()):
        raise ValueError(f"iteration {iteration} already exists in {RESULTS_PATH}")


def write_iteration_markdown(iteration: int, result: dict[str, Any], interpretation: str) -> None:
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    architecture_json = json.dumps(result["architecture"], indent=2)
    content = f"""# Iteration {iteration:02d}: {result["name"]}

Timestamp: {datetime.now().replace(microsecond=0).isoformat()}

## Decision
{result["decision"]}

## Constraints
- Output features <= 256: {"yes" if result["feature_count"] <= 256 else "no"}
- Optical nonlinearity: data re-encoding/repetition only
- Excluded: Williamson, activations, intensity re-upload, row/column token mixing

## Architecture
```json
{architecture_json}
```

## Metrics
- train_accuracy: {result["train_accuracy"]:.4f}
- val_accuracy: {result["val_accuracy"]:.4f}
- initial_ridge_mse: {result["initial_ridge_mse"]:.6f}
- final_ridge_mse: {result["final_ridge_mse"]:.6f}
- feature_count: {result["feature_count"]}
- patch_grid: {result["patch_grid"]}x{result["patch_grid"]}
- pool_grid: {result["pool_grid"]}x{result["pool_grid"]}
- channels: {result["channels"]}

## Interpretation
{interpretation}
"""
    (ITERATIONS_DIR / f"iteration_{iteration:02d}.md").write_text(content, encoding="utf-8")


def write_results(iteration: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    results = load_results()
    results.append({"iteration": iteration, **result})
    results.sort(key=lambda item: item["iteration"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def write_summary(results: list[dict[str, Any]]) -> None:
    best = max(results, key=lambda item: item["val_accuracy"])
    lines = [
        "# Imagenette Lumix Architecture Iterations",
        "",
        f"Completed iterations: {len(results)}",
        f"Best validation accuracy: {best['val_accuracy']:.4f} ({best['name']})",
        "",
        "| Iteration | Name | Patch | Block | Sharing | Layers | Features | Train Acc | Val Acc |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        architecture = item["architecture"]
        lines.append(
            f"| {item['iteration']:02d} | {item['name']} | "
            f"{architecture['patch_size']} / {architecture['patch_stride']} | "
            f"{architecture['block']} | {architecture['sharing']} | "
            f"{architecture['layers']} | {item['feature_count']} | "
            f"{item['train_accuracy']:.4f} | {item['val_accuracy']:.4f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one adaptive Imagenette Lumix architecture iteration.")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--architecture-json", required=True)
    parser.add_argument("--interpretation", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_iteration_is_new(args.iteration)
    architecture = Architecture.from_json(args.architecture_json)
    result = run_experiment(architecture)
    write_iteration_markdown(args.iteration, result, args.interpretation)
    results = write_results(args.iteration, result)
    write_summary(results)
    print(json.dumps({"iteration": args.iteration, **result}, indent=2))


if __name__ == "__main__":
    main()
