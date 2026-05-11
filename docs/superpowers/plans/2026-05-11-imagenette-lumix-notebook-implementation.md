# Imagenette Lumix Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/imagenette_repeated_encoding_lumix.ipynb`, a tutorial notebook that reproduces the Tidy3D grayscale Imagenette repeated-encoding baseline using Lumix, Flax, JAX, Optax, and notebook-local preprocessing.

**Architecture:** The notebook owns Imagenette download/loading, patch extraction, explicit feature standardization, training loop, and artifact saving. Lumix owns optical components (`InformationEncoder`, `UnitaryLinear`, `intensity`) and ridge readout application (`solve_ridge`, `RidgeReadout`). Flax owns module state, `nn.avg_pool`, and Optax updates the optical params.

**Tech Stack:** Jupyter notebook, JAX, Flax Linen, Optax, Lumix, NumPy, Pillow, matplotlib, uv.

---

## File Structure

- Create `notebooks/imagenette_repeated_encoding_lumix.ipynb`
  - Tutorial notebook with markdown and code cells.
- Create then delete `tmp/jupyter-notebook/build_imagenette_lumix_notebook.py`
  - Temporary generation script for controlled notebook JSON edits.
- Do not create library modules for Imagenette, standardization, pooling, or notebook-only helpers.
- Do not modify `src/lumix` for this notebook unless missing Lumix API work is discovered and explicitly handled separately.

## Prerequisites

Before implementation, the branch should include these Lumix APIs:

```python
from lumix.functional import encode_phase, solve_ridge
from lumix.functional.readout import intensity
from lumix.linen import InformationEncoder, RidgeReadout, UnitaryLinear
from lumix.metrics import mean_squared_error
```

If any import is missing, stop and report the missing prerequisite. Do not reimplement those APIs inside the notebook.

---

### Task 1: Scaffold the Tutorial Notebook

**Files:**
- Create: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Create notebook directory**

Run:

```bash
mkdir -p notebooks
```

- [ ] **Step 2: Scaffold with the Jupyter notebook helper**

Run:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export JUPYTER_NOTEBOOK_CLI="$CODEX_HOME/skills/jupyter-notebook/scripts/new_notebook.py"
uv run --python 3.12 python "$JUPYTER_NOTEBOOK_CLI" \
  --kind tutorial \
  --title "Imagenette Repeated-Encoding Optical Network with Lumix" \
  --out notebooks/imagenette_repeated_encoding_lumix.ipynb
```

Expected: notebook file exists at `notebooks/imagenette_repeated_encoding_lumix.ipynb`.

- [ ] **Step 3: Verify notebook JSON parses**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/imagenette_repeated_encoding_lumix.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["nbformat"] == 4
assert data["cells"]
print(f"ok: {path} has {len(data['cells'])} cells")
PY
```

Expected: prints `ok: notebooks/imagenette_repeated_encoding_lumix.ipynb has ... cells`.

- [ ] **Step 4: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: scaffold imagenette lumix notebook"
```

---

### Task 2: Add Setup, Configuration, and Dataset Cells

**Files:**
- Modify: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Replace notebook body with tutorial cells**

Use a small Python notebook-editing script rather than hand-editing JSON. Create `tmp/jupyter-notebook/build_imagenette_lumix_notebook.py` with the complete cell list. The first cells must be:

```python
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax

from lumix.functional import solve_ridge
from lumix.functional.readout import intensity
from lumix.linen import InformationEncoder, RidgeReadout, UnitaryLinear
from lumix.metrics import mean_squared_error
```

Add a configuration cell:

```python
DATA_ROOT = Path("data")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIR = DATA_ROOT / "imagenette2-160"
ARTIFACT_DIR = Path("artifacts/imagenette_lumix_repeated_encoding")

MAX_TRAIN = 5000
MAX_VAL = 1000
IMAGE_SIZE = 160
POOLED_IMAGE_SIZE = 64
COLOR_MODE = "grayscale"
PATCH_SIZE = 4
PATCH_STRIDE = 4
PATCH_GRID = 16
NUM_PATCHES = PATCH_GRID * PATCH_GRID
CHANNELS = PATCH_SIZE * PATCH_SIZE
NUM_CLASSES = 10

NUM_LAYERS = 4
PHASE_SCALE = jnp.pi
RIDGE_ALPHA = 1.0e-3
UNITARY_STEPS = 50
UNITARY_LR = 1.0e-2
RANDOM_SEED = 7
TRAIN_UNITARY_ON_SUBSET = None
```

- [ ] **Step 2: Add dataset download and loading helpers**

Add code cells with these exact helpers:

```python
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_imagenette(url: str = IMAGENETTE_URL, target_dir: Path = IMAGENETTE_DIR) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        return target_dir

    archive_path = DATA_ROOT / Path(url).name
    if not archive_path.exists():
        print(f"Downloading {url} -> {archive_path}")
        urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(DATA_ROOT)
    return target_dir


def class_directories(split_dir: Path) -> list[Path]:
    classes = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not classes:
        raise RuntimeError(f"No class directories found in {split_dir}")
    return classes


def stratified_take(paths: list[tuple[Path, int]], *, count: int, random_state: int) -> list[tuple[Path, int]]:
    if count >= len(paths):
        return paths
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
    rng.shuffle(selected)
    return selected


def load_image(path: Path, *, image_size: int, pooled_image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image = image.resize((pooled_image_size, pooled_image_size), Image.Resampling.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_imagefolder_split(
    root: Path,
    split: str,
    *,
    image_size: int,
    pooled_image_size: int,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split_dir = root / split
    classes = class_directories(split_dir)
    class_names = [path.name for path in classes]
    paths: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(classes):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append((path, label))
    if not paths:
        raise RuntimeError(f"No images found in {split_dir}")
    selected = stratified_take(paths, count=max_samples, random_state=random_state)
    images = np.stack(
        [load_image(path, image_size=image_size, pooled_image_size=pooled_image_size) for path, _label in selected]
    )
    labels = np.asarray([label for _path, label in selected], dtype=np.int32)
    return images, labels, class_names
```

- [ ] **Step 3: Add patch extraction helpers**

Add:

```python
def image_patch_matrix(images: np.ndarray, *, patch_size: int, stride: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (samples, height, width)")
    if patch_size > images.shape[1] or patch_size > images.shape[2]:
        raise ValueError("patch_size cannot exceed image height or width")
    windows = np.lib.stride_tricks.sliding_window_view(
        images,
        (patch_size, patch_size),
        axis=(1, 2),
    )
    windows = windows[:, ::stride, ::stride, :, :]
    return windows.reshape(images.shape[0], -1, patch_size * patch_size)
```

- [ ] **Step 4: Add dataset load cell**

Add:

```python
imagenette_root = download_imagenette()

x_train_images, y_train_np, class_names = load_imagefolder_split(
    imagenette_root,
    "train",
    image_size=IMAGE_SIZE,
    pooled_image_size=POOLED_IMAGE_SIZE,
    max_samples=MAX_TRAIN,
    random_state=RANDOM_SEED,
)
x_val_images, y_val_np, val_class_names = load_imagefolder_split(
    imagenette_root,
    "val",
    image_size=IMAGE_SIZE,
    pooled_image_size=POOLED_IMAGE_SIZE,
    max_samples=MAX_VAL,
    random_state=RANDOM_SEED,
)
assert class_names == val_class_names

train_patches_np = image_patch_matrix(x_train_images, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
val_patches_np = image_patch_matrix(x_val_images, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)

assert train_patches_np.shape == (MAX_TRAIN, NUM_PATCHES, CHANNELS)
assert val_patches_np.shape == (MAX_VAL, NUM_PATCHES, CHANNELS)
assert y_train_np.shape == (MAX_TRAIN,)
assert y_val_np.shape == (MAX_VAL,)

print("train_patches:", train_patches_np.shape)
print("val_patches:", val_patches_np.shape)
print("classes:", class_names)
```

- [ ] **Step 5: Validate notebook syntax by parsing JSON**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/imagenette_repeated_encoding_lumix.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))
assert any("download_imagenette" in "".join(cell.get("source", "")) for cell in data["cells"])
assert any("image_patch_matrix" in "".join(cell.get("source", "")) for cell in data["cells"])
print("dataset cells present")
PY
```

Expected: `dataset cells present`.

- [ ] **Step 6: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: add imagenette notebook data pipeline"
```

---

### Task 3: Add Lumix Optical Model and Feature Helpers

**Files:**
- Modify: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Add the repeated-encoding model cell**

Add this code cell:

```python
def uniform_complex_field(prefix_shape: tuple[int, ...], channels: int) -> jnp.ndarray:
    amplitude = jnp.sqrt(jnp.asarray(1.0 / channels, dtype=jnp.float32))
    return jnp.full((*prefix_shape, channels), amplitude, dtype=jnp.complex64)


class RepeatedEncodingPatchEncoder(nn.Module):
    channels: int = CHANNELS
    layers: int = NUM_LAYERS
    phase_scale: float = float(np.pi)

    @nn.compact
    def __call__(self, patch_values: jnp.ndarray) -> jnp.ndarray:
        fields = uniform_complex_field(patch_values.shape[:-1], self.channels)
        phase_fields = InformationEncoder(mode="phase", normalize=False)(
            self.phase_scale * patch_values
        )
        shared_unitary = UnitaryLinear(width=self.channels, name="shared_unitary")
        for _layer_index in range(self.layers):
            fields = fields * phase_fields
            fields = shared_unitary(fields)
        return intensity(fields)
```

This deliberately creates `shared_unitary` once and reuses it in the loop.

- [ ] **Step 2: Add pooling and feature extraction helpers**

Add:

```python
def pool_patch_intensities(patch_intensities: jnp.ndarray) -> jnp.ndarray:
    if patch_intensities.shape[-2:] != (NUM_PATCHES, CHANNELS):
        raise ValueError("patch_intensities must have shape (..., 256, 16)")
    feature_map = patch_intensities.reshape(
        patch_intensities.shape[0],
        PATCH_GRID,
        PATCH_GRID,
        CHANNELS,
    )
    pooled = nn.avg_pool(
        feature_map,
        window_shape=(4, 4),
        strides=(4, 4),
        padding="VALID",
    )
    return pooled.reshape(pooled.shape[0], -1)


def optical_features(model: nn.Module, params, patches: jnp.ndarray) -> jnp.ndarray:
    patch_intensities = model.apply({"params": params}, patches)
    return pool_patch_intensities(patch_intensities)


def standardize_from_train(train_features: jnp.ndarray, val_features: jnp.ndarray):
    mean = jnp.mean(train_features, axis=0, keepdims=True)
    std = jnp.std(train_features, axis=0, keepdims=True)
    std = jnp.where(std == 0.0, 1.0, std)
    return (train_features - mean) / std, (val_features - mean) / std, mean, std
```

- [ ] **Step 3: Add ridge helpers**

Add:

```python
def one_hot(labels: np.ndarray | jnp.ndarray, num_classes: int = NUM_CLASSES) -> jnp.ndarray:
    labels = jnp.asarray(labels, dtype=jnp.int32)
    return jax.nn.one_hot(labels, num_classes, dtype=jnp.float32)


def ridge_logits(train_features: jnp.ndarray, labels: jnp.ndarray, features: jnp.ndarray):
    targets = one_hot(labels)
    ridge_params = solve_ridge(train_features, targets, alpha=RIDGE_ALPHA, use_bias=True)
    readout = RidgeReadout(features=NUM_CLASSES)
    return readout.apply({"params": ridge_params}, features), ridge_params


def accuracy_from_logits(logits: jnp.ndarray, labels: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    labels = jnp.asarray(labels)
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)
```

- [ ] **Step 4: Validate helper names exist**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

source = "\n".join(
    "".join(cell.get("source", ""))
    for cell in json.loads(Path("notebooks/imagenette_repeated_encoding_lumix.ipynb").read_text())["cells"]
)
for name in [
    "RepeatedEncodingPatchEncoder",
    "pool_patch_intensities",
    "standardize_from_train",
    "ridge_logits",
]:
    assert name in source, name
print("model/helper cells present")
PY
```

Expected: `model/helper cells present`.

- [ ] **Step 5: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: add lumix optical model notebook cells"
```

---

### Task 4: Add Smoke Checks

**Files:**
- Modify: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Add conversion and smoke-check cell**

Add:

```python
train_patches = jnp.asarray(train_patches_np, dtype=jnp.float32)
val_patches = jnp.asarray(val_patches_np, dtype=jnp.float32)
y_train = jnp.asarray(y_train_np, dtype=jnp.int32)
y_val = jnp.asarray(y_val_np, dtype=jnp.int32)

model = RepeatedEncodingPatchEncoder()
smoke_patches = train_patches[:8]
variables = model.init(jax.random.key(RANDOM_SEED), smoke_patches)
params = variables["params"]
smoke_intensities = model.apply({"params": params}, smoke_patches)
smoke_features = pool_patch_intensities(smoke_intensities)

assert smoke_intensities.shape == (8, NUM_PATCHES, CHANNELS)
assert smoke_features.shape == (8, 4 * 4 * CHANNELS)

smoke_targets = one_hot(y_train[:8])
smoke_ridge_params = solve_ridge(smoke_features, smoke_targets, alpha=RIDGE_ALPHA, use_bias=True)
smoke_logits = RidgeReadout(features=NUM_CLASSES).apply({"params": smoke_ridge_params}, smoke_features)
assert smoke_logits.shape == (8, NUM_CLASSES)

print("smoke intensities:", smoke_intensities.shape)
print("smoke pooled features:", smoke_features.shape)
print("smoke logits:", smoke_logits.shape)
```

- [ ] **Step 2: Add one-update smoke cell**

Add:

```python
optimizer = optax.adam(UNITARY_LR)
opt_state = optimizer.init(params)


def objective(optical_params, patches, labels):
    features = optical_features(model, optical_params, patches)
    features_std, _, _, _ = standardize_from_train(features, features)
    targets = one_hot(labels)
    logits, _ridge_params = ridge_logits(features_std, labels, features_std)
    return mean_squared_error(targets, logits)


@jax.jit
def train_step(optical_params, opt_state, patches, labels):
    loss, grads = jax.value_and_grad(objective)(optical_params, patches, labels)
    updates, opt_state = optimizer.update(grads, opt_state, optical_params)
    optical_params = optax.apply_updates(optical_params, updates)
    return optical_params, opt_state, loss


next_params, next_opt_state, smoke_loss = train_step(params, opt_state, smoke_patches, y_train[:8])
flat_before, _ = jax.tree_util.tree_flatten(params)
flat_after, _ = jax.tree_util.tree_flatten(next_params)
delta = sum(float(jnp.sum(jnp.abs(after - before))) for before, after in zip(flat_before, flat_after))
assert delta > 0.0
print("one-step smoke loss:", float(smoke_loss))
print("parameter delta:", delta)
```

- [ ] **Step 3: Run a lightweight notebook validation**

Do not execute the full notebook here. Parse the notebook and verify the smoke code is present:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

source = "\n".join(
    "".join(cell.get("source", ""))
    for cell in json.loads(Path("notebooks/imagenette_repeated_encoding_lumix.ipynb").read_text())["cells"]
)
assert "one-step smoke loss" in source
assert "parameter delta" in source
print("smoke cells present")
PY
```

Expected: `smoke cells present`.

- [ ] **Step 4: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: add imagenette notebook smoke checks"
```

---

### Task 5: Add Full Training and Evaluation Cells

**Files:**
- Modify: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Add training subset helper cell**

Add:

```python
def stratified_indices(labels: np.ndarray, *, count: int | None, random_state: int) -> np.ndarray:
    labels = np.asarray(labels)
    if count is None or count >= labels.shape[0]:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(random_state)
    selected = []
    classes = sorted(np.unique(labels).tolist())
    base = count // len(classes)
    remainder = count % len(classes)
    for class_index, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        order = rng.permutation(candidates.shape[0])
        take = min(candidates.shape[0], base + (1 if class_index < remainder else 0))
        selected.extend(candidates[order[:take]].tolist())
    selected = np.asarray(selected, dtype=np.int32)
    rng.shuffle(selected)
    return selected


train_unitary_indices = stratified_indices(
    y_train_np,
    count=TRAIN_UNITARY_ON_SUBSET,
    random_state=RANDOM_SEED,
)
unitary_train_patches = train_patches[train_unitary_indices]
unitary_train_labels = y_train[train_unitary_indices]
print("unitary training patches:", unitary_train_patches.shape)
```

- [ ] **Step 2: Add full objective and train loop cell**

Add:

```python
model = RepeatedEncodingPatchEncoder()
params = model.init(jax.random.key(RANDOM_SEED), unitary_train_patches[:8])["params"]
optimizer = optax.adam(UNITARY_LR)
opt_state = optimizer.init(params)


def training_objective(optical_params):
    features = optical_features(model, optical_params, unitary_train_patches)
    features_std, _, _, _ = standardize_from_train(features, features)
    targets = one_hot(unitary_train_labels)
    logits, _ridge_params = ridge_logits(features_std, unitary_train_labels, features_std)
    return mean_squared_error(targets, logits)


@jax.jit
def full_train_step(optical_params, opt_state):
    loss, grads = jax.value_and_grad(training_objective)(optical_params)
    updates, opt_state = optimizer.update(grads, opt_state, optical_params)
    optical_params = optax.apply_updates(optical_params, updates)
    return optical_params, opt_state, loss


loss_history = []
for step in range(UNITARY_STEPS + 1):
    loss = training_objective(params)
    loss_history.append(float(loss))
    if step == UNITARY_STEPS:
        break
    params, opt_state, _step_loss = full_train_step(params, opt_state)
    if step == 0 or (step + 1) % 10 == 0:
        print(f"step {step + 1:03d} ridge_mse={float(_step_loss):.6f}")

print(f"initial ridge_mse={loss_history[0]:.6f}")
print(f"final ridge_mse={loss_history[-1]:.6f}")
```

- [ ] **Step 3: Add evaluation cell**

Add:

```python
train_features_raw = optical_features(model, params, train_patches)
val_features_raw = optical_features(model, params, val_patches)
train_features_std, val_features_std, feature_mean, feature_std = standardize_from_train(
    train_features_raw,
    val_features_raw,
)

train_logits, final_ridge_params = ridge_logits(train_features_std, y_train, train_features_std)
val_logits = RidgeReadout(features=NUM_CLASSES).apply({"params": final_ridge_params}, val_features_std)

train_accuracy = accuracy_from_logits(train_logits, y_train)
val_accuracy = accuracy_from_logits(val_logits, y_val)

metrics = {
    "train_accuracy": float(train_accuracy),
    "val_accuracy": float(val_accuracy),
    "max_train": int(MAX_TRAIN),
    "max_val": int(MAX_VAL),
    "image_size": int(IMAGE_SIZE),
    "pooled_image_size": int(POOLED_IMAGE_SIZE),
    "color_mode": COLOR_MODE,
    "patch_size": int(PATCH_SIZE),
    "patch_stride": int(PATCH_STRIDE),
    "num_layers": int(NUM_LAYERS),
    "phase_scale_pi": float(PHASE_SCALE / jnp.pi),
    "ridge_alpha": float(RIDGE_ALPHA),
    "unitary_steps": int(UNITARY_STEPS),
    "unitary_lr": float(UNITARY_LR),
    "num_optical_features": int(train_features_std.shape[-1]),
    "random_seed": int(RANDOM_SEED),
    "initial_ridge_mse": float(loss_history[0]),
    "final_ridge_mse": float(loss_history[-1]),
}

for key, value in metrics.items():
    print(f"{key}: {value}")
```

- [ ] **Step 4: Add accuracy reference markdown**

Add a markdown cell after evaluation:

```markdown
The Tidy3D Hackathon reference run for this grayscale capped baseline reported about 50.6% validation accuracy with a tied trainable 16x16 unitary, four repeated layers, phase scale pi, 4x4 spatial pooling, and ridge readout. Small differences are expected because this notebook uses Lumix's unitary parameterization and the local JAX/Flax environment.
```

- [ ] **Step 5: Validate training cell presence**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

source = "\n".join(
    "".join(cell.get("source", ""))
    for cell in json.loads(Path("notebooks/imagenette_repeated_encoding_lumix.ipynb").read_text())["cells"]
)
for phrase in ["full_train_step", "loss_history", "val_accuracy", "final_ridge_mse"]:
    assert phrase in source, phrase
print("training/evaluation cells present")
PY
```

Expected: `training/evaluation cells present`.

- [ ] **Step 6: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: add imagenette notebook training loop"
```

---

### Task 6: Add Artifact Saving and Final Tutorial Polish

**Files:**
- Modify: `notebooks/imagenette_repeated_encoding_lumix.ipynb`

- [ ] **Step 1: Add save cell**

Add:

```python
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = ARTIFACT_DIR / "metrics.json"
params_path = ARTIFACT_DIR / "optical_params.msgpack"

metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
params_path.write_bytes(serialization.to_bytes(params))

print(f"saved metrics to {metrics_path}")
print(f"saved optical params to {params_path}")
```

- [ ] **Step 2: Add compact plot cell**

Add:

```python
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(loss_history, marker="o", linewidth=1.5)
ax.set_xlabel("unitary update")
ax.set_ylabel("ridge MSE")
ax.set_title("Optical unitary training objective")
ax.grid(True, alpha=0.3)
plt.show()
```

- [ ] **Step 3: Add final notes markdown**

Add:

```markdown
This notebook intentionally keeps Imagenette preprocessing and feature standardization outside Lumix. The reusable library pieces are the optical encoding/layer/readout components; the dataset-specific workflow remains explicit in the notebook.
```

- [ ] **Step 4: Validate notebook contains save outputs**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

source = "\n".join(
    "".join(cell.get("source", ""))
    for cell in json.loads(Path("notebooks/imagenette_repeated_encoding_lumix.ipynb").read_text())["cells"]
)
assert "metrics.json" in source
assert "optical_params.msgpack" in source
assert "serialization.to_bytes(params)" in source
print("artifact saving cells present")
PY
```

Expected: `artifact saving cells present`.

- [ ] **Step 5: Commit**

Run:

```bash
git add notebooks/imagenette_repeated_encoding_lumix.ipynb
git commit -m "docs: finish imagenette lumix notebook"
```

---

### Task 7: Validate Notebook and Focused Tests

**Files:**
- Read: `notebooks/imagenette_repeated_encoding_lumix.ipynb`
- Read: Lumix tests

- [ ] **Step 1: Validate notebook JSON and required sections**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/imagenette_repeated_encoding_lumix.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))
source = "\n".join("".join(cell.get("source", "")) for cell in data["cells"])
required = [
    "Imagenette Repeated-Encoding Optical Network with Lumix",
    "download_imagenette",
    "image_patch_matrix",
    "RepeatedEncodingPatchEncoder",
    "InformationEncoder",
    "UnitaryLinear",
    "nn.avg_pool",
    "standardize_from_train",
    "solve_ridge",
    "RidgeReadout",
    "optax.adam",
    "serialization.to_bytes",
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit(f"missing notebook content: {missing}")
print(f"notebook ok: {len(data['cells'])} cells")
PY
```

Expected: `notebook ok: ... cells`.

- [ ] **Step 2: Run Lumix focused tests**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_encoding.py tests/test_ridge.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run full Lumix tests**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass. If this fails because local uncommitted cleanup removed a non-library GAN test or because the Imagenette notebook is not executed in CI, report the exact failure rather than claiming success.

- [ ] **Step 4: Optional smoke execution with tiny subset**

If execution time allows, temporarily patch the notebook config in-memory or run the notebook manually with:

```python
MAX_TRAIN = 20
MAX_VAL = 10
UNITARY_STEPS = 1
TRAIN_UNITARY_ON_SUBSET = 20
```

Expected smoke result:

```text
train_patches: (20, 256, 16)
val_patches: (10, 256, 16)
smoke pooled features: (8, 256)
parameter delta: positive
metrics saved
```

Do not commit smoke-only config values.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git diff --stat HEAD
git status --short
```

Expected: notebook changes are intentional. Existing unrelated dirty files from prior Lumix work may still appear; do not revert them.

---

## Self-Review

- Spec coverage:
  - Faithful 50.6% baseline settings are represented: grayscale, capped 5000/1000 split, 64x64 images, 4x4 patches, 16x16 token grid, 4 layers, phase scale pi, 4x4 pooling, 256 features, ridge readout.
  - Notebook uses Lumix-native optical/readout components and Flax-compatible workflow.
  - Standardization remains notebook-local.
  - Pooling uses Flax `nn.avg_pool`.
  - No axial token mixer is included.
- Completeness scan:
  - No task uses vague markers or unspecified helper names.
  - Each required notebook helper has concrete code.
- Type consistency:
  - Patches use `(samples, 256, 16)`.
  - Optical outputs use `(batch, 256, 16)`.
  - Pooled features use `(batch, 256)`.
  - Ridge params come from `solve_ridge` and are applied by `RidgeReadout`.
  - Commands use `uv` per project rules.
