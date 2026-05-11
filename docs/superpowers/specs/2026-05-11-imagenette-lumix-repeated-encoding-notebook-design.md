# Imagenette Lumix Repeated-Encoding Notebook Design

## Goal

Create a tutorial Jupyter notebook that reproduces the Tidy3D Hackathon Imagenette repeated-encoding baseline using native Lumix and Flax conventions.

The target reference is the grayscale capped Imagenette run that reported approximately 50.6% validation accuracy:

- capped samples: 5000 train, 1000 validation
- grayscale images
- resize to 160x160, then BOX downsample to 64x64
- non-overlapping 4x4 patches, flattened to 16 optical channels
- 16x16 patch-token grid, 256 patches per image
- tied trainable 16x16 unitary across repeated optical layers
- repeated phase encoding with phase scale pi
- 4 optical layers
- 4x4 spatial pooling over the 16x16 patch map
- 256 final optical features
- train-set feature standardization before ridge
- closed-form ridge readout

The notebook must be Lumix-native for optical layers and readout application. Dataset loading, image preprocessing, patch extraction, and feature standardization remain notebook-local tutorial code.

## Non-Goals

- Do not add dataset loaders, standardizers, pooling layers, or Imagenette-specific helpers to the Lumix library.
- Do not implement the axial token-mixer extension in the main notebook.
- Do not add a custom Lumix training framework.
- Do not put ridge fitting inside `RidgeReadout.__call__`.
- Do not use the old Tidy3D helper functions as runtime dependencies. The notebook can cite them as provenance, but should implement the workflow with Lumix, Flax, JAX, Optax, and small local notebook utilities.

## Notebook Type and Location

The artifact should be a tutorial notebook tracked in the repository.

Notebook path:

```text
notebooks/imagenette_repeated_encoding_lumix.ipynb
```

## Dependencies

Use the project environment and `uv` commands.

Expected Python packages:

- `jax`
- `flax`
- `optax`
- `numpy`
- `Pillow`
- `matplotlib`
- `lumix`

The notebook may use Python standard library download/extraction utilities for Imagenette.

## Dataset Design

The notebook should download or reuse Imagenette from the FastAI Imagenette source.

Default dataset:

```text
imagenette2-160
```

Default local cache:

```text
data/imagenette2-160
```

The notebook should expose these configuration values near the top:

```python
MAX_TRAIN = 5000
MAX_VAL = 1000
IMAGE_SIZE = 160
POOLED_IMAGE_SIZE = 64
COLOR_MODE = "grayscale"
PATCH_SIZE = 4
PATCH_STRIDE = 4
RANDOM_SEED = 7
```

The sample cap must be stratified per class, matching the Tidy3D reference behavior. This is important because the target accuracy is measured on a capped validation subset, not necessarily the full validation split.

## Preprocessing Design

For each image:

1. Load from the ImageFolder-style split directory.
2. Convert to grayscale.
3. Resize to 160x160 with bilinear interpolation.
4. Downsample to 64x64 with BOX interpolation.
5. Convert to float in `[0, 1]`.
6. Extract non-overlapping 4x4 patches with stride 4.
7. Flatten each patch to 16 values.

The resulting tensor shapes should be:

```text
train_patches: (5000, 256, 16)
val_patches:   (1000, 256, 16)
y_train:       (5000,)
y_val:         (1000,)
```

The notebook should include a shape-check cell that asserts these dimensions.

## Optical Model Design

The optical model should be a Flax `nn.Module` built from Lumix layers and functions.

Conceptual structure:

```python
class RepeatedEncodingPatchEncoder(nn.Module):
    channels: int = 16
    layers: int = 4
    phase_scale: float = jnp.pi

    @nn.compact
    def __call__(self, patch_values):
        fields = uniform_complex_field(patch_values.shape[:-1], self.channels)
        phase_fields = InformationEncoder(mode="phase")(self.phase_scale * patch_values)
        unitary = UnitaryLinear(width=self.channels, name="shared_unitary")

        for _ in range(self.layers):
            fields = fields * phase_fields
            fields = unitary(fields)

        return intensity(fields)
```

The implementation must instantiate `UnitaryLinear(name="shared_unitary")` once and reuse the module object in the loop so the same trainable 16x16 unitary is tied across all layers.

The model input and output shapes should be:

```text
input:  (batch, 256, 16)
output: (batch, 256, 16)
```

The initial field should match the Tidy3D reference:

```text
sqrt(1 / channels)
```

for each optical channel.

## Repeated Encoding Semantics

The notebook target is repeated phase encoding, not single-upload encoding.

For every optical layer:

```text
fields = fields * exp(1j * phase_scale * patch_values)
fields = tied_unitary(fields)
```

Use Lumix `InformationEncoder(mode="phase", normalize=False)` for the phase factor. The patch values are already in `[0, 1]`, and the phase scale is applied explicitly as `phase_scale * patch_values`.

## Pooling Design

Do not add or use a Lumix pooling abstraction.

After optical detection, reshape the patch features:

```text
(batch, 256, 16) -> (batch, 16, 16, 16)
```

Then use Flax pooling:

```python
pooled = nn.avg_pool(
    feature_map,
    window_shape=(4, 4),
    strides=(4, 4),
    padding="VALID",
)
```

Final feature shape:

```text
(batch, 4, 4, 16) -> (batch, 256)
```

This matches the Tidy3D reference `spatial_pool_grid=4` and `num_optical_features=256`.

## Feature Standardization Design

Standardization is notebook-local and occurs after Lumix optical features and before ridge.

It must not be implemented inside Lumix and must not be hidden inside the optical model.

Use train-set statistics only:

```python
mean = train_features.mean(axis=0, keepdims=True)
std = train_features.std(axis=0, keepdims=True)
std = jnp.where(std == 0, 1.0, std)

train_features_std = (train_features - mean) / std
val_features_std = (val_features - mean) / std
```

This reproduces the Tidy3D reference workflow while keeping the Lumix library clean.

## Ridge Readout Design

Use Lumix ridge as a Flax-compatible two-part workflow:

1. `lumix.functional.solve_ridge` solves closed-form params.
2. `lumix.linen.RidgeReadout` applies those params.

Example:

```python
ridge_params = solve_ridge(train_features_std, targets, alpha=RIDGE_ALPHA, use_bias=True)
readout = RidgeReadout(features=NUM_CLASSES)
val_logits = readout.apply({"params": ridge_params}, val_features_std)
```

The ridge solve should not be embedded inside `RidgeReadout.__call__`.

## Training Objective

The trainable parameters are the Flax params of the Lumix optical encoder, specifically the tied unitary params.

The objective should match the Tidy3D baseline:

1. Apply the optical encoder to all capped training patches.
2. Pool to 256 features.
3. Standardize train features with train-set mean/std.
4. Solve ridge params against one-hot class targets.
5. Apply `RidgeReadout` to standardized train features.
6. Compute MSE between ridge scores and one-hot targets.
7. Backpropagate through the feature extraction and ridge solve into optical params.
8. Update optical params with Optax.

Recommended defaults:

```python
NUM_LAYERS = 4
PHASE_SCALE = jnp.pi
RIDGE_ALPHA = 1.0e-3
UNITARY_STEPS = 50
UNITARY_LR = 1.0e-2
```

The notebook should use `jax.jit` for the objective/update step where practical.

## Memory and Runtime Design

The reference run is full-batch over 5000 images. The notebook should default to the full comparable run, but include one explicit smoke-run switch:

```python
TRAIN_UNITARY_ON_SUBSET = None
```

When set to an integer, it limits unitary training to a stratified subset while still preserving the same code path.

For the comparable target run, this value should remain `None`.

## Evaluation Design

After training:

1. Extract raw train and validation optical features.
2. Compute train-set mean/std.
3. Standardize train and validation features.
4. Solve ridge on all capped training features.
5. Apply `RidgeReadout` to train and validation features.
6. Report train and validation accuracy.

The notebook should print a compact metrics table:

```text
train_accuracy
val_accuracy
ridge_alpha
unitary_steps
unitary_lr
num_layers
phase_scale_pi
num_optical_features
```

Expected reference target:

```text
validation accuracy around 50%
```

Small differences are acceptable because Lumix unitary parametrization, random seed handling, JAX precision, and environment details may differ from the Tidy3D script.

## Saved Outputs

The notebook should save:

```text
artifacts/imagenette_lumix_repeated_encoding/metrics.json
artifacts/imagenette_lumix_repeated_encoding/optical_params.msgpack
```

Use Flax serialization for the msgpack params file.

The metrics JSON should include:

```text
train_accuracy
val_accuracy
max_train
max_val
image_size
pooled_image_size
color_mode
patch_size
patch_stride
num_layers
phase_scale_pi
ridge_alpha
unitary_steps
unitary_lr
num_optical_features
random_seed
```

## Notebook Section Outline

1. Title and objective
2. Imports and configuration
3. Download or locate Imagenette
4. Load capped grayscale train/validation splits
5. Patch extraction and shape checks
6. Define the Lumix repeated-encoding patch encoder
7. Define pooling and ridge helper functions
8. Run smoke checks on a tiny batch
9. Initialize optical Flax params and optimizer
10. Train the tied unitary with ridge-MSE
11. Evaluate final ridge readout
12. Save metrics and params
13. Brief comparison to Tidy3D reference result

## Testing and Validation

Before executing the full training run, the notebook should include smoke checks that:

- confirm patch tensor shape is `(samples, 256, 16)`
- confirm optical encoder output shape is `(batch, 256, 16)`
- confirm pooled feature shape is `(batch, 256)`
- confirm `RidgeReadout` can apply params from `solve_ridge`
- confirm one update step changes optical params

The implementation plan should include a small automated test or notebook execution check if practical. If full notebook execution is too slow for CI, validate helper cells with a small synthetic batch and report that full training is manual.

## Flax Compatibility Requirements

- Optical model params are stored in a normal Flax variable tree.
- Optax updates only optical params.
- `RidgeReadout` is applied through `.apply({"params": ridge_params}, features)`.
- Ridge fitting remains a pure functional solve outside the module.
- Feature standardization remains notebook-local and explicit.
- Pooling uses Flax `nn.avg_pool`.

## Implementation Decisions

- The notebook lives at `notebooks/imagenette_repeated_encoding_lumix.ipynb`.
- Optical params are saved with Flax serialization to `optical_params.msgpack`.
- The default implementation uses float32 for Lumix consistency.
- Do not enable x64 in the first implementation. If accuracy is materially below the reference after a completed full run, treat x64 as a later ablation rather than a hidden default.
