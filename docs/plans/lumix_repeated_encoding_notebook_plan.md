# Lumix Repeated-Encoding Notebook Plan

## Goal

Create a clean tutorial notebook series that explains and reproduces the repeated data re-encoding experiments in Lumix. The notebooks must read like Flax neural-network tutorials, not like collections of functional helper calls or one-off scripts.

The notebooks should:

- Use Lumix through `flax.linen.Module` architectures.
- Keep model definitions readable and close to the math.
- Use Optax training loops with explicit `params`, `opt_state`, `train_step`, and `eval_step`.
- Prefer structured modules and named model configs over ad hoc function pipelines.
- Include useful visualizations whenever they clarify the experiment.
- Separate conceptual explanation, dataset construction, training, ablation, and Imagenette application into focused notebooks.
- Avoid tied-model discussion in the core MNIST notebooks.
- Discuss parameter sharing only in the Imagenette context, where patch/time multiplexing gives enough diversity that sharing can reduce parameters and overfitting.

## Non-Goals

- Do not create the notebooks yet.
- Do not add new Lumix library APIs as part of notebook creation unless a notebook exposes a real missing abstraction.
- Do not make the notebooks depend on private state from previous terminal runs.
- Do not use low-level functional calls as the main user-facing architecture style.
- Do not turn every experiment into a full hyperparameter sweep.
- Do not present tables alone when a plot would make the comparison clearer.

## Shared Notebook Conventions

### Flax-Style Architecture

Each notebook that defines a model should use `flax.linen.Module` classes. The architecture should be readable from `setup` or `@nn.compact` blocks.

Preferred style:

```python
class RepeatedOpticalClassifier(nn.Module):
    width: int
    depth: int
    readout_features: int

    @nn.compact
    def __call__(self, x):
        encoder = InformationEncoder(mode="phase", normalize=False)
        fields = initialize_uniform_field(x, width=self.width)
        phase_mask = encoder(jnp.pi * x)

        for layer_index in range(self.depth):
            fields = UnitaryLinear(self.width, name=f"unitary_{layer_index}")(fields * phase_mask)

        intensities = IntensityReadout()(fields)
        return TemperatureSoftmaxReadout(self.readout_features)(intensities)
```

The final notebooks should avoid exposing internals like manual matrix constructors, singular-value utilities, or custom functional readout helpers unless the notebook is specifically explaining that low-level object.

### Training Workflow

Use standard Flax/Optax training shape in every training notebook:

1. Define a dataclass config.
2. Instantiate the Flax module.
3. Initialize with `model.init(rng, sample_batch)`.
4. Create `optax.adam` or the chosen optimizer.
5. Define `loss_fn(params, batch)`.
6. Define `@jax.jit train_step`.
7. Define `@jax.jit eval_step`.
8. Train for a controlled number of epochs.
9. Store history as a tidy dataframe-like list of dicts.
10. Plot metrics.

Required training objects:

- `params`
- `opt_state`
- `train_step`
- `eval_step`
- `history`
- `metrics`

Avoid hidden global mutation. Avoid training functions that implicitly read globals except for notebook-level constants such as `NUM_CLASSES` and `WIDTH`.

### Visualizations

Every notebook should include plots where useful. Good default visualizations:

- PCA feature histograms or min/max ranges.
- Phase values before encoding.
- Complex phase-mask unit-circle scatter for a sample.
- Intermediate optical field magnitude/phase heatmaps.
- Depth vs validation accuracy.
- Depth vs validation CE loss.
- Depth vs insertion loss.
- Learned temperature `gamma` vs depth.
- Parameter count vs accuracy.
- Train/validation gap for Imagenette sharing experiments.

Use compact plots with labeled axes and captions. Avoid dumping large raw arrays unless the notebook is explicitly demonstrating a small numerical example.

### Artifact Policy

Notebooks should be reproducible from source data when practical, but can also load existing experiment JSON artifacts for summary notebooks.

Allowed artifact inputs:

- `artifacts/mnist_pca_phase_comparison/three_way_depth_curve_trainable_softmax_results.json`
- `artifacts/mnist_pca_phase_comparison/objective_readout_results.json`
- `artifacts/mnist_pca_phase_comparison/ridge_readout_results.json`
- `artifacts/imagenette_williamson_vs_tied/summary.md`
- Imagenette architecture iteration artifacts under `artifacts/imagenette_lumix_architecture_iterations/`

When a notebook loads an artifact, it should clearly state whether it is:

- Re-running the experiment.
- Loading cached results for explanation.
- Doing both, with a small rerun option.

## Notebook 1: Repeated Encoding Basics

### Purpose

Explain the repeated data re-encoding mechanism with minimal machinery.

### Discuss

- Optical field vector `h`.
- Data vector `x`.
- Phase mask vector `m(x) = exp(i alpha x)`.
- Diagonal phase operator `D(x) = diag(m(x))`.
- Elementwise equivalence: `D(x) h = m(x) * h`.
- Repeated update:

```text
h_l = U_l D(x) h_{l-1}
```

- Why the optical transform is linear in `h` for fixed `x`.
- Why the model is nonlinear as a function of `x`.
- Why `U_l` is trainable during training and fixed after training.

### Must Avoid

- No tied-model discussion.
- No Williamson comparison.
- No dataset training.

### Visualizations

- Unit-circle plot of `m(x)`.
- Bar plot of final intensities for a hand-picked small example.
- Optional heatmap showing magnitude/phase of `h_0`, `h_1`, `h_2`, `h_3`.

### Flax Style

Even though this is conceptual, include a tiny `nn.Module` for a repeated optical block so the tutorial style matches later notebooks.

## Notebook 2: MNIST PCA Phase Dataset

### Purpose

Explain the phase-only input representation used for the MNIST comparison.

### Discuss

- Load MNIST.
- Flatten images.
- Fit PCA on train split only.
- Project to 16 PCA components.
- Standardize using train statistics.
- Map each component with train-set min/max without clipping.
- Encode phase with `alpha = pi`.

### Must State

- This is phase-only encoding.
- The input to the optical model is real PCA features.
- The optical field starts as a uniform complex vector.
- The data affects the field through phase masks only.

### Visualizations

- Example MNIST images.
- PCA explained variance bar plot.
- Feature histogram before and after min/max mapping.
- Phase histogram after multiplying by `pi`.
- Unit-circle scatter of encoded phase values for a few samples.

### Flax Style

Dataset utilities can be plain functions, but model-facing dataset batches should be shaped and named clearly:

- `x_train: [num_examples, 16]`
- `y_train: [num_examples, 10]`
- `x_test: [num_examples, 16]`
- `y_test: [num_examples, 10]`

## Notebook 3: MNIST Repeated vs Williamson Core Benchmark

### Purpose

Present the main controlled scientific comparison.

### Models

Use Flax modules for:

- `UnitaryRepeatedClassifier`
- `SubunitaryRepeatedClassifier`
- `WilliamsonClassifier`

Each model should expose the same call signature:

```python
logits, aux = model(x, return_aux=True)
```

where `aux` can contain:

- output intensities
- learned temperature
- insertion loss

### Discuss

- Unitary repeated:

```text
h_l = U_l D(x) h_{l-1}
```

- Subunitary repeated:

```text
h_l = S_l D(x) h_{l-1}
```

- Williamson:

```text
h_0 = D(x) h_in
h_l = W_l(U_l h_{l-1})
```

- Same width 16.
- Same depth 1-5.
- Same optimizer and epochs.
- Same trainable-temperature softmax objective.
- Accuracy, CE loss, learned `gamma`, and insertion loss.

### Visualizations

- Depth vs validation accuracy.
- Depth vs validation loss.
- Depth vs mean insertion loss.
- Learned `gamma` vs depth.
- Optional scatter: accuracy vs insertion loss.

### Artifact Source

Use:

```text
artifacts/mnist_pca_phase_comparison/three_way_depth_curve_trainable_softmax_results.json
```

The notebook can include an optional cell to rerun the benchmark, but the default tutorial path should load the JSON to avoid long runtime.

## Notebook 4: Objective and Readout Ablation

### Purpose

Explain why the final objective matters and why trainable-temperature softmax was used.

### Discuss

- Physical normalized intensity:

```text
p_c = I_c / sum_j I_j
```

- Fixed-temperature softmax:

```text
p = softmax(gamma I)
```

- Trainable-temperature softmax:

```text
p = softmax(exp(log_gamma) I)
```

- Why plain softmax on bounded intensities can be weak.
- Why `gamma` is a calibration/readout parameter.
- Why both repeated and Williamson receive the same readout treatment.
- Ridge readout as post-hoc linear readout, not the main optical objective.

### Visualizations

- Bar chart comparing objectives by accuracy and loss.
- Learned `gamma` by model.
- Calibration-style plot of confidence distribution if available.
- Ridge vs softmax readout comparison table.

### Artifact Sources

Use:

```text
artifacts/mnist_pca_phase_comparison/objective_readout_results.json
artifacts/mnist_pca_phase_comparison/ridge_readout_results.json
```

## Notebook 5: Imagenette Repeated Optical Network

### Purpose

Explain the Imagenette pipeline separately from the MNIST comparison.

### Discuss

- Imagenette loading.
- Downsample to the chosen resolution.
- Non-overlapping patch strategy.
- Time multiplexing.
- Width-16 optical core.
- Repeated phase encoding.
- Pooling/output construction.
- Digital readout.

### Architecture Requirements

Use a Flax module that makes the optical core readable:

```python
class ImagenetteRepeatedOpticalCore(nn.Module):
    width: int
    depth: int
    sharing: str

    @nn.compact
    def __call__(self, patch_vectors):
        ...
```

Use named submodules for:

- encoder
- optical core
- pooling
- readout

The notebook should not hide the architecture in a single monolithic function.

### Visualizations

- Example original/downsampled images.
- Patch grid overlay.
- Time-multiplexed token layout diagram.
- Training/validation accuracy curves.
- Confusion matrix.
- Per-class accuracy.
- Feature or output-vector heatmap after pooling.

## Notebook 6: Imagenette Parameter Sharing and Generalization

### Purpose

Show that parameter sharing can be useful in Imagenette when the data path already supplies enough diversity through patches and time multiplexing.

This replaces any tied-model discussion in the MNIST core notebooks.

### Discuss

- Why tying hurt small MNIST width-16 PCA experiments.
- Why Imagenette is different:
  - many patches/tokens
  - richer spatial variation
  - larger effective dataset structure
  - more opportunities for overfitting in untied optical cores
- Parameter sharing as hardware simplification and regularization.
- Tradeoff between expressivity and generalization.

### Required Comparisons

Include, when artifact coverage exists:

- tied vs untied unitary
- parameter count
- train accuracy
- validation accuracy
- train-validation gap
- hardware implication of fewer programmed optical elements

### Visualizations

- Parameter count vs validation accuracy.
- Train-validation gap by sharing mode.
- Accuracy curves for tied and untied models.
- Bar chart of overfitting gap.

### Artifact Sources

Use:

```text
artifacts/imagenette_williamson_vs_tied/summary.md
artifacts/imagenette_lumix_architecture_iterations/
```

If the artifact format is inconsistent, normalize results into a tidy dataframe inside the notebook with clear column names.

## Recommended Notebook Order

1. `01_repeated_encoding_basics.ipynb`
2. `02_mnist_pca_phase_dataset.ipynb`
3. `03_mnist_repeated_vs_williamson.ipynb`
4. `04_objective_and_readout_ablation.ipynb`
5. `05_imagenette_repeated_optical_network.ipynb`
6. `06_imagenette_parameter_sharing_generalization.ipynb`

## Acceptance Criteria

The notebook series is ready when:

- Every architecture notebook defines models as `flax.linen.Module`.
- Training notebooks use explicit Flax/Optax workflow.
- The main MNIST comparison can be understood without reading experiment scripts.
- The Imagenette sharing discussion is separated from the MNIST core story.
- Each notebook has at least one meaningful visualization.
- Long-running cells are optional or clearly marked.
- Cached-result paths are documented.
- Tables include units where applicable, especially insertion loss in dB.
- The notebooks do not rely on hidden state from prior terminal sessions.

## Open Decisions Before Notebook Creation

- Whether notebooks should re-run full training by default or load artifacts by default.
- Whether final notebooks should live under `notebooks/`, `examples/`, or `docs/tutorials/`.
- Whether plots should use Matplotlib only or allow Seaborn.
- Whether Imagenette notebooks should use full data by default or a small fast subset with full-run artifact loading.
