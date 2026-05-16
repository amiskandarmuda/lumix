# Ridge and Information Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic optical information encoding and a Flax-compatible ridge readout workflow to Lumix without adding custom pooling or training abstractions.

**Architecture:** Keep Lumix's existing split: pure JAX math in `lumix.functional`, Flax modules in `lumix.linen`, and generic metrics in `lumix.metrics`. Ridge fitting is a closed-form functional solver that returns ordinary Flax params; `RidgeReadout` is a normal linear Flax module that can be initialized from those params and fine-tuned with Optax.

**Tech Stack:** JAX, Flax Linen, Optax, pytest, uv.

---

## File Structure

- Create `src/lumix/functional/encoding.py`
  - Stateless conversion from real-valued data to complex optical fields.
  - Exposes `encode_phase`, `encode_amplitude`, and `encode_complex`.
- Create `src/lumix/functional/ridge.py`
  - Closed-form ridge solver returning Flax-compatible `kernel` and optional `bias` params.
- Create `src/lumix/linen/encoding.py`
  - `InformationEncoder`, a param-free `nn.Module` wrapper around functional encoding.
- Modify `src/lumix/linen/readout.py`
  - Add `RidgeReadout` with direct `kernel` and `bias` params.
- Modify `src/lumix/functional/__init__.py`
  - Export encoding and ridge functions.
- Modify `src/lumix/linen/__init__.py`
  - Export `InformationEncoder` and `RidgeReadout`.
- Modify `src/lumix/metrics.py`
  - Add generic `mean_squared_error`.
- Create `tests/test_encoding.py`
  - Verify deterministic encoding semantics and Linen wrapper behavior.
- Create `tests/test_ridge.py`
  - Verify closed-form solve, `RidgeReadout` param compatibility, and Optax update compatibility.
- Create `tests/test_metrics.py`
  - Verify MSE for real and complex arrays.

Do not add pooling files. Use Flax directly:

```python
from flax import linen as nn
from lumix.functional.readout import intensity

pooled = nn.avg_pool(intensity(fields), window_shape=(4, 4), strides=(4, 4), padding="VALID")
```

---

### Task 1: Add Generic Mean Squared Error Metric

**Files:**
- Modify: `src/lumix/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
import jax.numpy as jnp

from lumix.metrics import mean_squared_error


def test_mean_squared_error_real_values():
    targets = jnp.array([1.0, 2.0, 4.0])
    predictions = jnp.array([1.0, 4.0, 1.0])

    error = mean_squared_error(targets, predictions)

    assert jnp.allclose(error, jnp.array(13.0 / 3.0))


def test_mean_squared_error_complex_values_uses_squared_magnitude():
    targets = jnp.array([1.0 + 1.0j, 2.0 + 0.0j])
    predictions = jnp.array([2.0 + 3.0j, 0.0 + 0.0j])

    error = mean_squared_error(targets, predictions)

    assert jnp.allclose(error, jnp.array(4.5))
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: fail with `ImportError` or `NameError` for `mean_squared_error`.

- [ ] **Step 3: Implement the metric**

Modify `src/lumix/metrics.py`:

```python
import jax.numpy as jnp


def accuracy(probabilities_target: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(probabilities_target, axis=-1) == jnp.argmax(predictions, axis=-1))


def mean_squared_error(targets: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    error = predictions - targets
    return jnp.mean(jnp.square(jnp.abs(error)))
```

- [ ] **Step 4: Run the metric tests**

Run:

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lumix/metrics.py tests/test_metrics.py
git commit -m "feat: add mean squared error metric"
```

---

### Task 2: Add Functional Information Encoding

**Files:**
- Create: `src/lumix/functional/encoding.py`
- Modify: `src/lumix/functional/__init__.py`
- Test: `tests/test_encoding.py`

- [ ] **Step 1: Write failing functional encoding tests**

Create `tests/test_encoding.py`:

```python
import jax.numpy as jnp

from lumix.functional.encoding import encode_amplitude, encode_complex, encode_phase


def test_encode_phase_returns_unit_magnitude_complex_field():
    phases = jnp.array([0.0, jnp.pi / 2.0, jnp.pi])

    encoded = encode_phase(phases)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(jnp.abs(encoded), jnp.ones_like(phases))
    assert jnp.allclose(encoded, jnp.exp(1j * phases).astype(jnp.complex64))


def test_encode_phase_normalizes_configured_input_range():
    values = jnp.array([0.0, 0.5, 1.0])

    encoded = encode_phase(
        values,
        normalize=True,
        input_range=(0.0, 1.0),
        phase_range=(0.0, jnp.pi),
    )

    assert jnp.allclose(encoded, jnp.exp(1j * jnp.array([0.0, jnp.pi / 2.0, jnp.pi])).astype(jnp.complex64))


def test_encode_amplitude_returns_complex_amplitude_field():
    amplitudes = jnp.array([0.0, 0.5, 1.0])

    encoded = encode_amplitude(amplitudes)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(encoded, amplitudes.astype(jnp.complex64))


def test_encode_amplitude_can_clip_normalized_values():
    values = jnp.array([-1.0, 0.5, 2.0])

    encoded = encode_amplitude(
        values,
        normalize=True,
        input_range=(0.0, 1.0),
        amplitude_range=(0.0, 1.0),
        clip=True,
    )

    assert jnp.allclose(encoded, jnp.array([0.0, 0.5, 1.0], dtype=jnp.complex64))


def test_encode_complex_uses_amplitude_and_phase():
    phases = jnp.array([0.0, jnp.pi])

    encoded = encode_complex(phases, amplitude=0.5)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(encoded, 0.5 * jnp.exp(1j * phases).astype(jnp.complex64))


def test_functional_encoding_exports():
    from lumix.functional import encode_amplitude as exported_amplitude
    from lumix.functional import encode_complex as exported_complex
    from lumix.functional import encode_phase as exported_phase

    assert exported_phase is encode_phase
    assert exported_amplitude is encode_amplitude
    assert exported_complex is encode_complex
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/test_encoding.py -v
```

Expected: fail because `lumix.functional.encoding` does not exist.

- [ ] **Step 3: Implement functional encoding**

Create `src/lumix/functional/encoding.py`:

```python
import jax.numpy as jnp


def _scale_range(
    values: jnp.ndarray,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    clip: bool,
) -> jnp.ndarray:
    input_min, input_max = input_range
    output_min, output_max = output_range
    if input_max == input_min:
        raise ValueError("input_range endpoints must be different")

    normalized = (values - input_min) / (input_max - input_min)
    if clip:
        normalized = jnp.clip(normalized, 0.0, 1.0)
    return output_min + normalized * (output_max - output_min)


def encode_phase(
    values: jnp.ndarray,
    phase_range: tuple[float, float] = (0.0, jnp.pi),
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    phase = values
    if normalize:
        phase = _scale_range(values, input_range, phase_range, clip)
    return jnp.exp(1j * phase).astype(jnp.complex64)


def encode_amplitude(
    values: jnp.ndarray,
    amplitude_range: tuple[float, float] = (0.0, 1.0),
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    amplitude = values
    if normalize:
        amplitude = _scale_range(values, input_range, amplitude_range, clip)
    return amplitude.astype(jnp.complex64)


def encode_complex(
    values: jnp.ndarray,
    phase_range: tuple[float, float] = (0.0, jnp.pi),
    amplitude: float | jnp.ndarray = 1.0,
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    phase_field = encode_phase(
        values,
        phase_range=phase_range,
        normalize=normalize,
        input_range=input_range,
        clip=clip,
    )
    return amplitude * phase_field
```

- [ ] **Step 4: Export functional encoding**

Modify `src/lumix/functional/__init__.py` by adding imports:

```python
from lumix.functional.encoding import encode_amplitude, encode_complex, encode_phase
```

Add names to `__all__`:

```python
"encode_amplitude",
"encode_complex",
"encode_phase",
```

- [ ] **Step 5: Run the encoding tests**

Run:

```bash
uv run pytest tests/test_encoding.py -v
```

Expected: all tests in `tests/test_encoding.py` pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/lumix/functional/encoding.py src/lumix/functional/__init__.py tests/test_encoding.py
git commit -m "feat: add deterministic information encoding"
```

---

### Task 3: Add Linen InformationEncoder

**Files:**
- Create: `src/lumix/linen/encoding.py`
- Modify: `src/lumix/linen/__init__.py`
- Modify: `tests/test_encoding.py`

- [ ] **Step 1: Add failing Linen encoder tests**

Append to `tests/test_encoding.py`:

```python
from jax import random

from lumix.linen.encoding import InformationEncoder


def test_information_encoder_has_no_params():
    encoder = InformationEncoder(mode="phase")
    variables = encoder.init(random.key(0), jnp.array([0.0, jnp.pi]))

    assert variables == {}


def test_information_encoder_phase_mode_matches_functional_encoding():
    values = jnp.array([0.0, 1.0])
    encoder = InformationEncoder(
        mode="phase",
        normalize=True,
        input_range=(0.0, 1.0),
        phase_range=(0.0, jnp.pi),
    )

    encoded = encoder.apply({}, values)

    assert jnp.allclose(encoded, encode_phase(values, normalize=True, input_range=(0.0, 1.0), phase_range=(0.0, jnp.pi)))


def test_information_encoder_amplitude_mode_matches_functional_encoding():
    values = jnp.array([0.0, 1.0])
    encoder = InformationEncoder(mode="amplitude")

    encoded = encoder.apply({}, values)

    assert jnp.allclose(encoded, encode_amplitude(values))


def test_information_encoder_complex_mode_matches_functional_encoding():
    values = jnp.array([0.0, jnp.pi])
    encoder = InformationEncoder(mode="complex", amplitude=0.25)

    encoded = encoder.apply({}, values)

    assert jnp.allclose(encoded, encode_complex(values, amplitude=0.25))


def test_information_encoder_rejects_unknown_mode():
    encoder = InformationEncoder(mode="unknown")

    try:
        encoder.apply({}, jnp.array([0.0]))
    except ValueError as error:
        assert str(error) == "mode must be one of 'phase', 'amplitude', or 'complex'"
    else:
        raise AssertionError("expected ValueError")


def test_linen_encoding_exports():
    from lumix.linen import InformationEncoder as exported_encoder

    assert exported_encoder is InformationEncoder
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/test_encoding.py -v
```

Expected: fail because `lumix.linen.encoding` does not exist.

- [ ] **Step 3: Implement Linen encoder**

Create `src/lumix/linen/encoding.py`:

```python
from flax import linen as nn

import jax.numpy as jnp

from lumix.functional.encoding import encode_amplitude, encode_complex, encode_phase


class InformationEncoder(nn.Module):
    mode: str = "phase"
    normalize: bool = False
    input_range: tuple[float, float] = (0.0, 1.0)
    phase_range: tuple[float, float] = (0.0, jnp.pi)
    amplitude_range: tuple[float, float] = (0.0, 1.0)
    amplitude: float = 1.0
    clip: bool = False

    @nn.compact
    def __call__(self, values):
        if self.mode == "phase":
            return encode_phase(
                values,
                phase_range=self.phase_range,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        if self.mode == "amplitude":
            return encode_amplitude(
                values,
                amplitude_range=self.amplitude_range,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        if self.mode == "complex":
            return encode_complex(
                values,
                phase_range=self.phase_range,
                amplitude=self.amplitude,
                normalize=self.normalize,
                input_range=self.input_range,
                clip=self.clip,
            )
        raise ValueError("mode must be one of 'phase', 'amplitude', or 'complex'")
```

- [ ] **Step 4: Export Linen encoder**

Modify `src/lumix/linen/__init__.py` by adding:

```python
from lumix.linen.encoding import InformationEncoder
```

Add to `__all__`:

```python
"InformationEncoder",
```

- [ ] **Step 5: Run encoding tests**

Run:

```bash
uv run pytest tests/test_encoding.py -v
```

Expected: all encoding tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/lumix/linen/encoding.py src/lumix/linen/__init__.py tests/test_encoding.py
git commit -m "feat: add flax information encoder"
```

---

### Task 4: Add Functional Ridge Solver

**Files:**
- Create: `src/lumix/functional/ridge.py`
- Modify: `src/lumix/functional/__init__.py`
- Test: `tests/test_ridge.py`

- [ ] **Step 1: Write failing ridge solver tests**

Create `tests/test_ridge.py`:

```python
import jax.numpy as jnp

from lumix.functional.ridge import solve_ridge


def test_solve_ridge_recovers_linear_map_without_bias():
    inputs = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    kernel = jnp.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
        ]
    )
    targets = inputs @ kernel

    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=False)

    assert set(params.keys()) == {"kernel"}
    assert params["kernel"].shape == (2, 2)
    assert jnp.allclose(params["kernel"], kernel, atol=1e-5)


def test_solve_ridge_recovers_linear_map_with_bias():
    inputs = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    kernel = jnp.array([[2.0], [-1.0]])
    bias = jnp.array([0.75])
    targets = inputs @ kernel + bias

    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=True)

    assert set(params.keys()) == {"kernel", "bias"}
    assert params["kernel"].shape == (2, 1)
    assert params["bias"].shape == (1,)
    assert jnp.allclose(params["kernel"], kernel, atol=1e-5)
    assert jnp.allclose(params["bias"], bias, atol=1e-5)


def test_solve_ridge_accepts_vector_targets():
    inputs = jnp.array([[1.0], [2.0], [3.0]])
    targets = jnp.array([2.0, 4.0, 6.0])

    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=False)

    assert params["kernel"].shape == (1, 1)
    assert jnp.allclose(inputs @ params["kernel"], targets[:, None], atol=1e-5)


def test_functional_ridge_exports():
    from lumix.functional import solve_ridge as exported_solve_ridge

    assert exported_solve_ridge is solve_ridge
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_ridge.py -v
```

Expected: fail because `lumix.functional.ridge` does not exist.

- [ ] **Step 3: Implement ridge solver**

Create `src/lumix/functional/ridge.py`:

```python
import jax.numpy as jnp


def solve_ridge(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    alpha: float = 1e-3,
    use_bias: bool = True,
) -> dict[str, jnp.ndarray]:
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (samples, features)")

    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2:
        raise ValueError("targets must have shape (samples,) or (samples, outputs)")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count")

    design = inputs
    feature_count = inputs.shape[-1]
    if use_bias:
        ones = jnp.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        design = jnp.concatenate([inputs, ones], axis=-1)

    regularizer = alpha * jnp.eye(design.shape[-1], dtype=design.dtype)
    if use_bias:
        regularizer = regularizer.at[-1, -1].set(0.0)

    solution = jnp.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    if use_bias:
        return {
            "kernel": solution[:feature_count],
            "bias": solution[feature_count],
        }
    return {"kernel": solution}
```

- [ ] **Step 4: Export ridge solver**

Modify `src/lumix/functional/__init__.py` by adding:

```python
from lumix.functional.ridge import solve_ridge
```

Add to `__all__`:

```python
"solve_ridge",
```

- [ ] **Step 5: Run ridge solver tests**

Run:

```bash
uv run pytest tests/test_ridge.py -v
```

Expected: all current ridge tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/lumix/functional/ridge.py src/lumix/functional/__init__.py tests/test_ridge.py
git commit -m "feat: add closed form ridge solver"
```

---

### Task 5: Add RidgeReadout Linen Module

**Files:**
- Modify: `src/lumix/linen/readout.py`
- Modify: `src/lumix/linen/__init__.py`
- Modify: `tests/test_ridge.py`

- [ ] **Step 1: Add failing RidgeReadout tests**

Append to `tests/test_ridge.py`:

```python
import optax
from flax import linen as nn
from jax import grad, random

from lumix.linen.readout import RidgeReadout


def test_ridge_readout_applies_closed_form_params():
    inputs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    kernel = jnp.array([[2.0], [-1.0]])
    bias = jnp.array([0.5])
    variables = {"params": {"kernel": kernel, "bias": bias}}

    predictions = RidgeReadout(features=1).apply(variables, inputs)

    assert jnp.allclose(predictions, inputs @ kernel + bias)


def test_ridge_readout_params_match_closed_form_solver():
    inputs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    kernel = jnp.array([[2.0], [-1.0]])
    bias = jnp.array([0.5])
    targets = inputs @ kernel + bias
    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=True)

    predictions = RidgeReadout(features=1).apply({"params": params}, inputs)

    assert jnp.allclose(predictions, targets, atol=1e-5)


def test_ridge_readout_can_disable_bias():
    inputs = jnp.array([[1.0, 2.0]])
    kernel = jnp.array([[3.0], [4.0]])

    predictions = RidgeReadout(features=1, use_bias=False).apply({"params": {"kernel": kernel}}, inputs)

    assert jnp.allclose(predictions, jnp.array([[11.0]]))


def test_ridge_readout_params_are_optax_trainable():
    inputs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    targets = jnp.array([[1.0], [-1.0]])
    module = RidgeReadout(features=1)
    variables = module.init(random.key(0), inputs)
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state = optimizer.init(variables["params"])

    def loss_fn(params):
        predictions = module.apply({"params": params}, inputs)
        return jnp.mean(jnp.square(predictions - targets))

    gradients = grad(loss_fn)(variables["params"])
    updates, opt_state = optimizer.update(gradients, opt_state, variables["params"])
    updated_params = optax.apply_updates(variables["params"], updates)

    assert set(updated_params.keys()) == {"kernel", "bias"}
    assert updated_params["kernel"].shape == variables["params"]["kernel"].shape
    assert updated_params["bias"].shape == variables["params"]["bias"].shape


def test_ridge_readout_matches_dense_param_names_and_forward_pass():
    inputs = jnp.array([[1.0, 2.0]])
    kernel = jnp.array([[3.0], [4.0]])
    bias = jnp.array([5.0])
    variables = {"params": {"kernel": kernel, "bias": bias}}

    ridge_predictions = RidgeReadout(features=1).apply(variables, inputs)
    dense_predictions = nn.Dense(features=1).apply(variables, inputs)

    assert jnp.allclose(ridge_predictions, dense_predictions)


def test_linen_ridge_readout_exports():
    from lumix.linen import RidgeReadout as exported_readout

    assert exported_readout is RidgeReadout
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_ridge.py -v
```

Expected: fail because `RidgeReadout` does not exist.

- [ ] **Step 3: Implement RidgeReadout**

Modify `src/lumix/linen/readout.py` by adding imports:

```python
import jax.numpy as jnp
from flax.linen.initializers import Initializer
```

Add this class after `LogitReadout`:

```python
class RidgeReadout(nn.Module):
    features: int
    use_bias: bool = True
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, values):
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (values.shape[-1], self.features),
            values.dtype,
        )
        outputs = jnp.matmul(values, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), values.dtype)
            outputs = outputs + bias
        return outputs
```

- [ ] **Step 4: Export RidgeReadout**

Modify `src/lumix/linen/__init__.py`:

```python
from lumix.linen.readout import IntensityReadout, LogitReadout, ProbabilityReadout, RidgeReadout
```

Add to `__all__`:

```python
"RidgeReadout",
```

- [ ] **Step 5: Run ridge tests**

Run:

```bash
uv run pytest tests/test_ridge.py -v
```

Expected: all ridge tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/lumix/linen/readout.py src/lumix/linen/__init__.py tests/test_ridge.py
git commit -m "feat: add flax ridge readout"
```

---

### Task 6: Run Full Verification and Update Top-Level Exports Deliberately

**Files:**
- Optional Modify: `src/lumix/__init__.py`

- [ ] **Step 1: Decide whether top-level exports are wanted**

Default decision: do not export `InformationEncoder`, `RidgeReadout`, or `solve_ridge` from `lumix.__init__` unless the project already treats every Linen layer as top-level public API.

If top-level exports are required, modify `src/lumix/__init__.py` by adding:

```python
from lumix.functional.ridge import solve_ridge
from lumix.linen.encoding import InformationEncoder
from lumix.linen.readout import RidgeReadout
```

and add these strings to `__all__`:

```python
"InformationEncoder",
"RidgeReadout",
"solve_ridge",
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_encoding.py tests/test_ridge.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run pytest
```

Expected: all tests pass. If the deleted non-library GAN test is still present in the working tree, remove it or keep it excluded according to the earlier cleanup decision before claiming success.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff -- src/lumix tests
```

Expected:
- New encoding functions are deterministic and param-free.
- Ridge solver returns plain `kernel` and optional `bias`.
- `RidgeReadout` has no ridge-specific training behavior.
- No pooling abstraction was added.
- No standardizer or trainable normalization was added.

- [ ] **Step 5: Commit final export decision if needed**

If `src/lumix/__init__.py` was changed, run:

```bash
git add src/lumix/__init__.py
git commit -m "feat: export ridge and encoding public api"
```

If `src/lumix/__init__.py` was not changed, do not make an empty commit.

---

## Self-Review

- Spec coverage:
  - Ridge closed-form fitting plus Optax compatibility is covered by Tasks 4 and 5.
  - `RidgeReadout` is included and remains Flax-compatible.
  - Information encoding supports phase, amplitude, and complex deterministic modes in Tasks 2 and 3.
  - Optional stateless normalization is included with `normalize`, `input_range`, and `clip`.
  - Generic MSE is included in Task 1.
  - Pooling is intentionally excluded; Flax `nn.avg_pool` is used directly.
- Placeholder scan:
  - No task relies on unspecified implementation details.
  - Every test and implementation step includes concrete code.
- Type consistency:
  - Ridge params use `kernel` and `bias` consistently across solver, module, and tests.
  - Encoding mode strings are consistently `phase`, `amplitude`, and `complex`.
  - Python commands use `uv run` per project rules.
