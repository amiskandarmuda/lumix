import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import pytest

from lumix.functional.ridge import solve_ridge
from lumix.linen import RidgeReadout


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


def test_solve_ridge_recovers_complex_kernel():
    inputs = jnp.array(
        [
            [1.0 + 0.0j],
            [0.0 + 1.0j],
        ],
        dtype=jnp.complex64,
    )
    kernel = jnp.array([[2.0 - 3.0j]], dtype=jnp.complex64)
    targets = inputs @ kernel

    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=False)

    assert params["kernel"].shape == (1, 1)
    assert jnp.allclose(params["kernel"], kernel, atol=1e-5)


def test_solve_ridge_rejects_negative_alpha():
    inputs = jnp.array([[1.0], [2.0]])
    targets = jnp.array([1.0, 2.0])

    with pytest.raises(ValueError) as error:
        solve_ridge(inputs, targets, alpha=-1e-3)

    assert error.value.args[0] == "alpha must be non-negative"


def test_solve_ridge_rejects_inputs_without_sample_feature_shape():
    with pytest.raises(ValueError) as error:
        solve_ridge(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))

    assert error.value.args[0] == "inputs must have shape (samples, features)"


def test_solve_ridge_rejects_targets_without_sample_output_shape():
    inputs = jnp.array([[1.0], [2.0]])
    targets = jnp.ones((2, 1, 1))

    with pytest.raises(ValueError) as error:
        solve_ridge(inputs, targets)

    assert error.value.args[0] == "targets must have shape (samples,) or (samples, outputs)"


def test_solve_ridge_rejects_mismatched_sample_counts():
    inputs = jnp.array([[1.0], [2.0]])
    targets = jnp.array([1.0])

    with pytest.raises(ValueError) as error:
        solve_ridge(inputs, targets)

    assert error.value.args[0] == "inputs and targets must have the same sample count"


def test_functional_ridge_exports():
    from lumix.functional import solve_ridge as exported_solve_ridge

    assert exported_solve_ridge is solve_ridge


def test_ridge_readout_params_match_dense_shapes_and_names():
    values = jnp.ones((2, 4))

    variables = RidgeReadout(features=3).init(jax.random.key(0), values)

    assert set(variables["params"].keys()) == {"kernel", "bias"}
    assert variables["params"]["kernel"].shape == (4, 3)
    assert variables["params"]["bias"].shape == (3,)


def test_ridge_readout_default_param_dtype_matches_dense_for_float16_input():
    values = jnp.ones((2, 4), dtype=jnp.float16)

    ridge_params = RidgeReadout(features=3).init(jax.random.key(3), values)["params"]
    dense_params = nn.Dense(features=3).init(jax.random.key(3), values)["params"]

    assert ridge_params["kernel"].dtype == dense_params["kernel"].dtype == jnp.float32
    assert ridge_params["bias"].dtype == dense_params["bias"].dtype == jnp.float32


def test_ridge_readout_integer_input_initializes_and_applies_with_float32_params():
    values = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    readout = RidgeReadout(features=2)

    variables = readout.init(jax.random.key(4), values)
    outputs = readout.apply(variables, values)

    assert variables["params"]["kernel"].dtype == jnp.float32
    assert variables["params"]["bias"].dtype == jnp.float32
    assert outputs.shape == (2, 2)
    assert outputs.dtype == jnp.float32


def test_ridge_readout_complex_input_initializes_float32_params_and_computes():
    values = jnp.array([[1.0 + 2.0j, 3.0 - 1.0j]], dtype=jnp.complex64)
    readout = RidgeReadout(features=2)

    variables = readout.init(jax.random.key(5), values)
    outputs = readout.apply(variables, values)

    assert variables["params"]["kernel"].dtype == jnp.float32
    assert variables["params"]["bias"].dtype == jnp.float32
    assert outputs.shape == (1, 2)
    assert outputs.dtype == jnp.complex64


def test_ridge_readout_can_omit_bias_param():
    values = jnp.ones((2, 4))

    variables = RidgeReadout(features=3, use_bias=False).init(jax.random.key(1), values)

    assert set(variables["params"].keys()) == {"kernel"}
    assert variables["params"]["kernel"].shape == (4, 3)


def test_ridge_readout_matches_dense_forward_with_same_params():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    kernel = jnp.array([[2.0, -1.0, 0.5], [1.5, 3.0, -2.0]])
    bias = jnp.array([0.25, -0.5, 1.0])
    variables = {"params": {"kernel": kernel, "bias": bias}}

    ridge_outputs = RidgeReadout(features=3).apply(variables, values)
    dense_outputs = nn.Dense(features=3).apply(variables, values)

    assert jnp.allclose(ridge_outputs, dense_outputs)


def test_ridge_readout_applies_complex_solve_ridge_params():
    inputs = jnp.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
            [1.0 - 1.0j, 2.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    kernel = jnp.array([[2.0 - 1.0j], [0.5 + 3.0j]], dtype=jnp.complex64)
    bias = jnp.array([1.0 + 0.25j], dtype=jnp.complex64)
    targets = inputs @ kernel + bias
    params = solve_ridge(inputs, targets, alpha=0.0, use_bias=True)

    outputs = RidgeReadout(features=1).apply({"params": params}, inputs)

    assert jnp.allclose(outputs, targets, atol=1e-5)


def test_ridge_readout_params_are_compatible_with_optax_updates():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    targets = jnp.array([[1.0], [2.0]])
    readout = RidgeReadout(features=1)
    params = readout.init(jax.random.key(2), values)["params"]
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state = optimizer.init(params)

    def loss_fn(current_params):
        outputs = readout.apply({"params": current_params}, values)
        return jnp.mean(jnp.square(outputs - targets))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    assert jnp.isfinite(loss)
    assert set(updated_params.keys()) == {"kernel", "bias"}
    assert updated_params["kernel"].shape == params["kernel"].shape
    assert updated_params["bias"].shape == params["bias"].shape


def test_linen_ridge_readout_export():
    from lumix.linen import RidgeReadout as exported_ridge_readout

    assert exported_ridge_readout is RidgeReadout
