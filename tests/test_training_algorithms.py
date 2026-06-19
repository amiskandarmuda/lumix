import jax
import jax.numpy as jnp
import optax
import pytest

from lumix.functional.clements import clements_pair, init_clements
from lumix.functional.clements_convert import clements_transfer_matrix, identity_clements_params
from lumix.functional.unitary import unitary_linear
from lumix.linen.unitary import UnitaryLinear
from lumix.training.directional import bp_dd_step, directional_derivative_gradient
from lumix.training.forward_forward import (
    ff_ad_step,
    ff_dd_step,
    ffzero_margin_loss,
    ffzero_onn_simplex_loss,
)
from lumix.training.insitu import (
    clements_square_law_logits,
    insitu_classification_step,
    insitu_mse_gradients,
)
from lumix.training.physical import unitary_linear_to_clements_params


def _three_class_points():
    x = jnp.array(
        [
            [2.0, 0.0],
            [2.2, 0.2],
            [0.0, 2.0],
            [0.2, 2.2],
            [-2.0, -2.0],
            [-2.2, -1.8],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 0, 1, 1, 2, 2])
    return x, jnp.eye(3, dtype=jnp.float32)[labels]


def _linear_logits(params, x):
    return x @ params["kernel"] + params["bias"]


def _linear_cross_entropy(params, x, y):
    return optax.softmax_cross_entropy(_linear_logits(params, x), y).mean()


def _linear_accuracy(params, x, y):
    return jnp.mean(jnp.argmax(_linear_logits(params, x), axis=-1) == jnp.argmax(y, axis=-1))


def _tree_l2(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree)))


def test_ffzero_margin_loss_matches_reference_formula():
    activations = jnp.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=jnp.float32)
    references = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1, 0])

    normalized = activations / jnp.linalg.norm(activations, axis=1, keepdims=True)
    similarities = normalized @ references.T
    true_sim = similarities[jnp.arange(labels.shape[0]), labels]
    expected = jnp.maximum(0.3 + similarities - true_sim[:, None], 0.0).sum()

    assert jnp.allclose(ffzero_margin_loss(activations, labels, references), expected)


def test_ffzero_onn_simplex_loss_matches_reference_formula():
    activations = jnp.array([[3.0, 4.0], [1.0, -1.0]], dtype=jnp.float32)
    references = jnp.array([[0.6, 0.8], [1.0, 0.0]], dtype=jnp.float32)
    labels = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    normalized = activations / jnp.linalg.norm(activations, axis=1, keepdims=True)
    similarities = normalized @ references.T
    true_sim = jnp.sum(similarities * labels, axis=1)
    expected = jnp.mean(1.0 - true_sim)

    assert jnp.allclose(ffzero_onn_simplex_loss(activations, labels, references), expected)


def test_directional_derivative_with_basis_directions_matches_jax_grad_update():
    params = {
        "kernel": jnp.array([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float32),
        "bias": jnp.array([0.05, -0.2], dtype=jnp.float32),
    }

    def loss_fn(current):
        flat = jnp.concatenate([current["kernel"].reshape(-1), current["bias"]])
        return jnp.sum(flat * flat)

    exact_grad = jax.grad(loss_fn)(params)
    result = directional_derivative_gradient(
        params,
        loss_fn,
        key=jax.random.key(0),
        eps=1e-3,
        num_directions=6,
        directions=jnp.eye(6, dtype=jnp.float32),
    )

    assert jnp.allclose(result.gradient["kernel"], exact_grad["kernel"], atol=2e-4)
    assert jnp.allclose(result.gradient["bias"], exact_grad["bias"], atol=2e-4)


def test_bp_dd_trains_deterministic_three_class_task():
    x, y = _three_class_points()
    params = {
        "kernel": jnp.zeros((2, 3), dtype=jnp.float32),
        "bias": jnp.zeros((3,), dtype=jnp.float32),
    }
    key = jax.random.key(1)
    initial_loss = _linear_cross_entropy(params, x, y)

    for _ in range(80):
        result = bp_dd_step(
            params,
            lambda current: _linear_cross_entropy(current, x, y),
            key=key,
            eps=1e-3,
            learning_rate=0.5,
            num_directions=96,
        )
        params = result.params
        key = result.key

    assert float(_linear_cross_entropy(params, x, y)) < float(initial_loss) * 0.05
    assert float(_linear_accuracy(params, x, y)) == 1.0


def test_ff_ad_and_ff_dd_train_local_simplex_layer():
    x, y = _three_class_points()
    references = jnp.eye(3, dtype=jnp.float32)
    params = {
        "kernel": jnp.array(
            [
                [0.2, -0.1, 0.05],
                [-0.2, 0.15, 0.1],
            ],
            dtype=jnp.float32,
        ),
        "bias": jnp.zeros((3,), dtype=jnp.float32),
    }

    def apply_fn(current, batch):
        return batch @ current["kernel"] + current["bias"]

    initial_loss = ffzero_onn_simplex_loss(apply_fn(params, x), y, references)
    ad_params = params
    dd_params = params
    key = jax.random.key(2)

    for _ in range(60):
        ad_params, _ = ff_ad_step(
            ad_params,
            apply_fn,
            x,
            y,
            references,
            learning_rate=0.2,
            loss_mode="ffzero_onn_simplex",
        )
        dd_result = ff_dd_step(
            dd_params,
            apply_fn,
            x,
            y,
            references,
            key=key,
            eps=1e-3,
            learning_rate=0.2,
            num_directions=96,
            loss_mode="ffzero_onn_simplex",
        )
        dd_params = dd_result.params
        key = dd_result.key

    ad_loss = ffzero_onn_simplex_loss(apply_fn(ad_params, x), y, references)
    dd_loss = ffzero_onn_simplex_loss(apply_fn(dd_params, x), y, references)
    assert float(ad_loss) < float(initial_loss) * 0.25
    assert float(dd_loss) < float(initial_loss) * 0.25
    assert float(_linear_accuracy(ad_params, x, y)) == 1.0
    assert float(_linear_accuracy(dd_params, x, y)) == 1.0


def test_unitary_linear_physical_mapping_matches_clements_transfer_and_outputs():
    width = 4
    values = jnp.array(
        [[1.0 + 0.5j, -0.25 + 0.1j, 0.5 - 0.5j, 0.75 + 0.0j]],
        dtype=jnp.complex64,
    )
    unitary = UnitaryLinear(width=width, init_scale=1e-2)
    unitary_params = unitary.init(jax.random.key(3), values)["params"]

    mapped = unitary_linear_to_clements_params(unitary_params, width=width)
    target = mapped.target_matrix
    transfer = clements_transfer_matrix(mapped.clements_params, width)

    assert float(mapped.relative_frobenius_error) < 1e-5
    assert jnp.allclose(transfer, target, atol=1e-5)
    assert jnp.allclose(
        clements_pair(values, **mapped.clements_params),
        unitary_linear(values, target),
        atol=1e-5,
    )


def test_unitary_linear_physical_mapping_rejects_rectangular_isometry():
    params = {
        "left_re": jnp.zeros((3, 3), dtype=jnp.float32),
        "left_im": jnp.zeros((3, 3), dtype=jnp.float32),
        "right_re": jnp.zeros((2, 2), dtype=jnp.float32),
        "right_im": jnp.zeros((2, 2), dtype=jnp.float32),
    }

    with pytest.raises(ValueError, match="physical Clements mapping requires square UnitaryLinear params"):
        unitary_linear_to_clements_params(params, width=3)


def test_insitu_mse_gradients_match_jax_grad_for_clements_phases():
    params = identity_clements_params(2, depth=2)
    inputs = jnp.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=jnp.complex64)
    targets = jnp.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=jnp.complex64)

    def loss_fn(current):
        outputs = clements_pair(inputs, current["theta"], current["phi"], current["gamma"])
        return jnp.mean(jnp.square(jnp.abs(outputs - targets)))

    expected = jax.grad(loss_fn)(params)
    actual = insitu_mse_gradients(params, inputs, targets)

    assert jnp.allclose(actual["theta"], expected["theta"], atol=1e-6)
    assert jnp.allclose(actual["phi"], expected["phi"], atol=1e-6)
    assert jnp.allclose(actual["gamma"], expected["gamma"], atol=1e-6)


def test_insitu_gradients_do_not_call_jax_ad(monkeypatch):
    params = identity_clements_params(2, depth=2)
    params = {**params, "theta": params["theta"].at[0, 0].set(jnp.pi - 0.35)}
    inputs = jnp.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=jnp.complex64)
    targets = jnp.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=jnp.complex64)
    labels = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    def fail_ad(*args, **kwargs):
        raise AssertionError("production in-situ gradients must not use JAX AD")

    monkeypatch.setattr(jax, "grad", fail_ad)
    monkeypatch.setattr(jax, "value_and_grad", fail_ad)

    gradients = insitu_mse_gradients(params, inputs, targets)
    result = insitu_classification_step(params, inputs, labels, learning_rate=1.0)

    assert float(_tree_l2(gradients)) > 1e-2
    assert float(_tree_l2(result.gradients)) > 1e-2


@pytest.mark.parametrize("width", [2, 3, 4])
def test_insitu_field_interference_gradients_match_jax_grad_for_multiple_widths(width):
    depth = width
    params = init_clements(jax.random.key(width), width, depth)
    batch = 3
    real = jax.random.normal(jax.random.key(10 + width), (batch, width), dtype=jnp.float32)
    imag = jax.random.normal(jax.random.key(20 + width), (batch, width), dtype=jnp.float32)
    inputs = (real + 1j * imag).astype(jnp.complex64)
    target_real = jax.random.normal(jax.random.key(30 + width), (batch, width), dtype=jnp.float32)
    target_imag = jax.random.normal(jax.random.key(40 + width), (batch, width), dtype=jnp.float32)
    targets = (target_real + 1j * target_imag).astype(jnp.complex64)

    def loss_fn(current):
        outputs = clements_pair(inputs, current["theta"], current["phi"], current["gamma"])
        return jnp.mean(jnp.square(jnp.abs(outputs - targets)))

    expected = jax.grad(loss_fn)(params)
    actual = insitu_mse_gradients(params, inputs, targets)

    assert float(_tree_l2(actual)) > 1e-2
    assert jnp.allclose(actual["theta"], expected["theta"], atol=5e-5)
    assert jnp.allclose(actual["phi"], expected["phi"], atol=5e-5)
    assert jnp.allclose(actual["gamma"], expected["gamma"], atol=5e-5)


def test_insitu_field_interference_gradients_match_central_finite_difference():
    width = 3
    params = init_clements(jax.random.key(12), width, depth=width)
    inputs = jnp.array(
        [
            [0.8 + 0.2j, -0.3 + 0.5j, 0.6 - 0.1j],
            [-0.4 + 0.1j, 0.7 - 0.2j, 0.2 + 0.3j],
        ],
        dtype=jnp.complex64,
    )
    targets = jnp.array(
        [
            [-0.1 + 0.4j, 0.6 + 0.2j, 0.3 - 0.7j],
            [0.5 - 0.3j, -0.2 + 0.1j, 0.4 + 0.6j],
        ],
        dtype=jnp.complex64,
    )

    def loss_fn(current):
        outputs = clements_pair(inputs, current["theta"], current["phi"], current["gamma"])
        return jnp.mean(jnp.square(jnp.abs(outputs - targets)))

    gradients = insitu_mse_gradients(params, inputs, targets)
    eps = 1e-3
    checks = [("theta", (0, 0)), ("phi", (0, 0)), ("gamma", (0, 1))]
    for name, index in checks:
        plus = {**params, name: params[name].at[index].add(eps)}
        minus = {**params, name: params[name].at[index].add(-eps)}
        finite_difference = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)
        assert jnp.allclose(gradients[name][index], finite_difference, atol=2e-3)


def test_insitu_training_reduces_physical_clements_classification_loss():
    params = identity_clements_params(2, depth=2)
    params = {
        **params,
        # Exact identity is a stationary point for this square-law task.
        "theta": params["theta"].at[0, 0].set(jnp.pi - 0.35),
    }
    inputs = jnp.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=jnp.complex64)
    labels = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    def loss(current):
        return optax.softmax_cross_entropy(clements_square_law_logits(current, inputs), labels).mean()

    initial = loss(params)
    first_step = insitu_classification_step(params, inputs, labels, learning_rate=1.0)
    first_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(grad)) for grad in jax.tree_util.tree_leaves(first_step.gradients))
    )
    assert float(first_grad_norm) > 1e-2
    assert float(loss(first_step.params)) < float(initial)

    params = first_step.params
    for _ in range(79):
        result = insitu_classification_step(params, inputs, labels, learning_rate=1.0)
        params = result.params

    final_logits = clements_square_law_logits(params, inputs)
    final_loss = loss(params)
    final_acc = jnp.mean(jnp.argmax(final_logits, axis=1) == jnp.argmax(labels, axis=1))

    assert float(final_loss) < float(initial) * 0.35
    assert float(final_acc) == 1.0
