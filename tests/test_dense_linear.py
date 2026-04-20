import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.functional.subunitary import (
    insertion_loss_amplitude,
    insertion_loss_bounds,
    project_subunitary_to_bounds,
)
from lumix.functional.unitary import unitary_matrix
from lumix.linen.readout import LogitReadout, ProbabilityReadout
from lumix.linen.subunitary import SubUnitaryLinear
from lumix.linen.unitary import UnitaryLinear
from lumix.train import create_state, train_step_logits


class UnitaryNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = UnitaryLinear(width=16)(values)
        return ProbabilityReadout(classes=self.classes)(values)


class SubUnitaryNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = SubUnitaryLinear(width=16)(values)
        return ProbabilityReadout(classes=self.classes)(values)


def test_unitary_linear_forward_shape():
    model = UnitaryNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(0), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)


def test_subunitary_linear_forward_shape():
    model = SubUnitaryNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(1), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)


def test_subunitary_linear_supports_rectangular_maps():
    layer = SubUnitaryLinear(width=8, out_features=6, insertion_loss_db=(0.0, 3.0))
    values = (jnp.ones((4, 8)) + 1j * jnp.ones((4, 8))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(11), values)
    outputs = layer.apply(variables, values)
    assert outputs.shape == (4, 6)


def test_unitary_matrix_is_unitary():
    model = UnitaryLinear(width=8)
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(2), values)
    raw = variables["params"]["raw_re"] + 1j * variables["params"]["raw_im"]
    matrix = unitary_matrix(raw)
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    error = jnp.linalg.norm(jnp.conj(matrix.T) @ matrix - identity)
    assert float(error) < 1e-4


def test_subunitary_matrix_has_bounded_singular_values():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(0.5, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(3), values)
    raw = variables["params"]["raw_re"] + 1j * variables["params"]["raw_im"]
    matrix = project_subunitary_to_bounds(
        raw,
        variables["params"]["singular_min"],
        variables["params"]["singular_max"],
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    singular_min, singular_max = insertion_loss_bounds((0.5, 2.0))
    assert float(jnp.max(singular)) <= float(singular_max) + 1e-5
    assert float(jnp.min(singular)) >= float(singular_min) - 1e-5


def test_subunitary_fixed_loss_sets_exact_singular_values():
    model = SubUnitaryLinear(width=8, insertion_loss_db=1.5)
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(4), values)
    raw = variables["params"]["raw_re"] + 1j * variables["params"]["raw_im"]
    matrix = project_subunitary_to_bounds(
        raw,
        variables["params"]["singular_min"],
        variables["params"]["singular_max"],
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    target = insertion_loss_amplitude(1.5)
    assert float(jnp.max(jnp.abs(singular - target))) < 1e-5


def test_subunitary_supports_lower_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(1.0, None))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(5), values)
    raw = variables["params"]["raw_re"] + 1j * variables["params"]["raw_im"]
    matrix = project_subunitary_to_bounds(
        raw,
        variables["params"]["singular_min"],
        variables["params"]["singular_max"],
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    _, singular_max = insertion_loss_bounds((1.0, None))
    assert float(jnp.max(singular)) <= float(singular_max) + 1e-5


def test_subunitary_supports_upper_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(None, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(6), values)
    raw = variables["params"]["raw_re"] + 1j * variables["params"]["raw_im"]
    matrix = project_subunitary_to_bounds(
        raw,
        variables["params"]["singular_min"],
        variables["params"]["singular_max"],
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    singular_min, _ = insertion_loss_bounds((None, 2.0))
    assert float(jnp.min(singular)) >= float(singular_min) - 1e-5


def test_dense_layers_compose_with_readout():
    class MixedNet(nn.Module):
        classes: int = 10

        @nn.compact
        def __call__(self, values):
            values = UnitaryLinear(width=16)(values)
            values = SubUnitaryLinear(width=16, insertion_loss_db=(0.5, 2.0))(values)
            return ProbabilityReadout(classes=self.classes)(values)

    model = MixedNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(7), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)


def test_subunitary_train_step_preserves_passive_bounds():
    class LogitNet(nn.Module):
        classes: int = 10

        @nn.compact
        def __call__(self, values):
            values = SubUnitaryLinear(width=16, out_features=self.classes, insertion_loss_db=(0.5, 2.0))(values)
            return jnp.real(jnp.conj(values) * values)

    values = (jnp.ones((8, 16)) + 1j * jnp.ones((8, 16))).astype(jnp.complex64)
    labels = jnp.eye(10, dtype=jnp.float32)[jnp.arange(8) % 10]
    state = create_state(LogitNet(), jax.random.key(8), values, learning_rate=5e-3)

    next_state, _, _ = train_step_logits(state, values, labels)
    params = next_state.params["SubUnitaryLinear_0"]
    raw = params["raw_re"] + 1j * params["raw_im"]
    singular = jnp.linalg.svd(raw, compute_uv=False)

    singular_min, singular_max = insertion_loss_bounds((0.5, 2.0))
    assert float(jnp.max(singular)) <= float(singular_max) + 1e-5
    assert float(jnp.min(singular)) >= float(singular_min) - 1e-5


def test_probability_and_logit_readout_shapes():
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    prob_readout = ProbabilityReadout(classes=10)
    logit_readout = LogitReadout(classes=10)

    prob_variables = prob_readout.init(jax.random.key(9), values)
    logit_variables = logit_readout.init(jax.random.key(10), values)

    probs = prob_readout.apply(prob_variables, values)
    logits = logit_readout.apply(logit_variables, values)

    assert probs.shape == (4, 10)
    assert logits.shape == (4, 10)
