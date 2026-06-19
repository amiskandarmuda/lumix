import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from lumix.functional.readout import intensity
from lumix.functional.subunitary import (
    insertion_loss_amplitude,
    insertion_loss_bounds,
    singular_values_in_bounds,
    subunitary_matrix,
)
from lumix.functional.unitary import combine_complex_parts, isometric_matrix
from lumix.linen.readout import IntensityReadout, LogitReadout, ProbabilityReadout
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
    params = variables["params"]
    matrix = isometric_matrix(
        combine_complex_parts(params["left_re"], params["left_im"]),
        combine_complex_parts(params["right_re"], params["right_im"]),
        8,
        8,
    )
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    error = jnp.linalg.norm(jnp.conj(matrix.T) @ matrix - identity)
    assert float(error) < 1e-4


def test_unitary_linear_supports_rectangular_maps():
    layer = UnitaryLinear(width=8, out_features=6)
    values = (jnp.ones((4, 8)) + 1j * jnp.ones((4, 8))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(12), values)
    outputs = layer.apply(variables, values)
    assert outputs.shape == (4, 6)


def test_subunitary_matrix_has_bounded_singular_values():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(0.5, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(3), values)
    params = variables["params"]
    matrix = subunitary_matrix(
        combine_complex_parts(params["left_re"], params["left_im"]),
        combine_complex_parts(params["right_re"], params["right_im"]),
        params["singular_raw"],
        *insertion_loss_bounds((0.5, 2.0)),
        8,
        8,
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    singular_min, singular_max = insertion_loss_bounds((0.5, 2.0))
    assert float(jnp.max(singular)) <= float(singular_max) + 1e-5
    assert float(jnp.min(singular)) >= float(singular_min) - 1e-5


def test_subunitary_fixed_loss_sets_exact_singular_values():
    model = SubUnitaryLinear(width=8, insertion_loss_db=1.5)
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(4), values)
    params = variables["params"]
    singular = singular_values_in_bounds(params["singular_raw"], *insertion_loss_bounds(1.5))
    target = insertion_loss_amplitude(1.5)
    assert float(jnp.max(jnp.abs(singular - target))) < 1e-5


def test_subunitary_supports_lower_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(1.0, None))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(5), values)
    params = variables["params"]
    singular = singular_values_in_bounds(params["singular_raw"], *insertion_loss_bounds((1.0, None)))
    _, singular_max = insertion_loss_bounds((1.0, None))
    assert float(jnp.max(singular)) <= float(singular_max) + 1e-5


def test_subunitary_supports_upper_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(None, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(6), values)
    params = variables["params"]
    singular = singular_values_in_bounds(params["singular_raw"], *insertion_loss_bounds((None, 2.0)))
    singular_min, _ = insertion_loss_bounds((None, 2.0))
    assert float(jnp.min(singular)) >= float(singular_min) - 1e-5


def test_subunitary_loss_bounds_are_not_trainable_params():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(0.5, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)

    variables = model.init(jax.random.key(16), values)
    params = variables["params"]

    assert "singular_raw" in params
    assert "singular_min" not in params
    assert "singular_max" not in params


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
    matrix = subunitary_matrix(
        combine_complex_parts(params["left_re"], params["left_im"]),
        combine_complex_parts(params["right_re"], params["right_im"]),
        params["singular_raw"],
        *insertion_loss_bounds((0.5, 2.0)),
        10,
        16,
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)

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


def test_intensity_readout_without_projection_matches_raw_intensity():
    values = (jnp.ones((4, 16)) + 2j * jnp.ones((4, 16))).astype(jnp.complex64)
    readout = IntensityReadout()

    variables = readout.init(jax.random.key(14), values)
    outputs = readout.apply(variables, values)

    assert "params" not in variables
    assert outputs.shape == (4, 16)
    assert jnp.allclose(outputs, intensity(values))
    assert outputs.dtype == jnp.float32


def test_intensity_readout_projects_to_requested_width():
    values = (jnp.ones((2, 256)) + 1j * jnp.ones((2, 256))).astype(jnp.complex64)
    readout = IntensityReadout(out_features=784)

    variables = readout.init(jax.random.key(15), values)
    outputs = readout.apply(variables, values)

    assert variables["params"]["Dense_0"]["kernel"].shape == (256, 784)
    assert outputs.shape == (2, 784)


def test_intensity_readout_sigmoid_constrains_output_range():
    values = (jnp.ones((2, 32)) + 1j * jnp.ones((2, 32))).astype(jnp.complex64)
    readout = IntensityReadout(out_features=64, activation="sigmoid")

    variables = readout.init(jax.random.key(16), values)
    outputs = readout.apply(variables, values)

    assert outputs.shape == (2, 64)
    assert float(jnp.min(outputs)) >= 0.0
    assert float(jnp.max(outputs)) <= 1.0


def test_intensity_readout_softmax_normalizes_last_axis():
    values = (jnp.ones((2, 32)) + 1j * jnp.ones((2, 32))).astype(jnp.complex64)
    readout = IntensityReadout(out_features=16, activation="softmax")

    variables = readout.init(jax.random.key(17), values)
    outputs = readout.apply(variables, values)

    assert outputs.shape == (2, 16)
    assert jnp.allclose(jnp.sum(outputs, axis=-1), jnp.ones((2,), dtype=outputs.dtype), atol=1e-6)


def test_intensity_readout_reshapes_output():
    values = (jnp.ones((2, 256)) + 1j * jnp.ones((2, 256))).astype(jnp.complex64)
    readout = IntensityReadout(out_features=784, activation="sigmoid", output_shape=(28, 28))

    variables = readout.init(jax.random.key(18), values)
    outputs = readout.apply(variables, values)

    assert outputs.shape == (2, 28, 28)
    assert float(jnp.min(outputs)) >= 0.0
    assert float(jnp.max(outputs)) <= 1.0


def test_intensity_readout_rejects_invalid_activation():
    values = (jnp.ones((1, 8)) + 1j * jnp.ones((1, 8))).astype(jnp.complex64)
    readout = IntensityReadout(activation="tanh")

    with pytest.raises(ValueError, match="activation must be one of None, 'sigmoid', or 'softmax'"):
        readout.init(jax.random.key(19), values)


def test_intensity_readout_rejects_mismatched_output_shape():
    values = (jnp.ones((1, 16)) + 1j * jnp.ones((1, 16))).astype(jnp.complex64)
    readout = IntensityReadout(out_features=20, output_shape=(3, 7))

    with pytest.raises(ValueError, match="output_shape product must match the effective output width"):
        readout.init(jax.random.key(20), values)


def test_optical_stack_can_end_with_intensity_readout():
    class GeneratorTail(nn.Module):
        @nn.compact
        def __call__(self, values):
            values = UnitaryLinear(width=256)(values)
            values = SubUnitaryLinear(width=256, insertion_loss_db=(0.5, 2.0))(values)
            return IntensityReadout(out_features=784, activation="sigmoid", output_shape=(28, 28))(values)

    model = GeneratorTail()
    values = (jnp.ones((2, 256)) + 1j * jnp.ones((2, 256))).astype(jnp.complex64)

    variables = model.init(jax.random.key(21), values)
    outputs = model.apply(variables, values)

    assert outputs.shape == (2, 28, 28)
    assert float(jnp.min(outputs)) >= 0.0
    assert float(jnp.max(outputs)) <= 1.0
