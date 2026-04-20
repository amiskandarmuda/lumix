import jax
import jax.numpy as jnp
from flax import linen as nn

from lumix.functional.subunitary import resolve_insertion_loss, subunitary_matrix
from lumix.functional.unitary import unitary_matrix
from lumix.linen.readout import PowerReadout
from lumix.linen.subunitary import SubUnitaryLinear
from lumix.linen.unitary import UnitaryLinear


class UnitaryNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = UnitaryLinear(width=16)(values)
        return PowerReadout(classes=self.classes)(values)


class SubUnitaryNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        values = SubUnitaryLinear(width=16)(values)
        return PowerReadout(classes=self.classes)(values)


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


def test_unitary_matrix_is_unitary():
    model = UnitaryLinear(width=8)
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(2), values)
    raw = variables["params"]["raw"]
    matrix = unitary_matrix(raw)
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    error = jnp.linalg.norm(jnp.conj(matrix.T) @ matrix - identity)
    assert float(error) < 1e-4


def test_subunitary_matrix_has_bounded_singular_values():
    model = SubUnitaryLinear(width=8)
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(3), values)
    matrix = subunitary_matrix(
        variables["params"]["raw"],
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    assert float(jnp.max(singular)) <= 1.0 + 1e-5


def test_subunitary_default_initialization_starts_near_zero_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(0.0, 3.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(4), values)
    loss_db = resolve_insertion_loss(variables["params"]["loss_raw"], (0.0, 3.0))
    assert 0.0 <= float(loss_db) < 0.1


def test_subunitary_supports_lower_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(1.0, None))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(5), values)
    loss_db = resolve_insertion_loss(variables["params"]["loss_raw"], (1.0, None))
    assert float(loss_db) >= 1.0


def test_subunitary_supports_upper_bounded_loss():
    model = SubUnitaryLinear(width=8, insertion_loss_db=(None, 2.0))
    values = (jnp.ones((2, 8)) + 1j * jnp.ones((2, 8))).astype(jnp.complex64)
    variables = model.init(jax.random.key(6), values)
    loss_db = resolve_insertion_loss(variables["params"]["loss_raw"], (None, 2.0))
    assert 0.0 <= float(loss_db) <= 2.0


def test_dense_layers_compose_with_readout():
    class MixedNet(nn.Module):
        classes: int = 10

        @nn.compact
        def __call__(self, values):
            values = UnitaryLinear(width=16)(values)
            values = SubUnitaryLinear(width=16, insertion_loss_db=(0.5, 2.0))(values)
            return PowerReadout(classes=self.classes)(values)

    model = MixedNet()
    values = (jnp.ones((4, 16)) + 1j * jnp.ones((4, 16))).astype(jnp.complex64)
    variables = model.init(jax.random.key(7), values)
    probs = model.apply(variables, values)
    assert probs.shape == (4, 10)
