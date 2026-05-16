import jax
import jax.numpy as jnp

from lumix.functional.routing import routing_leakage, routing_mask
from lumix.linen.subunitary import SubUnitaryLinear
from lumix.linen.unitary import UnitaryLinear


def test_symmetric_routing_limit_allows_ports_within_distance():
    mask = routing_mask(output_features=6, input_features=6, routing_limit=2)

    assert mask[:, 3].tolist() == [False, True, True, True, True, True]
    assert mask[:, 0].tolist() == [True, True, True, False, False, False]


def test_asymmetric_routing_limit_uses_left_and_right_distances():
    mask = routing_mask(output_features=8, input_features=8, routing_limit=(2, 3))

    assert mask[:, 3].tolist() == [False, True, True, True, True, True, True, False]


def test_routing_leakage_cannot_be_reduced_by_global_attenuation():
    matrix = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=jnp.complex64,
    )

    leakage = routing_leakage(matrix, routing_limit=0)
    attenuated_leakage = routing_leakage(0.25 * matrix, routing_limit=0)

    assert jnp.allclose(leakage, attenuated_leakage)


def test_unitary_linear_sows_routing_metrics():
    layer = UnitaryLinear(width=6, routing_limit=2)
    values = (jnp.ones((2, 6)) + 1j * jnp.ones((2, 6))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(0), values)

    _, state = layer.apply(variables, values, mutable=["metrics"])

    assert "routing_leakage" in state["metrics"]
    assert "routing_penalty" not in state["metrics"]


def test_subunitary_linear_sows_fractional_routing_leakage():
    layer = SubUnitaryLinear(width=6, insertion_loss_db=(0.0, 1.5), routing_limit=2)
    values = (jnp.ones((2, 6)) + 1j * jnp.ones((2, 6))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(1), values)

    _, state = layer.apply(variables, values, mutable=["metrics"])

    assert "routing_leakage" in state["metrics"]
    assert "routing_penalty" not in state["metrics"]
