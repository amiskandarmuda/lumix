import jax
import jax.numpy as jnp

from lumix.linen.clements import ClementsLinear


def test_clements_linear_sows_routing_metrics_for_local_mesh():
    layer = ClementsLinear(width=6, depth=2, routing_limit=2)
    values = (jnp.ones((2, 6)) + 1j * jnp.ones((2, 6))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(0), values)

    _, state = layer.apply(variables, values, mutable=["metrics"])

    assert "routing_leakage" in state["metrics"]
    assert "routing_penalty" not in state["metrics"]
