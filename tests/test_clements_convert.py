import jax
import jax.numpy as jnp

from lumix.functional.clements_convert import (
    clements_fit_metrics,
    decompose_unitary_to_clements,
    identity_clements_params,
    unitary_linear_matrix_from_params,
)
from lumix.linen.unitary import UnitaryLinear


def test_analytic_clements_decomposition_reconstructs_unitary_linear_matrix():
    width = 8
    layer = UnitaryLinear(width=width, init_scale=1e-2)
    params = layer.init(jax.random.PRNGKey(0), jnp.ones((1, width), dtype=jnp.complex64))["params"]
    target = unitary_linear_matrix_from_params(params, width)
    clements_params = decompose_unitary_to_clements(target)
    metrics = clements_fit_metrics(target, clements_params, depth=width)
    assert float(metrics["relative_frobenius_error"]) < 1e-5


def test_identity_clements_params_shapes():
    params = identity_clements_params(width=8)
    assert params["theta"].shape == (8, 4)
    assert params["phi"].shape == (8, 4)
    assert params["gamma"].shape == (1, 8)
