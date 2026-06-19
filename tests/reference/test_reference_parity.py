import importlib
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from lumix.functional.clements_convert import clements_fit_metrics, decompose_unitary_to_clements
from lumix.training.forward_forward import ffzero_margin_loss, ffzero_onn_simplex_loss


def _add_reference_path(env_name: str):
    path = os.environ.get(env_name)
    if not path:
        pytest.skip(f"{env_name} is not set")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def test_ffzero_margin_formula_matches_reference_torch_expression():
    _add_reference_path("FFZERO_PATH")
    torch = pytest.importorskip("torch")

    activations = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=torch.float32)
    references = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    z = activations / activations.norm(p=2, dim=1, keepdim=True)
    sim = torch.matmul(z, references.T.float())
    true_sim = sim[torch.arange(len(labels)), labels]
    expected = torch.relu(0.3 + sim - true_sim.unsqueeze(1)).sum()

    actual = ffzero_margin_loss(
        jnp.asarray(activations.numpy()),
        jnp.asarray(labels.numpy()),
        jnp.asarray(references.numpy()),
    )

    assert np.allclose(np.asarray(actual), expected.detach().numpy())


def test_ffzero_onn_simplex_formula_matches_reference_tensorflow_expression():
    _add_reference_path("FFZERO_PATH")
    tf = pytest.importorskip("tensorflow")

    activations = tf.constant([[3.0, 4.0], [1.0, -1.0]], dtype=tf.float32)
    references = tf.constant([[0.6, 0.8], [1.0, 0.0]], dtype=tf.float32)
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    out = tf.math.l2_normalize(activations, axis=1)
    cos_sim = tf.matmul(out, references, transpose_b=True)
    true_sim = tf.reduce_sum(cos_sim * labels, axis=1)
    expected = tf.reduce_mean(1.0 - true_sim)

    actual = ffzero_onn_simplex_loss(
        jnp.asarray(activations.numpy()),
        jnp.asarray(labels.numpy()),
        jnp.asarray(references.numpy()),
    )

    assert np.allclose(np.asarray(actual), expected.numpy())


def test_neurophox_clements_decomposition_convention_matches_square_unitary():
    _add_reference_path("NEUROPHOX_PATH")
    pytest.importorskip("neurophox")

    target = jnp.asarray(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    params = decompose_unitary_to_clements(target)
    metrics = clements_fit_metrics(target, params, depth=2)

    assert float(metrics["relative_frobenius_error"]) < 1e-5


def test_neuroptica_import_available_for_external_insitu_parity():
    _add_reference_path("NEUROPTICA_PATH")
    module = importlib.import_module("neuroptica.component_layers")

    assert hasattr(module, "OpticalMesh")
