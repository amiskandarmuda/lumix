import jax
import jax.numpy as jnp
import numpy as np


def test_spm_response_matches_neuroptica_self_phase_modulation():
    from lumix.functional.spm import spm_response

    values = jnp.asarray([[1.0 + 2.0j, -0.5 + 0.25j]], dtype=jnp.complex64)
    gain = jnp.asarray(0.7, dtype=jnp.float32)

    outputs = spm_response(values, gain)
    expected = values * jnp.exp(-1j * gain * jnp.square(jnp.abs(values)))

    np.testing.assert_allclose(np.asarray(outputs), np.asarray(expected), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.abs(outputs)), np.asarray(jnp.abs(values)), rtol=1e-6, atol=1e-6)


def test_spm_nonlinearity_exposes_fixed_or_trainable_gain():
    from lumix.linen.spm import SPMNonlinearity

    values = jnp.asarray([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=jnp.complex64)
    fixed = SPMNonlinearity(gain=0.25, train_gain=False)
    trainable = SPMNonlinearity(gain=0.25, train_gain=True)

    fixed_variables = fixed.init(jax.random.key(0), values)
    trainable_variables = trainable.init(jax.random.key(1), values)

    assert "gain" in fixed_variables["params"]
    assert "gain" in trainable_variables["params"]
    np.testing.assert_allclose(float(fixed_variables["params"]["gain"]), 0.25, rtol=1e-6)
    np.testing.assert_allclose(float(trainable_variables["params"]["gain"]), 0.25, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(fixed.apply(fixed_variables, values)),
        np.asarray(trainable.apply(trainable_variables, values)),
        rtol=1e-6,
        atol=1e-6,
    )
