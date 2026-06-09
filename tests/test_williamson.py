import jax.numpy as jnp
import numpy as np

from lumix.functional.williamson import electro_optic_phase_parameters, williamson_response


def test_electro_optic_phase_parameters_match_neuroptica_mapping():
    gain, bias = electro_optic_phase_parameters(
        tap=0.2,
        responsivity=0.9,
        area=1.5,
        v_pi=4.0,
        v_bias=2.0,
        resistance=2.5e3,
        impedance=120.0 * np.pi,
    )

    expected_gain = np.pi * 0.2 * 2.5e3 * 0.9 * 1.5e-12 / (2.0 * 4.0 * 120.0 * np.pi)
    expected_bias = np.pi * 2.0 / 4.0
    np.testing.assert_allclose(float(gain), expected_gain, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(float(bias), expected_bias, rtol=1e-6, atol=1e-12)


def test_williamson_response_matches_neuroptica_electro_optic_activation():
    values = jnp.asarray(
        [[1.0 + 0.2j, -0.4 + 1.3j], [0.7 - 0.8j, -1.1 - 0.5j]],
        dtype=jnp.complex64,
    )
    tap = 0.37
    gain = 0.81
    bias = 1.23

    theta = gain * np.square(np.abs(np.asarray(values))) + bias
    neuroptica = (
        1j
        * np.sqrt(1.0 - tap)
        * np.exp(-0.5j * theta)
        * np.cos(0.5 * theta)
        * np.asarray(values)
    )
    lumix = np.asarray(williamson_response(values, gain, bias, tap))

    np.testing.assert_allclose(lumix, neuroptica, rtol=1e-6, atol=1e-6)
