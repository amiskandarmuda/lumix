import jax
import jax.numpy as jnp

from lumix.functional.waveguide import (
    symmetric_delta_profile,
    symmetric_kappa_profile,
    waveguide_hamiltonian,
    waveguide_propagator,
)
from lumix.linen.waveguide import FixedWaveguideArray


def _rk4_propagate(hamiltonian: jnp.ndarray, values: jnp.ndarray, length: float, steps: int) -> jnp.ndarray:
    step_size = length / steps

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        return -1j * (state @ hamiltonian.T)

    state = values
    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step_size * k1)
        k3 = rhs(state + 0.5 * step_size * k2)
        k4 = rhs(state + step_size * k3)
        state = state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def test_waveguide_hamiltonian_places_delta_and_kappa():
    delta = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    kappa = jnp.array([0.1, 0.2], dtype=jnp.float32)

    hamiltonian = waveguide_hamiltonian(delta, kappa)
    expected = jnp.array(
        [
            [1.0, 0.1, 0.0],
            [0.1, 2.0, 0.2],
            [0.0, 0.2, 3.0],
        ],
        dtype=jnp.float32,
    )

    assert jnp.array_equal(hamiltonian, expected)
    assert jnp.array_equal(hamiltonian, hamiltonian.T)


def test_symmetric_delta_profile_supports_even_and_odd_widths():
    left = jnp.array([1.0, 2.0], dtype=jnp.float32)

    even_profile = symmetric_delta_profile(left)
    odd_profile = symmetric_delta_profile(left, center=3.0)

    assert jnp.array_equal(even_profile, jnp.array([1.0, 2.0, 2.0, 1.0], dtype=jnp.float32))
    assert jnp.array_equal(odd_profile, jnp.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=jnp.float32))


def test_symmetric_kappa_profile_supports_even_and_odd_widths():
    left = jnp.array([0.1, 0.2], dtype=jnp.float32)

    odd_profile = symmetric_kappa_profile(left)
    even_profile = symmetric_kappa_profile(left, center=0.3)

    assert jnp.array_equal(odd_profile, jnp.array([0.1, 0.2, 0.2, 0.1], dtype=jnp.float32))
    assert jnp.array_equal(even_profile, jnp.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=jnp.float32))


def test_waveguide_propagator_is_unitary():
    delta = jnp.array([0.3, -0.1, 0.5, 0.2], dtype=jnp.float32)
    kappa = jnp.array([0.4, 0.2, 0.1], dtype=jnp.float32)

    propagator = waveguide_propagator(delta, kappa, length=1.7)
    identity = jnp.eye(propagator.shape[0], dtype=propagator.dtype)
    error = jnp.linalg.norm(jnp.conj(propagator.T) @ propagator - identity)

    assert float(error) < 1e-4
    assert float(jnp.max(jnp.abs(propagator - propagator.T))) < 1e-6


def test_waveguide_propagation_preserves_total_power():
    delta = jnp.array([0.0, 0.2, -0.2, 0.1], dtype=jnp.float32)
    kappa = jnp.array([0.3, 0.4, 0.3], dtype=jnp.float32)
    values = (
        jax.random.normal(jax.random.key(0), (5, 4), dtype=jnp.float32)
        + 1j * jax.random.normal(jax.random.key(1), (5, 4), dtype=jnp.float32)
    ).astype(jnp.complex64)

    propagator = waveguide_propagator(delta, kappa, length=0.9)
    outputs = values @ propagator.T
    input_power = jnp.sum(jnp.abs(values) ** 2, axis=-1)
    output_power = jnp.sum(jnp.abs(outputs) ** 2, axis=-1)

    assert float(jnp.max(jnp.abs(input_power - output_power))) < 1e-4


def test_waveguide_propagator_returns_identity_at_zero_length():
    delta = jnp.array([0.2, -0.4, 0.6], dtype=jnp.float32)
    kappa = jnp.array([0.1, 0.3], dtype=jnp.float32)

    propagator = waveguide_propagator(delta, kappa, length=0.0)
    identity = jnp.eye(3, dtype=propagator.dtype)

    assert float(jnp.max(jnp.abs(propagator - identity))) < 1e-6


def test_waveguide_propagator_reduces_to_independent_phase_shifts_when_uncoupled():
    delta = jnp.array([0.2, -0.4, 0.6], dtype=jnp.float32)
    kappa = jnp.zeros((2,), dtype=jnp.float32)
    length = 1.3

    propagator = waveguide_propagator(delta, kappa, length=length)
    expected = jnp.diag(jnp.exp(-1j * delta * length))

    assert float(jnp.max(jnp.abs(propagator - expected))) < 1e-6


def test_waveguide_eigenmodes_only_pick_up_phase():
    delta = jnp.array([0.3, -0.1, 0.5, 0.2], dtype=jnp.float32)
    kappa = jnp.array([0.4, 0.2, 0.1], dtype=jnp.float32)
    length = 1.7

    hamiltonian = waveguide_hamiltonian(delta, kappa)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
    propagator = waveguide_propagator(delta, kappa, length=length)

    for mode_index in range(eigenvectors.shape[1]):
        eigenmode = eigenvectors[:, mode_index]
        propagated = propagator @ eigenmode
        expected = jnp.exp(-1j * eigenvalues[mode_index] * length) * eigenmode
        assert float(jnp.max(jnp.abs(propagated - expected))) < 1e-5
        assert float(jnp.max(jnp.abs(jnp.abs(propagated) ** 2 - jnp.abs(eigenmode) ** 2))) < 1e-6


def test_waveguide_propagator_composes_over_length():
    delta = jnp.array([0.2, -0.1, 0.4, -0.3, 0.1], dtype=jnp.float32)
    kappa = jnp.array([0.3, 0.15, 0.2, 0.1], dtype=jnp.float32)
    first_length = 0.8
    second_length = 1.1

    combined = waveguide_propagator(delta, kappa, length=first_length + second_length)
    staged = waveguide_propagator(delta, kappa, length=second_length) @ waveguide_propagator(
        delta,
        kappa,
        length=first_length,
    )

    assert float(jnp.max(jnp.abs(combined - staged))) < 1e-5


def test_waveguide_propagator_matches_coupled_mode_ode_integration():
    delta = jnp.array([0.2, -0.4, 0.3, 0.1], dtype=jnp.float32)
    kappa = jnp.array([0.15, 0.25, 0.1], dtype=jnp.float32)
    length = 1.25
    values = (
        jax.random.normal(jax.random.key(11), (3, 4), dtype=jnp.float32)
        + 1j * jax.random.normal(jax.random.key(12), (3, 4), dtype=jnp.float32)
    ).astype(jnp.complex64)

    hamiltonian = waveguide_hamiltonian(delta, kappa)
    direct = values @ waveguide_propagator(delta, kappa, length=length).T
    integrated = _rk4_propagate(hamiltonian, values, length=length, steps=4096)

    assert float(jnp.max(jnp.abs(direct - integrated))) < 5e-4


def test_center_symmetric_profiles_preserve_mirror_symmetric_intensity():
    delta = symmetric_delta_profile(jnp.array([-0.3, -0.1, 0.2], dtype=jnp.float32), center=0.4)
    kappa = symmetric_kappa_profile(jnp.array([0.2, 0.25, 0.3], dtype=jnp.float32))
    values = jnp.array(
        [[1.0 + 0.0j, 0.5 + 0.25j, -0.3 + 0.2j, 0.1 - 0.4j, -0.3 + 0.2j, 0.5 + 0.25j, 1.0 + 0.0j]],
        dtype=jnp.complex64,
    )

    outputs = values @ waveguide_propagator(delta, kappa, length=0.7).T
    intensity = jnp.abs(outputs[0]) ** 2

    assert float(jnp.max(jnp.abs(intensity - jnp.flip(intensity)))) < 1e-5


def test_fixed_waveguide_array_has_only_constants_and_preserves_shape():
    layer = FixedWaveguideArray(delta=[0.1, 0.0, -0.1, 0.2], kappa=[0.3, 0.4, 0.3], length=1.1)
    values = (jnp.ones((4, 4)) + 1j * jnp.ones((4, 4))).astype(jnp.complex64)

    variables = layer.init(jax.random.key(2), values)
    outputs = layer.apply(variables, values)

    assert "params" not in variables
    assert variables["constants"]["delta"].shape == (4,)
    assert variables["constants"]["kappa"].shape == (3,)
    assert variables["constants"]["propagator"].shape == (4, 4)
    assert outputs.shape == (4, 4)


def test_fixed_waveguide_array_reuses_stored_constants():
    layer = FixedWaveguideArray(delta=[0.1, 0.0, -0.1], kappa=[0.2, 0.2], length=1.1)
    values = (jnp.ones((2, 3)) + 1j * jnp.ones((2, 3))).astype(jnp.complex64)
    variables = layer.init(jax.random.key(3), values)

    first = layer.apply(variables, values)
    second = layer.apply(variables, values)

    assert float(jnp.max(jnp.abs(first - second))) < 1e-7


def test_fixed_waveguide_array_supports_256_mode_profiles():
    delta = symmetric_delta_profile(jnp.linspace(-0.5, 0.5, 128, dtype=jnp.float32))
    kappa = symmetric_kappa_profile(jnp.linspace(0.1, 0.4, 127, dtype=jnp.float32), center=0.45)
    layer = FixedWaveguideArray(delta=tuple(delta.tolist()), kappa=tuple(kappa.tolist()), length=0.5)
    values = (jnp.ones((2, 256)) + 1j * jnp.ones((2, 256))).astype(jnp.complex64)

    variables = layer.init(jax.random.key(4), values)
    outputs = layer.apply(variables, values)

    assert variables["constants"]["propagator"].shape == (256, 256)
    assert outputs.shape == (2, 256)
