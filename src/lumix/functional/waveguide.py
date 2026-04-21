from collections.abc import Sequence

import jax.numpy as jnp


def _ensure_real_vector(values: Sequence[float] | jnp.ndarray, name: str) -> jnp.ndarray:
    array = jnp.asarray(values)
    if jnp.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")

    vector = jnp.asarray(array, dtype=jnp.float32)
    if not bool(jnp.all(jnp.isfinite(vector))):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _ensure_real_scalar(value: float, name: str) -> jnp.ndarray:
    scalar = jnp.asarray(value)
    if jnp.iscomplexobj(scalar):
        raise ValueError(f"{name} must be real-valued")
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar")

    real_scalar = jnp.asarray(scalar, dtype=jnp.float32)
    if not bool(jnp.isfinite(real_scalar)):
        raise ValueError(f"{name} must be finite")
    return real_scalar


def _symmetric_profile(left: Sequence[float] | jnp.ndarray, center: float | None = None) -> jnp.ndarray:
    left_vector = _ensure_real_vector(left, "left")
    mirrored = jnp.flip(left_vector)
    if center is None:
        return jnp.concatenate((left_vector, mirrored))

    center_value = _ensure_real_scalar(center, "center")
    return jnp.concatenate((left_vector, center_value[None], mirrored))


def _validated_waveguide_inputs(
    delta: Sequence[float] | jnp.ndarray,
    kappa: Sequence[float] | jnp.ndarray,
    length: float | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    delta_vector = _ensure_real_vector(delta, "delta")
    kappa_vector = _ensure_real_vector(kappa, "kappa")
    if kappa_vector.shape[0] != max(delta_vector.shape[0] - 1, 0):
        raise ValueError("kappa must have length len(delta) - 1")

    if length is None:
        return delta_vector, kappa_vector, None

    length_scalar = _ensure_real_scalar(length, "length")
    if float(length_scalar) < 0.0:
        raise ValueError("length must be non-negative")
    return delta_vector, kappa_vector, length_scalar


def waveguide_hamiltonian(delta: Sequence[float] | jnp.ndarray, kappa: Sequence[float] | jnp.ndarray) -> jnp.ndarray:
    delta_vector, kappa_vector, _ = _validated_waveguide_inputs(delta, kappa)
    width = delta_vector.shape[0]
    hamiltonian = jnp.diag(delta_vector)
    if width <= 1:
        return hamiltonian
    upper = jnp.diag(kappa_vector, k=1)
    lower = jnp.diag(kappa_vector, k=-1)
    return hamiltonian + upper + lower


def waveguide_propagator(
    delta: Sequence[float] | jnp.ndarray,
    kappa: Sequence[float] | jnp.ndarray,
    length: float | jnp.ndarray,
) -> jnp.ndarray:
    delta_vector, kappa_vector, length_scalar = _validated_waveguide_inputs(delta, kappa, length)
    hamiltonian = waveguide_hamiltonian(delta_vector, kappa_vector)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
    phases = jnp.exp(-1j * eigenvalues * length_scalar)
    return eigenvectors @ jnp.diag(phases) @ eigenvectors.T


def waveguide_linear(values: jnp.ndarray, propagator: jnp.ndarray) -> jnp.ndarray:
    if propagator.ndim != 2 or propagator.shape[0] != propagator.shape[1]:
        raise ValueError("propagator must be a square matrix")
    if values.shape[-1] != propagator.shape[0]:
        raise ValueError("values width must match propagator width")
    return values @ propagator.T


def symmetric_delta_profile(left: Sequence[float] | jnp.ndarray, center: float | None = None) -> jnp.ndarray:
    return _symmetric_profile(left, center=center)


def symmetric_kappa_profile(left: Sequence[float] | jnp.ndarray, center: float | None = None) -> jnp.ndarray:
    return _symmetric_profile(left, center=center)
