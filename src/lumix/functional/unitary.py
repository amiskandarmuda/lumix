import jax.numpy as jnp
from jax.scipy.linalg import expm


def combine_complex_parts(real: jnp.ndarray, imag: jnp.ndarray) -> jnp.ndarray:
    return real + 1j * imag


def skew_hermitian(raw: jnp.ndarray) -> jnp.ndarray:
    return raw - jnp.conj(raw.T)


def unitary_matrix(raw: jnp.ndarray) -> jnp.ndarray:
    return expm(skew_hermitian(raw))


def isometric_matrix(
    left_raw: jnp.ndarray,
    right_raw: jnp.ndarray,
    output_features: int,
    input_features: int,
) -> jnp.ndarray:
    rank = min(output_features, input_features)
    left = unitary_matrix(left_raw)
    right = unitary_matrix(right_raw)
    return left[:, :rank] @ jnp.conj(right[:, :rank]).T


def unitary_linear(values: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    return values @ matrix.T
