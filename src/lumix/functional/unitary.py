import jax.numpy as jnp
from jax.scipy.linalg import expm


def skew_hermitian(raw: jnp.ndarray) -> jnp.ndarray:
    return raw - jnp.conj(raw.T)


def unitary_matrix(raw: jnp.ndarray) -> jnp.ndarray:
    return expm(skew_hermitian(raw))


def unitary_linear(values: jnp.ndarray, raw: jnp.ndarray) -> jnp.ndarray:
    return values @ unitary_matrix(raw)
