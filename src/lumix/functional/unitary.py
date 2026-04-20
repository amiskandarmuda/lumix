import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import expm


def init_unitary(key, width: int, init_scale: float = 1e-2) -> jnp.ndarray:
    real_key, imag_key = random.split(key)
    real = random.normal(real_key, (width, width), dtype=jnp.float32)
    imag = random.normal(imag_key, (width, width), dtype=jnp.float32)
    return init_scale * (real + 1j * imag)


def skew_hermitian(raw: jnp.ndarray) -> jnp.ndarray:
    return raw - jnp.conj(raw.T)


def unitary_matrix(raw: jnp.ndarray) -> jnp.ndarray:
    return expm(skew_hermitian(raw))


def unitary_linear(values: jnp.ndarray, raw: jnp.ndarray) -> jnp.ndarray:
    return values @ unitary_matrix(raw)
