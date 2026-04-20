import jax.numpy as jnp
from jax import random


def channel_permutation(width: int, depth: int) -> jnp.ndarray:
    ordered = jnp.arange(width)
    left = jnp.roll(ordered, -1)
    right = jnp.roll(ordered, 1)
    layers = []
    for index in range(depth):
        layers.append(left if index % 2 == 0 else right)
    return jnp.vstack([ordered, jnp.stack(layers), ordered])


def phase_layout(phase: jnp.ndarray, width: int) -> jnp.ndarray:
    stripe = jnp.zeros((width - 1, phase.shape[0]), dtype=phase.dtype)
    stripe = stripe.at[::2, :].set(phase.T)
    tail = jnp.zeros((1, phase.shape[0]), dtype=phase.dtype)
    return jnp.concatenate([stripe, tail], axis=0)


def split_phase_layout(phase: jnp.ndarray, width: int) -> jnp.ndarray:
    stripe = phase_layout(phase, width)
    return stripe / 2.0 - jnp.roll(stripe / 2.0, 1, axis=0)


def sample_theta(key, width: int, depth: int) -> jnp.ndarray:
    return random.uniform(key, (depth, width // 2), minval=0.0, maxval=jnp.pi)


def sample_phi(key, width: int, depth: int) -> jnp.ndarray:
    return random.uniform(key, (depth, width // 2), minval=0.0, maxval=2.0 * jnp.pi)


def sample_gamma(key, width: int) -> jnp.ndarray:
    return random.uniform(key, (width,), minval=0.0, maxval=2.0 * jnp.pi)


def init_clements(key, width: int, depth: int) -> dict[str, jnp.ndarray]:
    theta_key, phi_key, gamma_key = random.split(key, 3)
    return {
        "theta": sample_theta(theta_key, width, depth),
        "phi": sample_phi(phi_key, width, depth),
        "gamma": sample_gamma(gamma_key, width),
    }


def _pair_matrix(internal_upper, internal_lower, output_upper, output_lower) -> tuple[jnp.ndarray, ...]:
    mix_cos = 0.5
    mix_sin = 0.5
    upper = jnp.exp(1j * internal_upper)
    lower = jnp.exp(1j * internal_lower)
    out_upper = jnp.exp(1j * output_upper)
    out_lower = jnp.exp(1j * output_lower)
    u11 = (mix_cos * upper - mix_sin * lower) * out_upper
    u12 = 1j * (mix_cos * upper + mix_sin * lower) * out_lower
    u21 = 1j * (mix_cos * upper + mix_sin * lower) * out_upper
    u22 = (mix_cos * lower - mix_sin * upper) * out_lower
    return u11, u12, u21, u22


def _apply_pair_layer(
    values: jnp.ndarray,
    internal: jnp.ndarray,
    output: jnp.ndarray,
) -> jnp.ndarray:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    upper = internal[0::2]
    lower = internal[1::2]
    out_upper = output[0::2]
    out_lower = output[1::2]
    u11, u12, u21, u22 = _pair_matrix(upper, lower, out_upper, out_lower)
    next_even = even * u11 + odd * u21
    next_odd = even * u12 + odd * u22
    result = jnp.empty_like(values)
    result = result.at[..., 0::2].set(next_even)
    result = result.at[..., 1::2].set(next_odd)
    return result


def clements_pair(
    values: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    width = values.shape[-1]
    depth = theta.shape[0]
    perms = channel_permutation(width, depth)
    values = values * jnp.exp(1j * gamma)
    internal = split_phase_layout(theta, width)
    output = phase_layout(phi, width)

    for index in range(depth):
        if index == 0:
            values = values[..., perms[0]]
        values = _apply_pair_layer(values, internal[:, index], output[:, index])
        values = values[..., perms[index + 1]]
    return values
