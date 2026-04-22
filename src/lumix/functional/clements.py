from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import lax, random

from lumix.spec import ClementsSpec


def layer_mask(width: int, depth: int) -> jnp.ndarray:
    full_pairs = width // 2
    odd_pairs = (width - 1) // 2
    pair_counts = jnp.full((depth,), full_pairs, dtype=jnp.int32)
    pair_counts = pair_counts.at[1::2].set(odd_pairs)
    pair_index = jnp.arange(full_pairs)[None, :]
    return (pair_index < pair_counts[:, None]).astype(jnp.float32)


def grid_permutation(width: int, depth: int) -> jnp.ndarray:
    ordered = jnp.arange(width, dtype=jnp.int32)
    left = jnp.roll(ordered, -1)
    right = jnp.roll(ordered, 1)
    permuted = jnp.zeros((depth, width), dtype=jnp.int32)
    permuted = permuted.at[::2].set(jnp.broadcast_to(left, (depth - depth // 2, width)))
    permuted = permuted.at[1::2].set(jnp.broadcast_to(right, (depth // 2, width)))
    if depth % 2:
        return jnp.vstack((ordered[None, :], permuted[:-1], ordered[None, :]))
    return jnp.vstack((ordered[None, :], permuted))


def stripe_layout(phase: jnp.ndarray, width: int) -> jnp.ndarray:
    depth = phase.shape[0]
    stripe = jnp.zeros((width - 1, depth), dtype=phase.dtype)
    stripe = stripe.at[::2, :].set(phase.T)
    tail = jnp.zeros((1, depth), dtype=phase.dtype)
    return jnp.vstack((stripe, tail))


def differential_layout(phase: jnp.ndarray, width: int) -> jnp.ndarray:
    stripe = stripe_layout(phase, width)
    half = stripe / 2.0
    return half - jnp.roll(half, 1, axis=0)


def haar_diagonal_sequence(diagonal_length: int, reverse: bool = False) -> jnp.ndarray:
    odd_values = diagonal_length + 1 - jnp.flip(jnp.arange(1, diagonal_length + 1, 2), axis=0)
    even_stop = 2 * (diagonal_length - odd_values.shape[0]) + 1
    even_values = diagonal_length + 1 - jnp.arange(2, even_stop, 2)
    values = jnp.concatenate((odd_values, even_values))
    return values[::-1] if reverse else values


def alpha_checkerboard(width: int, depth: int) -> jnp.ndarray:
    if width < depth:
        raise ValueError("width must be at least depth")
    board = jnp.zeros((width - 1, depth), dtype=jnp.float32)
    sequences = [haar_diagonal_sequence(index, reverse=bool(depth % 2)) for index in range(1, depth + 1)]
    for row in range(width - 1):
        for column in range(depth):
            if (row + column) % 2 != 0:
                continue
            if row < depth and column > row:
                diagonal = depth - abs(row - column)
            elif row > width - depth and column < row - width + depth:
                diagonal = depth - abs(row - column - width + depth) - int(width == depth)
            else:
                diagonal = depth - int(width == depth)
            value = (
                jnp.float32(1.0)
                if diagonal == 1
                else jnp.asarray(sequences[diagonal - 1][min(row, column)], dtype=jnp.float32)
            )
            board = board.at[row, column].set(value)
    if width != depth:
        board = (board + jnp.flipud(board)) / 2.0
    return board


def alpha_checkerboard_stack(width: int, depth: int) -> jnp.ndarray:
    blocks = []
    full_blocks = depth // width
    for block_index in range(full_blocks):
        flip = bool(block_index % 2 and width % 2)
        block = alpha_checkerboard(width, width)
        blocks.append(jnp.flipud(block) if flip else block)
    extra_depth = depth - full_blocks * width
    if extra_depth:
        flip = bool((not full_blocks % 2) and width % 2)
        block = alpha_checkerboard(width, extra_depth)
        blocks.append(jnp.flipud(block) if flip else block)
    return jnp.hstack(blocks) if blocks else jnp.zeros((width - 1, 0), dtype=jnp.float32)


def _haar_diagonal_sequence_np(diagonal_length: int, reverse: bool = False) -> np.ndarray:
    odd_values = diagonal_length + 1 - np.flip(np.arange(1, diagonal_length + 1, 2), axis=0)
    even_stop = 2 * (diagonal_length - odd_values.shape[0]) + 1
    even_values = diagonal_length + 1 - np.arange(2, even_stop, 2)
    values = np.concatenate((odd_values, even_values)).astype(np.float32, copy=False)
    return values[::-1] if reverse else values


def _alpha_checkerboard_np(width: int, depth: int) -> np.ndarray:
    if width < depth:
        raise ValueError("width must be at least depth")
    board = np.zeros((width - 1, depth), dtype=np.float32)
    sequences = [
        _haar_diagonal_sequence_np(index, reverse=bool(depth % 2))
        for index in range(1, depth + 1)
    ]
    equal_sides = int(width == depth)
    for row in range(width - 1):
        for column in range(depth):
            if (row + column) % 2 != 0:
                continue
            if row < depth and column > row:
                diagonal = depth - abs(row - column)
            elif row > width - depth and column < row - width + depth:
                diagonal = depth - abs(row - column - width + depth) - equal_sides
            else:
                diagonal = depth - equal_sides
            if diagonal == 1:
                value = np.float32(1.0)
            else:
                value = np.float32(sequences[diagonal - 1][min(row, column)])
            board[row, column] = value
    if width != depth:
        board = (board + np.flipud(board)) / 2.0
    return board.astype(np.float32, copy=False)


@lru_cache(maxsize=None)
def _alpha_checkerboard_stack_cached(width: int, depth: int) -> np.ndarray:
    blocks = []
    full_blocks = depth // width
    square_block = _alpha_checkerboard_np(width, width)
    for block_index in range(full_blocks):
        flip = bool(block_index % 2 and width % 2)
        blocks.append(np.flipud(square_block) if flip else square_block)
    extra_depth = depth - full_blocks * width
    if extra_depth:
        flip = bool((full_blocks % 2 == 0) and width % 2)
        block = _alpha_checkerboard_np(width, extra_depth)
        blocks.append(np.flipud(block) if flip else block)
    if not blocks:
        return np.zeros((width - 1, 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32, copy=False)


def sample_theta(key, width: int, depth: int, hadamard: bool = False) -> jnp.ndarray:
    roots = jnp.asarray(_alpha_checkerboard_stack_cached(width, depth)).T
    even_roots = 2.0 * roots[::2, ::2]
    odd_roots = 2.0 * roots[1::2, 1::2]
    even_shape = even_roots.shape
    odd_shape = odd_roots.shape
    even_key, odd_key = random.split(key)
    even_uniform = random.uniform(even_key, even_shape, minval=0.0, maxval=1.0)
    odd_uniform = random.uniform(odd_key, odd_shape, minval=0.0, maxval=1.0)
    even_theta = 2.0 * jnp.arcsin(even_uniform ** (1.0 / even_roots))
    odd_theta = 2.0 * jnp.arcsin(odd_uniform ** (1.0 / odd_roots))
    if not hadamard:
        even_theta = jnp.pi - even_theta
        odd_theta = jnp.pi - odd_theta
    theta = jnp.zeros((depth, width // 2), dtype=jnp.float32)
    theta = theta.at[::2, :].set(even_theta.astype(jnp.float32))
    if width % 2:
        theta = theta.at[1::2, :].set(odd_theta.astype(jnp.float32))
    else:
        theta = theta.at[1::2, :-1].set(odd_theta.astype(jnp.float32))
    return theta


def sample_phi(key, width: int, depth: int) -> jnp.ndarray:
    return random.uniform(key, (depth, width // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32)


def sample_gamma(key, width: int) -> jnp.ndarray:
    return random.uniform(key, (1, width), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32)


def build_clements_spec(width: int, depth: int) -> ClementsSpec:
    return ClementsSpec(
        width=width,
        depth=depth,
        perm=grid_permutation(width, depth),
        mask=layer_mask(width, depth),
    )


def init_clements(key, width: int, depth: int, hadamard: bool = False) -> dict[str, jnp.ndarray]:
    theta_key, phi_key, gamma_key = random.split(key, 3)
    return {
        "theta": sample_theta(theta_key, width, depth, hadamard=hadamard),
        "phi": sample_phi(phi_key, width, depth),
        "gamma": sample_gamma(gamma_key, width),
    }


def mask_phases(
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    mask: jnp.ndarray,
    hadamard: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mask = mask.astype(theta.dtype)
    bar_state = 0.0 if hadamard else jnp.pi
    theta_masked = theta * mask + (1.0 - mask) * bar_state
    phi_masked = phi * mask + (1.0 - mask) * bar_state
    return theta_masked, phi_masked


def pair_coefficients(
    internal_upper: jnp.ndarray,
    internal_lower: jnp.ndarray,
    external_upper: jnp.ndarray,
    external_lower: jnp.ndarray,
    hadamard: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cc = jnp.float32(0.5)
    cs = jnp.float32(0.5)
    sc = jnp.float32(0.5)
    ss = jnp.float32(0.5)
    upper = jnp.exp(1j * internal_upper)
    lower = jnp.exp(1j * internal_lower)
    output_upper = jnp.exp(1j * external_upper)
    output_lower = jnp.exp(1j * external_lower)
    if hadamard:
        return (
            (cc * upper + ss * lower) * output_upper,
            (cs * upper - sc * lower) * output_lower,
            (sc * upper - cs * lower) * output_upper,
            (ss * upper + cc * lower) * output_lower,
        )
    return (
        (cc * upper - ss * lower) * output_upper,
        1j * (cs * upper + sc * lower) * output_lower,
        1j * (sc * upper + cs * lower) * output_upper,
        (cc * lower - ss * upper) * output_lower,
    )


def pair_coefficients_layout(
    internal: jnp.ndarray,
    output: jnp.ndarray,
    hadamard: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    upper = internal[0::2, :]
    lower = internal[1::2, :]
    output_upper = output[0::2, :]
    output_lower = output[1::2, :]
    return pair_coefficients(
        upper,
        lower,
        output_upper,
        output_lower,
        hadamard=hadamard,
    )


def apply_pair_layer(
    values: jnp.ndarray,
    internal: jnp.ndarray,
    output: jnp.ndarray,
    hadamard: bool = False,
) -> jnp.ndarray:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    upper = internal[0::2]
    lower = internal[1::2]
    output_upper = output[0::2]
    output_lower = output[1::2]
    u11, u12, u21, u22 = pair_coefficients(
        upper,
        lower,
        output_upper,
        output_lower,
        hadamard=hadamard,
    )
    next_even = even * u11 + odd * u21
    next_odd = even * u12 + odd * u22
    next_values = jnp.empty_like(values)
    next_values = next_values.at[..., 0::2].set(next_even)
    next_values = next_values.at[..., 1::2].set(next_odd)
    return next_values


def apply_pair_coefficients(
    values: jnp.ndarray,
    u11: jnp.ndarray,
    u12: jnp.ndarray,
    u21: jnp.ndarray,
    u22: jnp.ndarray,
) -> jnp.ndarray:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    next_even = even * u11 + odd * u21
    next_odd = even * u12 + odd * u22
    next_values = jnp.empty_like(values)
    next_values = next_values.at[..., 0::2].set(next_even)
    next_values = next_values.at[..., 1::2].set(next_odd)
    return next_values


def clements_pair(
    values: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    gamma: jnp.ndarray,
    spec: ClementsSpec | None = None,
    hadamard: bool = False,
) -> jnp.ndarray:
    mesh_spec = build_clements_spec(values.shape[-1], theta.shape[0]) if spec is None else spec
    perm = mesh_spec.perm
    mask = mesh_spec.mask
    width = perm.shape[-1]
    depth = theta.shape[0]

    theta_masked, phi_masked = mask_phases(theta, phi, mask, hadamard=hadamard)
    internal = differential_layout(theta_masked, width)
    output = stripe_layout(phi_masked, width)
    u11, u12, u21, u22 = pair_coefficients_layout(
        internal,
        output,
        hadamard=hadamard,
    )
    next_values = values * jnp.exp(1j * gamma)
    next_values = jnp.take(next_values, perm[0], axis=-1)

    def apply_layer(carry: jnp.ndarray, layer_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        u11_layer, u12_layer, u21_layer, u22_layer, permutation = layer_inputs
        updated = apply_pair_coefficients(
            carry,
            u11_layer,
            u12_layer,
            u21_layer,
            u22_layer,
        )
        return jnp.take(updated, permutation, axis=-1), None

    next_values, _ = lax.scan(
        apply_layer,
        next_values,
        xs=(u11.T, u12.T, u21.T, u22.T, perm[1 : depth + 1]),
    )
    return next_values
