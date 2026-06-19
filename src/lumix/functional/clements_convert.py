"""Conversion utilities from dense unitary layers to Clements meshes."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lumix.functional.clements import build_clements_spec, clements_pair, init_clements
from lumix.functional.unitary import combine_complex_parts, isometric_matrix


@dataclass(frozen=True)
class ClementsFitResult:
    """Result of fitting a Clements mesh to a target transfer matrix."""

    params: dict[str, jnp.ndarray]
    loss: float
    relative_frobenius_error: float
    max_abs_error: float
    iterations: int


def unitary_linear_matrix_from_params(params: dict[str, jnp.ndarray], width: int, out_features: int | None = None) -> jnp.ndarray:
    """Return the transfer matrix represented by a ``UnitaryLinear`` param dict.

    The returned matrix follows Lumix's dense convention: applying it to row
    vectors is ``values @ matrix.T``.
    """

    output_features = width if out_features is None else out_features
    input_features = params["right_re"].shape[0]
    left = combine_complex_parts(params["left_re"], params["left_im"])
    right = combine_complex_parts(params["right_re"], params["right_im"])
    return isometric_matrix(left, right, output_features, input_features)


def clements_transfer_matrix(params: dict[str, jnp.ndarray], width: int, depth: int | None = None, hadamard: bool = False) -> jnp.ndarray:
    """Return the dense transfer matrix implemented by Clements phase params."""

    mesh_depth = width if depth is None else depth
    basis = jnp.eye(width, dtype=jnp.complex64)
    spec = build_clements_spec(width, mesh_depth)
    return clements_pair(
        basis,
        params["theta"],
        params["phi"],
        params["gamma"],
        spec=spec,
        hadamard=hadamard,
    ).T


def identity_clements_params(width: int, depth: int | None = None, hadamard: bool = False) -> dict[str, jnp.ndarray]:
    """Return Clements phase params that implement the identity matrix."""

    mesh_depth = width if depth is None else depth
    theta_value = 0.0 if hadamard else jnp.pi
    return {
        "theta": jnp.full((mesh_depth, width // 2), theta_value, dtype=jnp.float32),
        "phi": jnp.zeros((mesh_depth, width // 2), dtype=jnp.float32),
        "gamma": jnp.zeros((1, width), dtype=jnp.float32),
    }


def perturbed_identity_clements_params(
    key: jax.Array,
    width: int,
    depth: int | None = None,
    hadamard: bool = False,
    scale: float = 1e-2,
) -> dict[str, jnp.ndarray]:
    """Return identity Clements params plus small phase perturbations."""

    params = identity_clements_params(width, depth=depth, hadamard=hadamard)
    theta_key, phi_key, gamma_key = jax.random.split(key, 3)
    return {
        "theta": params["theta"] + scale * jax.random.normal(theta_key, params["theta"].shape, dtype=jnp.float32),
        "phi": params["phi"] + scale * jax.random.normal(phi_key, params["phi"].shape, dtype=jnp.float32),
        "gamma": params["gamma"] + scale * jax.random.normal(gamma_key, params["gamma"].shape, dtype=jnp.float32),
    }


def _bloch_mzi_matrix(theta: float, phi: float, hadamard: bool = False) -> np.ndarray:
    if hadamard:
        return np.asarray(
            [
                [np.exp(1j * phi) * np.cos(theta / 2.0), 1j * np.sin(theta / 2.0)],
                [1j * np.exp(1j * phi) * np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ],
            dtype=np.complex128,
        )
    return 1j * np.asarray(
        [
            [np.exp(1j * phi) * np.sin(theta / 2.0), np.cos(theta / 2.0)],
            [np.exp(1j * phi) * np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _givens_rotation(unitary_2x2: np.ndarray, units: int, mode: int) -> np.ndarray:
    rotation = np.eye(units, dtype=np.complex128)
    rotation[mode, mode] = unitary_2x2[0, 0]
    rotation[mode, mode + 1] = unitary_2x2[0, 1]
    rotation[mode + 1, mode] = unitary_2x2[1, 0]
    rotation[mode + 1, mode + 1] = unitary_2x2[1, 1]
    return rotation


def _checkerboard_to_param(checkerboard: np.ndarray, units: int) -> np.ndarray:
    params = np.zeros((units, units // 2), dtype=np.float64)
    if units % 2:
        params[::2, :] = checkerboard.T[::2, :-1:2]
    else:
        params[::2, :] = checkerboard.T[::2, ::2]
    params[1::2, :] = checkerboard.T[1::2, 1::2]
    return params


def _grid_common_mode_flow(external_phases: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    units, num_layers = external_phases.shape
    phase_shifts = np.hstack((gamma[:, np.newaxis], external_phases)).T
    new_phase_shifts = np.zeros_like(external_phases.T)
    for i in range(num_layers):
        current_layer = num_layers - i
        start_idx = (current_layer - 1) % 2
        end_idx = units - (current_layer + units - 1) % 2
        upper_phase = np.mod(phase_shifts[current_layer][start_idx:end_idx][::2], 2 * np.pi)
        lower_phase = np.mod(phase_shifts[current_layer][start_idx:end_idx][1::2], 2 * np.pi)
        new_phase_shifts[-i - 1][start_idx:end_idx][::2] = upper_phase - lower_phase
        phase_shifts[current_layer] -= new_phase_shifts[-i - 1]
        phase_shifts[current_layer - 1] += np.mod(phase_shifts[current_layer], 2 * np.pi)
        phase_shifts[current_layer] = 0
    new_gamma = np.mod(phase_shifts[0], 2 * np.pi)
    return np.mod(new_phase_shifts.T, 2 * np.pi), new_gamma


def decompose_unitary_to_clements(target: np.ndarray | jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Analytically decompose a unitary matrix into Lumix Clements phases.

    This follows the rectangular Clements/Bloch-MZI convention used by
    Neurophox and Lumix's ``clements_pair`` implementation.
    """

    # Neurophox's decomposition routine is written for the opposite matrix
    # orientation from Lumix's row-vector transfer convention.
    matrix = np.asarray(target, dtype=np.complex128).T
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("target must be a square unitary matrix")
    n = matrix.shape[0]
    u_hat = matrix.T.copy()
    theta_checkerboard = np.zeros_like(matrix, dtype=np.float64)
    phi_checkerboard = np.zeros_like(matrix, dtype=np.float64)
    phi_checkerboard = np.hstack((np.zeros((n, 1), dtype=np.float64), phi_checkerboard))

    for i in range(n - 1):
        if i % 2:
            for j in range(i + 1):
                pairwise_index = n + j - i - 2
                target_row, target_col = n + j - i - 1, j
                theta = 2.0 * np.arctan(np.abs(u_hat[target_row - 1, target_col] / u_hat[target_row, target_col]))
                phi = np.angle(u_hat[target_row, target_col] / u_hat[target_row - 1, target_col])
                left_multiplier = _givens_rotation(_bloch_mzi_matrix(theta, phi, hadamard=False), n, pairwise_index)
                u_hat = left_multiplier @ u_hat
                theta_checkerboard[pairwise_index, j] = theta
                phi_checkerboard[pairwise_index, j] = -phi + np.pi
                phi_checkerboard[pairwise_index + 1, j] = np.pi
        else:
            for j in range(i + 1):
                pairwise_index = i - j
                target_row, target_col = n - j - 1, i - j
                theta = 2.0 * np.arctan(np.abs(u_hat[target_row, target_col + 1] / u_hat[target_row, target_col]))
                phi = np.angle(-u_hat[target_row, target_col] / u_hat[target_row, target_col + 1])
                right_multiplier = _givens_rotation(_bloch_mzi_matrix(theta, phi, hadamard=False), n, pairwise_index)
                u_hat = u_hat @ right_multiplier.conj().T
                theta_checkerboard[pairwise_index, -j - 1] = theta
                phi_checkerboard[pairwise_index, -j - 1] = phi + np.pi

    diag_phases = np.angle(np.diag(u_hat))
    theta = _checkerboard_to_param(np.fliplr(theta_checkerboard), n)
    phi_checkerboard = np.fliplr(phi_checkerboard)
    if n % 2:
        phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    else:
        phi_checkerboard[:, 1:] += np.fliplr(np.diag(diag_phases))
    phi_checkerboard[-1, 2::2] += np.pi / 2.0
    phi_checkerboard[0, 2::2] += np.pi / 2.0
    gamma = phi_checkerboard[:, 0]
    external_phases = phi_checkerboard[:, 1:]
    phi, gamma = _grid_common_mode_flow(external_phases, gamma=gamma)
    phi = _checkerboard_to_param(phi, n)
    gamma_adj = np.zeros_like(gamma)
    gamma_adj[1::4] = 1
    gamma_adj[2::4] = 1
    gamma += np.pi * (1 - gamma_adj) if (n // 2) % 2 else np.pi * gamma_adj
    gamma = np.mod(gamma, 2 * np.pi)
    return {
        "theta": jnp.asarray(theta, dtype=jnp.float32),
        "phi": jnp.asarray(phi, dtype=jnp.float32),
        "gamma": jnp.asarray(gamma[None, :], dtype=jnp.float32),
    }


def clements_fit_metrics(
    target: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    depth: int | None = None,
    hadamard: bool = False,
) -> dict[str, jnp.ndarray]:
    """Measure how closely Clements params reproduce a target matrix."""

    width = target.shape[0]
    matrix = clements_transfer_matrix(params, width, depth=depth, hadamard=hadamard)
    diff = matrix - target
    loss = jnp.mean(jnp.square(jnp.abs(diff)))
    relative = jnp.linalg.norm(diff) / (jnp.linalg.norm(target) + 1e-12)
    return {
        "loss": loss,
        "relative_frobenius_error": relative,
        "max_abs_error": jnp.max(jnp.abs(diff)),
    }


def fit_clements_to_unitary(
    target: jnp.ndarray,
    key: jax.Array,
    depth: int | None = None,
    iterations: int = 2000,
    learning_rate: float = 2e-2,
    hadamard: bool = False,
    init_params: dict[str, jnp.ndarray] | None = None,
) -> ClementsFitResult:
    """Fit Clements mesh phases to a target unitary transfer matrix.

    This is hardware phase calibration, not task retraining. The target is a
    fixed dense unitary matrix, and the objective is only matrix mismatch.
    """

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target must be a square dense unitary matrix")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    width = int(target.shape[0])
    mesh_depth = width if depth is None else int(depth)
    target = target.astype(jnp.complex64)
    params = init_clements(key, width, mesh_depth, hadamard=hadamard) if init_params is None else init_params
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)
    spec = build_clements_spec(width, mesh_depth)
    basis = jnp.eye(width, dtype=jnp.complex64)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(current):
            matrix = clements_pair(
                basis,
                current["theta"],
                current["phi"],
                current["gamma"],
                spec=spec,
                hadamard=hadamard,
            ).T
            return jnp.mean(jnp.square(jnp.abs(matrix - target)))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss = jnp.asarray(jnp.inf, dtype=jnp.float32)
    for _ in range(iterations):
        params, opt_state, loss = step(params, opt_state)

    metrics = clements_fit_metrics(target, params, depth=mesh_depth, hadamard=hadamard)
    return ClementsFitResult(
        params=params,
        loss=float(metrics["loss"]),
        relative_frobenius_error=float(metrics["relative_frobenius_error"]),
        max_abs_error=float(metrics["max_abs_error"]),
        iterations=int(iterations),
    )


def fit_clements_to_unitary_linear_params(
    unitary_params: dict[str, jnp.ndarray],
    key: jax.Array,
    width: int,
    depth: int | None = None,
    iterations: int = 2000,
    learning_rate: float = 2e-2,
    hadamard: bool = False,
    init_params: dict[str, jnp.ndarray] | None = None,
) -> ClementsFitResult:
    """Fit Clements mesh params to a saved ``UnitaryLinear`` param dict."""

    target = unitary_linear_matrix_from_params(unitary_params, width)
    return fit_clements_to_unitary(
        target,
        key,
        depth=depth,
        iterations=iterations,
        learning_rate=learning_rate,
        hadamard=hadamard,
        init_params=init_params,
    )
