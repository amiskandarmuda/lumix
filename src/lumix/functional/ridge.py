import jax.numpy as jnp


def solve_ridge(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    alpha: float = 1e-3,
    use_bias: bool = True,
) -> dict[str, jnp.ndarray]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (samples, features)")

    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2:
        raise ValueError("targets must have shape (samples,) or (samples, outputs)")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count")

    feature_count = inputs.shape[-1]
    design = inputs
    if use_bias:
        ones = jnp.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        design = jnp.concatenate([inputs, ones], axis=-1)

    regularizer = alpha * jnp.eye(design.shape[-1], dtype=design.dtype)
    if use_bias:
        regularizer = regularizer.at[-1, -1].set(0.0)

    design_hermitian = jnp.conj(design.T)
    solution = jnp.linalg.solve(
        design_hermitian @ design + regularizer,
        design_hermitian @ targets,
    )
    if use_bias:
        return {
            "kernel": solution[:feature_count],
            "bias": solution[feature_count],
        }
    return {"kernel": solution}
