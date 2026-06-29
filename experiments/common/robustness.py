from __future__ import annotations

import jax
import jax.numpy as jnp

from experiments.common.models import (
    SubUnitarySurrogateConfig,
    _active_layer_count,
    routed_subunitary_matrix,
)
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear
from lumix.functional.unitary import combine_complex_parts


def surrogate_layer_matrix_from_params(
    config: SubUnitarySurrogateConfig,
    params,
    layer_index: int,
) -> jnp.ndarray:
    layer_params = params[f"{config.layer_name_prefix}_{int(layer_index)}"]
    return routed_subunitary_matrix(
        combine_complex_parts(layer_params["left_re"], layer_params["left_im"]),
        combine_complex_parts(layer_params["right_re"], layer_params["right_im"]),
        layer_params["singular_raw"],
        insertion_loss_db=config.loss_db,
        output_features=config.width,
        input_features=config.width,
        routing_limit=config.routing_limit,
        hard_routing=config.hard_routing,
    )


def relative_frobenius_perturbation(
    matrix: jnp.ndarray,
    rng,
    *,
    relative_error: float,
    rescale_to_passive: bool = False,
    singular_max: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if relative_error <= 0.0:
        return matrix

    real_key, imag_key = jax.random.split(rng)
    real = jax.random.normal(real_key, matrix.shape, dtype=jnp.float32)
    imag = jax.random.normal(imag_key, matrix.shape, dtype=jnp.float32)
    noise = (real + 1j * imag).astype(matrix.dtype)
    noise_norm = jnp.linalg.norm(noise)
    matrix_norm = jnp.linalg.norm(matrix)
    scaled_noise = noise * (
        jnp.asarray(relative_error, dtype=jnp.float32)
        * matrix_norm
        / jnp.maximum(noise_norm, jnp.asarray(1e-12, dtype=jnp.float32))
    ).astype(matrix.dtype)
    perturbed = matrix + scaled_noise

    if not rescale_to_passive:
        return perturbed

    max_allowed = jnp.asarray(1.0, dtype=jnp.float32) if singular_max is None else singular_max
    spectral_norm = jnp.linalg.svd(perturbed, compute_uv=False)[0]
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        max_allowed / jnp.maximum(spectral_norm, jnp.asarray(1e-12, dtype=jnp.float32)),
    )
    return perturbed * scale.astype(perturbed.dtype)


def noisy_surrogate_forward(
    model,
    params,
    values: jnp.ndarray,
    rng,
    *,
    relative_error: float,
    depth: int | None = None,
    readout_name: str | None = None,
    rescale_to_passive: bool = False,
    return_aux: bool = False,
):
    active_layers = _active_layer_count(model.config, depth)
    _, singular_max = insertion_loss_bounds(model.config.loss_db)
    fields = model.input_fields(values)
    for index in range(active_layers):
        matrix = surrogate_layer_matrix_from_params(model.config, params, index)
        matrix = relative_frobenius_perturbation(
            matrix,
            jax.random.fold_in(rng, index),
            relative_error=relative_error,
            rescale_to_passive=rescale_to_passive,
            singular_max=singular_max,
        )
        fields = subunitary_linear(fields * model.phase_mask_for_layer(values, index), matrix)

    return model.apply(
        {"params": params},
        fields,
        return_aux=return_aux,
        readout_name=readout_name,
        method=type(model).readout_fields,
    )
