import jax.numpy as jnp


def _scale_range(
    values: jnp.ndarray,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    clip: bool,
) -> jnp.ndarray:
    input_min, input_max = input_range
    output_min, output_max = output_range
    if input_max == input_min:
        raise ValueError("input_range endpoints must be different")

    normalized = (values - input_min) / (input_max - input_min)
    normalized = jnp.where(clip, jnp.clip(normalized, 0.0, 1.0), normalized)
    return output_min + normalized * (output_max - output_min)


def _maybe_scale_range(
    values: jnp.ndarray,
    normalize: bool,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    clip: bool,
) -> jnp.ndarray:
    if isinstance(normalize, bool):
        if not normalize:
            return values
        return _scale_range(values, input_range, output_range, clip)

    return jnp.where(
        normalize,
        _scale_range(values, input_range, output_range, clip),
        values,
    )


def encode_phase(
    values: jnp.ndarray,
    phase_range: tuple[float, float] = (0.0, jnp.pi),
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    phase = _maybe_scale_range(values, normalize, input_range, phase_range, clip)
    return jnp.exp(1j * phase).astype(jnp.complex64)


def encode_amplitude(
    values: jnp.ndarray,
    amplitude_range: tuple[float, float] = (0.0, 1.0),
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    amplitude = _maybe_scale_range(values, normalize, input_range, amplitude_range, clip)
    return amplitude.astype(jnp.complex64)


def encode_complex(
    values: jnp.ndarray,
    phase_range: tuple[float, float] = (0.0, jnp.pi),
    amplitude: float | jnp.ndarray = 1.0,
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = False,
) -> jnp.ndarray:
    phase_field = encode_phase(
        values,
        phase_range=phase_range,
        normalize=normalize,
        input_range=input_range,
        clip=clip,
    )
    return (amplitude * phase_field).astype(jnp.complex64)
