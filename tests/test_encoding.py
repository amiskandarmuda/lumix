from jax import config, jit, random
import jax.numpy as jnp
import pytest

from lumix.functional.encoding import encode_amplitude, encode_complex, encode_phase
from lumix.linen.encoding import InformationEncoder


def test_public_encoders_reject_matching_input_range_endpoints():
    with pytest.raises(ValueError, match="input_range endpoints must be different"):
        encode_phase(jnp.array([0.0]), normalize=True, input_range=(1.0, 1.0))

    with pytest.raises(ValueError, match="input_range endpoints must be different"):
        encode_amplitude(jnp.array([0.0]), normalize=True, input_range=(1.0, 1.0))


def test_encode_phase_ignores_input_range_when_not_normalizing():
    phases = jnp.array([0.0, jnp.pi])

    encoded = encode_phase(phases, normalize=False, input_range=(1.0, 1.0))

    assert jnp.allclose(encoded, jnp.exp(1j * phases).astype(jnp.complex64))


def test_encode_phase_returns_unit_magnitude_complex_field():
    phases = jnp.array([0.0, jnp.pi / 2.0, jnp.pi])

    encoded = encode_phase(phases)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(jnp.abs(encoded), jnp.ones_like(phases))
    assert jnp.allclose(encoded, jnp.exp(1j * phases).astype(jnp.complex64))


def test_encode_phase_normalizes_configured_input_range():
    values = jnp.array([0.0, 0.5, 1.0])

    encoded = encode_phase(
        values,
        normalize=True,
        input_range=(0.0, 1.0),
        phase_range=(0.0, jnp.pi),
    )

    assert jnp.allclose(encoded, jnp.exp(1j * jnp.array([0.0, jnp.pi / 2.0, jnp.pi])).astype(jnp.complex64))


def test_encode_amplitude_returns_complex_amplitude_field():
    amplitudes = jnp.array([0.0, 0.5, 1.0])

    encoded = encode_amplitude(amplitudes)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(encoded, amplitudes.astype(jnp.complex64))


def test_encode_amplitude_can_clip_normalized_values():
    values = jnp.array([-1.0, 0.5, 2.0])

    encoded = encode_amplitude(
        values,
        normalize=True,
        input_range=(0.0, 1.0),
        amplitude_range=(0.0, 1.0),
        clip=True,
    )

    assert jnp.allclose(encoded, jnp.array([0.0, 0.5, 1.0], dtype=jnp.complex64))


def test_encode_amplitude_accepts_traced_normalize_and_clip_controls():
    values = jnp.array([-1.0, 0.5, 2.0])

    def encode_with_dynamic_controls(values, normalize, clip):
        return encode_amplitude(values, normalize=normalize, clip=clip)

    encoded = jit(encode_with_dynamic_controls)(values, jnp.array(True), jnp.array(True))

    assert jnp.allclose(encoded, jnp.array([0.0, 0.5, 1.0], dtype=jnp.complex64))


def test_encode_complex_uses_amplitude_and_phase():
    phases = jnp.array([0.0, jnp.pi])

    encoded = encode_complex(phases, amplitude=0.5)

    assert encoded.dtype == jnp.complex64
    assert jnp.allclose(encoded, 0.5 * jnp.exp(1j * phases).astype(jnp.complex64))


def test_encode_complex_returns_complex64_with_float64_amplitude():
    config.update("jax_enable_x64", True)
    phases = jnp.array([0.0, jnp.pi])
    amplitude = jnp.array([0.5, 0.25], dtype=jnp.float64)

    encoded = encode_complex(phases, amplitude=amplitude)

    assert encoded.dtype == jnp.complex64


def test_functional_encoding_exports():
    from lumix.functional import encode_amplitude as exported_amplitude
    from lumix.functional import encode_complex as exported_complex
    from lumix.functional import encode_phase as exported_phase

    assert exported_phase is encode_phase
    assert exported_amplitude is encode_amplitude
    assert exported_complex is encode_complex


def test_information_encoder_has_no_params():
    encoder = InformationEncoder(mode="phase")
    variables = encoder.init(random.key(0), jnp.array([0.0, jnp.pi]))

    assert variables == {}


def test_information_encoder_phase_mode_matches_functional_encoding():
    values = jnp.array([0.0, 1.0])
    encoder = InformationEncoder(
        mode="phase",
        normalize=True,
        input_range=(0.0, 1.0),
        phase_range=(0.0, jnp.pi),
    )

    encoded = encoder.apply({}, values)

    assert jnp.allclose(
        encoded,
        encode_phase(values, normalize=True, input_range=(0.0, 1.0), phase_range=(0.0, jnp.pi)),
    )


def test_information_encoder_amplitude_mode_matches_functional_encoding():
    values = jnp.array([0.0, 1.0])
    encoder = InformationEncoder(mode="amplitude")

    encoded = encoder.apply({}, values)

    assert jnp.allclose(encoded, encode_amplitude(values))


def test_information_encoder_complex_mode_matches_functional_encoding():
    values = jnp.array([0.0, jnp.pi])
    encoder = InformationEncoder(mode="complex", amplitude=0.25)

    encoded = encoder.apply({}, values)

    assert jnp.allclose(encoded, encode_complex(values, amplitude=0.25))


def test_information_encoder_rejects_unknown_mode():
    encoder = InformationEncoder(mode="unknown")

    with pytest.raises(ValueError) as error:
        encoder.apply({}, jnp.array([0.0]))
    assert error.value.args[0] == "mode must be one of 'phase', 'amplitude', or 'complex'"


def test_linen_encoding_exports():
    from lumix.linen import InformationEncoder as exported_encoder

    assert exported_encoder is InformationEncoder
