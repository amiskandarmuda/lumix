import jax
import jax.numpy as jnp


def electro_optic_phase_parameters(
    tap: float | jnp.ndarray = 0.1,
    responsivity: float | jnp.ndarray = 0.8,
    area: float | jnp.ndarray = 1.0,
    v_pi: float | jnp.ndarray = 10.0,
    v_bias: float | jnp.ndarray = 10.0,
    resistance: float | jnp.ndarray = 1e3,
    impedance: float | jnp.ndarray = 120.0 * jnp.pi,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return Neuroptica electro-optic activation phase gain and bias.

    This maps the physical parameters from fancompute/neuroptica's
    ``ElectroOpticActivation`` to the direct Williamson response parameters:
    ``gain`` and ``bias``.
    """

    gain = jnp.pi * tap * resistance * responsivity * area * 1e-12 / (2.0 * v_pi * impedance)
    bias = jnp.pi * v_bias / v_pi
    return jnp.asarray(gain, dtype=jnp.float32), jnp.asarray(bias, dtype=jnp.float32)


@jax.custom_jvp
def williamson_response(
    values: jnp.ndarray,
    gain: jnp.ndarray,
    bias: jnp.ndarray,
    tap: float,
) -> jnp.ndarray:
    power = values.real * values.real + values.imag * values.imag
    phase = gain * power + bias
    scale = 0.5j * jnp.sqrt(1.0 - tap)
    return scale * (1.0 + jnp.exp(-1j * phase)) * values


@williamson_response.defjvp
def williamson_response_jvp(primals, tangents):
    values, gain, bias, tap = primals
    dvalues, dgain, dbias, dtap = tangents

    power = values.real * values.real + values.imag * values.imag
    phase = gain * power + bias
    exp_term = jnp.exp(-1j * phase)
    head = 1.0 + exp_term
    sqrt_term = jnp.sqrt(1.0 - tap)
    scale = 0.5j * sqrt_term
    primal_out = scale * head * values

    dpower = 2.0 * jnp.real(jnp.conj(values) * dvalues)
    dphase = gain * dpower + power * dgain + dbias
    dscale = -0.25j * dtap / sqrt_term
    tangent_out = dscale * head * values + scale * (head * dvalues - 1j * exp_term * dphase * values)
    return primal_out, tangent_out
