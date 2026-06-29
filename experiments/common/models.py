from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from flax import linen as nn
from flax.core import unfreeze
from flax.linen import nowrap
from jax import random
import jax.numpy as jnp

from lumix.functional.routing import RoutingLimit, routing_leakage, routing_mask
from lumix.functional.readout import intensity
from lumix.functional.subunitary import insertion_loss_bounds, subunitary_linear, subunitary_matrix
from lumix.functional.unitary import combine_complex_parts


@dataclass(frozen=True)
class SubUnitarySurrogateConfig:
    width: int = 16
    layers: int = 3
    loss_db: tuple[float | None, float | None] = (2.0, 3.0)
    classes: int = 10
    routing_limit: RoutingLimit | None = None
    hard_routing: bool = False
    input_amplitude: float = 1.0
    phase_scale: float = math.pi
    nonlinearity: str = "repeated_phase_mask"
    readout: str = "intensity_logits"
    init_gamma: float = 10.0
    layer_name_prefix: str = "optical"
    readout_port_start: int | None = None
    bias_ports: int = 0


class RoutedSubUnitaryLinear(nn.Module):
    """Subunitary layer with optional hard masking of nonlocal routes."""

    width: int
    out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    routing_limit: RoutingLimit | None = None
    hard_routing: bool = False
    init_scale: float = 1e-2
    singular_bias: float = 3.0

    @nowrap
    def _complex_param(self, name: str, size: int) -> jnp.ndarray:
        real = self.param(
            f"{name}_re",
            lambda key: self.init_scale * random.normal(key, (size, size), dtype=jnp.float32),
        )
        imag = self.param(
            f"{name}_im",
            lambda key: self.init_scale * random.normal(key, (size, size), dtype=jnp.float32),
        )
        return combine_complex_parts(real, imag)

    @nn.compact
    def __call__(self, values):
        output_features = self.width if self.out_features is None else self.out_features
        input_features = values.shape[-1]
        rank = min(output_features, input_features)
        singular_raw = self.param(
            "singular_raw",
            lambda key: jnp.full((rank,), self.singular_bias, dtype=jnp.float32),
        )
        matrix = routed_subunitary_matrix(
            self._complex_param("left", output_features),
            self._complex_param("right", input_features),
            singular_raw,
            insertion_loss_db=self.insertion_loss_db,
            output_features=output_features,
            input_features=input_features,
            routing_limit=self.routing_limit,
            hard_routing=self.hard_routing,
        )

        if self.is_mutable_collection("lumix_inverse_design"):
            self.sow("lumix_inverse_design", "matrix", matrix)
        if self.routing_limit is not None and self.is_mutable_collection("metrics"):
            self.sow("metrics", "routing_leakage", routing_leakage(matrix, self.routing_limit))
        return subunitary_linear(values, matrix)


def routed_subunitary_matrix(
    left: jnp.ndarray,
    right: jnp.ndarray,
    singular_raw: jnp.ndarray,
    *,
    insertion_loss_db: float | tuple[float | None, float | None],
    output_features: int,
    input_features: int,
    routing_limit: RoutingLimit | None,
    hard_routing: bool,
) -> jnp.ndarray:
    singular_min, singular_max = insertion_loss_bounds(insertion_loss_db)
    matrix = subunitary_matrix(
        left,
        right,
        singular_raw,
        singular_min,
        singular_max,
        output_features,
        input_features,
    )
    if routing_limit is not None and hard_routing:
        mask = routing_mask(output_features, input_features, routing_limit)
        matrix = matrix * mask.astype(matrix.dtype)
        masked_singular_max = jnp.linalg.svd(matrix, compute_uv=False)[0]
        scale = jnp.minimum(
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(singular_max, dtype=jnp.float32) / jnp.maximum(masked_singular_max, 1e-12),
        )
        matrix = matrix * scale.astype(matrix.dtype)
    return matrix


def _active_layer_count(config: SubUnitarySurrogateConfig, depth: int | None) -> int:
    active_layers = config.layers if depth is None else int(depth)
    if active_layers < 0 or active_layers > config.layers:
        raise ValueError(f"depth must be between 0 and {config.layers}")
    return active_layers


def surrogate_routing_leakage_from_params(
    config: SubUnitarySurrogateConfig,
    params,
    *,
    depth: int | None = None,
) -> jnp.ndarray:
    if config.routing_limit is None:
        return jnp.asarray(0.0, dtype=jnp.float32)

    active_layers = _active_layer_count(config, depth)
    if active_layers == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    leakages = []
    for index in range(active_layers):
        layer_params = params[f"{config.layer_name_prefix}_{index}"]
        matrix = routed_subunitary_matrix(
            combine_complex_parts(layer_params["left_re"], layer_params["left_im"]),
            combine_complex_parts(layer_params["right_re"], layer_params["right_im"]),
            layer_params["singular_raw"],
            insertion_loss_db=config.loss_db,
            output_features=config.width,
            input_features=config.width,
            routing_limit=config.routing_limit,
            hard_routing=config.hard_routing,
        )
        leakages.append(routing_leakage(matrix, config.routing_limit))
    return jnp.mean(jnp.stack(leakages))


class DataRepetitionNonlinearity(nn.Module):
    """Square-law data re-encoding followed by width-preserving repetition."""

    width: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, values):
        intensities = jnp.real(jnp.conj(values) * values)
        normalized = intensities / jnp.sqrt(jnp.mean(jnp.square(intensities), axis=-1, keepdims=True) + self.eps)
        if normalized.shape[-1] < self.width:
            repeats = math.ceil(self.width / normalized.shape[-1])
            normalized = jnp.tile(normalized, (1,) * (normalized.ndim - 1) + (repeats,))
        return normalized[..., : self.width].astype(jnp.complex64)


class TemperatureSoftmaxReadout(nn.Module):
    classes: int = 10
    init_gamma: float = 10.0
    port_start: int | None = None

    @nn.compact
    def __call__(self, intensities):
        log_gamma = self.param(
            "log_gamma",
            lambda key: jnp.asarray(jnp.log(self.init_gamma), dtype=jnp.float32),
        )
        gamma = jnp.exp(log_gamma)
        return gamma * select_readout_ports(intensities, self.classes, self.port_start), gamma


def readout_port_start(width: int, classes: int, configured_start: int | None) -> int:
    if classes > width:
        raise ValueError("classes must be less than or equal to width")
    start = (width - classes) // 2 if configured_start is None else int(configured_start)
    if start < 0 or start + classes > width:
        raise ValueError("readout_port_start selects ports outside the output width")
    return start


def select_readout_ports(values: jnp.ndarray, classes: int, configured_start: int | None) -> jnp.ndarray:
    start = readout_port_start(values.shape[-1], classes, configured_start)
    return values[..., start : start + classes]


class SubUnitarySurrogate(nn.Module):
    config: SubUnitarySurrogateConfig

    def _phase_values(self, values):
        phase_values = jnp.asarray(jnp.real(values), dtype=jnp.float32)
        if self.config.bias_ports == 0:
            return phase_values
        data_width = self.config.width - self.config.bias_ports
        if data_width < 0:
            raise ValueError("bias_ports must be less than or equal to width")
        if phase_values.shape[-1] != data_width:
            raise ValueError("input feature width must equal width - bias_ports")
        bias_phase = jnp.zeros((*phase_values.shape[:-1], self.config.bias_ports), dtype=phase_values.dtype)
        return jnp.concatenate([phase_values, bias_phase], axis=-1)

    def input_fields(self, values):
        return jnp.full(
            (*values.shape[:-1], self.config.width),
            self.config.input_amplitude,
            dtype=jnp.float32,
        ).astype(jnp.complex64)

    def repeated_phase_mask(self, values):
        phase_values = self._phase_values(values)
        return jnp.exp(1j * jnp.asarray(self.config.phase_scale, dtype=jnp.float32) * phase_values).astype(jnp.complex64)

    def phase_mask_for_layer(self, values, layer_index: int):
        if self.config.nonlinearity == "repeated_phase_mask":
            return self.repeated_phase_mask(values)
        if self.config.nonlinearity == "input_phase_once":
            if layer_index == 0:
                return self.repeated_phase_mask(values)
            return jnp.ones((*values.shape[:-1], self.config.width), dtype=jnp.complex64)
        raise ValueError("nonlinearity must be one of 'repeated_phase_mask' or 'input_phase_once'")

    @nn.compact
    def readout_fields(self, fields, return_aux: bool = False, readout_name: str | None = None):
        intensities = intensity(fields)
        selected_readout_name = "readout" if readout_name is None else readout_name
        if self.config.readout == "intensity_logits":
            logits = select_readout_ports(
                intensities,
                self.config.classes,
                self.config.readout_port_start,
            )
            gamma = jnp.asarray(1.0, dtype=jnp.float32)
        elif self.config.readout == "temperature_softmax":
            logits, gamma = TemperatureSoftmaxReadout(
                classes=self.config.classes,
                init_gamma=self.config.init_gamma,
                port_start=self.config.readout_port_start,
                name=selected_readout_name,
            )(intensities)
        else:
            raise ValueError("readout must be one of 'intensity_logits' or 'temperature_softmax'")

        aux = {"intensities": intensities, "gamma": gamma}
        return (logits, aux) if return_aux else logits

    @nn.compact
    def __call__(
        self,
        values,
        return_aux: bool = False,
        depth: int | None = None,
        readout_name: str | None = None,
    ):
        active_layers = _active_layer_count(self.config, depth)
        fields = self.input_fields(values)
        for index in range(active_layers):
            phase_mask = self.phase_mask_for_layer(values, index)
            fields = RoutedSubUnitaryLinear(
                width=self.config.width,
                out_features=self.config.width,
                insertion_loss_db=self.config.loss_db,
                routing_limit=self.config.routing_limit,
                hard_routing=self.config.hard_routing,
                name=f"{self.config.layer_name_prefix}_{index}",
            )(fields * phase_mask)
        return self.readout_fields(fields, return_aux=return_aux, readout_name=readout_name)


def build_subunitary_surrogate(config: SubUnitarySurrogateConfig) -> SubUnitarySurrogate:
    return SubUnitarySurrogate(config=config)


def _routing_limit_from_value(value: object) -> RoutingLimit | None:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("routing_limit list must contain exactly two values")
        return (int(value[0]), int(value[1]))
    return int(value)


def _loss_bound_from_value(value: object) -> float | None:
    return None if value is None else float(value)


def subunitary_surrogate_config_from_mapping(model_config: Mapping[str, object]) -> SubUnitarySurrogateConfig:
    return SubUnitarySurrogateConfig(
        width=int(model_config["width"]),
        layers=int(model_config["layers"]),
        loss_db=tuple(_loss_bound_from_value(value) for value in model_config["loss_db"]),
        classes=int(model_config["classes"]),
        routing_limit=_routing_limit_from_value(model_config.get("routing_limit")),
        hard_routing=bool(model_config.get("hard_routing", False)),
        input_amplitude=float(model_config.get("input_amplitude", 1.0)),
        phase_scale=float(model_config.get("phase_scale", math.pi)),
        nonlinearity=str(model_config.get("nonlinearity", "repeated_phase_mask")),
        readout=str(model_config.get("readout", "intensity_logits")),
        init_gamma=float(model_config.get("init_gamma", 10.0)),
        layer_name_prefix=str(model_config.get("layer_name_prefix", "optical")),
        readout_port_start=_readout_port_start_from_value(model_config.get("readout_port_start")),
        bias_ports=int(model_config.get("bias_ports", 0)),
    )


def _readout_port_start_from_value(value: object) -> int | None:
    if value is None or value == "center":
        return None
    if value == "first":
        return 0
    return int(value)


def _collect_matrices(node: Any, path: tuple[str, ...], output: dict[str, jnp.ndarray]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "matrix":
                matrix = value[-1] if isinstance(value, tuple) else value
                layer_name = next(
                    (
                        part
                        for part in reversed(path)
                        if part.startswith("optical_") or part.startswith("subunitary_")
                    ),
                    ".".join(path),
                )
                output[layer_name] = matrix
            else:
                _collect_matrices(value, (*path, key), output)


def extract_inverse_design_matrices(
    model: nn.Module,
    params,
    sample_x: jnp.ndarray,
    *,
    depth: int | None = None,
    readout_name: str | None = None,
) -> dict[str, jnp.ndarray]:
    _, collections = model.apply(
        {"params": params},
        sample_x,
        depth=depth,
        readout_name=readout_name,
        mutable=["lumix_inverse_design"],
    )
    matrices: dict[str, jnp.ndarray] = {}
    _collect_matrices(unfreeze(collections.get("lumix_inverse_design", {})), (), matrices)
    return dict(sorted(matrices.items()))
