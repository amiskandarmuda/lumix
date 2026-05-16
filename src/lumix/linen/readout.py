import math

from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, Initializer
import jax.numpy as jnp

from lumix.functional.readout import class_logits, class_probs, intensity


class IntensityReadout(nn.Module):
    out_features: int | None = None
    activation: str | None = None
    output_shape: tuple[int, ...] | None = None

    def _validate_activation(self) -> None:
        if self.activation not in {None, "sigmoid", "softmax"}:
            raise ValueError("activation must be one of None, 'sigmoid', or 'softmax'")

    def _validate_out_features(self) -> None:
        if self.out_features is not None and self.out_features < 1:
            raise ValueError("out_features must be at least 1")

    def _apply_activation(self, values):
        if self.activation is None:
            return values
        if self.activation == "sigmoid":
            return nn.sigmoid(values)
        return nn.softmax(values, axis=-1)

    def _reshape_output(self, values):
        if self.output_shape is None:
            return values

        target_size = math.prod(self.output_shape)
        if values.shape[-1] != target_size:
            raise ValueError("output_shape product must match the effective output width")
        return values.reshape(*values.shape[:-1], *self.output_shape)

    @nn.compact
    def __call__(self, values):
        self._validate_activation()
        self._validate_out_features()

        intensities = intensity(values)
        outputs = intensities
        if self.out_features is not None:
            outputs = nn.Dense(features=self.out_features)(outputs)

        outputs = self._apply_activation(outputs)
        return self._reshape_output(outputs)


class ProbabilityReadout(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        return class_probs(intensity(values), self.classes)


class LogitReadout(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        return class_logits(intensity(values), self.classes)


class RidgeReadout(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, values):
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (values.shape[-1], self.features),
            self.param_dtype,
        )
        bias = None
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)

        values, kernel, bias = promote_dtype(values, kernel, bias, dtype=self.dtype)
        outputs = jnp.matmul(values, kernel)
        if bias is not None:
            outputs = outputs + bias
        return outputs
