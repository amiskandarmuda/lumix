from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

from lumix.linen.clements import ClementsLinear
from lumix.linen.subunitary import SubUnitaryLinear
from lumix.linen.unitary import UnitaryLinear


BlockType = Literal["unitary", "subunitary", "clements"]


class BlockParallelLinear(nn.Module):
    block_type: BlockType
    num_blocks: int
    block_in_features: int
    block_out_features: int | None = None
    block_kwargs: Mapping[str, Any] | None = None

    def _resolved_block_out_features(self) -> int:
        return self.block_in_features if self.block_out_features is None else self.block_out_features

    def _validated_block_out_features(self) -> int:
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if self.block_in_features < 1:
            raise ValueError("block_in_features must be at least 1")

        block_out_features = self._resolved_block_out_features()
        if block_out_features < 1:
            raise ValueError("block_out_features must be at least 1")

        if self.block_type not in {"unitary", "subunitary", "clements"}:
            raise ValueError(f"unsupported block_type: {self.block_type}")
        if self.block_type == "clements" and block_out_features != self.block_in_features:
            raise ValueError("Clements blocks must be square")
        return block_out_features

    def _block_kwargs(self) -> dict[str, Any]:
        return dict(self.block_kwargs or {})

    def _make_block(self, block_out_features: int, index: int) -> nn.Module:
        kwargs = self._block_kwargs()
        name = f"block_{index}"

        if self.block_type == "unitary":
            return UnitaryLinear(
                width=self.block_in_features,
                out_features=block_out_features,
                name=name,
                **kwargs,
            )
        if self.block_type == "subunitary":
            return SubUnitaryLinear(
                width=self.block_in_features,
                out_features=block_out_features,
                name=name,
                **kwargs,
            )
        return ClementsLinear(
            width=self.block_in_features,
            name=name,
            **kwargs,
        )

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        block_out_features = self._validated_block_out_features()
        expected_input_features = self.num_blocks * self.block_in_features
        if values.shape[-1] != expected_input_features:
            raise ValueError("values width must equal num_blocks * block_in_features")

        leading_shape = values.shape[:-1]
        blocked = values.reshape(*leading_shape, self.num_blocks, self.block_in_features)
        outputs = []
        for block_index in range(self.num_blocks):
            block = self._make_block(block_out_features, block_index)
            outputs.append(block(blocked[..., block_index, :]))
        return jnp.concatenate(outputs, axis=-1)


def _apply_block_parallel(
    values: jnp.ndarray,
    *,
    block_type: BlockType,
    num_blocks: int,
    block_in_features: int,
    block_out_features: int | None = None,
    block_kwargs: Mapping[str, Any] | None = None,
) -> jnp.ndarray:
    return BlockParallelLinear(
        block_type=block_type,
        num_blocks=num_blocks,
        block_in_features=block_in_features,
        block_out_features=block_out_features,
        block_kwargs=block_kwargs,
    )(values)


class BlockParallelUnitary(nn.Module):
    num_blocks: int
    block_in_features: int
    block_out_features: int | None = None
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        return _apply_block_parallel(
            values,
            block_type="unitary",
            num_blocks=self.num_blocks,
            block_in_features=self.block_in_features,
            block_out_features=self.block_out_features,
            block_kwargs={"init_scale": self.init_scale},
        )


class BlockParallelSubUnitary(nn.Module):
    num_blocks: int
    block_in_features: int
    block_out_features: int | None = None
    insertion_loss_db: float | tuple[float | None, float | None] = 0.0
    init_scale: float = 1e-2
    singular_bias: float = 3.0

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        return _apply_block_parallel(
            values,
            block_type="subunitary",
            num_blocks=self.num_blocks,
            block_in_features=self.block_in_features,
            block_out_features=self.block_out_features,
            block_kwargs={
                "insertion_loss_db": self.insertion_loss_db,
                "init_scale": self.init_scale,
                "singular_bias": self.singular_bias,
            },
        )


class BlockParallelClements(nn.Module):
    num_blocks: int
    block_in_features: int
    block_out_features: int | None = None
    depth: int | None = None
    hadamard: bool = False

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        return _apply_block_parallel(
            values,
            block_type="clements",
            num_blocks=self.num_blocks,
            block_in_features=self.block_in_features,
            block_out_features=self.block_out_features,
            block_kwargs={
                "depth": self.depth,
                "hadamard": self.hadamard,
            },
        )
