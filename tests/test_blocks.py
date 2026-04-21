import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from lumix.linen.blocks import (
    BlockParallelClements,
    BlockParallelLinear,
    BlockParallelSubUnitary,
    BlockParallelUnitary,
)
from lumix.linen.waveguide import FixedWaveguideArray
from lumix.linen.williamson import WilliamsonNonlinearity


def _complex_ones(batch: int, width: int) -> jnp.ndarray:
    return (jnp.ones((batch, width)) + 1j * jnp.ones((batch, width))).astype(jnp.complex64)


def test_block_parallel_unitary_preserves_square_width():
    layer = BlockParallelUnitary(num_blocks=4, block_in_features=4)
    values = _complex_ones(3, 16)

    variables = layer.init(jax.random.key(0), values)
    outputs = layer.apply(variables, values)

    assert outputs.shape == (3, 16)


def test_block_parallel_unitary_supports_rectangular_expansion():
    layer = BlockParallelUnitary(num_blocks=4, block_in_features=4, block_out_features=6)
    values = _complex_ones(3, 16)

    variables = layer.init(jax.random.key(1), values)
    outputs = layer.apply(variables, values)

    assert outputs.shape == (3, 24)


def test_block_parallel_subunitary_supports_rectangular_expansion():
    layer = BlockParallelSubUnitary(
        num_blocks=4,
        block_in_features=4,
        block_out_features=6,
        insertion_loss_db=(0.5, 2.0),
    )
    values = _complex_ones(3, 16)

    variables = layer.init(jax.random.key(2), values)
    outputs = layer.apply(variables, values)

    assert outputs.shape == (3, 24)


def test_block_parallel_clements_preserves_square_width():
    layer = BlockParallelClements(num_blocks=4, block_in_features=4)
    values = _complex_ones(3, 16)

    variables = layer.init(jax.random.key(3), values)
    outputs = layer.apply(variables, values)

    assert outputs.shape == (3, 16)


def test_block_parallel_blocks_have_independent_parameter_subtrees():
    layer = BlockParallelSubUnitary(num_blocks=3, block_in_features=4, block_out_features=5)
    values = _complex_ones(2, 12)

    variables = layer.init(jax.random.key(4), values)
    block_params = variables["params"]["BlockParallelLinear_0"]

    assert set(block_params.keys()) == {"block_0", "block_1", "block_2"}
    assert block_params["block_0"]["left_re"].shape == (5, 5)
    assert block_params["block_1"]["right_re"].shape == (4, 4)


def test_block_parallel_parameter_structure_changes_with_num_blocks():
    two_block_layer = BlockParallelUnitary(num_blocks=2, block_in_features=4)
    three_block_layer = BlockParallelUnitary(num_blocks=3, block_in_features=4)

    two_block_variables = two_block_layer.init(jax.random.key(5), _complex_ones(1, 8))
    three_block_variables = three_block_layer.init(jax.random.key(6), _complex_ones(1, 12))

    assert len(two_block_variables["params"]["BlockParallelLinear_0"]) == 2
    assert len(three_block_variables["params"]["BlockParallelLinear_0"]) == 3


def test_block_parallel_rejects_mismatched_input_width():
    layer = BlockParallelUnitary(num_blocks=4, block_in_features=4)
    values = _complex_ones(2, 15)

    with pytest.raises(ValueError, match="values width must equal num_blocks \\* block_in_features"):
        layer.init(jax.random.key(7), values)


def test_block_parallel_rejects_invalid_feature_configuration():
    layer = BlockParallelLinear(block_type="unitary", num_blocks=0, block_in_features=4)
    values = _complex_ones(1, 4)

    with pytest.raises(ValueError, match="num_blocks must be at least 1"):
        layer.init(jax.random.key(8), values)


def test_block_parallel_clements_rejects_rectangular_configuration():
    layer = BlockParallelClements(num_blocks=2, block_in_features=4, block_out_features=6)
    values = _complex_ones(1, 8)

    with pytest.raises(ValueError, match="Clements blocks must be square"):
        layer.init(jax.random.key(9), values)


def test_block_parallel_subunitary_supports_generator_lift_to_256():
    layer = BlockParallelSubUnitary(num_blocks=8, block_in_features=8, block_out_features=32)
    values = _complex_ones(2, 64)

    variables = layer.init(jax.random.key(10), values)
    outputs = layer.apply(variables, values)

    assert outputs.shape == (2, 256)


def test_block_parallel_composes_with_fixed_waveguide_array():
    delta = tuple(jnp.linspace(-0.3, 0.3, 16, dtype=jnp.float32).tolist())
    kappa = tuple(jnp.linspace(0.1, 0.4, 15, dtype=jnp.float32).tolist())

    class MixedNet(nn.Module):
        @nn.compact
        def __call__(self, values):
            values = BlockParallelSubUnitary(num_blocks=2, block_in_features=8)(values)
            values = FixedWaveguideArray(delta=delta, kappa=kappa, length=0.5)(values)
            return WilliamsonNonlinearity()(values)

    model = MixedNet()
    values = _complex_ones(2, 16)

    variables = model.init(jax.random.key(11), values)
    outputs = model.apply(variables, values)

    assert outputs.shape == (2, 16)


def test_block_parallel_clements_and_unitary_run_on_batched_complex_inputs():
    clements = BlockParallelClements(num_blocks=2, block_in_features=4)
    unitary = BlockParallelUnitary(num_blocks=2, block_in_features=4, block_out_features=6)
    values = _complex_ones(5, 8)

    clements_variables = clements.init(jax.random.key(12), values)
    unitary_variables = unitary.init(jax.random.key(13), values)

    clements_outputs = clements.apply(clements_variables, values)
    unitary_outputs = unitary.apply(unitary_variables, values)

    assert clements_outputs.shape == (5, 8)
    assert unitary_outputs.shape == (5, 12)
