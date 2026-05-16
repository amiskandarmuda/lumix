import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


def load_runner_module():
    module_path = Path(__file__).parents[1] / "scripts" / "run_imagenette_adaptive_iteration.py"
    spec = importlib.util.spec_from_file_location("run_imagenette_adaptive_iteration", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stride2_p4_architecture_is_rejected_by_nonoverlap_constraint():
    runner = load_runner_module()
    architecture = runner.Architecture(
        name="p4-stride2-depth4-adaptivepool4",
        decision="Preserve 4x4 optical tokens while adding spatial overlap.",
        patch_size=4,
        patch_stride=2,
        layers=4,
        channels=16,
        pool_grid=4,
    )

    try:
        runner.validate_architecture(architecture)
    except ValueError as error:
        assert "patch strategy must be either 4x4/stride4 or 16x16/stride16" in str(error)
    else:
        raise AssertionError("stride-2 p4 architecture should be rejected")


def test_phase_amplitude_encoding_applies_mild_amplitude_gate():
    runner = load_runner_module()
    model = runner.RepeatedEncodingPatchEncoder(
        channels=16,
        layers=1,
        phase_scales=(jnp.pi,),
        encoding_mode="phase_amplitude",
        amplitude_range=(0.5, 1.0),
        block="unitary",
        sharing="tied",
        post_encode=False,
        insertion_loss_db=0.0,
        clements_depth=None,
        clements_hadamard=False,
        block_count=None,
    )
    patches = jnp.stack(
        [
            jnp.zeros((2, 16), dtype=jnp.float32),
            jnp.ones((2, 16), dtype=jnp.float32),
        ]
    )

    variables = model.init(jax.random.key(0), patches)
    intensities = model.apply(variables, patches)

    total_power = jnp.sum(intensities, axis=-1)
    assert jnp.allclose(total_power[0], 0.25, atol=1e-5)
    assert jnp.allclose(total_power[1], 1.0, atol=1e-5)
