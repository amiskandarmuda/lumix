import importlib.util
import json
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


def test_phase_offset_is_serialized_and_applied_before_phase_scaling():
    runner = load_runner_module()
    architecture = runner.Architecture.from_json(
        json.dumps(
            {
                "name": "offset-check",
                "decision": "Check centered phase encoding.",
                "patch_size": 4,
                "patch_stride": 4,
                "layers": 1,
                "channels": 16,
                "pool_grid": 4,
                "phase_offset": -0.5,
            }
        )
    )
    assert architecture.as_dict()["phase_offset"] == -0.5

    base_model = runner.RepeatedEncodingPatchEncoder(
        channels=16,
        layers=1,
        phase_scales=(jnp.pi,),
        block="unitary",
        sharing="tied",
        post_encode=False,
        insertion_loss_db=0.0,
        clements_depth=None,
        clements_hadamard=False,
        block_count=None,
    )
    offset_model = runner.RepeatedEncodingPatchEncoder(
        channels=16,
        layers=1,
        phase_scales=(jnp.pi,),
        phase_offset=-0.5,
        block="unitary",
        sharing="tied",
        post_encode=False,
        insertion_loss_db=0.0,
        clements_depth=None,
        clements_hadamard=False,
        block_count=None,
    )
    patches = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32).reshape(1, 2, 16)
    variables = base_model.init(jax.random.key(0), patches)

    expected = base_model.apply(variables, patches - 0.5)
    actual = offset_model.apply(variables, patches)

    assert jnp.allclose(actual, expected, atol=1e-6)
