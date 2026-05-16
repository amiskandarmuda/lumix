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


def test_depth4_palindrome_sharing_uses_two_unitary_modules():
    runner = load_runner_module()
    architecture = runner.Architecture(
        name="p4-palindrome-depth4",
        decision="Use A/B/B/A two-unitary sharing under repeated data encoding.",
        patch_size=4,
        patch_stride=4,
        layers=4,
        channels=16,
        pool_grid=4,
        sharing="palindrome",
    )

    runner.validate_architecture(architecture)
    model = runner.RepeatedEncodingPatchEncoder(
        channels=architecture.channels,
        layers=architecture.layers,
        phase_scales=(jnp.pi,),
        block=architecture.block,
        sharing=architecture.sharing,
        post_encode=architecture.post_encode,
        insertion_loss_db=architecture.insertion_loss_db,
        clements_depth=architecture.clements_depth,
        clements_hadamard=architecture.clements_hadamard,
        block_count=architecture.block_count,
    )

    variables = model.init(jax.random.key(0), jnp.zeros((2, 4, 16), dtype=jnp.float32))

    assert set(variables["params"]) == {"palindrome_a", "palindrome_b"}
