import importlib.util
import json
import sys
from pathlib import Path


def load_runner_module():
    module_path = Path(__file__).parents[1] / "scripts" / "run_imagenette_adaptive_iteration.py"
    spec = importlib.util.spec_from_file_location("run_imagenette_adaptive_iteration", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def architecture_payload(**overrides):
    payload = {
        "name": "p4-subunitary-range",
        "decision": "Test trainable bounded passive loss.",
        "patch_size": 4,
        "patch_stride": 4,
        "layers": 4,
        "channels": 16,
        "pool_grid": 4,
        "block": "subunitary",
        "sharing": "alternating",
        "insertion_loss_db": [0.0, 1.5],
    }
    payload.update(overrides)
    return payload


def test_subunitary_insertion_loss_range_validates_and_serializes_as_json_list():
    runner = load_runner_module()

    architecture = runner.Architecture.from_json(json.dumps(architecture_payload()))

    runner.validate_architecture(architecture)
    serialized = json.loads(json.dumps(architecture.as_dict()))
    assert serialized["insertion_loss_db"] == [0.0, 1.5]


def test_insertion_loss_range_is_rejected_for_non_subunitary_blocks():
    runner = load_runner_module()
    architecture = runner.Architecture.from_json(json.dumps(architecture_payload(block="unitary")))

    try:
        runner.validate_architecture(architecture)
    except ValueError as error:
        assert "ranges are only supported for subunitary" in str(error)
    else:
        raise AssertionError("unitary blocks must reject insertion_loss_db ranges")


def test_subunitary_insertion_loss_range_must_be_ordered_and_nonnegative():
    runner = load_runner_module()
    architecture = runner.Architecture.from_json(
        json.dumps(architecture_payload(insertion_loss_db=[1.5, 0.0]))
    )

    try:
        runner.validate_architecture(architecture)
    except ValueError as error:
        assert "ordered and nonnegative" in str(error)
    else:
        raise AssertionError("subunitary ranges must be ordered")
