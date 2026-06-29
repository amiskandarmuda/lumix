from __future__ import annotations

import argparse
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.models import (
    build_subunitary_surrogate,
    extract_inverse_design_matrices,
    subunitary_surrogate_config_from_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained subunitary layer matrices for inverse design.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--params", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    model_config = subunitary_surrogate_config_from_mapping(config["model"])
    model = build_subunitary_surrogate(model_config)
    sample_x = jnp.zeros((1, model_config.width), dtype=jnp.float32)
    initialized = model.init(jax.random.key(0), sample_x)["params"]

    run_dir = resolve_config_path(config_path, config["outputs"]["run_dir"])
    params_path = args.params.resolve() if args.params else run_dir / "params.msgpack"
    params = serialization.from_bytes(initialized, params_path.read_bytes())
    matrices = extract_inverse_design_matrices(model, params, sample_x)

    target_dir = run_dir / "inverse_design_targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale_target in target_dir.glob("layer_*.npy"):
        stale_target.unlink()
    payload = {}
    for name, matrix in matrices.items():
        array = jax.device_get(matrix)
        payload[name] = array
        np.save(target_dir / f"layer_{name}.npy", array)
    np.savez_compressed(target_dir / "targets.npz", **payload)
    print(f"saved {len(payload)} target matrices to {target_dir}")


if __name__ == "__main__":
    main()
