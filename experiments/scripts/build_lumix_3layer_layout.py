from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.photonforge_lumix_layout import (
    LumixLayoutConfig,
    build_lumix_module,
    write_lumix_layout_artifacts,
)


DEFAULT_CASE_CONFIG = (
    PROJECT_ROOT
    / "experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 3-layer Lumix PhotonForge circuit layout.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-circuit-sweep", action="store_true")
    parser.add_argument("--inverse-design-targets", type=Path, default=None)
    return parser.parse_args()


def _load_inverse_design_targets(config_path: Path, target_path: Path | None, layers: int):
    if target_path is None:
        config = load_json_config(config_path)
        run_dir = resolve_config_path(config_path, config["outputs"]["run_dir"])
        target_path = run_dir / "inverse_design_targets" / "targets.npz"
    if not target_path.exists():
        return None
    data = np.load(target_path)
    keys = sorted(data.files)
    if len(keys) < layers:
        raise ValueError(f"Expected at least {layers} matrices in {target_path}, found {len(keys)}.")
    return [np.asarray(data[key], dtype=np.complex128) for key in keys[:layers]]


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config_path.parent / "layouts" / f"{args.layers}layer_module"
    output_dir = output_dir.resolve()

    layout = build_lumix_module(LumixLayoutConfig(width=args.width, layers=args.layers))
    inverse_design_matrices = (
        None
        if args.no_circuit_sweep
        else _load_inverse_design_targets(config_path, args.inverse_design_targets, args.layers)
    )
    artifacts = write_lumix_layout_artifacts(
        layout,
        output_dir,
        write_preview=not args.no_preview,
        write_circuit_sweep=not args.no_circuit_sweep,
        inverse_design_matrices=inverse_design_matrices,
    )

    print(f"wrote layout artifacts to {artifacts.output_dir}")
    print(f"gds: {artifacts.gds_path}")
    print(f"oas: {artifacts.oas_path}")
    print(f"summary: {artifacts.summary_path}")
    print(f"connectivity: {artifacts.connectivity_path}")
    if artifacts.circuit_sweep_path is not None:
        print(f"circuit sweep: {artifacts.circuit_sweep_path}")
    if artifacts.preview_path is not None:
        print(f"preview: {artifacts.preview_path}")
    print(
        "summary: "
        f"{layout.summary['layers']} layers, "
        f"{layout.summary['width']} channels, "
        f"{layout.summary['phase_modulators']} modulators, "
        f"{layout.summary['inverse_design_regions']} passive ID regions, "
        f"{layout.summary['optical_routes']} optical routes"
    )


if __name__ == "__main__":
    main()
