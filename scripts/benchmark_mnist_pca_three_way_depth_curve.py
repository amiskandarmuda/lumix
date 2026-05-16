from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


def load_objective_module():
    module_path = Path(__file__).resolve().with_name("benchmark_mnist_pca_objectives.py")
    spec = importlib.util.spec_from_file_location("benchmark_mnist_pca_objectives", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("could not load objective benchmark module")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OBJECTIVES = load_objective_module()

ARTIFACT_DIR = Path("artifacts/mnist_pca_phase_comparison")
RESULTS_PATH = ARTIFACT_DIR / "three_way_depth_curve_trainable_softmax_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "three_way_depth_curve_trainable_softmax_summary.md"


VARIANTS = {
    "Unitary repeated": "repeated_phase",
    "Subunitary repeated": "repeated_subunitary",
    "Williamson": "williamson",
}


def _result_by_name(results: list[dict], label: str, depth: int) -> dict:
    kind = VARIANTS[label]
    for result in results:
        if result["kind"] == kind and result["layers"] == depth:
            return result
    raise KeyError((label, depth))


def write_report(results: list[dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MNIST PCA Three-Way Depth Curve: Trainable-Temperature Softmax",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset/encoding: MNIST projected to 16 PCA components, train-set min/max mapped without clipping, then phase encoded.",
        "",
        f"Training: {OBJECTIVES.EPOCHS} epochs, batch size {OBJECTIVES.BATCH_SIZE}, Adam lr={OBJECTIVES.LEARNING_RATE}.",
        "",
        "Objective: CE on `softmax(gamma * class_intensity)` with one trainable scalar gamma initialized at 10.",
        "",
        "Variants:",
        "",
        "- `Unitary repeated`: repeated phase encoding before each unitary layer.",
        "- `Subunitary repeated`: repeated phase encoding before each subunitary layer with 0-1.5 dB per-layer loss bounds.",
        "- `Williamson`: one phase encoding at input, then unitary plus Williamson response each layer with trainable gain and bias.",
        "",
        "| Depth | Variant | Val Acc | Val Loss | Gamma | Mean IL (dB) | IL Range (dB) | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in range(1, 6):
        for label in VARIANTS:
            result = _result_by_name(results, label, depth)
            lines.append(
                f"| {depth} | {label} | {result['val_accuracy']:.4f} | {result['val_loss']:.4f} | "
                f"{result['final_logit_scale']:.3f} | {result['val_mean_insertion_loss_db']:.3f} | "
                f"{result['val_min_insertion_loss_db']:.3f}-{result['val_max_insertion_loss_db']:.3f} | "
                f"{result['stored_param_count']} |"
            )

    lines.extend(["", "## Best Points", ""])
    for label in VARIANTS:
        kind = VARIANTS[label]
        best = max((result for result in results if result["kind"] == kind), key=lambda result: result["val_accuracy"])
        lines.append(
            f"- {label}: depth {best['layers']}, {100.0 * best['val_accuracy']:.2f}% accuracy, "
            f"{best['val_loss']:.4f} loss, gamma {best['final_logit_scale']:.3f}, "
            f"mean IL {best['val_mean_insertion_loss_db']:.3f} dB."
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = OBJECTIVES.PCA.load_or_create_no_clip_cache()
    configs = []
    for depth in range(1, 6):
        configs.append(
            OBJECTIVES.ObjectiveConfig(
                f"unitary-repeated-depth{depth}-softmax-trainable",
                "repeated_phase",
                depth,
                "softmax_trainable",
            )
        )
        configs.append(
            OBJECTIVES.ObjectiveConfig(
                f"subunitary-repeated-depth{depth}-softmax-trainable",
                "repeated_subunitary",
                depth,
                "softmax_trainable",
            )
        )
        configs.append(
            OBJECTIVES.ObjectiveConfig(
                f"williamson-depth{depth}-softmax-trainable",
                "williamson",
                depth,
                "softmax_trainable",
                train_williamson_gain=True,
                train_williamson_bias=True,
            )
        )

    results = [OBJECTIVES.run_config(config, data) for config in configs]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
