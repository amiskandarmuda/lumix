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
RESULTS_PATH = ARTIFACT_DIR / "depth_curve_trainable_softmax_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "depth_curve_trainable_softmax_summary.md"


def write_report(results: list[dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MNIST PCA Depth Curve: Trainable-Temperature Softmax",
        "",
        f"Timestamp: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "Dataset/encoding: MNIST projected to 16 PCA components, train-set min/max mapped without clipping, then phase encoded.",
        "",
        f"Training: {OBJECTIVES.EPOCHS} epochs, batch size {OBJECTIVES.BATCH_SIZE}, Adam lr={OBJECTIVES.LEARNING_RATE}.",
        "",
        "Objective: CE on `softmax(gamma * class_intensity)` with one trainable scalar gamma initialized at 10.",
        "",
        "| Depth | Repeated Acc | Repeated Loss | Repeated Gamma | Repeated IL (dB) | Williamson Acc | Williamson Loss | Williamson Gamma | Williamson IL (dB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(result["kind"], result["layers"]): result for result in results}
    for depth in range(1, 6):
        repeated = by_key[("repeated_final_subunitary_trainable_phase", depth)]
        williamson = by_key[("williamson", depth)]
        lines.append(
            f"| {depth} | {repeated['val_accuracy']:.4f} | {repeated['val_loss']:.4f} | "
            f"{repeated['final_logit_scale']:.3f} | {repeated['val_mean_insertion_loss_db']:.3f} | "
            f"{williamson['val_accuracy']:.4f} | {williamson['val_loss']:.4f} | "
            f"{williamson['final_logit_scale']:.3f} | {williamson['val_mean_insertion_loss_db']:.3f} |"
        )

    repeated_best = max(
        (result for result in results if result["kind"] == "repeated_final_subunitary_trainable_phase"),
        key=lambda result: result["val_accuracy"],
    )
    williamson_best = max(
        (result for result in results if result["kind"] == "williamson"),
        key=lambda result: result["val_accuracy"],
    )
    lines.extend(
        [
            "",
            "## Best Points",
            "",
            f"- Repeated: depth {repeated_best['layers']}, {100.0 * repeated_best['val_accuracy']:.2f}% accuracy, {repeated_best['val_loss']:.4f} loss, gamma {repeated_best['final_logit_scale']:.3f}, mean IL {repeated_best['val_mean_insertion_loss_db']:.3f} dB.",
            f"- Williamson: depth {williamson_best['layers']}, {100.0 * williamson_best['val_accuracy']:.2f}% accuracy, {williamson_best['val_loss']:.4f} loss, gamma {williamson_best['final_logit_scale']:.3f}, mean IL {williamson_best['val_mean_insertion_loss_db']:.3f} dB.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = OBJECTIVES.PCA.load_or_create_no_clip_cache()
    configs = []
    for depth in range(1, 6):
        configs.append(
            OBJECTIVES.ObjectiveConfig(
                f"repeated-final-subunitary-trainable-phase-depth{depth}-softmax-trainable",
                "repeated_final_subunitary_trainable_phase",
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
