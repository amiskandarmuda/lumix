# MNIST PCA-16 Z-Score Default

This case is the default MNIST PCA-16 repeated-phase surrogate experiment.

Architecture:

- Input: 16 PCA features standardized with train-set mean and standard deviation.
- Field initialization: constant amplitude `sqrt(1 / 16) = 0.25` on each input port.
- Data re-encoding: multiply by the same `exp(1j * 0.5 * z)` phase mask before every optical layer.
- Optical stack: 5 passive subunitary 16x16 layers with `insertion_loss_db=(0.0, None)`.
- Layer naming: `subunitary_0` through `subunitary_4`.
- Routing: `routing_limit=4` with 10% fractional leakage target.
- Readout: trainable-temperature intensity logits over the center 10 ports (`3..12` for width 16).
- Objective: cross entropy plus `0.5 * max(leakage - 0.10, 0)`.
- Checkpointing: save the best validation-accuracy checkpoint among epochs `1, 25, 50, 100, 150, 200, 250, 300`.

Run from the repository root:

```bash
rtk uv run python experiments/scripts/train_surrogate.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json
```

To train one reusable 8-layer stack and evaluate shared prefixes from depths 1-8:

```bash
rtk uv run python experiments/scripts/train_shared_prefix_surrogate.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json --max-depth 8 --run-dir runs/shared_prefix_depth_1_8
```

This writes `prefix_metrics.csv`, `per_layer_insertion_loss.csv`, shared inverse-design targets, and plasma matrix heatmaps in the run directory.

The tuned shared-prefix setting that reaches at least 94% at depths 4 and 5 uses a depth-focused objective and looser soft routing:

```bash
rtk uv run python experiments/scripts/train_shared_prefix_surrogate.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json --epochs 139 --max-depth 8 --prefix-weights 0.0,0.0,0.05,0.5,1.0,0.2,0.2,0.2 --selection-metric prefix_5_val_accuracy --routing-limit 7 --routing-weight 0.1 --routing-target 0.2 --run-dir runs/shared_prefix_max8_focus45_r7_loose_e139_bestd5
```
