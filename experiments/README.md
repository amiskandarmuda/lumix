# Experiments

This folder separates reusable experiment utilities from concrete study cases.

- `common/`: shared Python helpers for datasets, surrogate models, and exports.
- `datasets/`: dataset-specific preparation folders. Keep raw downloads and generated `.npz` files here.
- `cases/`: concrete experiment cases, organized by dataset and architecture.
- `scripts/`: command-line entry points that read case configs.

## MNIST PCA-16 Default

The remaining case trains a 5-layer surrogate Lumix neural network on MNIST reduced to 16 PCA dimensions. It uses train-set z-score PCA features with repeated phase encoding `exp(1j * 0.5 * z)` and passive open-loss subunitary layers.

```bash
rtk uv run python experiments/scripts/prepare_mnist_pca.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json
rtk uv run python experiments/scripts/train_surrogate.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json
rtk uv run python experiments/scripts/train_shared_prefix_surrogate.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json --max-depth 8 --run-dir runs/shared_prefix_depth_1_8
rtk uv run python experiments/scripts/export_inverse_design_targets.py --config experiments/cases/mnist_pca16/local_routing_5x16_reference_93/config.json
rtk uv run --extra tidy3d python experiments/scripts/build_lumix_3layer_layout.py
```

The layout command emits a PhotonForge-generated CORNERSTONE SOI 220 nm active
module under the case `layouts/3layer_module/` folder, including GDS, OAS,
connectivity JSON, summary JSON, compact-model `circuit_sweep.json`, and a PNG
preview. Pass `--no-circuit-sweep` to skip the local PhotonForge circuit solve.
