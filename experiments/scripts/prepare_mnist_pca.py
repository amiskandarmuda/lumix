from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import load_json_config, resolve_config_path
from experiments.common.mnist_pca import fit_pca16_dataset, load_mnist, save_pca_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MNIST reduced to PCA features.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    dataset_config = config["dataset"]
    raw_dir = resolve_config_path(config_path, dataset_config["raw_dir"])
    output_path = resolve_config_path(config_path, dataset_config["processed_path"])

    train_images, train_labels, test_images, test_labels = load_mnist(raw_dir)
    dataset = fit_pca16_dataset(
        train_images,
        train_labels,
        test_images,
        test_labels,
        components=int(dataset_config["components"]),
        preprocessing=str(dataset_config.get("preprocessing", "standardize_minmax_no_clip")),
    )
    save_pca_dataset(dataset, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
