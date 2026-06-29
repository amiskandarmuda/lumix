from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import struct
from urllib.request import urlretrieve

import numpy as np


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}
PCA_PREPROCESSING = "standardize_minmax_no_clip"
PCA_ZSCORE_PREPROCESSING = "pca_zscore"


@dataclass(frozen=True)
class PcaDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_min: np.ndarray
    feature_range: np.ndarray
    explained_variance: np.ndarray
    preprocessing: str = PCA_PREPROCESSING


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    return encoded


def _flatten_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32).reshape(images.shape[0], -1) / 255.0


def fit_pca16_dataset(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    *,
    components: int = 16,
    preprocessing: str = PCA_PREPROCESSING,
) -> PcaDataset:
    train = _flatten_images(train_images)
    test = _flatten_images(test_images)
    if components < 1 or components > train.shape[1]:
        raise ValueError("components must be between 1 and the flattened image width")

    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_centered = train - mean
    test_centered = test - mean

    covariance = (train_centered.T @ train_centered) / train_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    basis = eigenvectors[:, order].T.astype(np.float32)
    explained = eigenvalues[order].astype(np.float32)

    train_scores = train_centered @ basis.T
    test_scores = test_centered @ basis.T
    feature_mean = train_scores.mean(axis=0).astype(np.float32)
    feature_std = train_scores.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

    train_standardized = (train_scores - feature_mean) / feature_std
    test_standardized = (test_scores - feature_mean) / feature_std
    feature_min = train_standardized.min(axis=0).astype(np.float32)
    feature_max = train_standardized.max(axis=0).astype(np.float32)
    feature_range = np.where(feature_max == feature_min, 1.0, feature_max - feature_min).astype(np.float32)
    if preprocessing == PCA_PREPROCESSING:
        x_train = ((train_standardized - feature_min) / feature_range).astype(np.float32)
        x_test = ((test_standardized - feature_min) / feature_range).astype(np.float32)
    elif preprocessing == PCA_ZSCORE_PREPROCESSING:
        x_train = train_standardized.astype(np.float32)
        x_test = test_standardized.astype(np.float32)
    else:
        raise ValueError("preprocessing must be one of 'standardize_minmax_no_clip' or 'pca_zscore'")

    return PcaDataset(
        x_train=x_train,
        y_train=one_hot(train_labels),
        x_test=x_test,
        y_test=one_hot(test_labels),
        mean=mean,
        components=basis,
        scale=feature_std.astype(np.float32),
        feature_mean=feature_mean,
        feature_std=feature_std.astype(np.float32),
        feature_min=feature_min,
        feature_range=feature_range,
        explained_variance=explained,
        preprocessing=preprocessing,
    )


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(url, path)


def download_mnist(raw_dir: Path) -> None:
    for name, url in MNIST_URLS.items():
        _download_file(url, raw_dir / f"{name}.gz")


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_mnist(raw_dir)
    return (
        _read_idx_images(raw_dir / "train_images.gz"),
        _read_idx_labels(raw_dir / "train_labels.gz"),
        _read_idx_images(raw_dir / "test_images.gz"),
        _read_idx_labels(raw_dir / "test_labels.gz"),
    )


def save_pca_dataset(dataset: PcaDataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        mean=dataset.mean,
        components=dataset.components,
        scale=dataset.scale,
        feature_mean=dataset.feature_mean,
        feature_std=dataset.feature_std,
        feature_min=dataset.feature_min,
        feature_range=dataset.feature_range,
        explained_variance=dataset.explained_variance,
        preprocessing=np.asarray(dataset.preprocessing),
    )
    metadata = {
        "train_samples": int(dataset.x_train.shape[0]),
        "test_samples": int(dataset.x_test.shape[0]),
        "components": int(dataset.components.shape[0]),
        "flattened_image_width": int(dataset.mean.shape[0]),
        "preprocessing": dataset.preprocessing,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def _scalar_string(data, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return str(value)


def is_current_pca_dataset(path: Path, *, expected_preprocessing: str = PCA_PREPROCESSING) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return _scalar_string(data, "preprocessing") == expected_preprocessing
    except (OSError, ValueError):
        return False


def load_pca_dataset(path: Path) -> PcaDataset:
    with np.load(path) as data:
        feature_std = data["feature_std"].astype(np.float32) if "feature_std" in data else data["scale"].astype(np.float32)
        feature_count = data["x_train"].shape[-1]
        return PcaDataset(
            x_train=data["x_train"].astype(np.float32),
            y_train=data["y_train"].astype(np.float32),
            x_test=data["x_test"].astype(np.float32),
            y_test=data["y_test"].astype(np.float32),
            mean=data["mean"].astype(np.float32),
            components=data["components"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            feature_mean=(
                data["feature_mean"].astype(np.float32)
                if "feature_mean" in data
                else np.zeros((feature_count,), dtype=np.float32)
            ),
            feature_std=feature_std,
            feature_min=(
                data["feature_min"].astype(np.float32)
                if "feature_min" in data
                else np.zeros((feature_count,), dtype=np.float32)
            ),
            feature_range=(
                data["feature_range"].astype(np.float32)
                if "feature_range" in data
                else np.ones((feature_count,), dtype=np.float32)
            ),
            explained_variance=data["explained_variance"].astype(np.float32),
            preprocessing=_scalar_string(data, "preprocessing", "standardize_only"),
        )
