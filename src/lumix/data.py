from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def normalize_features(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.complex64, copy=True)
    scale = np.mean(np.abs(values), axis=0, keepdims=True)
    scale = np.where(scale == 0, 1, scale)
    return values / scale


def crop_spectrum(array: np.ndarray, radius: int) -> np.ndarray:
    center = array.shape[-1] // 2
    start = center - radius
    stop = center + radius
    return array[:, start:stop, start:stop]


def load_mnist_fourier(radius: int = 2) -> DataSplit:
    from tensorflow.keras.datasets import mnist

    (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()
    x_train_fft = np.fft.fftshift(np.fft.fft2(x_train_raw), axes=(1, 2))
    x_test_fft = np.fft.fftshift(np.fft.fft2(x_test_raw), axes=(1, 2))

    x_train = crop_spectrum(x_train_fft, radius).reshape(x_train_fft.shape[0], -1)
    x_test = crop_spectrum(x_test_fft, radius).reshape(x_test_fft.shape[0], -1)

    return DataSplit(
        x_train=normalize_features(x_train).astype(np.complex64),
        y_train=np.eye(10, dtype=np.float32)[y_train],
        x_test=normalize_features(x_test).astype(np.complex64),
        y_test=np.eye(10, dtype=np.float32)[y_test],
    )
