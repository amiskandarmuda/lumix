# MNIST PCA-16

This dataset reduces flattened MNIST images from 784 pixels to 16 PCA features. The default experiment uses train-set z-score PCA features directly for phase encoding. Labels are stored as 10-way one-hot vectors.

Expected generated files:

- `raw/*.gz`: downloaded MNIST IDX files.
- `processed/mnist_pca16_zscore.npz`: `x_train`, `y_train`, `x_test`, `y_test`, PCA mean, basis, feature standardization/min-max stats, explained variance, and preprocessing metadata.
- `processed/mnist_pca16_zscore.json`: shape and preprocessing metadata.
