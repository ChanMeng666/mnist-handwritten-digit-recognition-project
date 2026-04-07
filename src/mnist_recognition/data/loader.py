"""MNIST dataset loading and exploration utilities."""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from sklearn.feature_selection import SelectKBest, f_classif


def load_mnist():
    """Load the MNIST dataset and print an overview.

    Returns:
        Tuple of (x_train, y_train), (x_test, y_test) as raw numpy arrays.
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    print("\nDataset Overview:")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image dimensions: {x_train.shape[1]}x{x_train.shape[2]}")

    print("\nDigit Distribution in Training Set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count}")

    return (x_train, y_train), (x_test, y_test)


def show_sample_images(x_train, y_train, n=9):
    """Display a grid of sample images from the dataset.

    Args:
        x_train: Training images array.
        y_train: Training labels array.
        n: Number of images to display (default 9).
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def explore_dataset(x_train, y_train):
    """Perform pixel intensity analysis and feature importance visualization.

    Args:
        x_train: Raw training images (N, 28, 28).
        y_train: Raw training labels.
    """
    plt.figure(figsize=(12, 4))

    # Histogram of pixel intensities
    plt.subplot(1, 2, 1)
    plt.hist(x_train.ravel(), bins=50)
    plt.title("Distribution of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Average image across all samples
    plt.subplot(1, 2, 2)
    mean_image = np.mean(x_train, axis=0)
    plt.imshow(mean_image, cmap="gray")
    plt.title("Average Digit Image")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Feature importance analysis
    X_flat = x_train.reshape(x_train.shape[0], -1)
    feature_variance = np.var(X_flat, axis=0)
    non_constant_features = feature_variance > 0
    X_flat_filtered = X_flat[:, non_constant_features]

    print(f"Original features: {X_flat.shape[1]}")
    print(f"Features after removing constants: {X_flat_filtered.shape[1]}")

    n_features = min(196, X_flat_filtered.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_features)
    y_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    selector.fit(X_flat_filtered, y_labels)

    importance_map = np.zeros((28, 28))
    non_constant_indices = np.where(non_constant_features)[0]
    importance_map.ravel()[non_constant_indices] = selector.scores_

    importance_map = np.nan_to_num(importance_map)
    importance_map = (importance_map - np.min(importance_map)) / (
        np.max(importance_map) - np.min(importance_map)
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(importance_map, cmap="hot")
    plt.title("Pixel Importance Heatmap")
    plt.colorbar()
    plt.show()
