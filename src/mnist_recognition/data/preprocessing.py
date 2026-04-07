"""Data preprocessing utilities for MNIST."""

import numpy as np
from tensorflow import keras


def preprocess_data(x_train, x_test, y_train, y_test, validation_split=0.1):
    """Preprocess MNIST data for neural network training.

    Normalizes pixel values, reshapes to include channel dimension,
    one-hot encodes labels, and creates a validation split.

    Args:
        x_train: Raw training images.
        x_test: Raw test images.
        y_train: Raw training labels.
        y_test: Raw test labels.
        validation_split: Fraction of training data to use for validation.

    Returns:
        Tuple of ((x_train, y_train), (x_val, y_val), (x_test, y_test)).
    """
    print("Starting data preprocessing...")

    print("\nChecking for missing values...")
    print(f"Training data NaN values: {np.isnan(x_train).any()}")
    print(f"Test data NaN values: {np.isnan(x_test).any()}")

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to include channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # One-hot encode labels
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    # Validation split
    val_size = int(x_train.shape[0] * validation_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print("\nData shapes after preprocessing:")
    print(f"Training data: {x_train.shape}")
    print(f"Validation data: {x_val.shape}")
    print(f"Test data: {x_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
