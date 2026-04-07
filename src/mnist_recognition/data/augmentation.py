"""Data augmentation utilities for MNIST training."""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_data_augmentation():
    """Create a data augmentation pipeline for training.

    Returns:
        A Keras Sequential model that applies random transformations.
    """
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    return data_augmentation


def visualize_augmentation(data_augmentation, x_train, n=9):
    """Display augmented versions of a single training image.

    Args:
        data_augmentation: Keras augmentation pipeline.
        x_train: Training images (N, 28, 28, 1).
        n: Number of augmented samples to show.
    """
    plt.figure(figsize=(10, 10))
    for i in range(n):
        augmented_image = data_augmentation(x_train[0:1])
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle("Augmented Image Examples")
    plt.show()


def create_data_generators(x_train, y_train, x_val, y_val, batch_size):
    """Create data generators for training and validation.

    Args:
        x_train: Training images.
        y_train: Training labels.
        x_val: Validation images.
        y_val: Validation labels.
        batch_size: Batch size for generators.

    Returns:
        Tuple of (train_generator, val_generator).
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_generator = train_datagen.flow(
        x_train, y_train, batch_size=batch_size
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow(
        x_val, y_val, batch_size=batch_size
    )

    return train_generator, val_generator
