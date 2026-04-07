"""Optimized model with feature selection for MNIST classification."""

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from mnist_recognition.data.feature_selection import calculate_feature_importance


def create_and_train_optimized_model(config, x_train, y_train, x_val, y_val, x_test):
    """Create and train an optimized model using feature selection.

    Reduces input dimensionality from 784 to 196 features using ANOVA
    F-value selection, then trains a compact MLP.

    Args:
        config: Configuration dictionary.
        x_train: Training images (N, 28, 28, 1).
        y_train: One-hot training labels.
        x_val: Validation images.
        y_val: One-hot validation labels.
        x_test: Test images.

    Returns:
        Tuple of (model, reduced_data, results_dict) where reduced_data is
        (x_train_reduced, x_val_reduced, x_test_reduced).
    """
    print("\nCreating optimized model with selected features...")

    importance_map, important_pixels = calculate_feature_importance(
        x_train, y_train, n_features=196
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(importance_map, cmap="hot")
    plt.title("Feature Importance Heatmap")
    plt.colorbar()
    plt.show()

    # Create reduced datasets
    x_train_reduced = x_train.reshape(x_train.shape[0], -1)[:, important_pixels]
    x_val_reduced = x_val.reshape(x_val.shape[0], -1)[:, important_pixels]
    x_test_reduced = x_test.reshape(x_test.shape[0], -1)[:, important_pixels]

    print(f"\nReduced feature dimensionality:")
    print(f"Training data: {x_train.shape} -> {x_train_reduced.shape}")
    print(f"Validation data: {x_val.shape} -> {x_val_reduced.shape}")
    print(f"Test data: {x_test.shape} -> {x_test_reduced.shape}")

    # Build model
    inputs = keras.Input(shape=(196,))
    x = Dense(256, activation="relu")(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(config["num_classes"], activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="optimized_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train_reduced, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(x_val_reduced, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    results = {
        "history": history.history,
        "final_accuracy": history.history["val_accuracy"][-1],
        "best_accuracy": max(history.history["val_accuracy"]),
        "selected_features": important_pixels,
        "importance_map": importance_map,
    }

    reduced_data = (x_train_reduced, x_val_reduced, x_test_reduced)
    return model, reduced_data, results
