"""Deep Convolutional Neural Network model for MNIST classification."""

from tensorflow import keras
from tensorflow.keras import layers


def create_and_train_deep_model(config, train_generator, val_generator):
    """Create and train a deep CNN with three convolutional blocks.

    Architecture: 3 x (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm ->
    ReLU -> MaxPool -> Dropout) followed by dense layers.

    Args:
        config: Configuration dictionary with input_shape, num_classes,
            learning_rate, and epochs.
        train_generator: Training data generator.
        val_generator: Validation data generator.

    Returns:
        Tuple of (model, results_dict).
    """
    print("\nCreating and training deep neural network model...")

    inputs = layers.Input(shape=config["input_shape"])

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(config["num_classes"], activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="deep_model")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            clipnorm=1.0,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        epochs=config["epochs"],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
    )

    results = {
        "history": history.history,
        "final_accuracy": history.history["val_accuracy"][-1],
        "best_accuracy": max(history.history["val_accuracy"]),
    }

    return model, results
