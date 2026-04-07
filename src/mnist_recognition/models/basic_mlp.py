"""Basic Multi-Layer Perceptron model for MNIST classification."""

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_and_train_basic_model(
    config, train_generator, val_generator, x_val, y_val,
    neuron_counts=None,
):
    """Create and evaluate basic MLP models with different neuron counts.

    Trains one model per neuron count and returns the best performing one.

    Args:
        config: Configuration dictionary with input_shape, num_classes,
            learning_rate, and epochs.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        x_val: Validation images for evaluation.
        y_val: Validation labels for evaluation.
        neuron_counts: List of hidden-layer sizes to try (default [128, 256, 512]).

    Returns:
        Tuple of (best_model, results_dict) where results_dict contains
        training history and configuration details.
    """
    if neuron_counts is None:
        neuron_counts = [128, 256, 512]

    best_model = None
    best_val_acc = 0
    results = {"configurations": []}

    for neurons in neuron_counts:
        print(f"\nTraining basic model with {neurons} neurons...")

        inputs = keras.Input(shape=config["input_shape"])
        x = Flatten()(inputs)
        x = Dense(neurons, activation="relu")(x)
        outputs = Dense(config["num_classes"], activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"basic_model_{neurons}")

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
            train_generator,
            epochs=config["epochs"],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        results["configurations"].append({
            "neurons": neurons,
            "val_accuracy": val_acc,
            "history": history.history,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            results["history"] = history.history
            results["final_accuracy"] = history.history["val_accuracy"][-1]
            results["best_accuracy"] = max(history.history["val_accuracy"])

        print(f"Validation accuracy with {neurons} neurons: {val_acc:.4f}")

    return best_model, results
