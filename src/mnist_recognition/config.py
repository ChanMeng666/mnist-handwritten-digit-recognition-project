"""Default configuration and hyperparameters."""

DEFAULT_CONFIG = {
    "input_shape": (28, 28, 1),
    "batch_size": 64,
    "epochs": 20,
    "num_classes": 10,
    "learning_rate": 0.001,
    "dropout_rate": 0.25,
}


def store_training_results(results, model_name, history):
    """Store training results in the results dictionary.

    Args:
        results: Dictionary to store results in.
        model_name: Key name for the model.
        history: Keras training history object.
    """
    results[model_name]["history"] = history.history
    results[model_name]["final_accuracy"] = history.history["val_accuracy"][-1]
    results[model_name]["best_accuracy"] = max(history.history["val_accuracy"])
