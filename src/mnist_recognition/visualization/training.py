"""Training history visualization."""

import matplotlib.pyplot as plt


def plot_training_history(results, model_names):
    """Plot accuracy and loss curves for all models.

    Args:
        results: Dictionary keyed by model name, each containing a
            'history' sub-dict with accuracy, val_accuracy, loss,
            and val_loss lists.
        model_names: List of model name keys to plot.
    """
    for model_name in model_names:
        if model_name not in results:
            print(f"Warning: {model_name} not found in results")
            return
        if "history" not in results[model_name]:
            print(f"Warning: No training history found for {model_name}")
            return

    plt.figure(figsize=(15, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    for model_name in model_names:
        try:
            history = results[model_name]["history"]
            plt.plot(history["accuracy"], label=f"{model_name} (train)")
            plt.plot(history["val_accuracy"], label=f"{model_name} (val)")
        except KeyError as e:
            print(f"Error plotting {model_name}: {e}")
            continue
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    for model_name in model_names:
        try:
            history = results[model_name]["history"]
            plt.plot(history["loss"], label=f"{model_name} (train)")
            plt.plot(history["val_loss"], label=f"{model_name} (val)")
        except KeyError as e:
            print(f"Error plotting {model_name}: {e}")
            continue
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
