"""Error analysis and misclassification visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def analyze_errors(model, x_test, y_test, model_name, is_optimized=False):
    """Analyze and visualize model prediction errors.

    Args:
        model: Trained Keras model.
        x_test: Test data (images or reduced features).
        y_test: One-hot test labels.
        model_name: Display name for the model.
        is_optimized: If True, treats x_test as reduced feature vectors
            instead of images.

    Returns:
        Dictionary with error_rate, error_count, and error_matrix.
    """
    print(f"\nAnalyzing prediction errors for {model_name}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    errors = y_pred_classes != y_true_classes
    x_errors = x_test[errors]
    y_pred_errors = y_pred_classes[errors]
    y_true_errors = y_true_classes[errors]

    error_rate = np.mean(errors)
    error_count = np.sum(errors)

    print(f"\nError Analysis Statistics:")
    print(f"Total errors: {error_count}")
    print(f"Error rate: {error_rate:.4f}")

    # Plot misclassified examples
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Misclassified Examples by {model_name}")

    for i in range(min(10, len(x_errors))):
        plt.subplot(2, 5, i + 1)
        if is_optimized:
            plt.plot(x_errors[i])
            plt.title(f"True: {y_true_errors[i]}\nPred: {y_pred_errors[i]}")
        else:
            if len(x_errors[i].shape) == 3:
                plt.imshow(x_errors[i][:, :, 0], cmap="gray")
            else:
                plt.imshow(x_errors[i].reshape(28, 28), cmap="gray")
            plt.title(f"True: {y_true_errors[i]}\nPred: {y_pred_errors[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Error pattern confusion matrix
    error_matrix = confusion_matrix(y_true_errors, y_pred_errors)
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix, annot=True, fmt="d", cmap="Reds")
    plt.title(f"Error Pattern Analysis for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return {
        "error_rate": error_rate,
        "error_count": error_count,
        "error_matrix": error_matrix,
    }
