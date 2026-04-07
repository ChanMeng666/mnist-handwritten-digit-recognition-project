"""Comprehensive model evaluation metrics."""

import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def comprehensive_evaluation(model, x_test, y_test, model_name):
    """Perform comprehensive evaluation of a trained model.

    Computes accuracy, prediction timing, confusion matrix, and
    classification report, then visualizes the confusion matrix.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: One-hot test labels.
        model_name: Display name for the model.

    Returns:
        Dictionary with accuracy, prediction_time, confusion_matrix,
        classification_report, predictions, true_labels, and model_size.
    """
    print(f"\nComprehensive Evaluation for {model_name}")
    print("-" * 50)

    start_time = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start_time

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    report = classification_report(y_true_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    evaluation = {
        "accuracy": accuracy,
        "prediction_time": prediction_time,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "predictions": y_pred,
        "true_labels": y_test,
        "model_size": model.count_params(),
    }

    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"Model Size: {model.count_params():,} parameters")
    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return evaluation
