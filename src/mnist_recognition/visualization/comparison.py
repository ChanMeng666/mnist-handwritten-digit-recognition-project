"""Model comparison visualizations."""

import matplotlib.pyplot as plt
import pandas as pd


def visualize_model_comparison(results):
    """Create bar charts comparing accuracy, speed, and size across models.

    Args:
        results: Dictionary keyed by model name, each containing an
            'evaluation' sub-dict with accuracy, prediction_time,
            and model_size.
    """
    model_names = ["basic_model", "optimized_model", "deep_model"]

    metrics = {
        "Accuracy": [results[m]["evaluation"]["accuracy"] for m in model_names],
        "Prediction Time": [
            results[m]["evaluation"]["prediction_time"] for m in model_names
        ],
        "Model Size": [results[m]["evaluation"]["model_size"] for m in model_names],
    }

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.bar(model_names, metrics["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    plt.subplot(132)
    plt.bar(model_names, metrics["Prediction Time"])
    plt.title("Prediction Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)

    plt.subplot(133)
    plt.bar(model_names, metrics["Model Size"])
    plt.title("Model Size Comparison")
    plt.ylabel("Number of Parameters")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame({
        "Model": model_names,
        "Accuracy": metrics["Accuracy"],
        "Prediction Time (s)": metrics["Prediction Time"],
        "Model Size": metrics["Model Size"],
    })

    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
