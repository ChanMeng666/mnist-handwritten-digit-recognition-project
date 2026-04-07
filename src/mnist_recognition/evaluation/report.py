"""Final report generation."""

import pandas as pd


def generate_final_report(results, config):
    """Generate a comprehensive summary report of all models.

    Args:
        results: Dictionary keyed by model name, each containing an
            'evaluation' sub-dict with accuracy, prediction_time, and
            model_size, plus optionally a 'sensitivity_report'.
        config: Configuration dictionary used for training.

    Returns:
        A pandas DataFrame comparing the models.
    """
    model_names = ["basic_model", "optimized_model", "deep_model"]

    print("\nFinal Report: MNIST Handwritten Digit Recognition")
    print("=" * 80)

    # Model architecture summary
    print("\n1. Model Architectures:")
    for model_name in model_names:
        print(f"\n{model_name.upper()}:")
        print("-" * 40)
        if model_name in results:
            eval_data = results[model_name]["evaluation"]
            print(f"Parameters: {eval_data['model_size']:,}")
            print(f"Accuracy: {eval_data['accuracy']:.4f}")
            print(f"Prediction Time: {eval_data['prediction_time']:.4f} seconds")

    # Performance comparison
    print("\n2. Performance Comparison:")
    print("-" * 40)
    comparison_df = pd.DataFrame({
        "Model": ["Basic", "Optimized", "Deep"],
        "Accuracy": [results[m]["evaluation"]["accuracy"] for m in model_names],
        "Prediction Time (s)": [
            results[m]["evaluation"]["prediction_time"] for m in model_names
        ],
        "Model Size": [results[m]["evaluation"]["model_size"] for m in model_names],
    })
    print(comparison_df.to_string(index=False))

    # Key findings
    print("\n3. Key Findings:")
    print("-" * 40)
    best_model = max(
        model_names, key=lambda x: results[x]["evaluation"]["accuracy"]
    )
    print(f"Best performing model: {best_model}")
    print(f"Best accuracy achieved: {results[best_model]['evaluation']['accuracy']:.4f}")

    # Sensitivity analysis summary (if available)
    if all("sensitivity_report" in results.get(m, {}) for m in model_names):
        print("\n4. Sensitivity Analysis Summary:")
        print("-" * 40)
        for model_name in model_names:
            print(f"\n{model_name.upper()}:")
            report = results[model_name]["sensitivity_report"]
            print(
                f"Gradient Importance Mean: "
                f"{report['gradient_importance_stats']['mean']:.4f}"
            )
            print(
                f"Perturbation Robustness: "
                f"{report['perturbation_robustness']['accuracy_drop_rate']:.4f}"
            )
            print(
                f"Occlusion Sensitivity Mean: "
                f"{report['occlusion_sensitivity']['mean']:.4f}"
            )

    # Recommendations
    print("\n5. Recommendations:")
    print("-" * 40)
    print("Based on the analysis, we recommend:")
    if results["deep_model"]["evaluation"]["accuracy"] > 0.99:
        print("- Using the deep model for highest accuracy")
    fastest = min(
        model_names, key=lambda m: results[m]["evaluation"]["prediction_time"]
    )
    if fastest == "optimized_model":
        print("- Using the optimized model for fastest prediction time")
    print(
        "- Consider model size vs. performance tradeoffs based on "
        "deployment constraints"
    )

    return comparison_df
