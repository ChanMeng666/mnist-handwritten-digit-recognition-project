"""Sensitivity and robustness analysis visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def perform_occlusion_analysis(model, x_samples, y_samples, patch_size=3, is_optimized=False):
    """Perform occlusion sensitivity analysis.

    For CNN models, slides an occlusion patch across the image.
    For the optimized model, masks groups of features.

    Args:
        model: Trained Keras model.
        x_samples: Input samples.
        y_samples: Corresponding labels.
        patch_size: Size of the occlusion patch (for CNN models).
        is_optimized: If True, uses feature-masking instead of spatial patches.

    Returns:
        Sensitivity map as a numpy array.
    """
    if is_optimized:
        n_features = x_samples.shape[1]
        sensitivity_map = np.zeros(n_features)

        patch_size = min(10, n_features // 10)
        for i in range(0, n_features, patch_size):
            end_idx = min(i + patch_size, n_features)
            occluded = x_samples.copy()
            occluded[:, i:end_idx] = 0

            original_pred = model.predict(x_samples, verbose=0)
            occluded_pred = model.predict(occluded, verbose=0)
            sensitivity = np.mean(np.abs(original_pred - occluded_pred))
            sensitivity_map[i:end_idx] = sensitivity

        return sensitivity_map.reshape(-1, 1)
    else:
        sensitivity_map = np.zeros((28, 28))

        for i in range(0, 28 - patch_size + 1, 2):
            for j in range(0, 28 - patch_size + 1, 2):
                occluded = x_samples.copy()
                occluded[:, i:i + patch_size, j:j + patch_size, :] = 0

                original_pred = model.predict(x_samples, verbose=0)
                occluded_pred = model.predict(occluded, verbose=0)
                sensitivity = np.mean(np.abs(original_pred - occluded_pred))
                sensitivity_map[i:i + patch_size, j:j + patch_size] = sensitivity

        return sensitivity_map


def calculate_gradient_importance(model, x_data):
    """Calculate gradient-based feature importance using GradientTape.

    Args:
        model: Trained Keras model.
        x_data: Input samples.

    Returns:
        Importance map averaged over samples.
    """
    x_tensor = tf.cast(x_data, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor, training=False)
        max_class = tf.reduce_max(predictions, axis=1)
    gradients = tape.gradient(max_class, x_tensor)
    importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
    return importance.squeeze()


def perform_perturbation_analysis(model, x_data, y_data):
    """Measure model robustness under varying noise levels.

    Args:
        model: Trained Keras model.
        x_data: Input samples.
        y_data: One-hot labels.

    Returns:
        List of dicts with noise_level and accuracy for each level.
    """
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []

    y_true = np.argmax(y_data, axis=1)

    for noise_level in noise_levels:
        noisy_data = x_data + np.random.normal(0, noise_level, x_data.shape)
        noisy_data = np.clip(noisy_data, 0, 1)

        predictions = model.predict(noisy_data, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = np.mean(y_pred == y_true)

        results.append({"noise_level": noise_level, "accuracy": accuracy})

    return results


def generate_sensitivity_report(model_name, sensitivity_data):
    """Aggregate sensitivity analysis statistics into a summary report.

    Args:
        model_name: Display name for the model.
        sensitivity_data: Dictionary with gradient_importance,
            perturbation_scores, and occlusion_map.

    Returns:
        Summary dictionary with gradient_importance_stats,
        perturbation_robustness, and occlusion_sensitivity.
    """
    grad = sensitivity_data["gradient_importance"]
    pert = sensitivity_data["perturbation_scores"]
    occ = sensitivity_data["occlusion_map"]

    baseline_acc = pert[0]["accuracy"] if pert else 0
    worst_acc = min(r["accuracy"] for r in pert) if pert else 0

    return {
        "gradient_importance_stats": {
            "mean": float(np.mean(grad)),
            "std": float(np.std(grad)),
            "max": float(np.max(grad)),
        },
        "perturbation_robustness": {
            "accuracy_drop_rate": float(baseline_acc - worst_acc),
            "scores": pert,
        },
        "occlusion_sensitivity": {
            "mean": float(np.mean(occ)),
            "std": float(np.std(occ)),
            "max": float(np.max(occ)),
        },
    }


def perform_sensitivity_analysis(model, x_test, y_test, model_name, is_optimized=False):
    """Run the full sensitivity analysis pipeline for a model.

    Args:
        model: Trained Keras model.
        x_test: Test images (or reduced features for optimized model).
        y_test: One-hot test labels.
        model_name: Display name for the model.
        is_optimized: Whether this is the feature-selected optimized model.

    Returns:
        Sensitivity report dictionary.
    """
    print(f"\nPerforming Sensitivity Analysis for {model_name}")
    print("-" * 50)

    sensitivity_data = {}

    print("Calculating gradient-based importance...")
    gradients = calculate_gradient_importance(model, x_test[:1000])
    sensitivity_data["gradient_importance"] = gradients

    print("Performing perturbation analysis...")
    perturbation_scores = perform_perturbation_analysis(
        model, x_test[:1000], y_test[:1000]
    )
    sensitivity_data["perturbation_scores"] = perturbation_scores

    print("Calculating occlusion sensitivity...")
    occlusion_map = perform_occlusion_analysis(
        model, x_test[:100], y_test[:100], is_optimized=is_optimized
    )
    sensitivity_data["occlusion_map"] = occlusion_map

    visualize_sensitivity_results(model_name, sensitivity_data)

    report = generate_sensitivity_report(model_name, sensitivity_data)
    return report


def visualize_sensitivity_results(model_name, sensitivity_results):
    """Visualize all sensitivity analysis results in a single figure.

    Args:
        model_name: Display name for the model.
        sensitivity_results: Dictionary with gradient_importance,
            perturbation_scores, and occlusion_map.
    """
    plt.figure(figsize=(15, 5))

    # Gradient importance
    plt.subplot(131)
    grad = sensitivity_results["gradient_importance"]
    if grad.ndim == 1:
        plt.plot(grad)
        plt.title("Feature Importance")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
    else:
        plt.imshow(grad, cmap="hot")
        plt.title("Gradient-based Importance")
        plt.colorbar()

    # Perturbation analysis
    plt.subplot(132)
    pert_results = sensitivity_results["perturbation_scores"]
    noise_levels = [r["noise_level"] for r in pert_results]
    accuracies = [r["accuracy"] for r in pert_results]
    plt.plot(noise_levels, accuracies)
    plt.title("Perturbation Sensitivity")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")

    # Occlusion sensitivity
    plt.subplot(133)
    occ = sensitivity_results["occlusion_map"]
    if occ.ndim == 1 or (occ.ndim == 2 and occ.shape[1] == 1):
        plt.plot(occ.ravel())
        plt.title("Feature Sensitivity")
        plt.xlabel("Feature Index")
        plt.ylabel("Sensitivity")
    else:
        plt.imshow(occ, cmap="hot")
        plt.title("Occlusion Sensitivity")
        plt.colorbar()

    plt.suptitle(f"Sensitivity Analysis for {model_name}")
    plt.tight_layout()
    plt.show()
