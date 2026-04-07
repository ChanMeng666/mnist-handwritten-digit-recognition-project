"""Feature importance and selection utilities."""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def calculate_feature_importance(x_train, y_train, n_features=196):
    """Calculate feature importance using ANOVA F-value selection.

    Args:
        x_train: Training images (N, 28, 28, 1) or (N, 784).
        y_train: Training labels (one-hot or integer).
        n_features: Number of top features to select.

    Returns:
        Tuple of (importance_map, important_pixel_indices).
        importance_map has shape (28, 28).
    """
    X_flat = x_train.reshape(x_train.shape[0], -1)

    # Remove constant features
    feature_variance = np.var(X_flat, axis=0)
    non_constant_features = feature_variance > 0
    X_flat_filtered = X_flat[:, non_constant_features]

    print(f"Original features: {X_flat.shape[1]}")
    print(f"Non-constant features: {X_flat_filtered.shape[1]}")

    n_features = min(n_features, X_flat_filtered.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_features)

    y_indices = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    selector.fit(X_flat_filtered, y_indices)

    # Build importance map
    importance_map = np.zeros(X_flat.shape[1])
    non_constant_indices = np.where(non_constant_features)[0]
    importance_map[non_constant_indices] = selector.scores_
    importance_map = importance_map.reshape(28, 28)

    # Get top feature indices
    important_pixels = np.argsort(importance_map.ravel())[-n_features:]

    return importance_map, important_pixels
