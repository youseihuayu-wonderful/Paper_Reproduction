"""
category_influence.py - Category-Wise Influence Vector Estimation

This module extends standard influence functions to category-wise analysis.
Instead of computing a single scalar influence score, we compute a K-dimensional
influence VECTOR where each component measures influence on a specific class.

Key Concept:
    For K classes, each training sample z has an influence vector:
    P(z) = [P¹(z), P²(z), ..., Pᴷ(z)] ∈ ℝᴷ
    
    Where Pᵏ(z) = ℐ(z, Sᵏ) is the influence of z on validation samples of class k.

Paper Reference:
    Nahin et al. (2025). "What Is The Performance Ceiling of My Classifier?"
    arXiv:2510.03950
"""

import numpy as np
from typing import List, Tuple, Dict
from .influence import (
    compute_hessian,
    compute_inverse_hessian,
    compute_sample_gradient,
    compute_influence_score
)


def split_by_class(X: np.ndarray, y: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into per-class subsets.
    
    This is needed because category-wise influence is computed by measuring
    the influence of each training sample on each class's validation samples
    separately.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        
    Returns:
        Dictionary mapping class label -> (X_class, y_class) tuple
    """
    classes = np.unique(y)
    class_data = {}
    
    for c in classes:
        mask = (y == c)
        class_data[int(c)] = (X[mask], y[mask])
    
    return class_data


def compute_influence_vector(
    train_x: np.ndarray,
    train_y: float,
    val_class_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    H_inv: np.ndarray
) -> np.ndarray:
    """
    Compute the K-dimensional influence VECTOR for a single training sample.
    
    The influence vector captures how this sample affects EACH class:
    - P(z) = [influence on class 0, influence on class 1, ...]
    
    INTERPRETATION:
    - Pᵏ(z) > 0: Sample z is beneficial for class k (removing z hurts class k)
    - Pᵏ(z) < 0: Sample z is detrimental for class k (removing z helps class k)
    
    This is the KEY INNOVATION of the paper - moving from scalar to vector
    influence enables Pareto frontier analysis across classes.
    
    Args:
        train_x: Training sample features of shape (n_features,)
        train_y: Training sample label (0 or 1)
        val_class_data: Dictionary of {class_label: (X_val_class, y_val_class)}
        weights: Model weights of shape (n_features,)
        H_inv: Inverse Hessian of shape (n_features, n_features)
        
    Returns:
        Influence vector of shape (n_classes,)
    """
    n_classes = len(val_class_data)
    influence_vector = np.zeros(n_classes)
    
    # Compute influence on each class separately
    for k, (val_X_k, val_y_k) in val_class_data.items():
        # Influence of this training sample on class k's validation data
        influence_k = compute_influence_score(
            train_x, train_y,
            val_X_k, val_y_k,
            weights, H_inv
        )
        influence_vector[k] = influence_k
    
    return influence_vector


def compute_all_influence_vectors(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    weights: np.ndarray,
    damping: float = 1e-4
) -> np.ndarray:
    """
    Compute influence vectors for ALL training samples.
    
    This is the main function for category-wise influence analysis.
    It returns a matrix where:
    - Row i is the influence vector for training sample i
    - Column k is the influence of all samples on class k
    
    The result can be used to:
    1. Identify samples in joint positive/negative regions
    2. Analyze the Pareto frontier
    3. Determine if performance ceiling is reached
    
    Args:
        train_X: Training feature matrix of shape (n_train, n_features)
        train_y: Training labels of shape (n_train,)
        val_X: Validation feature matrix of shape (n_val, n_features)
        val_y: Validation labels of shape (n_val,)
        weights: Model weights of shape (n_features,)
        damping: Hessian regularization (default: 1e-4)
        
    Returns:
        Influence vectors of shape (n_train, n_classes)
    """
    n_train = train_X.shape[0]
    
    # Split validation set by class
    val_class_data = split_by_class(val_X, val_y)
    n_classes = len(val_class_data)
    
    # Compute Hessian inverse (shared across all samples)
    H = compute_hessian(train_X, weights)
    H_inv = compute_inverse_hessian(H, damping)
    
    # Compute influence vector for each training sample
    influence_vectors = np.zeros((n_train, n_classes))
    
    for j in range(n_train):
        influence_vectors[j] = compute_influence_vector(
            train_X[j], train_y[j],
            val_class_data,
            weights, H_inv
        )
    
    return influence_vectors


def classify_samples_by_region(
    influence_vectors: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Classify training samples into Pareto regions based on their influence vectors.
    
    For 2-class problems, we have 4 regions:
    1. JOINT POSITIVE: P⁰(z) > 0 AND P¹(z) > 0
       - Sample is beneficial to BOTH classes
       - Indicates room for Pareto improvement (should keep/upweight)
       
    2. JOINT NEGATIVE: P⁰(z) < 0 AND P¹(z) < 0
       - Sample is detrimental to BOTH classes
       - Indicates room for Pareto improvement (should remove/downweight)
       
    3. TRADEOFF CLASS 0: P⁰(z) > 0 AND P¹(z) < 0
       - Improves class 0 but hurts class 1
       - Pure tradeoff region
       
    4. TRADEOFF CLASS 1: P⁰(z) < 0 AND P¹(z) > 0
       - Improves class 1 but hurts class 0
       - Pure tradeoff region
    
    PARETO CEILING CONDITION:
    If ALL samples are in tradeoff regions (no joint positive/negative),
    AND they lie approximately on the line y = -x,
    then the classifier has reached its performance ceiling.
    
    Args:
        influence_vectors: Influence vectors of shape (n_train, n_classes)
        
    Returns:
        Dictionary mapping region name -> array of sample indices
    """
    n_samples = influence_vectors.shape[0]
    
    # Extract per-class influences (assuming 2 classes for clarity)
    # This generalizes to K classes with quadrant analysis
    P0 = influence_vectors[:, 0]  # Influence on class 0
    P1 = influence_vectors[:, 1]  # Influence on class 1
    
    # Classify into regions
    joint_positive_mask = (P0 > 0) & (P1 > 0)
    joint_negative_mask = (P0 < 0) & (P1 < 0)
    tradeoff_0_mask = (P0 > 0) & (P1 < 0)  # Helps class 0, hurts class 1
    tradeoff_1_mask = (P0 < 0) & (P1 > 0)  # Helps class 1, hurts class 0
    
    regions = {
        'joint_positive': np.where(joint_positive_mask)[0],
        'joint_negative': np.where(joint_negative_mask)[0],
        'tradeoff_class_0': np.where(tradeoff_0_mask)[0],
        'tradeoff_class_1': np.where(tradeoff_1_mask)[0]
    }
    
    return regions


def check_pareto_ceiling(
    influence_vectors: np.ndarray,
    threshold: float = 0.1
) -> Tuple[bool, float]:
    """
    Check if the classifier has reached its performance ceiling.
    
    The Pareto ceiling is reached when all training samples approximately
    lie on a hyperplane, meaning:
    - No pure Pareto improvements are possible via individual sample manipulation
    - Σₖ Pᵏ(z) ≈ 0 for all samples z
    
    For 2 classes, this means all points lie on the line y = -x.
    
    HOW WE CHECK:
    1. Compute the "Pareto residual" = P⁰(z) + P¹(z) for each sample
    2. If all residuals are near zero (relative to the influence magnitudes),
       then the ceiling is reached
    
    Args:
        influence_vectors: Influence vectors of shape (n_train, n_classes)
        threshold: Relative threshold for determining if ceiling is reached
        
    Returns:
        Tuple of (is_at_ceiling, explained_variance_ratio)
        - is_at_ceiling: True if ceiling is reached
        - explained_variance_ratio: How much variance is explained by
          the first principal component (high = samples form a line)
    """
    from sklearn.decomposition import PCA
    
    # Fit PCA to influence vectors
    pca = PCA(n_components=1)
    pca.fit(influence_vectors)
    
    # Explained variance ratio of first PC
    # High ratio means vectors lie approximately on a line (1D structure)
    explained_ratio = pca.explained_variance_ratio_[0]
    
    # Also check if samples are NOT in joint positive/negative regions
    regions = classify_samples_by_region(influence_vectors)
    n_joint = len(regions['joint_positive']) + len(regions['joint_negative'])
    n_total = influence_vectors.shape[0]
    
    joint_fraction = n_joint / n_total if n_total > 0 else 0
    
    # Ceiling is reached if:
    # 1. Most variance is on a single axis (line-like structure)
    # 2. Very few samples in joint positive/negative regions
    is_at_ceiling = (explained_ratio > 0.9) and (joint_fraction < threshold)
    
    return is_at_ceiling, explained_ratio


def identify_noisy_samples(
    influence_vectors: np.ndarray,
    percentile: float = 10.0
) -> np.ndarray:
    """
    Identify potentially noisy/mislabeled samples based on influence vectors.
    
    Noisy samples are typically in the JOINT NEGATIVE region because:
    - A mislabeled sample hurts its "wrong" class (obvious)
    - It also confuses the model, often hurting the other class too
    
    We also look at samples with extreme negative influences on their own class.
    
    This is one of the KEY APPLICATIONS of the paper:
    Using category-wise influence to detect data quality issues.
    
    Args:
        influence_vectors: Influence vectors of shape (n_train, n_classes)
        percentile: Consider samples below this percentile as "noisy"
        
    Returns:
        Array of indices of suspected noisy samples
    """
    # Get samples in joint negative region
    regions = classify_samples_by_region(influence_vectors)
    joint_negative = regions['joint_negative']
    
    # Also consider samples with very low sum of influences
    # (very negative total impact on the model)
    total_influence = np.sum(influence_vectors, axis=1)
    threshold = np.percentile(total_influence, percentile)
    very_negative = np.where(total_influence < threshold)[0]
    
    # Union of both criteria
    noisy_samples = np.union1d(joint_negative, very_negative)
    
    return noisy_samples
