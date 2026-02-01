"""
pareto_lp_ga.py - Combined Pareto-LP-GA Framework

This module combines all components into the complete Pareto-LP-GA algorithm
as described in Algorithm 1 of the paper.

The framework:
1. Computes category-wise influence vectors
2. Uses GA to search for optimal class thresholds
3. Uses LP to find optimal sample weights given thresholds
4. Applies weighted training to achieve Pareto improvement

Two modes:
- Direct Improvement (DI): Improve specific underperforming classes
- Course Correction (CC): Reverse accuracy drops that occurred in training

Reference:
    Nahin et al. (2025). Algorithm 1 "Pareto-LP-GA"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .influence import (
    train_logistic_regression,
    train_weighted_logistic_regression,
    compute_predictions
)
from .category_influence import compute_all_influence_vectors
from .lp_reweight import solve_lp_weights, validate_lp_solution
from .ga_search import GeneticAlgorithm, simple_threshold_search


def compute_class_accuracies(
    X: np.ndarray, 
    y: np.ndarray, 
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute per-class accuracy for a trained classifier.
    
    This is the key metric we're trying to optimize.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: True labels of shape (n_samples,)
        weights: Model weights for prediction
        
    Returns:
        Per-class accuracy array of shape (n_classes,)
    """
    # Get predictions
    probs = compute_predictions(X, weights)
    preds = (probs >= 0.5).astype(int)
    
    classes = np.unique(y)
    n_classes = len(classes)
    accuracies = np.zeros(n_classes)
    
    for i, k in enumerate(classes):
        mask = (y == k)
        if np.sum(mask) > 0:
            accuracies[i] = np.mean(preds[mask] == y[mask])
    
    return accuracies


def pareto_lp_ga_direct_improvement(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    target_classes: List[int],
    current_weights: np.ndarray,
    n_ga_iterations: int = 20,
    ga_population_size: int = 30,
    verbose: bool = True
) -> Dict:
    """
    Direct Improvement (DI) mode of Pareto-LP-GA.
    
    Use this when you observe that certain classes have low accuracy
    and you want to improve them while maintaining performance on others.
    
    ALGORITHM:
    1. Compute influence vectors for all training samples
    2. Initialize GA population of threshold vectors
    3. For each GA iteration:
       a. For each threshold vector, solve LP to get sample weights
       b. Estimate improvement using influence vectors
       c. Evaluate fitness based on target improvement + non-target preservation
    4. Apply best weights to retrain the model
    
    Args:
        train_X: Training features of shape (n_train, n_features)
        train_y: Training labels of shape (n_train,)
        val_X: Validation features for influence computation
        val_y: Validation labels
        target_classes: List of class indices to improve
        current_weights: Current model weights
        n_ga_iterations: Number of GA generations
        ga_population_size: GA population size
        verbose: Print progress
        
    Returns:
        Dictionary with:
        - 'optimal_sample_weights': Best sample weights from LP
        - 'optimal_thresholds': Best thresholds from GA
        - 'new_model_weights': Retrained model weights
        - 'baseline_accuracies': Original per-class accuracies
        - 'new_accuracies': Per-class accuracies after reweighting
        - 'improvement': Change in accuracy for each class
    """
    n_classes = len(np.unique(train_y))
    
    if verbose:
        print("=" * 50)
        print("PARETO-LP-GA: Direct Improvement Mode")
        print(f"Target classes: {target_classes}")
        print("=" * 50)
    
    # Step 1: Compute baseline accuracies
    if verbose:
        print("\nStep 1: Computing baseline accuracies...")
    baseline_acc = compute_class_accuracies(val_X, val_y, current_weights)
    if verbose:
        for k in range(n_classes):
            print(f"  Class {k}: {baseline_acc[k]:.4f}")
    
    # Step 2: Compute influence vectors
    if verbose:
        print("\nStep 2: Computing category-wise influence vectors...")
    influence_vectors = compute_all_influence_vectors(
        train_X, train_y, val_X, val_y, current_weights
    )
    if verbose:
        print(f"  Computed {influence_vectors.shape[0]} influence vectors")
    
    # Step 3: Run simplified threshold search (faster for demonstration)
    if verbose:
        print("\nStep 3: Searching for optimal thresholds...")
    
    def solve_lp_wrapper(inf_vec, targets, alpha):
        return solve_lp_weights(inf_vec, targets, alpha)
    
    best_thresholds, best_sample_weights = simple_threshold_search(
        influence_vectors, target_classes, solve_lp_wrapper, n_trials=ga_population_size * n_ga_iterations
    )
    
    if verbose:
        print(f"  Best thresholds: {best_thresholds}")
    
    # Step 4: Verify LP solution
    if verbose:
        print("\nStep 4: Validating LP solution...")
    validation = validate_lp_solution(
        influence_vectors, best_sample_weights, target_classes, best_thresholds
    )
    if verbose:
        print(f"  Constraints satisfied: {validation['constraints_satisfied']}")
        print(f"  Non-zero weights: {validation['weight_stats']['nonzero']}")
    
    # Step 5: Retrain with weighted loss
    if verbose:
        print("\nStep 5: Retraining with optimal sample weights...")
    new_model_weights = train_weighted_logistic_regression(
        train_X, train_y, best_sample_weights,
        learning_rate=0.1, n_iterations=1000
    )
    
    # Step 6: Evaluate new accuracies
    if verbose:
        print("\nStep 6: Evaluating new model...")
    new_acc = compute_class_accuracies(val_X, val_y, new_model_weights)
    improvement = new_acc - baseline_acc
    
    if verbose:
        print("\nResults:")
        print("-" * 40)
        for k in range(n_classes):
            marker = "TARGET" if k in target_classes else "      "
            sign = "+" if improvement[k] >= 0 else ""
            print(f"  Class {k} [{marker}]: {baseline_acc[k]:.4f} → {new_acc[k]:.4f} "
                  f"({sign}{improvement[k]*100:.2f}%)")
    
    return {
        'optimal_sample_weights': best_sample_weights,
        'optimal_thresholds': best_thresholds,
        'new_model_weights': new_model_weights,
        'baseline_accuracies': baseline_acc,
        'new_accuracies': new_acc,
        'improvement': improvement,
        'influence_vectors': influence_vectors
    }


def pareto_lp_ga_course_correction(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    prev_epoch_weights: np.ndarray,
    current_epoch_weights: np.ndarray,
    degradation_threshold: float = 0.03,
    verbose: bool = True
) -> Dict:
    """
    Course Correction (CC) mode of Pareto-LP-GA.
    
    Use this when you detect that training caused performance to DROP
    for certain classes, and you want to correct the training trajectory.
    
    ALGORITHM:
    1. Identify classes that degraded between epochs
    2. Set those as target classes
    3. Run Pareto-LP-GA to find weights that reverse the degradation
    
    Args:
        train_X: Training features
        train_y: Training labels
        val_X: Validation features
        val_y: Validation labels
        prev_epoch_weights: Model weights from previous epoch
        current_epoch_weights: Model weights from current epoch
        degradation_threshold: Minimum accuracy drop to consider as degradation
        verbose: Print progress
        
    Returns:
        Same as pareto_lp_ga_direct_improvement
    """
    n_classes = len(np.unique(train_y))
    
    # Detect degraded classes
    prev_acc = compute_class_accuracies(val_X, val_y, prev_epoch_weights)
    curr_acc = compute_class_accuracies(val_X, val_y, current_epoch_weights)
    
    degraded_classes = []
    for k in range(n_classes):
        if prev_acc[k] - curr_acc[k] > degradation_threshold:
            degraded_classes.append(k)
    
    if verbose:
        print("=" * 50)
        print("PARETO-LP-GA: Course Correction Mode")
        print("-" * 50)
        print("Accuracy comparison (prev → current):")
        for k in range(n_classes):
            delta = curr_acc[k] - prev_acc[k]
            sign = "+" if delta >= 0 else ""
            marker = "DEGRADED" if k in degraded_classes else ""
            print(f"  Class {k}: {prev_acc[k]:.4f} → {curr_acc[k]:.4f} "
                  f"({sign}{delta*100:.2f}%) {marker}")
    
    if len(degraded_classes) == 0:
        if verbose:
            print("\nNo significant degradation detected. No correction needed.")
        return {
            'optimal_sample_weights': np.ones(len(train_y)) / len(train_y),
            'target_classes': [],
            'baseline_accuracies': curr_acc,
            'new_accuracies': curr_acc,
            'improvement': np.zeros(n_classes)
        }
    
    if verbose:
        print(f"\nDegraded classes detected: {degraded_classes}")
        print("Running correction...")
    
    # Run direct improvement on degraded classes
    result = pareto_lp_ga_direct_improvement(
        train_X, train_y, val_X, val_y,
        target_classes=degraded_classes,
        current_weights=current_epoch_weights,
        verbose=verbose
    )
    
    result['target_classes'] = degraded_classes
    return result


def quick_pareto_analysis(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    model_weights: np.ndarray
) -> Dict:
    """
    Quick Pareto analysis without retraining.
    
    This is useful for understanding the current state of the model
    without committing to retraining.
    
    Returns:
        Dictionary with influence vectors, region classifications, and statistics
    """
    from .category_influence import classify_samples_by_region, check_pareto_ceiling
    from .pareto import analyze_pareto_statistics
    
    # Compute influence vectors
    influence_vectors = compute_all_influence_vectors(
        train_X, train_y, val_X, val_y, model_weights
    )
    
    # Classify regions
    regions = classify_samples_by_region(influence_vectors)
    
    # Check ceiling
    is_ceiling, explained_ratio = check_pareto_ceiling(influence_vectors)
    
    # Statistics
    stats = analyze_pareto_statistics(influence_vectors, train_y)
    
    return {
        'influence_vectors': influence_vectors,
        'regions': regions,
        'pareto_ceiling_reached': is_ceiling,
        'explained_variance_ratio': explained_ratio,
        'statistics': stats
    }
