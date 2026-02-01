"""
influence.py - Standard Influence Function Computation

This module implements influence functions for logistic regression following
Koh & Liang (2017). Influence functions measure the effect of removing or
downweighting a training sample on model predictions.

Key Formula:
    ℐ(z_j, V) = Σ_{z∈V} ∇ℓ(z; θ)ᵀ H⁻¹ ∇ℓ(z_j; θ)

Where:
    - z_j is the training sample whose influence we want to measure
    - V is the validation set
    - H is the Hessian of the training loss
    - ∇ℓ is the gradient of the loss w.r.t. model parameters

Reference:
    Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
    influence functions. ICML.
"""

import numpy as np
from scipy.special import expit  # Sigmoid function
from typing import Tuple, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    
    The sigmoid function σ(x) = 1 / (1 + exp(-x)) is the core of logistic
    regression, converting linear predictions to probabilities.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Array of same shape with values in (0, 1)
    """
    return expit(x)


def compute_predictions(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute predicted probabilities for logistic regression.
    
    For a sample x with weights w, we compute:
        p(y=1|x) = σ(w·x) = 1 / (1 + exp(-w·x))
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        weights: Model weights of shape (n_features,)
        
    Returns:
        Predicted probabilities of shape (n_samples,)
    """
    # Linear combination: z = X @ w
    linear_pred = X @ weights
    
    # Apply sigmoid to get probabilities
    return sigmoid(linear_pred)


def compute_loss(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss for logistic regression.
    
    The loss function is:
        L = -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
    
    This is the empirical risk that logistic regression minimizes.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        weights: Model weights of shape (n_features,)
        
    Returns:
        Scalar loss value
    """
    probs = compute_predictions(X, weights)
    
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    
    # Binary cross-entropy
    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    
    return loss


def compute_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute gradient of the loss w.r.t. model weights.
    
    For logistic regression, the gradient is:
        ∇L = (1/n) * Σ (p_i - y_i) * x_i
        
    This gradient tells us how each parameter affects the loss.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        weights: Model weights of shape (n_features,)
        
    Returns:
        Gradient vector of shape (n_features,)
    """
    n_samples = X.shape[0]
    probs = compute_predictions(X, weights)
    
    # Difference between predictions and true labels
    residuals = probs - y
    
    # Gradient: average of (residual * feature) across samples
    gradient = (1 / n_samples) * (X.T @ residuals)
    
    return gradient


def compute_sample_gradient(x: np.ndarray, y_true: float, weights: np.ndarray) -> np.ndarray:
    """
    Compute gradient contribution of a SINGLE sample.
    
    This is the key quantity for influence functions - we need to know
    how each individual sample contributes to the gradient.
    
    ∇ℓ(z; θ) = (p - y) * x
    
    Args:
        x: Single sample features of shape (n_features,)
        y_true: Binary label (0 or 1)
        weights: Model weights of shape (n_features,)
        
    Returns:
        Gradient vector of shape (n_features,)
    """
    # Probability prediction for this sample
    prob = sigmoid(np.dot(weights, x))
    
    # Gradient contribution: (prediction - true_label) * features
    gradient = (prob - y_true) * x
    
    return gradient


def compute_hessian(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian matrix of the loss w.r.t. model weights.
    
    For logistic regression, the Hessian is:
        H = (1/n) * Σ p_i * (1 - p_i) * x_i * x_i^T
        
    The Hessian captures the curvature of the loss landscape and is
    essential for influence functions. It is always positive semi-definite
    for logistic regression (convex loss).
    
    INTUITION: The Hessian tells us how "curved" the loss surface is.
    When the loss is very curved (high Hessian eigenvalues), the model
    is sensitive to perturbations. When flat, perturbations have less effect.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        weights: Model weights of shape (n_features,)
        
    Returns:
        Hessian matrix of shape (n_features, n_features)
    """
    n_samples, n_features = X.shape
    
    # Get predicted probabilities
    probs = compute_predictions(X, weights)
    
    # p * (1 - p) is the variance of a Bernoulli random variable
    # It's maximized at p=0.5 (maximum uncertainty)
    variance = probs * (1 - probs)
    
    # Weight each sample's outer product by its variance
    # H = Σ variance_i * x_i * x_i^T
    # This can be computed efficiently as: X^T @ diag(variance) @ X
    H = (1 / n_samples) * (X.T @ np.diag(variance) @ X)
    
    return H


def compute_inverse_hessian(H: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """
    Compute the inverse of the Hessian matrix with regularization.
    
    We add damping to ensure the Hessian is invertible:
        H_inv = (H + λI)^(-1)
    
    This corresponds to L2 regularization in the original problem and
    improves numerical stability.
    
    Args:
        H: Hessian matrix of shape (n_features, n_features)
        damping: Regularization strength (default: 1e-4)
        
    Returns:
        Inverse Hessian of shape (n_features, n_features)
    """
    n_features = H.shape[0]
    
    # Add damping term for numerical stability
    H_damped = H + damping * np.eye(n_features)
    
    # Compute inverse
    H_inv = np.linalg.inv(H_damped)
    
    return H_inv


def compute_influence_score(
    train_x: np.ndarray,
    train_y: float,
    val_X: np.ndarray,
    val_y: np.ndarray,
    weights: np.ndarray,
    H_inv: np.ndarray
) -> float:
    """
    Compute the influence of a SINGLE training sample on the validation loss.
    
    The influence score tells us: "If we remove this training sample and
    retrain, how much would the validation loss change?"
    
    Formula:
        ℐ(z_train, V) = Σ_{z∈V} ∇ℓ(z_val; θ)^T @ H^(-1) @ ∇ℓ(z_train; θ)
    
    POSITIVE influence = sample is BENEFICIAL (removing it increases loss)
    NEGATIVE influence = sample is DETRIMENTAL (removing it decreases loss)
    
    INTUITION: We're measuring how aligned the training sample's gradient is
    with the validation set's gradient, after accounting for the loss curvature.
    
    Args:
        train_x: Training sample features of shape (n_features,)
        train_y: Training sample label (0 or 1)
        val_X: Validation feature matrix of shape (n_val, n_features)
        val_y: Validation labels of shape (n_val,)
        weights: Model weights of shape (n_features,)
        H_inv: Inverse Hessian of shape (n_features, n_features)
        
    Returns:
        Scalar influence score
    """
    # Gradient of the training sample
    grad_train = compute_sample_gradient(train_x, train_y, weights)
    
    # Compute H^(-1) @ grad_train (called the "influence direction")
    # This tells us how the parameters would change if we upweight this sample
    influence_direction = H_inv @ grad_train
    
    # Sum influence over all validation samples
    total_influence = 0.0
    for i in range(val_X.shape[0]):
        # Gradient of validation sample
        grad_val = compute_sample_gradient(val_X[i], val_y[i], weights)
        
        # Dot product: how aligned is val gradient with influence direction?
        # This measures the indirect effect of train sample on val loss
        total_influence += np.dot(grad_val, influence_direction)
    
    return total_influence


def compute_all_influences(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    weights: np.ndarray,
    damping: float = 1e-4
) -> np.ndarray:
    """
    Compute influence scores for ALL training samples on the validation set.
    
    This is the main entry point for standard (non-category-wise) influence
    computation. It returns a vector where each element indicates how
    influential the corresponding training sample is.
    
    Args:
        train_X: Training feature matrix of shape (n_train, n_features)
        train_y: Training labels of shape (n_train,)
        val_X: Validation feature matrix of shape (n_val, n_features)
        val_y: Validation labels of shape (n_val,)
        weights: Model weights of shape (n_features,)
        damping: Hessian regularization (default: 1e-4)
        
    Returns:
        Influence scores of shape (n_train,)
    """
    n_train = train_X.shape[0]
    
    # Compute Hessian and its inverse (done once, shared across all samples)
    H = compute_hessian(train_X, weights)
    H_inv = compute_inverse_hessian(H, damping)
    
    # Compute influence for each training sample
    influences = np.zeros(n_train)
    for j in range(n_train):
        influences[j] = compute_influence_score(
            train_X[j], train_y[j],
            val_X, val_y,
            weights, H_inv
        )
    
    return influences


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
    l2_reg: float = 0.01,
    verbose: bool = False
) -> np.ndarray:
    """
    Train logistic regression using gradient descent.
    
    This is a simple implementation for our experiments. We use L2
    regularization for stability.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        learning_rate: Step size for gradient descent
        n_iterations: Number of training iterations
        l2_reg: L2 regularization strength
        verbose: Whether to print training progress
        
    Returns:
        Trained weights of shape (n_features,)
    """
    n_features = X.shape[1]
    
    # Initialize weights to zeros
    weights = np.zeros(n_features)
    
    for i in range(n_iterations):
        # Compute gradient
        grad = compute_gradient(X, y, weights)
        
        # Add L2 regularization gradient
        grad += l2_reg * weights
        
        # Gradient descent step
        weights -= learning_rate * grad
        
        if verbose and (i + 1) % 100 == 0:
            loss = compute_loss(X, y, weights)
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4f}")
    
    return weights


def train_weighted_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
    l2_reg: float = 0.01
) -> np.ndarray:
    """
    Train logistic regression with per-sample weights.
    
    This is used by the Pareto-LP-GA framework to apply the optimized
    sample weights during training.
    
    The weighted loss is:
        L = -Σ w_i * [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        sample_weights: Per-sample weights of shape (n_samples,)
        learning_rate: Step size for gradient descent
        n_iterations: Number of training iterations
        l2_reg: L2 regularization strength
        
    Returns:
        Trained weights of shape (n_features,)
    """
    n_samples, n_features = X.shape
    
    # Normalize sample weights to sum to n_samples
    sample_weights = sample_weights * (n_samples / np.sum(sample_weights))
    
    # Initialize weights
    weights = np.zeros(n_features)
    
    for _ in range(n_iterations):
        probs = compute_predictions(X, weights)
        residuals = probs - y
        
        # Weighted gradient
        weighted_residuals = sample_weights * residuals
        grad = (1 / n_samples) * (X.T @ weighted_residuals)
        
        # Add L2 regularization
        grad += l2_reg * weights
        
        # Update
        weights -= learning_rate * grad
    
    return weights
