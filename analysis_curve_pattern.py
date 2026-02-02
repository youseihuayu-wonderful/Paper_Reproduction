"""
Deep Analysis: Why Curve Patterns Emerge in Influence Space

This script analyzes the mathematical structure of category-wise influence
to understand why the original paper shows distinct "curved" patterns while
our reproduction shows scattered clusters.

Author: Reproduction Analysis
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from influence import (
    compute_hessian, compute_inverse_hessian,
    compute_sample_gradient, train_logistic_regression,
    compute_predictions
)


def analyze_influence_structure():
    """
    Analyze why influence vectors form curves vs scattered clusters.

    Key insight: The curve pattern emerges from the GEOMETRIC STRUCTURE
    of the problem. Let's understand this mathematically.
    """
    print("=" * 80)
    print("DEEP ANALYSIS: Influence Vector Curve Structure")
    print("=" * 80)

    # ============================================================
    # MATHEMATICAL BACKGROUND
    # ============================================================
    print("""
    MATHEMATICAL ANALYSIS
    =====================

    For logistic regression, the influence of training sample z on validation class k:

        P^k(z) = Σ_{z' ∈ V_k} ∇L(z')^T · H^{-1} · ∇L(z)

    Where:
        - ∇L(z) = (p_z - y_z) · x_z     [gradient of training sample]
        - ∇L(z') = (p_{z'} - y_{z'}) · x_{z'}  [gradient of validation sample]
        - H^{-1} is the inverse Hessian

    The CURVE PATTERN emerges because:

    1. For WELL-SEPARATED data with a WELL-TRAINED model:
       - Clean samples have p ≈ y (correct predictions)
       - Gradient magnitudes are small for well-classified samples
       - But the DIRECTION of gradients is consistent within each class

    2. The influence P^k(z) depends on:
       - Distance from decision boundary (affects |∇L(z)|)
       - Angle between ∇L(z) and ∇L(z') (determines sign and magnitude)
       - The Hessian structure (how curved the loss surface is)

    3. For samples of the SAME class:
       - Their feature vectors x_z are similar (same cluster)
       - Their gradients ∇L(z) point in similar directions
       - This creates a COHERENT structure in influence space

    The CURVE (not scatter) emerges because P^0 and P^1 are RELATED through
    the shared Hessian H^{-1} and the geometric arrangement of samples.
    """)

    # ============================================================
    # EXPERIMENT: Varying class separation
    # ============================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Effect of Class Separation on Curve Pattern")
    print("=" * 80)

    np.random.seed(42)
    n_per_class = 150

    separations = [1.0, 2.0, 3.0, 4.0]  # Different class separations

    fig, axes = plt.subplots(2, len(separations), figsize=(20, 10))

    for col, sep in enumerate(separations):
        # Generate data with given separation
        X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-sep, 0])
        X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([sep, 0])
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        # Add noise (flip 10% of labels)
        n_noise = int(0.1 * len(y))
        noise_idx = np.random.choice(len(y), n_noise, replace=False)
        y_noisy = y.copy()
        y_noisy[noise_idx] = 1 - y_noisy[noise_idx]

        # Split train/val
        perm = np.random.permutation(len(y))
        n_train = int(0.8 * len(y))
        train_X, train_y = X[perm[:n_train]], y_noisy[perm[:n_train]]
        val_X, val_y = X[perm[n_train:]], y_noisy[perm[n_train:]]

        # Train model
        weights = train_logistic_regression(train_X, train_y, n_iterations=500)

        # Compute influences
        H = compute_hessian(train_X, weights)
        H_inv = compute_inverse_hessian(H, damping=1e-3)

        # Compute P^0 and P^1 for each training sample
        n_train_samples = len(train_X)
        P0 = np.zeros(n_train_samples)
        P1 = np.zeros(n_train_samples)

        # Split validation by class
        val_X_0, val_y_0 = val_X[val_y == 0], val_y[val_y == 0]
        val_X_1, val_y_1 = val_X[val_y == 1], val_y[val_y == 1]

        for j in range(n_train_samples):
            grad_train = compute_sample_gradient(train_X[j], train_y[j], weights)
            inf_dir = H_inv @ grad_train

            # Influence on class 0
            for i in range(len(val_X_0)):
                grad_val = compute_sample_gradient(val_X_0[i], val_y_0[i], weights)
                P0[j] += np.dot(grad_val, inf_dir)

            # Influence on class 1
            for i in range(len(val_X_1)):
                grad_val = compute_sample_gradient(val_X_1[i], val_y_1[i], weights)
                P1[j] += np.dot(grad_val, inf_dir)

        # Plot data
        ax = axes[0, col]
        ax.scatter(train_X[train_y == 0, 0], train_X[train_y == 0, 1],
                  c='blue', alpha=0.6, label='Class 0')
        ax.scatter(train_X[train_y == 1, 0], train_X[train_y == 1, 1],
                  c='orange', alpha=0.6, label='Class 1')
        ax.set_title(f'Data (separation={sep})')
        ax.set_aspect('equal')
        ax.legend()

        # Plot influence space
        ax = axes[1, col]
        ax.scatter(P0[train_y == 0], P1[train_y == 0], c='blue', alpha=0.6, label='Class 0')
        ax.scatter(P0[train_y == 1], P1[train_y == 1], c='orange', alpha=0.6, label='Class 1')

        # Add Pareto line
        lim = max(abs(P0).max(), abs(P1).max()) * 1.2
        ax.plot([-lim, lim], [lim, -lim], 'k--', alpha=0.5)
        ax.axhline(0, color='gray', alpha=0.3)
        ax.axvline(0, color='gray', alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('P^0 (Influence on Class 0)')
        ax.set_ylabel('P^1 (Influence on Class 1)')
        ax.set_title(f'Influence Space (sep={sep})')
        ax.set_aspect('equal')
        ax.legend()

        # Compute statistics
        corr = np.corrcoef(P0, P1)[0, 1]
        print(f"\nSeparation = {sep}:")
        print(f"  Correlation(P^0, P^1) = {corr:.4f}")
        print(f"  Class 0 in Joint+: {np.sum((P0[train_y==0] > 0) & (P1[train_y==0] > 0))}/{np.sum(train_y==0)}")
        print(f"  Class 1 in Joint+: {np.sum((P0[train_y==1] > 0) & (P1[train_y==1] > 0))}/{np.sum(train_y==1)}")

    plt.tight_layout()
    plt.savefig('outputs/figures/analysis_separation_effect.png', dpi=200)
    print("\nFigure saved: outputs/figures/analysis_separation_effect.png")
    plt.close()

    # ============================================================
    # EXPERIMENT 2: The role of Hessian structure
    # ============================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Hessian Damping Effect")
    print("=" * 80)

    np.random.seed(42)
    sep = 2.0

    # Generate well-separated data
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-sep, 0])
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([sep, 0])
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Add noise
    noise_idx = np.random.choice(len(y), int(0.1 * len(y)), replace=False)
    y_noisy = y.copy()
    y_noisy[noise_idx] = 1 - y_noisy[noise_idx]

    # Split
    perm = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))
    train_X, train_y = X[perm[:n_train]], y_noisy[perm[:n_train]]
    val_X, val_y = X[perm[n_train:]], y_noisy[perm[n_train:]]

    weights = train_logistic_regression(train_X, train_y, n_iterations=500)
    H = compute_hessian(train_X, weights)

    dampings = [1e-5, 1e-3, 1e-1, 1.0]

    fig, axes = plt.subplots(1, len(dampings), figsize=(20, 5))

    for col, damp in enumerate(dampings):
        H_inv = compute_inverse_hessian(H, damping=damp)

        n_train_samples = len(train_X)
        P0 = np.zeros(n_train_samples)
        P1 = np.zeros(n_train_samples)

        val_X_0, val_y_0 = val_X[val_y == 0], val_y[val_y == 0]
        val_X_1, val_y_1 = val_X[val_y == 1], val_y[val_y == 1]

        for j in range(n_train_samples):
            grad_train = compute_sample_gradient(train_X[j], train_y[j], weights)
            inf_dir = H_inv @ grad_train

            for i in range(len(val_X_0)):
                grad_val = compute_sample_gradient(val_X_0[i], val_y_0[i], weights)
                P0[j] += np.dot(grad_val, inf_dir)

            for i in range(len(val_X_1)):
                grad_val = compute_sample_gradient(val_X_1[i], val_y_1[i], weights)
                P1[j] += np.dot(grad_val, inf_dir)

        ax = axes[col]
        ax.scatter(P0[train_y == 0], P1[train_y == 0], c='blue', alpha=0.6, label='Class 0')
        ax.scatter(P0[train_y == 1], P1[train_y == 1], c='orange', alpha=0.6, label='Class 1')

        lim = max(abs(P0).max(), abs(P1).max()) * 1.2
        ax.plot([-lim, lim], [lim, -lim], 'k--', alpha=0.5)
        ax.axhline(0, color='gray', alpha=0.3)
        ax.axvline(0, color='gray', alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('P^0')
        ax.set_ylabel('P^1')
        ax.set_title(f'Damping = {damp}')
        ax.set_aspect('equal')
        ax.legend()

        print(f"\nDamping = {damp}:")
        print(f"  Range P^0: [{P0.min():.2f}, {P0.max():.2f}]")
        print(f"  Range P^1: [{P1.min():.2f}, {P1.max():.2f}]")

    plt.tight_layout()
    plt.savefig('outputs/figures/analysis_damping_effect.png', dpi=200)
    print("\nFigure saved: outputs/figures/analysis_damping_effect.png")
    plt.close()

    # ============================================================
    # KEY INSIGHT: Matching the paper's setup
    # ============================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR REPRODUCING THE PAPER'S FIGURE")
    print("=" * 80)
    print("""
    Based on this analysis, to reproduce the paper's curve pattern:

    1. LARGER CLASS SEPARATION:
       - The paper's clusters appear well-separated (gap ≈ 2-3 units)
       - Use separation = 2.5 or higher

    2. LOWER DAMPING:
       - Lower damping preserves the natural structure
       - Try damping = 1e-4 or 1e-5

    3. NOISE INJECTION:
       - The paper injects noise for samples NEAR the boundary
       - Not random noise across all samples

    4. CONSISTENT CLUSTER SHAPE:
       - Both clusters should have similar variance
       - Use variance ≈ 0.8-1.0

    5. INFLUENCE SIGN CONVENTION:
       - Check if paper uses positive = beneficial or positive = harmful
       - May need to negate influences
    """)

    return


if __name__ == "__main__":
    analyze_influence_structure()
