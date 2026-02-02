"""
figure2_ceiling_comparison.py - Comprehensive Ceiling Analysis

This script generates a comparison figure showing two types of non-separable patterns:

1. PAPER-MATCHING CASE (Partial Ceiling):
   - Two curved arms in tradeoff quadrants
   - Positive correlation (~0.6)
   - ~30% samples in tradeoff regions
   - Shows CLASS-SPECIFIC tradeoff behavior

2. TRUE CEILING CASE (Complete Ceiling):
   - Single band along y = -x direction
   - Negative correlation (< -0.5)
   - >70% samples in tradeoff regions
   - Shows COMPLETE tradeoff - P0 + P1 ≈ 0

The key insight:
- "Line structure" alone does NOT indicate ceiling
- Must check: correlation sign, tradeoff fraction, P0+P1 residuals

Author: Paper Reproduction
Date: 2026-02
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.influence import (
    train_logistic_regression, compute_predictions,
    compute_hessian, compute_inverse_hessian, compute_sample_gradient
)


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 14,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })


def create_paper_colormap():
    """Blue → White → Red colormap matching paper."""
    colors = [
        (0.0, '#2166ac'),
        (0.15, '#4393c3'),
        (0.35, '#92c5de'),
        (0.45, '#d1e5f0'),
        (0.5, '#f7f7f7'),
        (0.55, '#fddbc7'),
        (0.65, '#f4a582'),
        (0.85, '#d6604d'),
        (1.0, '#b2182b'),
    ]

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

    positions = [c[0] for c in colors]
    rgb_colors = [hex_to_rgb(c[1]) for c in colors]

    cdict = {
        'red': [(p, c[0], c[0]) for p, c in zip(positions, rgb_colors)],
        'green': [(p, c[1], c[1]) for p, c in zip(positions, rgb_colors)],
        'blue': [(p, c[2], c[2]) for p, c in zip(positions, rgb_colors)]
    }

    return LinearSegmentedColormap('paper_cmap', cdict)


def compute_influence(train_X, train_y, val_X, val_y, weights, damping=1e-4):
    """Compute category-wise influence vectors P^0 and P^1."""
    H = compute_hessian(train_X, weights)
    H_inv = compute_inverse_hessian(H, damping=damping)

    n_train = len(train_X)
    P0 = np.zeros(n_train)
    P1 = np.zeros(n_train)

    val_X_0, val_y_0 = val_X[val_y == 0], val_y[val_y == 0]
    val_X_1, val_y_1 = val_X[val_y == 1], val_y[val_y == 1]

    for j in range(n_train):
        grad_train = compute_sample_gradient(train_X[j], train_y[j], weights)
        inf_dir = H_inv @ grad_train

        for i in range(len(val_X_0)):
            grad_val = compute_sample_gradient(val_X_0[i], val_y_0[i], weights)
            P0[j] -= np.dot(grad_val, inf_dir)

        for i in range(len(val_X_1)):
            grad_val = compute_sample_gradient(val_X_1[i], val_y_1[i], weights)
            P1[j] -= np.dot(grad_val, inf_dir)

    return np.column_stack([P0, P1])


# ==================== DATASET GENERATORS ====================

def generate_paper_matching(n_per_class=300, seed=42):
    """
    PAPER-MATCHING CASE: Two curved arms in tradeoff quadrants.

    - Offset Gaussians with moderate overlap
    - Produces positive correlation (~0.6)
    - Blue upper-left, orange lower-right
    """
    np.random.seed(seed)

    center_blue = np.array([-0.5, 0.5])
    center_orange = np.array([0.5, -0.5])
    spread = 1.0

    X_blue = np.random.randn(n_per_class, 2) * spread + center_blue
    X_orange = np.random.randn(n_per_class, 2) * spread + center_orange

    X = np.vstack([X_blue, X_orange])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    return X, y


def generate_true_ceiling(n_per_class=300, seed=42):
    """
    TRUE CEILING CASE: Complete overlap with no class structure.

    - Both classes centered at origin with same distribution
    - Linear classifier can only achieve ~50% accuracy
    - Every sample has P0 + P1 ≈ 0 (pure tradeoff)
    - Produces negative correlation (< -0.5)
    """
    np.random.seed(seed)

    # NESTED: Blue concentrated in center, Orange spread around
    # This creates complete ambiguity - every region has both classes
    X_blue = np.random.randn(n_per_class, 2) * 0.8  # Tight center
    X_orange = np.random.randn(n_per_class, 2) * 1.6  # Spread out

    X = np.vstack([X_blue, X_orange])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    return X, y


def compute_ceiling_metrics(influence):
    """Compute metrics that indicate ceiling behavior."""
    P0, P1 = influence[:, 0], influence[:, 1]
    n_samples = len(P0)

    # Correlation
    corr = np.corrcoef(P0, P1)[0, 1]

    # Pareto regions
    joint_pos = np.sum((P0 > 0) & (P1 > 0))
    joint_neg = np.sum((P0 < 0) & (P1 < 0))
    tradeoff_0 = np.sum((P0 > 0) & (P1 < 0))  # Helps class 0, hurts class 1
    tradeoff_1 = np.sum((P0 < 0) & (P1 > 0))  # Helps class 1, hurts class 0

    tradeoff_frac = (tradeoff_0 + tradeoff_1) / n_samples
    joint_frac = (joint_pos + joint_neg) / n_samples

    # P0 + P1 residual (should be ~0 for true ceiling)
    residuals = P0 + P1
    residual_std = np.std(residuals)
    residual_mean = np.mean(np.abs(residuals))

    # Explained variance (line structure)
    centered = influence - influence.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    explained_var = eigenvalues[-1] / eigenvalues.sum()

    return {
        'correlation': corr,
        'tradeoff_frac': tradeoff_frac,
        'joint_frac': joint_frac,
        'tradeoff_0': tradeoff_0,
        'tradeoff_1': tradeoff_1,
        'joint_pos': joint_pos,
        'joint_neg': joint_neg,
        'residual_std': residual_std,
        'residual_mean': residual_mean,
        'explained_var': explained_var
    }


def run_experiment(X, y, name):
    """Run full experiment on a dataset."""
    np.random.seed(42)
    perm = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))

    train_X, train_y = X[perm[:n_train]], y[perm[:n_train]]
    val_X, val_y = X[perm[n_train:]], y[perm[n_train:]]

    # Train
    weights = train_logistic_regression(train_X, train_y, n_iterations=1000, l2_reg=0.01)
    preds = (compute_predictions(train_X, weights) >= 0.5).astype(int)
    acc = np.mean(preds == train_y)

    # Compute influence
    influence = compute_influence(train_X, train_y, val_X, val_y, weights, damping=1e-4)

    # Metrics
    metrics = compute_ceiling_metrics(influence)
    metrics['accuracy'] = acc
    metrics['name'] = name

    return train_X, train_y, influence, weights, metrics


def get_decision_boundary(weights, xlim):
    """Get the learned decision boundary."""
    w0, w1 = weights[0], weights[1]
    x_vals = np.array(xlim)
    if abs(w1) > 1e-6:
        y_vals = -(w0 / w1) * x_vals
    else:
        y_vals = np.array(xlim)
    return x_vals, y_vals


def create_comparison_figure(results_partial, results_ceiling, save_path):
    """
    Create 2x3 comparison figure.

    Row 1: Paper-matching (partial ceiling)
    Row 2: True ceiling
    """
    setup_publication_style()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    color_blue = '#1f77b4'
    color_orange = '#ff7f0e'
    cmap = create_paper_colormap()
    marker_size = 25
    alpha = 0.8

    for row, (train_X, train_y, influence, weights, metrics) in enumerate([results_partial, results_ceiling]):
        P0, P1 = influence[:, 0], influence[:, 1]
        net_inf = P0 + P1

        mask_0 = train_y == 0
        mask_1 = train_y == 1

        row_label = "PARTIAL CEILING" if row == 0 else "TRUE CEILING"

        # ==================== Column 1: Data Distribution ====================
        ax = axes[row, 0]

        ax.scatter(train_X[mask_0, 0], train_X[mask_0, 1],
                   c=color_blue, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.3, label='Class 0')
        ax.scatter(train_X[mask_1, 0], train_X[mask_1, 1],
                   c=color_orange, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.3, label='Class 1')

        xlim = [-4, 4]
        x_bound, y_bound = get_decision_boundary(weights, xlim)
        ax.plot(x_bound, y_bound, 'k-', linewidth=1.5)

        ax.set_xlim(xlim)
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        panel = "D" if row == 0 else "D'"
        ax.set_title(f"{panel}: {row_label}\nAccuracy: {metrics['accuracy']:.1%}", fontweight='bold')

        # ==================== Column 2: Influence Space ====================
        ax = axes[row, 1]

        ax.scatter(P0[mask_0], P1[mask_0],
                   c=color_blue, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.3)
        ax.scatter(P0[mask_1], P1[mask_1],
                   c=color_orange, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.3)

        lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.15

        # Grid lines (4 quadrants)
        ax.axhline(0, color='#404040', alpha=0.6, linewidth=1.0, linestyle='-')
        ax.axvline(0, color='#404040', alpha=0.6, linewidth=1.0, linestyle='-')

        # y = -x reference line (ceiling line)
        ax.plot([-lim, lim], [lim, -lim], 'r--', linewidth=1.5, alpha=0.7, label='y = -x (ceiling)')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('Class 0 Influence (P⁰)')
        ax.set_ylabel('Class 1 Influence (P¹)')

        panel = "E" if row == 0 else "E'"
        ax.set_title(f"{panel}: Corr={metrics['correlation']:.3f}\nTradeoff: {metrics['tradeoff_frac']:.1%}", fontweight='bold')

        # Add quadrant labels
        ax.text(lim*0.7, lim*0.7, f"JP: {metrics['joint_pos']}", fontsize=8, ha='center', color='green')
        ax.text(-lim*0.7, -lim*0.7, f"JN: {metrics['joint_neg']}", fontsize=8, ha='center', color='red')
        ax.text(lim*0.7, -lim*0.7, f"T0: {metrics['tradeoff_0']}", fontsize=8, ha='center', color='blue')
        ax.text(-lim*0.7, lim*0.7, f"T1: {metrics['tradeoff_1']}", fontsize=8, ha='center', color='orange')

        # ==================== Column 3: Net Influence ====================
        ax = axes[row, 2]

        vmax = np.percentile(np.abs(net_inf), 95)
        if vmax < 1e-6:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        scatter = ax.scatter(train_X[:, 0], train_X[:, 1],
                             c=net_inf, cmap=cmap, norm=norm,
                             s=marker_size, alpha=0.9,
                             edgecolors='white', linewidth=0.3)

        ax.plot(x_bound, y_bound, 'k-', linewidth=1.5)

        ax.set_xlim(xlim)
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        panel = "F" if row == 0 else "F'"
        ax.set_title(f"{panel}: Net Influence (P⁰+P¹)\nResidual σ: {metrics['residual_std']:.1f}", fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    # Add main title
    fig.suptitle("Performance Ceiling Analysis: Partial vs Complete Ceiling",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("PERFORMANCE CEILING COMPARISON")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # ==================== PAPER-MATCHING (PARTIAL CEILING) ====================
    print("\n" + "=" * 50)
    print("CASE 1: PAPER-MATCHING (PARTIAL CEILING)")
    print("=" * 50)

    X_partial, y_partial = generate_paper_matching(n_per_class=300, seed=42)
    train_X_p, train_y_p, inf_p, weights_p, metrics_p = run_experiment(X_partial, y_partial, "Paper-Matching")

    print(f"  Accuracy: {metrics_p['accuracy']:.1%}")
    print(f"  Correlation: {metrics_p['correlation']:.3f}")
    print(f"  Tradeoff fraction: {metrics_p['tradeoff_frac']:.1%}")
    print(f"  Joint fraction: {metrics_p['joint_frac']:.1%}")
    print(f"  Explained variance: {metrics_p['explained_var']:.1%}")
    print(f"  Residual (P0+P1) std: {metrics_p['residual_std']:.2f}")

    # ==================== TRUE CEILING ====================
    print("\n" + "=" * 50)
    print("CASE 2: TRUE CEILING")
    print("=" * 50)

    X_ceiling, y_ceiling = generate_true_ceiling(n_per_class=300, seed=42)
    train_X_c, train_y_c, inf_c, weights_c, metrics_c = run_experiment(X_ceiling, y_ceiling, "True Ceiling")

    print(f"  Accuracy: {metrics_c['accuracy']:.1%}")
    print(f"  Correlation: {metrics_c['correlation']:.3f}")
    print(f"  Tradeoff fraction: {metrics_c['tradeoff_frac']:.1%}")
    print(f"  Joint fraction: {metrics_c['joint_frac']:.1%}")
    print(f"  Explained variance: {metrics_c['explained_var']:.1%}")
    print(f"  Residual (P0+P1) std: {metrics_c['residual_std']:.2f}")

    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Partial Ceiling':>18} {'True Ceiling':>18}")
    print("-" * 65)
    print(f"{'Accuracy':<25} {metrics_p['accuracy']:>17.1%} {metrics_c['accuracy']:>17.1%}")
    print(f"{'Correlation':<25} {metrics_p['correlation']:>18.3f} {metrics_c['correlation']:>18.3f}")
    print(f"{'Tradeoff fraction':<25} {metrics_p['tradeoff_frac']:>17.1%} {metrics_c['tradeoff_frac']:>17.1%}")
    print(f"{'Joint fraction':<25} {metrics_p['joint_frac']:>17.1%} {metrics_c['joint_frac']:>17.1%}")
    print(f"{'Explained variance':<25} {metrics_p['explained_var']:>17.1%} {metrics_c['explained_var']:>17.1%}")
    print(f"{'Residual std':<25} {metrics_p['residual_std']:>18.2f} {metrics_c['residual_std']:>18.2f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("  - PARTIAL CEILING: Positive correlation, curved arms in tradeoff quadrants")
    print("  - TRUE CEILING: Negative correlation, samples on y = -x line (P0+P1 ≈ 0)")
    print("  - Both show 'line structure' but meaning is DIFFERENT!")
    print("=" * 70)

    # Generate comparison figure
    results_partial = (train_X_p, train_y_p, inf_p, weights_p, metrics_p)
    results_ceiling = (train_X_c, train_y_c, inf_c, weights_c, metrics_c)

    save_path = os.path.join(output_dir, 'figure2_ceiling_comparison.png')
    create_comparison_figure(results_partial, results_ceiling, save_path)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
