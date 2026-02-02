"""
analysis_nonlinear_ceiling.py - Finding True Pareto Ceiling Configuration

The paper's Panel E shows influence vectors tightly aligned on y = -x (NEGATIVE correlation).
Our experiments show POSITIVE correlation. This script investigates why and finds the correct configuration.

KEY INSIGHT:
- Positive correlation (P0 ∝ P1): Changes affect both classes SIMILARLY (joint behavior)
- Negative correlation (P0 ∝ -P1): Changes affect classes OPPOSITELY (tradeoff behavior)

For TRUE ceiling (negative correlation), we need:
- Model at optimum (can't improve overall)
- Any weight change trades off between classes (helps one, hurts other)

Hypothesis: The paper uses a configuration where the boundary is the ONLY reasonable choice,
and samples near it create symmetric tradeoffs.

Author: Paper Reproduction Analysis
Date: 2026-02
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.influence import (
    train_logistic_regression, compute_predictions,
    compute_hessian, compute_inverse_hessian, compute_sample_gradient
)


def create_paper_colormap():
    colors = [
        (0.0, '#1f77b4'), (0.25, '#6baed6'), (0.45, '#c6dbef'),
        (0.5, '#ffffff'), (0.55, '#fdd0a2'), (0.75, '#fd8d3c'), (1.0, '#d62728'),
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


def uniform_disk(n, radius, center=(0, 0)):
    pts = []
    while len(pts) < n:
        x = np.random.uniform(-radius, radius, n * 2)
        y = np.random.uniform(-radius, radius, n * 2)
        mask = (x**2 + y**2) <= radius**2
        valid = np.column_stack([x[mask] + center[0], y[mask] + center[1]])
        pts.extend(valid[:n - len(pts)])
    return np.array(pts[:n])


def compute_influence(train_X, train_y, val_X, val_y, weights, damping=1e-4):
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


def check_ceiling(influence_vectors):
    P0, P1 = influence_vectors[:, 0], influence_vectors[:, 1]
    centered = influence_vectors - influence_vectors.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    explained_ratio = eigenvalues[-1] / eigenvalues.sum()
    corr = np.corrcoef(P0, P1)[0, 1]
    return explained_ratio, corr


def generate_ceiling_dataset(n_per_class=300, config="paper_interleaved", seed=42):
    """
    Generate datasets designed to achieve the Pareto ceiling (negative correlation).

    Configurations:
    - "paper_interleaved": Two heavily overlapping Gaussians centered on diagonal
    - "boundary_symmetric": Samples symmetric around the y=-x line
    - "random_labels": Same distribution, random labels (50% accuracy guaranteed)
    - "xor_boundary_aligned": XOR pattern aligned with diagonal boundary
    """
    np.random.seed(seed)

    if config == "paper_interleaved":
        # Match paper's Panel D: overlapping clusters along the diagonal
        # Key: both classes have substantial presence on BOTH sides of boundary
        center_offset = 0.5
        spread = 1.5

        # Class 0: centered below-left of boundary, but spread across
        X0 = np.random.randn(n_per_class, 2) * spread
        X0[:, 0] -= center_offset
        X0[:, 1] -= center_offset

        # Class 1: centered above-right of boundary, but spread across
        X1 = np.random.randn(n_per_class, 2) * spread
        X1[:, 0] += center_offset
        X1[:, 1] += center_offset

    elif config == "boundary_symmetric":
        # Samples clustered ALONG the boundary y = -x
        # Labels alternate based on perpendicular distance
        n_total = 2 * n_per_class

        # Generate points along y = -x + noise
        t = np.random.uniform(-3, 3, n_total)  # Position along boundary
        perp_noise = np.random.randn(n_total) * 0.5  # Perpendicular spread

        # Points on the line y = -x are at (t, -t)
        # Perpendicular direction is (1, 1) / sqrt(2)
        x = t + perp_noise / np.sqrt(2)
        y = -t + perp_noise / np.sqrt(2)

        X = np.column_stack([x, y])

        # Labels based on perpendicular distance (which side of boundary)
        # d = (x + y) / sqrt(2)
        d = (X[:, 0] + X[:, 1]) / np.sqrt(2)
        labels = (d > 0).astype(int)

        # Shuffle and balance
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        if len(idx_0) > n_per_class:
            idx_0 = np.random.choice(idx_0, n_per_class, replace=False)
        if len(idx_1) > n_per_class:
            idx_1 = np.random.choice(idx_1, n_per_class, replace=False)

        all_idx = np.concatenate([idx_0, idx_1])
        X = X[all_idx]
        y = labels[all_idx]

        return X, y

    elif config == "random_labels":
        # Both classes from SAME distribution
        # Labels are RANDOM → model can't do better than 50%
        X = np.random.randn(2 * n_per_class, 2) * 1.5
        y = np.random.randint(0, 2, 2 * n_per_class)
        return X, y

    elif config == "xor_boundary_aligned":
        # XOR pattern where diagonal boundary creates maximum tradeoff
        # Clusters at positions that the diagonal cuts through
        r = 1.2

        # Class 0: upper-left and lower-right quadrants relative to y=-x
        # These are positions where x + y > 0 or x + y < 0 alternating
        X0_1 = np.random.randn(n_per_class // 2, 2) * 0.5 + np.array([1.5, 0])
        X0_2 = np.random.randn(n_per_class // 2, 2) * 0.5 + np.array([0, -1.5])
        X0 = np.vstack([X0_1, X0_2])

        # Class 1: other two quadrants
        X1_1 = np.random.randn(n_per_class // 2, 2) * 0.5 + np.array([-1.5, 0])
        X1_2 = np.random.randn(n_per_class // 2, 2) * 0.5 + np.array([0, 1.5])
        X1 = np.vstack([X1_1, X1_2])

    elif config == "circular_xor":
        # Circular arrangement where diagonal creates tradeoffs
        angles = np.linspace(0, 2*np.pi, 2*n_per_class, endpoint=False)
        angles += np.random.randn(2*n_per_class) * 0.1  # Small noise
        r = 2.0

        X = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

        # Labels based on quadrant (XOR of x>0 and y>0)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

        return X, y

    elif config == "stripe_pattern":
        # Stripes perpendicular to the diagonal boundary
        n_total = 2 * n_per_class
        X = np.random.randn(n_total, 2) * 2

        # Distance from diagonal y = -x
        d = (X[:, 0] + X[:, 1]) / np.sqrt(2)

        # Labels based on which stripe (creates inherent tradeoff)
        stripe_idx = np.floor(d / 0.5).astype(int)
        y = (stripe_idx % 2).astype(int)

        # Balance classes
        idx_0 = np.where(y == 0)[0][:n_per_class]
        idx_1 = np.where(y == 1)[0][:n_per_class]
        all_idx = np.concatenate([idx_0, idx_1])

        return X[all_idx], y[all_idx]

    else:
        raise ValueError(f"Unknown config: {config}")

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def analyze_and_plot(config_name, X, y, ax_row, damping=1e-4):
    """Analyze a dataset and plot results."""
    np.random.seed(42)
    n = len(y)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)

    train_X, train_y = X[perm[:n_train]], y[perm[:n_train]]
    val_X, val_y = X[perm[n_train:]], y[perm[n_train:]]

    # Train
    weights = train_logistic_regression(train_X, train_y, n_iterations=1000, l2_reg=0.01)

    # Accuracy
    preds = (compute_predictions(train_X, weights) >= 0.5).astype(int)
    acc = np.mean(preds == train_y)

    # Influence
    influence = compute_influence(train_X, train_y, val_X, val_y, weights, damping)
    explained, corr = check_ceiling(influence)

    P0, P1 = influence[:, 0], influence[:, 1]
    net_inf = P0 + P1

    mask_0 = train_y == 0
    mask_1 = train_y == 1

    cmap = create_paper_colormap()

    # Panel A: Data
    ax = ax_row[0]
    ax.scatter(train_X[mask_0, 0], train_X[mask_0, 1], c='#1f77b4', s=25, alpha=0.7,
               edgecolors='white', linewidth=0.2, label='Class 0')
    ax.scatter(train_X[mask_1, 0], train_X[mask_1, 1], c='#ff7f0e', s=25, alpha=0.7,
               edgecolors='white', linewidth=0.2, label='Class 1')

    xlim = [train_X[:, 0].min() - 0.3, train_X[:, 0].max() + 0.3]
    ylim = [train_X[:, 1].min() - 0.3, train_X[:, 1].max() + 0.3]
    xx = np.array([min(xlim[0], ylim[0]) - 1, max(xlim[1], ylim[1]) + 1])
    ax.plot(xx, -xx, 'k-', linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'{config_name}\nAcc: {acc:.1%}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

    # Panel B: Influence Space
    ax = ax_row[1]
    ax.scatter(P0[mask_0], P1[mask_0], c='#1f77b4', s=25, alpha=0.7,
               edgecolors='white', linewidth=0.2)
    ax.scatter(P0[mask_1], P1[mask_1], c='#ff7f0e', s=25, alpha=0.7,
               edgecolors='white', linewidth=0.2)

    lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.1
    if lim < 1:
        lim = 10
    ax.plot([-lim, lim], [lim, -lim], 'k-', linewidth=2, label='Pareto frontier')
    ax.axhline(0, color='gray', alpha=0.3, linestyle='--')
    ax.axvline(0, color='gray', alpha=0.3, linestyle='--')

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_title(f'Exp: {explained:.1%}, Corr: {corr:.2f}', fontsize=11)
    ax.set_xlabel('P0', fontsize=9)
    ax.set_ylabel('P1', fontsize=9)

    # Panel C: Net Influence
    ax = ax_row[2]
    vmax = np.percentile(np.abs(net_inf), 97)
    if vmax < 1e-6:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    scatter = ax.scatter(train_X[:, 0], train_X[:, 1], c=net_inf, cmap=cmap, norm=norm,
                         s=25, alpha=0.9, edgecolors='white', linewidth=0.2)
    ax.plot(xx, -xx, 'k-', linewidth=2)

    plt.colorbar(scatter, ax=ax, shrink=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

    return {
        'accuracy': acc,
        'explained': explained,
        'correlation': corr,
        'is_ceiling': explained > 0.9 and corr < -0.8
    }


def run_ceiling_analysis():
    """Main analysis to find true ceiling configuration."""
    print("=" * 70)
    print("PARETO CEILING ANALYSIS - Finding Negative Correlation")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        "paper_interleaved",
        "boundary_symmetric",
        "random_labels",
        "xor_boundary_aligned",
        "circular_xor",
        "stripe_pattern",
    ]

    fig, axes = plt.subplots(len(configs), 3, figsize=(16, 4 * len(configs)))

    results = {}

    print("\nTesting configurations for negative correlation (ceiling)...")
    print("-" * 70)

    for idx, config in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] Testing: {config}")

        X, y = generate_ceiling_dataset(n_per_class=300, config=config, seed=42)
        res = analyze_and_plot(config, X, y, axes[idx], damping=1e-4)
        results[config] = res

        status = "✓ CEILING" if res['is_ceiling'] else "✗ Not ceiling"
        print(f"    Accuracy: {res['accuracy']:.1%}")
        print(f"    Explained Variance: {res['explained']:.1%}")
        print(f"    Correlation: {res['correlation']:.3f}")
        print(f"    Status: {status}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'analysis_ceiling_configs.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_path}")
    plt.close(fig)

    # Find best ceiling match
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    best_config = None
    best_score = float('inf')  # Lower correlation is better for ceiling

    for config, res in results.items():
        if res['explained'] > 0.7:  # Must have some linearity
            score = res['correlation']  # Want most negative
            if score < best_score:
                best_score = score
                best_config = config

    print(f"\nBest ceiling match: {best_config}")
    if best_config:
        print(f"  Correlation: {results[best_config]['correlation']:.3f}")
        print(f"  Explained: {results[best_config]['explained']:.1%}")
        print(f"  Accuracy: {results[best_config]['accuracy']:.1%}")

    return results


if __name__ == "__main__":
    results = run_ceiling_analysis()
