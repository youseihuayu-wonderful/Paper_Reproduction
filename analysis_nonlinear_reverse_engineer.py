"""
analysis_nonlinear_reverse_engineer.py - Deep Analysis of Non-Linear Dataset

This script systematically analyzes and reverse-engineers the dataset generation
process to match the original paper's Figure 2 D-F (non-linearly separable case).

Key Observations from Original Paper:
1. Panel D: Two overlapping circular clusters (not flower pattern)
2. Panel E: TIGHT LINE along y=-x (not scattered cloud)
3. Panel F: Mostly gray/neutral (not mixed colors)

Hypothesis: The paper uses heavily overlapping identical distributions,
making the linear classifier fundamentally unable to separate classes.

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
    """Blue → White → Red colormap."""
    colors = [
        (0.0, '#1f77b4'),
        (0.25, '#6baed6'),
        (0.45, '#c6dbef'),
        (0.5, '#ffffff'),
        (0.55, '#fdd0a2'),
        (0.75, '#fd8d3c'),
        (1.0, '#d62728'),
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
    """Generate uniformly distributed points in a disk."""
    pts = []
    while len(pts) < n:
        x = np.random.uniform(-radius, radius, n * 2)
        y = np.random.uniform(-radius, radius, n * 2)
        mask = (x**2 + y**2) <= radius**2
        valid = np.column_stack([x[mask] + center[0], y[mask] + center[1]])
        pts.extend(valid[:n - len(pts)])
    return np.array(pts[:n])


def compute_influence_fast(train_X, train_y, val_X, val_y, weights, damping=1e-4):
    """Compute category-wise influence vectors."""
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
    """Check if Pareto ceiling is reached (vectors form a line)."""
    P0, P1 = influence_vectors[:, 0], influence_vectors[:, 1]

    # PCA to check linearity
    centered = influence_vectors - influence_vectors.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    explained_ratio = eigenvalues[-1] / eigenvalues.sum()

    # Correlation (should be strongly negative for ceiling)
    corr = np.corrcoef(P0, P1)[0, 1]

    return explained_ratio, corr


def generate_dataset_type(dtype, n_per_class=300, seed=42):
    """Generate different dataset types for comparison."""
    np.random.seed(seed)

    if dtype == "paper_style":
        # Hypothesis: Paper uses heavily overlapping disks
        # Both classes centered near origin with small offset
        offset = 0.3
        radius = 1.8
        X0 = uniform_disk(n_per_class, radius, center=(-offset, -offset))
        X1 = uniform_disk(n_per_class, radius, center=(offset, offset))

    elif dtype == "high_overlap":
        # Even more overlap - nearly identical distributions
        offset = 0.2
        radius = 2.0
        X0 = uniform_disk(n_per_class, radius, center=(-offset, -offset))
        X1 = uniform_disk(n_per_class, radius, center=(offset, offset))

    elif dtype == "xor_classic":
        # Classic XOR pattern
        radius = 1.0
        X0_q1 = uniform_disk(n_per_class // 2, radius, center=(1.5, 1.5))
        X0_q3 = uniform_disk(n_per_class // 2, radius, center=(-1.5, -1.5))
        X1_q2 = uniform_disk(n_per_class // 2, radius, center=(-1.5, 1.5))
        X1_q4 = uniform_disk(n_per_class // 2, radius, center=(1.5, -1.5))
        X0 = np.vstack([X0_q1, X0_q3])
        X1 = np.vstack([X1_q2, X1_q4])

    elif dtype == "concentric":
        # Concentric circles
        angles = np.random.uniform(0, 2*np.pi, n_per_class)
        r0 = np.sqrt(np.random.uniform(0, 1, n_per_class)) * 1.0
        X0 = np.column_stack([r0 * np.cos(angles), r0 * np.sin(angles)])

        angles = np.random.uniform(0, 2*np.pi, n_per_class)
        r1 = np.sqrt(np.random.uniform(0.5, 1, n_per_class)) * 2.0
        X1 = np.column_stack([r1 * np.cos(angles), r1 * np.sin(angles)])

    elif dtype == "flower_original":
        # Our original flower pattern
        angles_blue = np.random.uniform(0, 2*np.pi, n_per_class)
        radii_blue = 1.5 * np.sqrt(np.random.uniform(0, 1, n_per_class))
        X0 = np.column_stack([radii_blue * np.cos(angles_blue),
                              radii_blue * np.sin(angles_blue)])

        angles_orange = np.random.uniform(0, 2*np.pi, n_per_class)
        radii_orange = 1.0 + 0.8 * np.sin(3 * angles_orange)
        radii_orange *= np.sqrt(np.random.uniform(0.5, 1, n_per_class))
        X1 = np.column_stack([radii_orange * np.cos(angles_orange),
                              radii_orange * np.sin(angles_orange)])
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    return X, y


def analyze_dataset(X, y, damping=1e-4, title="Dataset"):
    """Run full analysis on a dataset."""
    # Split
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
    influence = compute_influence_fast(train_X, train_y, val_X, val_y, weights, damping)

    # Ceiling check
    explained, corr = check_ceiling(influence)

    return {
        'train_X': train_X,
        'train_y': train_y,
        'weights': weights,
        'accuracy': acc,
        'influence': influence,
        'explained_variance': explained,
        'correlation': corr
    }


def create_comparison_figure(results, save_path):
    """Create comparison figure for different dataset types."""
    n_types = len(results)
    fig, axes = plt.subplots(n_types, 3, figsize=(18, 5 * n_types))

    if n_types == 1:
        axes = axes.reshape(1, -1)

    cmap = create_paper_colormap()

    for idx, (dtype, res) in enumerate(results.items()):
        X = res['train_X']
        y = res['train_y']
        P0, P1 = res['influence'][:, 0], res['influence'][:, 1]
        net_inf = P0 + P1

        mask_0 = y == 0
        mask_1 = y == 1

        # Panel A: Data
        ax = axes[idx, 0]
        ax.scatter(X[mask_0, 0], X[mask_0, 1], c='#1f77b4', s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.2, label='Class 0')
        ax.scatter(X[mask_1, 0], X[mask_1, 1], c='#ff7f0e', s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.2, label='Class 1')

        xlim = [X[:, 0].min() - 0.3, X[:, 0].max() + 0.3]
        ylim = [X[:, 1].min() - 0.3, X[:, 1].max() + 0.3]
        xx = np.array([xlim[0], xlim[1]])
        ax.plot(xx, -xx, 'k-', linewidth=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'{dtype}\nAcc: {res["accuracy"]:.1%}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Panel B: Influence Space
        ax = axes[idx, 1]
        ax.scatter(P0[mask_0], P1[mask_0], c='#1f77b4', s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.2)
        ax.scatter(P0[mask_1], P1[mask_1], c='#ff7f0e', s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.2)

        lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.1
        ax.plot([-lim, lim], [lim, -lim], 'k-', linewidth=2)
        ax.axhline(0, color='gray', alpha=0.3, linestyle='--')
        ax.axvline(0, color='gray', alpha=0.3, linestyle='--')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'Explained: {res["explained_variance"]:.1%}\nCorr: {res["correlation"]:.3f}',
                     fontsize=12)
        ax.set_xlabel('P0 (Class 0 Influence)')
        ax.set_ylabel('P1 (Class 1 Influence)')

        # Panel C: Net Influence
        ax = axes[idx, 2]
        vmax = np.percentile(np.abs(net_inf), 97)
        if vmax < 1e-6:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        scatter = ax.scatter(X[:, 0], X[:, 1], c=net_inf, cmap=cmap, norm=norm,
                             s=30, alpha=0.9, edgecolors='white', linewidth=0.2)
        ax.plot(xx, -xx, 'k-', linewidth=2)

        plt.colorbar(scatter, ax=ax, shrink=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig


def run_analysis():
    """Main analysis function."""
    print("=" * 70)
    print("NON-LINEAR DATASET REVERSE ENGINEERING ANALYSIS")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Test different dataset types
    dataset_types = [
        "paper_style",      # Hypothesis: heavily overlapping disks
        "high_overlap",     # Even more overlap
        "xor_classic",      # Classic XOR
        "concentric",       # Concentric circles
        "flower_original",  # Our current implementation
    ]

    results = {}

    print("\n[1] Analyzing different dataset types...")
    for dtype in dataset_types:
        print(f"\n  Testing: {dtype}")
        X, y = generate_dataset_type(dtype, n_per_class=300, seed=42)
        res = analyze_dataset(X, y, damping=1e-4)
        results[dtype] = res

        print(f"    Accuracy: {res['accuracy']:.1%}")
        print(f"    Explained Variance: {res['explained_variance']:.1%}")
        print(f"    Correlation(P0,P1): {res['correlation']:.3f}")

        # Ceiling detection
        is_ceiling = res['explained_variance'] > 0.9 and res['correlation'] < -0.9
        print(f"    Ceiling Reached: {'YES' if is_ceiling else 'NO'}")

    print("\n[2] Creating comparison figure...")
    save_path = os.path.join(output_dir, 'analysis_nonlinear_types.png')
    fig = create_comparison_figure(results, save_path)
    plt.close(fig)

    # Find best match to paper
    print("\n[3] Finding best match to paper's Panel E (tight line)...")
    best_type = None
    best_explained = 0

    for dtype, res in results.items():
        if res['explained_variance'] > best_explained and res['correlation'] < -0.5:
            best_explained = res['explained_variance']
            best_type = dtype

    print(f"\n  Best match: {best_type}")
    print(f"  Explained variance: {best_explained:.1%}")

    # Fine-tune the best type
    print("\n[4] Fine-tuning 'paper_style' parameters...")
    fine_tune_results = {}

    for offset in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for radius in [1.5, 1.8, 2.0, 2.2]:
            np.random.seed(42)
            X0 = uniform_disk(300, radius, center=(-offset, -offset))
            X1 = uniform_disk(300, radius, center=(offset, offset))
            X = np.vstack([X0, X1])
            y = np.array([0] * 300 + [1] * 300)

            res = analyze_dataset(X, y, damping=1e-4)
            key = f"offset={offset}_radius={radius}"
            fine_tune_results[key] = res

            if res['explained_variance'] > 0.85:
                print(f"    {key}: Exp={res['explained_variance']:.1%}, "
                      f"Corr={res['correlation']:.3f}, Acc={res['accuracy']:.1%}")

    # Find optimal parameters
    best_key = max(fine_tune_results.keys(),
                   key=lambda k: fine_tune_results[k]['explained_variance']
                                 if fine_tune_results[k]['correlation'] < -0.8 else 0)

    print(f"\n  OPTIMAL PARAMETERS: {best_key}")
    print(f"  Explained: {fine_tune_results[best_key]['explained_variance']:.1%}")
    print(f"  Correlation: {fine_tune_results[best_key]['correlation']:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results, fine_tune_results


if __name__ == "__main__":
    results, fine_tune = run_analysis()
