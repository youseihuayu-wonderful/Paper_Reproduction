"""
figure2_comprehensive.py - Publication Quality Figure 2 Reproduction

Carefully reverse-engineered from original paper's Figure 2 (arXiv:2510.03950):

Panel A-C (Linear Separable with Noise):
- Two circular clusters NEXT TO EACH OTHER
- Blue: upper-left, Orange: lower-right
- Label noise creates mislabeled samples
- Panel B: 4 quadrants with dark grid lines
- Panel C: Red dotted circles mark Joint Negative samples

Panel D-F (Non-Separable - Pareto Ceiling):
- Same cluster positions as A-C but MORE OVERLAP
- Blue: upper-left, Orange: lower-right
- Heavy overlap creates ceiling behavior
- Panel E: Curved bands along Pareto frontier
- Panel F: Mostly gray/neutral samples

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
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'figure.dpi': 150,
        'savefig.dpi': 1200,  # Super high resolution
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 2.0,
        'patch.linewidth': 1.5,
    })


def create_paper_colormap():
    """Blue → White → Red colormap matching paper exactly."""
    colors = [
        (0.0, '#2166ac'),   # Dark blue (negative/harmful)
        (0.15, '#4393c3'),
        (0.35, '#92c5de'),
        (0.45, '#d1e5f0'),
        (0.5, '#f7f7f7'),   # White (neutral)
        (0.55, '#fddbc7'),
        (0.65, '#f4a582'),
        (0.85, '#d6604d'),
        (1.0, '#b2182b'),   # Dark red (positive/beneficial)
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


def uniform_disk(n, radius, center):
    """Generate uniformly distributed points in a disk (circular cluster)."""
    pts = []
    while len(pts) < n:
        x = np.random.uniform(-radius, radius, n * 2)
        y = np.random.uniform(-radius, radius, n * 2)
        mask = (x**2 + y**2) <= radius**2
        valid = np.column_stack([x[mask] + center[0], y[mask] + center[1]])
        pts.extend(valid[:n - len(pts)])
    return np.array(pts[:n])


def generate_linear_separable_dataset(n_per_class=300, n_noise_blue=50, n_noise_orange=20, seed=42):
    """
    Generate LINEAR SEPARABLE dataset matching paper's Panel A.

    Reverse-engineered from original paper:
    - Two circular clusters NEXT TO EACH OTHER (touching)
    - Blue: upper-left at (-1, 1)
    - Orange: lower-right at (1, -1)
    - Radius ~1.0 so clusters touch at the diagonal
    """
    np.random.seed(seed)

    # Cluster positions matching original paper Panel A
    center_blue = np.array([-1.0, 1.0])    # Upper-left
    center_orange = np.array([1.0, -1.0])   # Lower-right
    radius = 1.0  # Clusters touch at the diagonal

    X_blue = uniform_disk(n_per_class, radius, center_blue)
    X_orange = uniform_disk(n_per_class, radius, center_orange)

    X = np.vstack([X_blue, X_orange])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Random label flips (creates mislabeled samples)
    blue_indices = np.arange(n_per_class)
    orange_indices = np.arange(n_per_class, 2 * n_per_class)

    flip_blue = np.random.choice(blue_indices, n_noise_blue, replace=False)
    flip_orange = np.random.choice(orange_indices, n_noise_orange, replace=False)

    y[flip_blue] = 1
    y[flip_orange] = 0

    noisy_indices = np.concatenate([flip_blue, flip_orange])

    return X, y, noisy_indices


def generate_nonlinear_dataset(n_per_class=300, seed=42):
    """
    Generate NON-SEPARABLE dataset for Pareto ceiling demonstration.

    The original paper's Panel D shows:
    - Two OVERLAPPING Gaussian clusters that interpenetrate
    - Blue cluster: upper-left bias (center at -0.5, 0.5)
    - Orange cluster: lower-right bias (center at 0.5, -0.5)
    - Moderate spread (σ=1.0) creates significant overlap in center
    - This creates ~77% accuracy ceiling for linear classifier

    This produces the two curved arms pattern in Panel E,
    demonstrating the Pareto frontier where improving one
    class necessarily hurts the other.
    """
    np.random.seed(seed)

    # Parameters tuned to match original paper's Panel D appearance
    # Offset 0.5 + Spread 1.0 gives best match:
    # - ~77% accuracy (ceiling behavior)
    # - ~0.60 correlation (moderate positive)
    # - Two curved arms in influence space
    center_blue = np.array([-0.5, 0.5])     # Upper-left bias
    center_orange = np.array([0.5, -0.5])    # Lower-right bias
    spread = 1.0  # Standard deviation - creates overlap in center

    X_blue = np.random.randn(n_per_class, 2) * spread + center_blue
    X_orange = np.random.randn(n_per_class, 2) * spread + center_orange

    X = np.vstack([X_blue, X_orange])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    return X, y


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


def get_decision_boundary(weights, xlim):
    """Get the learned decision boundary from logistic regression weights."""
    w0, w1 = weights[0], weights[1]
    x_vals = np.array(xlim)
    if abs(w1) > 1e-6:
        y_vals = -(w0 / w1) * x_vals
    else:
        y_vals = np.array(xlim)
    return x_vals, y_vals


def create_figure_abc(train_X, train_y, influence, weights, save_path):
    """
    Create Figure 2 A-C (Linear Separable Case) - Publication Quality.

    Features:
    - Panel A: Data with learned decision boundary
    - Panel B: Influence space with DARK grid lines (4 quadrants)
    - Panel C: Net influence with red dotted circles for Joint Negative
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    P0, P1 = influence[:, 0], influence[:, 1]
    net_inf = P0 + P1

    mask_0 = train_y == 0
    mask_1 = train_y == 1

    # Identify Joint Negative samples
    joint_negative_mask = (P0 < 0) & (P1 < 0)

    # Colors matching paper
    color_blue = '#1f77b4'
    color_orange = '#ff7f0e'
    marker_size = 35
    alpha = 0.85

    # ==================== Panel A: Data Distribution ====================
    ax = axes[0]

    ax.scatter(train_X[mask_0, 0], train_X[mask_0, 1],
               c=color_blue, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)
    ax.scatter(train_X[mask_1, 0], train_X[mask_1, 1],
               c=color_orange, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)

    # Learned decision boundary
    xlim = [-3, 3]
    x_bound, y_bound = get_decision_boundary(weights, xlim)
    ax.plot(x_bound, y_bound, 'k-', linewidth=2.0, zorder=1)

    ax.set_xlim(xlim)
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontweight='medium')
    ax.set_ylabel('Feature 2', fontweight='medium')
    ax.set_title('A', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    # ==================== Panel B: Influence Space ====================
    ax = axes[1]

    ax.scatter(P0[mask_0], P1[mask_0],
               c=color_blue, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)
    ax.scatter(P0[mask_1], P1[mask_1],
               c=color_orange, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)

    lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.15

    # DARK grid lines at x=0 and y=0 (4 quadrants) - MORE VISIBLE
    ax.axhline(0, color='#404040', alpha=0.8, linewidth=1.5, linestyle='-', zorder=1)
    ax.axvline(0, color='#404040', alpha=0.8, linewidth=1.5, linestyle='-', zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Class 0 Influence', fontweight='medium')
    ax.set_ylabel('Class 1 Influence', fontweight='medium')
    ax.set_title('B', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)

    # ==================== Panel C: Net Influence ====================
    ax = axes[2]

    cmap = create_paper_colormap()
    vmax = np.percentile(np.abs(net_inf), 95)
    if vmax < 1e-6:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Plot ALL samples with influence coloring
    scatter = ax.scatter(train_X[:, 0], train_X[:, 1],
                         c=net_inf, cmap=cmap, norm=norm,
                         s=marker_size, alpha=0.9,
                         edgecolors='white', linewidth=0.4, zorder=2)

    # Mark Joint Negative samples with RED DOTTED CIRCLES
    if np.sum(joint_negative_mask) > 0:
        ax.scatter(train_X[joint_negative_mask, 0], train_X[joint_negative_mask, 1],
                   facecolors='none', edgecolors='#b2182b',
                   s=marker_size*2.5, linewidth=2.0,
                   zorder=3)

    # Decision boundary
    ax.plot(x_bound, y_bound, 'k-', linewidth=2.0, zorder=1)

    ax.set_xlim(xlim)
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontweight='medium')
    ax.set_ylabel('Feature 2', fontweight='medium')
    ax.set_title('C', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cbar.ax.tick_params(labelsize=10, width=1.2)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.text(1.5, 0.92, 'Positive', transform=cbar.ax.transAxes,
                 fontsize=10, va='top', ha='left', color='#b2182b', fontweight='medium')
    cbar.ax.text(1.5, 0.5, 'Neutral', transform=cbar.ax.transAxes,
                 fontsize=10, va='center', ha='left', color='#666666', fontweight='medium')
    cbar.ax.text(1.5, 0.08, 'Negative', transform=cbar.ax.transAxes,
                 fontsize=10, va='bottom', ha='left', color='#2166ac', fontweight='medium')
    cbar.ax.set_ylabel('Class-Wise\nInfluence Score', fontsize=11, rotation=270,
                       labelpad=40, va='center', fontweight='medium')

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"Saved: {save_path} (1200 DPI)")
    return fig


def create_figure_def(train_X, train_y, influence, weights, save_path):
    """
    Create Figure 2 D-F (Non-Separable / Ceiling Case) - Publication Quality.

    Features:
    - Panel D: Overlapping clusters with visible structure
    - Panel E: Influence vectors with DARK grid lines
    - Panel F: Mostly gray/neutral samples
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    P0, P1 = influence[:, 0], influence[:, 1]
    net_inf = P0 + P1

    mask_0 = train_y == 0
    mask_1 = train_y == 1

    color_blue = '#1f77b4'
    color_orange = '#ff7f0e'
    marker_size = 35
    alpha = 0.85

    # ==================== Panel D: Data Distribution ====================
    ax = axes[0]

    ax.scatter(train_X[mask_0, 0], train_X[mask_0, 1],
               c=color_blue, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)
    ax.scatter(train_X[mask_1, 0], train_X[mask_1, 1],
               c=color_orange, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)

    # Learned decision boundary
    xlim = [-3, 3]
    x_bound, y_bound = get_decision_boundary(weights, xlim)
    ax.plot(x_bound, y_bound, 'k-', linewidth=2.0, zorder=1)

    ax.set_xlim(xlim)
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontweight='medium')
    ax.set_ylabel('Feature 2', fontweight='medium')
    ax.set_title('D', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    # ==================== Panel E: Influence Space ====================
    ax = axes[1]

    ax.scatter(P0[mask_0], P1[mask_0],
               c=color_blue, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)
    ax.scatter(P0[mask_1], P1[mask_1],
               c=color_orange, s=marker_size, alpha=alpha,
               edgecolors='white', linewidth=0.4, zorder=2)

    lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.15

    # DARK grid lines at x=0 and y=0 - MORE VISIBLE
    ax.axhline(0, color='#404040', alpha=0.8, linewidth=1.5, linestyle='-', zorder=1)
    ax.axvline(0, color='#404040', alpha=0.8, linewidth=1.5, linestyle='-', zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Class 0 Influence', fontweight='medium')
    ax.set_ylabel('Class 1 Influence', fontweight='medium')
    ax.set_title('E', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)

    # ==================== Panel F: Net Influence ====================
    ax = axes[2]

    cmap = create_paper_colormap()
    vmax = np.percentile(np.abs(net_inf), 95)
    if vmax < 1e-6:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    scatter = ax.scatter(train_X[:, 0], train_X[:, 1],
                         c=net_inf, cmap=cmap, norm=norm,
                         s=marker_size, alpha=0.9,
                         edgecolors='white', linewidth=0.4, zorder=2)

    ax.plot(x_bound, y_bound, 'k-', linewidth=2.0, zorder=1)

    ax.set_xlim(xlim)
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontweight='medium')
    ax.set_ylabel('Feature 2', fontweight='medium')
    ax.set_title('F', loc='left', fontsize=20, fontweight='bold', x=-0.02, y=1.02)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cbar.ax.tick_params(labelsize=10, width=1.2)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.text(1.5, 0.92, 'Positive', transform=cbar.ax.transAxes,
                 fontsize=10, va='top', ha='left', color='#b2182b', fontweight='medium')
    cbar.ax.text(1.5, 0.5, 'Neutral', transform=cbar.ax.transAxes,
                 fontsize=10, va='center', ha='left', color='#666666', fontweight='medium')
    cbar.ax.text(1.5, 0.08, 'Negative', transform=cbar.ax.transAxes,
                 fontsize=10, va='bottom', ha='left', color='#2166ac', fontweight='medium')
    cbar.ax.set_ylabel('Class-Wise\nInfluence Score', fontsize=11, rotation=270,
                       labelpad=40, va='center', fontweight='medium')

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"Saved: {save_path} (1200 DPI)")
    return fig


def main():
    print("=" * 70)
    print("PUBLICATION QUALITY FIGURE 2 REPRODUCTION")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # ==================== LINEAR SEPARABLE (A-C) ====================
    print("\n" + "=" * 50)
    print("PANEL A-C: LINEAR SEPARABLE CASE")
    print("=" * 50)

    print("\n[1] Generating dataset (matching paper's Panel A)...")
    print("    - Uniform disk clusters")
    print("    - Blue: center (-1.0, 1.0), radius 1.0")
    print("    - Orange: center (1.0, -1.0), radius 1.0")
    print("    - Flipping 50 blue + 20 orange labels")

    X, y, noisy_indices = generate_linear_separable_dataset(
        n_per_class=300, n_noise_blue=50, n_noise_orange=20, seed=42
    )

    np.random.seed(42)
    perm = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))

    train_X, train_y = X[perm[:n_train]], y[perm[:n_train]]
    val_X, val_y = X[perm[n_train:]], y[perm[n_train:]]
    train_noisy_mask = np.isin(perm[:n_train], noisy_indices)

    print(f"    Training: {len(train_X)}, Noisy: {np.sum(train_noisy_mask)}")

    print("\n[2] Training logistic regression...")
    weights = train_logistic_regression(train_X, train_y, n_iterations=1000, l2_reg=0.01)
    preds = (compute_predictions(train_X, weights) >= 0.5).astype(int)
    acc = np.mean(preds == train_y)
    print(f"    Accuracy: {acc:.1%}")

    print("\n[3] Computing influence (damping=1e-4)...")
    influence = compute_influence(train_X, train_y, val_X, val_y, weights, damping=1e-4)

    P0, P1 = influence[:, 0], influence[:, 1]
    corr = np.corrcoef(P0, P1)[0, 1]
    print(f"    Correlation(P0, P1): {corr:.3f}")

    joint_neg = (P0 < 0) & (P1 < 0)
    joint_pos = (P0 > 0) & (P1 > 0)
    noisy_in_jn = np.sum(train_noisy_mask & joint_neg)
    print(f"    Joint Positive: {np.sum(joint_pos)}")
    print(f"    Joint Negative: {np.sum(joint_neg)}")
    print(f"    Noisy in JN: {noisy_in_jn}/{np.sum(train_noisy_mask)} ({100*noisy_in_jn/np.sum(train_noisy_mask):.1f}%)")

    print("\n[4] Generating Figure A-C (1200 DPI)...")
    save_path = os.path.join(output_dir, 'figure2_ABC.png')
    fig = create_figure_abc(train_X, train_y, influence, weights, save_path)
    plt.close(fig)

    # ==================== NON-LINEAR (D-F) ====================
    print("\n" + "=" * 50)
    print("PANEL D-F: NON-SEPARABLE CASE (PARETO CEILING)")
    print("=" * 50)

    print("\n[1] Generating dataset (matching paper's Panel D)...")
    print("    - OVERLAPPING Gaussian clusters with interpenetration")
    print("    - Blue: center (-0.5, 0.5), Orange: center (0.5, -0.5)")
    print("    - Spread 1.0 for optimal overlap pattern")

    X_nl, y_nl = generate_nonlinear_dataset(n_per_class=300, seed=42)

    np.random.seed(42)
    perm_nl = np.random.permutation(len(y_nl))
    n_train_nl = int(0.8 * len(y_nl))

    train_X_nl, train_y_nl = X_nl[perm_nl[:n_train_nl]], y_nl[perm_nl[:n_train_nl]]
    val_X_nl, val_y_nl = X_nl[perm_nl[n_train_nl:]], y_nl[perm_nl[n_train_nl:]]

    print(f"    Training: {len(train_X_nl)}")

    print("\n[2] Training logistic regression...")
    weights_nl = train_logistic_regression(train_X_nl, train_y_nl, n_iterations=1000, l2_reg=0.01)
    preds_nl = (compute_predictions(train_X_nl, weights_nl) >= 0.5).astype(int)
    acc_nl = np.mean(preds_nl == train_y_nl)
    print(f"    Accuracy: {acc_nl:.1%}")

    print("\n[3] Computing influence (damping=1e-4)...")
    influence_nl = compute_influence(train_X_nl, train_y_nl, val_X_nl, val_y_nl, weights_nl, damping=1e-4)

    P0_nl, P1_nl = influence_nl[:, 0], influence_nl[:, 1]
    corr_nl = np.corrcoef(P0_nl, P1_nl)[0, 1]
    print(f"    Correlation(P0, P1): {corr_nl:.3f}")

    # Ceiling analysis
    centered = influence_nl - influence_nl.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    explained = eigenvalues[-1] / eigenvalues.sum()
    print(f"    Explained Variance: {explained:.1%}")

    # Tradeoff regions
    tradeoff_01 = (P0_nl > 0) & (P1_nl < 0)
    tradeoff_10 = (P0_nl < 0) & (P1_nl > 0)
    print(f"    Tradeoff (Class 0 up): {np.sum(tradeoff_01)}")
    print(f"    Tradeoff (Class 1 up): {np.sum(tradeoff_10)}")

    print("\n[4] Generating Figure D-F (1200 DPI)...")
    save_path = os.path.join(output_dir, 'figure2_DEF.png')
    fig = create_figure_def(train_X_nl, train_y_nl, influence_nl, weights_nl, save_path)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("COMPLETE - All figures saved at 1200 DPI")
    print("=" * 70)


if __name__ == "__main__":
    main()
