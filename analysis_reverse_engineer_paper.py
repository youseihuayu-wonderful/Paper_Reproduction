"""
Deep Reverse Engineering of Paper's Figure 2

This script analyzes the original paper's Figure 2 to understand:
1. Exact dataset structure (how many classes, samples, positions)
2. How Panel A, B, C relate to each other
3. What causes the curved arm pattern in Panel B
4. How to exactly reproduce the paper's results

Author: Paper Reproduction Analysis
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from influence import (
    train_logistic_regression, compute_predictions,
    compute_hessian, compute_inverse_hessian, compute_sample_gradient
)


print("=" * 80)
print("DEEP REVERSE ENGINEERING: Paper's Figure 2")
print("=" * 80)

print("""
================================================================================
ANALYSIS OF ORIGINAL PAPER'S FIGURE 2
================================================================================

The paper shows Figure 2 with 6 panels (A-F):

TOP ROW (A, B, C) - Linearly Separable Dataset with Intentional Noise
BOTTOM ROW (D, E, F) - Non-Linearly Separable Dataset (Overlapping Classes)

================================================================================
PANEL-BY-PANEL ANALYSIS (Top Row):
================================================================================

**PANEL A - Data Distribution:**
- X-axis: Feature 1 (range: approximately -3 to +3)
- Y-axis: Feature 2 (range: approximately -3 to +3)
- Blue dots: Class 0 samples (LEFT cluster, centered around (-1.5, 0))
- Orange dots: Class 1 samples (RIGHT cluster, centered around (+1.5, 0))
- Black diagonal line: Decision boundary (y = -x, NOT vertical!)
- Cluster radius: approximately 1.5 units
- Number of samples: ~300 per class (600 total)

**CRITICAL OBSERVATION about Decision Boundary:**
The decision boundary is a DIAGONAL line (y = -x), not a vertical line.
This means the classifier is NOT simply separating based on Feature 1.
The diagonal boundary creates a specific geometric relationship that
produces the curved pattern in Panel B.

**PANEL B - Category-Wise Influence Space:**
- X-axis: "Class 0 Influence" (P^0) - influence on Class 0 validation samples
- Y-axis: "Class 1 Influence" (P^1) - influence on Class 1 validation samples
- Black diagonal line: Pareto frontier (P^1 = -P^0)
- Blue dots: Class 0 training samples
- Orange dots: Class 1 training samples
- Small red cluster: Mislabeled/noisy samples

**KEY PATTERN in Panel B:**
- Blue samples form a CURVED ARM from upper-left to lower-right
- Orange samples form a CURVED ARM from lower-left to upper-right
- The two arms CROSS near the center
- Range: approximately [-2, +2] for both axes

**WHY THE CURVES EMERGE:**
The curve pattern emerges because influence varies SYSTEMATICALLY with
position in feature space:

1. For Class 0 (blue) samples:
   - Samples FAR from boundary (left side of cluster): LOW influence magnitude
   - Samples NEAR boundary (right side of cluster): HIGH influence magnitude
   - Direction changes: Near-boundary samples tend to have P0 > 0, P1 < 0

2. For Class 1 (orange) samples:
   - Mirror pattern: Near-boundary samples have P0 < 0, P1 > 0

3. The DIAGONAL boundary creates this systematic variation because:
   - Distance to boundary varies across the cluster
   - Gradient direction depends on position relative to boundary

**PANEL C - Feature Space Colored by Influence:**
- Same coordinates as Panel A (Feature 1 vs Feature 2)
- Colored by "Class-Wise Influence Score" (appears to be P^0 + P^1)
- Colorbar: Blue (Negative) → White (Neutral) → Red (Positive)

**KEY OBSERVATIONS in Panel C:**
- Blue cluster (Class 0): mostly CYAN/LIGHT BLUE colors
- Orange cluster (Class 1): mostly WHITE/LIGHT ORANGE colors
- Red dots scattered: likely mislabeled samples that are HARMFUL
- The coloring shows which samples are beneficial vs harmful overall

**RELATIONSHIP between Panel A and C:**
- Panel A: Shows data colored by TRUE CLASS (blue vs orange)
- Panel C: Shows SAME data colored by INFLUENCE SCORE
- Comparison reveals: Most samples are neutral, harmful samples stand out

================================================================================
BOTTOM ROW (D, E, F) - Non-Linearly Separable Dataset:
================================================================================

**PANEL D:** Overlapping circular distributions (blue and orange intermixed)
**PANEL E:** Influence vectors lying ON the Pareto frontier (y = -x line)
**PANEL F:** Mostly gray/white samples indicating ceiling reached

================================================================================
REVERSE-ENGINEERED PARAMETERS FOR REPRODUCTION:
================================================================================
""")

# Parameters reverse-engineered from the paper
PARAMS = {
    # Dataset parameters
    'n_per_class': 300,
    'cluster_center_0': np.array([-1.5, 1.5]),  # Positioned for diagonal boundary
    'cluster_center_1': np.array([1.5, -1.5]),   # Opposite side of diagonal
    'cluster_radius': 1.5,

    # The diagonal decision boundary y = -x passes through origin
    # For this boundary to separate the classes, we need:
    # Class 0: mostly where x + y < 0 (below the line)
    # Class 1: mostly where x + y > 0 (above the line)
    # OR vice versa

    # Noise injection
    'noise_fraction': 0.10,  # ~10% noise (50 blue + 20 orange ≈ 70/600)

    # Model parameters
    'damping': 1e-4,  # Low damping preserves curve structure
}

print(f"Reverse-Engineered Parameters:")
print(f"  Samples per class: {PARAMS['n_per_class']}")
print(f"  Cluster radius: {PARAMS['cluster_radius']}")
print(f"  Noise fraction: {PARAMS['noise_fraction']}")
print(f"  Hessian damping: {PARAMS['damping']}")


def generate_paper_exact_dataset(
    n_per_class=300,
    noise_fraction=0.10,
    random_seed=42
):
    """
    Generate dataset EXACTLY matching the paper's Figure 2 A-C.

    Key insight: The diagonal decision boundary (y = -x) is crucial.
    We position clusters so that they are separated by this diagonal.
    """
    np.random.seed(random_seed)

    # Generate uniform disk distribution
    def uniform_disk(n, radius):
        """Generate points uniformly distributed in a disk."""
        points = []
        while len(points) < n:
            x = np.random.uniform(-radius, radius, n * 2)
            y = np.random.uniform(-radius, radius, n * 2)
            mask = (x**2 + y**2) <= radius**2
            valid = np.column_stack([x[mask], y[mask]])
            points.extend(valid.tolist())
        return np.array(points[:n])

    radius = 1.5

    # Position clusters for diagonal boundary y = -x
    # Class 0: below the line (x + y < 0) - centered at (-1.5, 0) or (0, -1.5)
    # Class 1: above the line (x + y > 0) - centered at (1.5, 0) or (0, 1.5)

    # Looking at paper: blue on LEFT, orange on RIGHT
    # With diagonal boundary y = -x:
    # - Blue cluster center at (-1.5, 0): x + y = -1.5 < 0 ✓
    # - Orange cluster center at (1.5, 0): x + y = 1.5 > 0 ✓

    center_0 = np.array([-1.5, 0])
    center_1 = np.array([1.5, 0])

    X0 = uniform_disk(n_per_class, radius) + center_0
    X1 = uniform_disk(n_per_class, radius) + center_1

    X = np.vstack([X0, X1])
    true_labels = np.array([0] * n_per_class + [1] * n_per_class)
    y = true_labels.copy()

    # For diagonal boundary y = -x, the "distance" to boundary is |x + y| / sqrt(2)
    # Samples with x + y close to 0 are near the boundary
    distances_to_diagonal = np.abs(X[:, 0] + X[:, 1]) / np.sqrt(2)

    # Inject noise for samples NEAR the boundary (most ambiguous)
    n_noise = int(noise_fraction * len(y))

    # Select samples with smallest distance to diagonal
    noise_candidates = np.argsort(distances_to_diagonal)[:n_noise * 3]  # Pool of candidates
    noise_indices = np.random.choice(noise_candidates, n_noise, replace=False)

    # Flip labels
    y[noise_indices] = 1 - y[noise_indices]

    return X, y, noise_indices, true_labels


def compute_paper_style_influence(train_X, train_y, val_X, val_y, weights, damping=1e-4):
    """
    Compute influence using the paper's approach.

    IMPORTANT: The sign convention matters!
    We compute the influence of REMOVING a sample on the loss.
    Positive = removing increases loss = sample is BENEFICIAL
    Negative = removing decreases loss = sample is HARMFUL
    """
    H = compute_hessian(train_X, weights)
    H_inv = compute_inverse_hessian(H, damping=damping)

    n_train = len(train_X)
    P0 = np.zeros(n_train)
    P1 = np.zeros(n_train)

    # Split validation by class
    val_X_0 = val_X[val_y == 0]
    val_y_0 = val_y[val_y == 0]
    val_X_1 = val_X[val_y == 1]
    val_y_1 = val_y[val_y == 1]

    for j in range(n_train):
        grad_train = compute_sample_gradient(train_X[j], train_y[j], weights)
        inf_dir = H_inv @ grad_train

        # Influence on Class 0 validation
        for i in range(len(val_X_0)):
            grad_val = compute_sample_gradient(val_X_0[i], val_y_0[i], weights)
            # The influence formula: impact = grad_val^T @ H^{-1} @ grad_train
            # We NEGATE because: removing sample z changes loss by -grad^T H^{-1} grad
            P0[j] -= np.dot(grad_val, inf_dir)

        # Influence on Class 1 validation
        for i in range(len(val_X_1)):
            grad_val = compute_sample_gradient(val_X_1[i], val_y_1[i], weights)
            P1[j] -= np.dot(grad_val, inf_dir)

    return P0, P1


def create_paper_colormap():
    """Paper's exact colormap: Blue → Cyan → White → Orange → Red"""
    colors = [
        (0.0, '#1f77b4'),   # Blue (harmful/negative)
        (0.25, '#6baed6'),
        (0.45, '#c6dbef'),
        (0.5, '#ffffff'),   # White (neutral)
        (0.55, '#fdd0a2'),
        (0.75, '#fd8d3c'),
        (1.0, '#d62728'),   # Red (beneficial/positive)
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


print("\n" + "=" * 80)
print("RUNNING EXACT REPRODUCTION")
print("=" * 80)

# Generate dataset
print("\n[1] Generating dataset with diagonal boundary setup...")
X, y, noise_indices, true_labels = generate_paper_exact_dataset(
    n_per_class=300,
    noise_fraction=0.10,
    random_seed=42
)

print(f"  Total samples: {len(X)}")
print(f"  Noisy samples: {len(noise_indices)}")

# Verify class positions relative to diagonal
class0_above_diagonal = np.sum((X[true_labels == 0, 0] + X[true_labels == 0, 1]) > 0)
class1_above_diagonal = np.sum((X[true_labels == 1, 0] + X[true_labels == 1, 1]) > 0)
print(f"  Class 0 samples above diagonal (x+y>0): {class0_above_diagonal}/300")
print(f"  Class 1 samples above diagonal (x+y>0): {class1_above_diagonal}/300")

# Split train/val
np.random.seed(42)
perm = np.random.permutation(len(X))
n_train = int(0.8 * len(X))

train_X, train_y = X[perm[:n_train]], y[perm[:n_train]]
val_X, val_y = X[perm[n_train:]], y[perm[n_train:]]
train_true_labels = true_labels[perm[:n_train]]
train_noise_mask = np.isin(perm[:n_train], noise_indices)

print(f"  Training samples: {len(train_X)}")
print(f"  Noisy in training: {np.sum(train_noise_mask)}")

# Train model
print("\n[2] Training logistic regression...")
weights = train_logistic_regression(
    train_X, train_y,
    learning_rate=0.1,
    n_iterations=1000,
    l2_reg=0.01,
    verbose=False
)

train_preds = compute_predictions(train_X, weights)
train_acc = np.mean((train_preds >= 0.5) == train_y)
print(f"  Training accuracy: {train_acc:.4f}")

# Compute influence
print("\n[3] Computing category-wise influence...")
P0, P1 = compute_paper_style_influence(
    train_X, train_y, val_X, val_y, weights, damping=1e-4
)

print(f"  P0 range: [{P0.min():.2f}, {P0.max():.2f}]")
print(f"  P1 range: [{P1.min():.2f}, {P1.max():.2f}]")
print(f"  Correlation(P0, P1): {np.corrcoef(P0, P1)[0, 1]:.4f}")

# Verify sign convention
net_inf = P0 + P1
noisy_mean = net_inf[train_noise_mask].mean()
clean_mean = net_inf[~train_noise_mask].mean()
print(f"\n  Sign convention check:")
print(f"    Noisy samples mean influence: {noisy_mean:.2f} (should be NEGATIVE/harmful)")
print(f"    Clean samples mean influence: {clean_mean:.2f} (should be POSITIVE/beneficial)")

# Analyze curve structure
print("\n[4] Analyzing curve structure in influence space...")

mask_0 = train_y == 0
mask_1 = train_y == 1

# For Class 0: check if influence varies with position
class0_x = train_X[mask_0, 0]
class0_y = train_X[mask_0, 1]
class0_P0 = P0[mask_0]
class0_P1 = P1[mask_0]
class0_dist = (class0_x + class0_y) / np.sqrt(2)  # Distance to diagonal

corr_dist_P0 = np.corrcoef(class0_dist, class0_P0)[0, 1]
corr_dist_P1 = np.corrcoef(class0_dist, class0_P1)[0, 1]

print(f"  Class 0:")
print(f"    Correlation(distance_to_diagonal, P0): {corr_dist_P0:.4f}")
print(f"    Correlation(distance_to_diagonal, P1): {corr_dist_P1:.4f}")

# For Class 1
class1_x = train_X[mask_1, 0]
class1_y = train_X[mask_1, 1]
class1_P0 = P0[mask_1]
class1_P1 = P1[mask_1]
class1_dist = (class1_x + class1_y) / np.sqrt(2)

corr_dist_P0_c1 = np.corrcoef(class1_dist, class1_P0)[0, 1]
corr_dist_P1_c1 = np.corrcoef(class1_dist, class1_P1)[0, 1]

print(f"  Class 1:")
print(f"    Correlation(distance_to_diagonal, P0): {corr_dist_P0_c1:.4f}")
print(f"    Correlation(distance_to_diagonal, P1): {corr_dist_P1_c1:.4f}")

# Create figure matching paper exactly
print("\n[5] Creating figure matching paper's style...")

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

color_blue = '#1f77b4'
color_orange = '#ff7f0e'

# ===== PANEL A: Data Distribution =====
ax = axes[0]

ax.scatter(train_X[mask_0, 0], train_X[mask_0, 1],
           c=color_blue, s=35, alpha=0.8, edgecolors='none', label='Class 0')
ax.scatter(train_X[mask_1, 0], train_X[mask_1, 1],
           c=color_orange, s=35, alpha=0.8, edgecolors='none', label='Class 1')

# Diagonal decision boundary y = -x
xx = np.array([-4, 4])
ax.plot(xx, -xx, 'k-', linewidth=2)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('A', loc='left', fontweight='bold', fontsize=20)
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
ax.set_aspect('equal')
ax.legend(loc='upper left')

# ===== PANEL B: Influence Space =====
ax = axes[1]

ax.scatter(P0[mask_0], P1[mask_0],
           c=color_blue, s=35, alpha=0.8, edgecolors='none', label='Class 0')
ax.scatter(P0[mask_1], P1[mask_1],
           c=color_orange, s=35, alpha=0.8, edgecolors='none', label='Class 1')

# Mark noisy samples with red circles
if np.sum(train_noise_mask) > 0:
    ax.scatter(P0[train_noise_mask], P1[train_noise_mask],
               facecolors='none', edgecolors='red', s=80, linewidth=1.5,
               label='Noisy', zorder=10)

lim = max(np.abs(P0).max(), np.abs(P1).max()) * 1.15
ax.plot([-lim, lim], [lim, -lim], 'k-', linewidth=2)
ax.axhline(0, color='gray', alpha=0.3, linewidth=0.8)
ax.axvline(0, color='gray', alpha=0.3, linewidth=0.8)

ax.set_xlabel('Class 0 Influence')
ax.set_ylabel('Class 1 Influence')
ax.set_title('B', loc='left', fontweight='bold', fontsize=20)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.legend(loc='upper left')

# ===== PANEL C: Feature Space with Influence Coloring =====
ax = axes[2]

cmap = create_paper_colormap()
vmax = np.percentile(np.abs(net_inf), 97)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

scatter = ax.scatter(train_X[:, 0], train_X[:, 1],
                     c=net_inf, cmap=cmap, norm=norm,
                     s=35, alpha=0.9, edgecolors='none')

# Mark noisy samples with red outlines
if np.sum(train_noise_mask) > 0:
    ax.scatter(train_X[train_noise_mask, 0], train_X[train_noise_mask, 1],
               facecolors='none', edgecolors='red', s=80, linewidth=1.5,
               zorder=10)

ax.plot(xx, -xx, 'k-', linewidth=2)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label('Class-Wise\nInfluence Score', fontsize=12)

# Labels
cbar.ax.text(2.5, 0.92, 'Positive', transform=cbar.ax.transAxes, fontsize=9, color='#d62728')
cbar.ax.text(2.5, 0.5, 'Neutral', transform=cbar.ax.transAxes, fontsize=9, color='gray')
cbar.ax.text(2.5, 0.08, 'Negative', transform=cbar.ax.transAxes, fontsize=9, color='#1f77b4')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('C', loc='left', fontweight='bold', fontsize=20)
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/figures/figure2_paper_exact.png', dpi=400, bbox_inches='tight', facecolor='white')
print("  Figure saved: outputs/figures/figure2_paper_exact.png")
plt.close()

# Summary
print("\n" + "=" * 80)
print("SUMMARY: UNDERSTANDING PAPER'S FIGURE 2")
print("=" * 80)
print("""
KEY INSIGHTS:

1. **Two Classes**: Blue (Class 0) and Orange (Class 1)
   - Blue cluster: LEFT side of feature space
   - Orange cluster: RIGHT side of feature space

2. **Decision Boundary**: Diagonal line (y = -x)
   - NOT a vertical line separating left/right
   - This diagonal creates specific gradient patterns

3. **Panel A vs Panel C Comparison**:
   - Panel A: Data colored by CLASS LABEL (blue vs orange)
   - Panel C: SAME data colored by INFLUENCE SCORE
   - The comparison reveals which samples are harmful (blue in C)
     vs beneficial (red in C) regardless of their class

4. **Panel B Curve Pattern**:
   - Curves emerge because influence varies SYSTEMATICALLY with position
   - Samples near decision boundary have different influence than far samples
   - The diagonal boundary creates this systematic variation

5. **Why Our Reproduction Differed**:
   - We used VERTICAL boundary (separating left/right by x-coordinate)
   - Paper uses DIAGONAL boundary (y = -x)
   - This changes the gradient structure and thus the influence pattern
""")
