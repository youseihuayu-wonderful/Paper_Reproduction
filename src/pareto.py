"""
pareto.py - Pareto Frontier Analysis Utilities

This module provides tools for analyzing and visualizing the Pareto frontier
of a classifier's category-wise performance.

Key Concepts:
    - Pareto Frontier: The set of solutions where no class can be improved
      without degrading another class
    - Pareto Improvement: A change that improves at least one class without
      hurting any other class
    - Performance Ceiling: When the classifier is ON the Pareto frontier

Visualization:
    This module provides functions to visualize influence vectors in 2D,
    showing the four regions (joint positive, joint negative, two tradeoffs).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_influence_vectors(
    influence_vectors: np.ndarray,
    train_y: np.ndarray,
    noisy_indices: Optional[np.ndarray] = None,
    title: str = "Category-Wise Influence Vectors",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize 2D influence vectors showing the Pareto frontier.
    
    This reproduces Figure 1 from the paper - a scatter plot where:
    - X-axis: Influence on Class 0 (P⁰)
    - Y-axis: Influence on Class 1 (P¹)
    
    The plot is divided into 4 quadrants:
    - Upper Right (Q1): Joint Positive - beneficial to both classes
    - Lower Left (Q3): Joint Negative - detrimental to both classes
    - Lower Right (Q4): Tradeoff - helps class 0, hurts class 1
    - Upper Left (Q2): Tradeoff - helps class 1, hurts class 0
    
    The dashed line y = -x represents the Pareto frontier (pure tradeoff).
    Points ON this line indicate the performance ceiling is reached.
    
    Args:
        influence_vectors: Influence vectors of shape (n_train, 2)
        train_y: Training labels to color-code points
        noisy_indices: Optional array of noisy sample indices to highlight
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    P0 = influence_vectors[:, 0]  # Influence on class 0
    P1 = influence_vectors[:, 1]  # Influence on class 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by original class (blue = class 0, orange = class 1)
    colors = ['#3b82f6' if y == 0 else '#f97316' for y in train_y]
    
    # Plot all points
    scatter = ax.scatter(P0, P1, c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Highlight noisy samples if provided
    if noisy_indices is not None and len(noisy_indices) > 0:
        ax.scatter(P0[noisy_indices], P1[noisy_indices], 
                  c='red', s=100, marker='x', linewidth=2,
                  label='Noisy/Mislabeled', zorder=10)
    
    # Draw the y = -x line (Pareto frontier)
    lim = max(abs(P0).max(), abs(P1).max()) * 1.2
    ax.plot([-lim, lim], [lim, -lim], 'g--', alpha=0.7, linewidth=2, label='Pareto Frontier (y = -x)')
    
    # Draw quadrant lines (axes)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add quadrant labels
    text_offset = lim * 0.85
    ax.text(text_offset * 0.7, text_offset * 0.7, 'Joint\nPositive', 
           fontsize=12, ha='center', va='center', alpha=0.7,
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
    ax.text(-text_offset * 0.7, -text_offset * 0.7, 'Joint\nNegative', 
           fontsize=12, ha='center', va='center', alpha=0.7,
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    ax.text(text_offset * 0.7, -text_offset * 0.7, 'Tradeoff\n(Class 0 ↑)', 
           fontsize=10, ha='center', va='center', alpha=0.5)
    ax.text(-text_offset * 0.7, text_offset * 0.7, 'Tradeoff\n(Class 1 ↑)', 
           fontsize=10, ha='center', va='center', alpha=0.5)
    
    # Shade the joint regions
    ax.fill_between([0, lim*2], [0, 0], [lim*2, lim*2], alpha=0.05, color='green')
    ax.fill_between([-lim*2, 0], [-lim*2, -lim*2], [0, 0], alpha=0.05, color='red')
    
    # Labels and title
    ax.set_xlabel('Influence on Class 0 (P⁰)', fontsize=14)
    ax.set_ylabel('Influence on Class 1 (P¹)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', 
               markersize=10, label='Class 0 (Blue)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f97316', 
               markersize=10, label='Class 1 (Orange)'),
        Line2D([0], [0], color='g', linestyle='--', linewidth=2,
               label='Pareto Frontier')
    ]
    if noisy_indices is not None:
        legend_elements.append(
            Line2D([0], [0], marker='x', color='red', linestyle='None',
                   markersize=10, label='Noisy Samples')
        )
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_data_with_influence_colors(
    X: np.ndarray,
    y: np.ndarray,
    influence_vectors: np.ndarray,
    weights: np.ndarray,
    title: str = "Dataset with Influence-Based Coloring",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the original data points colored by their Pareto region.
    
    This shows WHERE in the feature space the different influence regions
    appear. It helps understand:
    - Where are the noisy/mislabeled samples located?
    - Where are the beneficial samples?
    - How does the decision boundary interact with influence?
    
    Args:
        X: Feature matrix of shape (n_samples, 2) - must be 2D for visualization
        y: Labels of shape (n_samples,)
        influence_vectors: Influence vectors of shape (n_samples, 2)
        weights: Model weights for drawing decision boundary
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    from .category_influence import classify_samples_by_region
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Classify samples by region
    regions = classify_samples_by_region(influence_vectors)
    
    # Color scheme for regions
    colors = np.array(['gray'] * len(y))
    colors[regions['joint_positive']] = 'green'
    colors[regions['joint_negative']] = 'red'
    colors[regions['tradeoff_class_0']] = 'skyblue'
    colors[regions['tradeoff_class_1']] = 'orange'
    
    # Marker by original class
    for i, (xi, yi, color) in enumerate(zip(X, y, colors)):
        marker = 'o' if yi == 0 else 's'
        ax.scatter(xi[0], xi[1], c=color, marker=marker, s=60, 
                  alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Draw decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    if len(weights) >= 2:  # Simple 2D case (no bias or bias included)
        if len(weights) == 2:
            # w0*x + w1*y = 0.5 (decision boundary for sigmoid = 0.5)
            # y = (0 - w0*x) / w1
            xx = np.linspace(x_min, x_max, 100)
            yy = -weights[0] / weights[1] * xx  # Assumes bias = 0 or included
            ax.plot(xx, yy, 'k--', linewidth=2, label='Decision Boundary')
    
    # Labels and title
    ax.set_xlabel('Feature 1', fontsize=14)
    ax.set_ylabel('Feature 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Joint Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Joint Negative'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
               markersize=10, label='Tradeoff (Class 0 ↑)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=10, label='Tradeoff (Class 1 ↑)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def create_figure2_style_plot(
    X: np.ndarray,
    y: np.ndarray,
    influence_vectors: np.ndarray,
    noisy_indices: Optional[np.ndarray] = None,
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a 3-panel figure in the style of Figure 2 from the paper.
    
    Panel A: Original data with classification boundary
    Panel B: Influence vector scatter plot (Pareto frontier)
    Panel C: Data colored by influence region
    
    This is the main visualization for the synthetic experiments.
    
    Args:
        X: Feature matrix of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
        influence_vectors: Influence vectors of shape (n_samples, 2)
        noisy_indices: Optional array of noisy sample indices
        dataset_name: Name for the figure title
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    from .category_influence import classify_samples_by_region
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === Panel A: Original Data ===
    ax = axes[0]
    colors = ['#3b82f6' if yi == 0 else '#f97316' for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
    
    if noisy_indices is not None and len(noisy_indices) > 0:
        ax.scatter(X[noisy_indices, 0], X[noisy_indices, 1], 
                  c='red', s=80, marker='x', linewidth=2, label='Noisy')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'A: {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # === Panel B: Influence Vector Scatter ===
    ax = axes[1]
    P0, P1 = influence_vectors[:, 0], influence_vectors[:, 1]
    ax.scatter(P0, P1, c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
    
    if noisy_indices is not None and len(noisy_indices) > 0:
        ax.scatter(P0[noisy_indices], P1[noisy_indices], 
                  c='red', s=80, marker='x', linewidth=2)
    
    # Pareto frontier line
    lim = max(abs(P0).max(), abs(P1).max()) * 1.3
    ax.plot([-lim, lim], [lim, -lim], 'g--', alpha=0.7, linewidth=2)
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.axvline(x=0, color='gray', alpha=0.3)
    
    ax.set_xlabel('Influence on Class 0', fontsize=12)
    ax.set_ylabel('Influence on Class 1', fontsize=12)
    ax.set_title('B: Category-Wise Influence Vectors', fontsize=14, fontweight='bold')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    # === Panel C: Data colored by region ===
    ax = axes[2]
    regions = classify_samples_by_region(influence_vectors)
    
    region_colors = np.array(['lightgray'] * len(y))
    region_colors[regions['joint_positive']] = '#22c55e'  # green
    region_colors[regions['joint_negative']] = '#ef4444'  # red
    region_colors[regions['tradeoff_class_0']] = '#60a5fa'  # light blue
    region_colors[regions['tradeoff_class_1']] = '#fb923c'  # light orange
    
    ax.scatter(X[:, 0], X[:, 1], c=region_colors, alpha=0.6, s=40, 
              edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('C: Samples Colored by Influence Region', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Summary statistics
    n_joint_pos = len(regions['joint_positive'])
    n_joint_neg = len(regions['joint_negative'])
    n_total = len(y)
    ax.text(0.02, 0.98, f"Joint+: {n_joint_pos} ({100*n_joint_pos/n_total:.1f}%)\n"
                        f"Joint-: {n_joint_neg} ({100*n_joint_neg/n_total:.1f}%)",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig


def analyze_pareto_statistics(
    influence_vectors: np.ndarray,
    train_y: np.ndarray
) -> Dict:
    """
    Compute statistics about the Pareto frontier for a trained classifier.
    
    This provides quantitative measures to complement the visualizations.
    
    Args:
        influence_vectors: Influence vectors of shape (n_train, n_classes)
        train_y: Training labels
        
    Returns:
        Dictionary with various Pareto frontier statistics
    """
    from .category_influence import classify_samples_by_region, check_pareto_ceiling
    
    n_samples = len(train_y)
    regions = classify_samples_by_region(influence_vectors)
    is_ceiling, explained_ratio = check_pareto_ceiling(influence_vectors)
    
    # Per-class analysis
    classes = np.unique(train_y)
    class_stats = {}
    for c in classes:
        class_mask = (train_y == c)
        class_influence = influence_vectors[class_mask]
        
        # How many samples from class c are in each region?
        class_in_joint_neg = len(np.intersect1d(
            regions['joint_negative'], 
            np.where(class_mask)[0]
        ))
        
        class_stats[int(c)] = {
            'n_samples': int(np.sum(class_mask)),
            'n_in_joint_negative': class_in_joint_neg,
            'mean_self_influence': float(np.mean(class_influence[:, int(c)])),
            'mean_other_influence': float(np.mean(class_influence[:, 1-int(c)]))
        }
    
    stats = {
        'n_total': n_samples,
        'n_joint_positive': len(regions['joint_positive']),
        'n_joint_negative': len(regions['joint_negative']),
        'n_tradeoff_0': len(regions['tradeoff_class_0']),
        'n_tradeoff_1': len(regions['tradeoff_class_1']),
        'pct_joint_positive': 100 * len(regions['joint_positive']) / n_samples,
        'pct_joint_negative': 100 * len(regions['joint_negative']) / n_samples,
        'pareto_ceiling_reached': is_ceiling,
        'explained_variance_ratio': explained_ratio,
        'per_class': class_stats
    }
    
    return stats
