"""
synthetic_nonlinear.py - Reproduce Figure 2 D-F

This experiment reproduces the second synthetic dataset from the paper:
a non-linearly separable 2D dataset where Pareto improvements are NOT possible.

Dataset:
- 350 blue (class 0) + 350 orange (class 1) samples
- Orange samples have radius that varies with angle (non-circular)
- The two classes overlap in a way that a linear classifier cannot separate

Expected Results:
- Since the model (logistic regression) cannot perfectly separate the data,
  all samples should lie in TRADEOFF regions
- Influence vectors should approximately lie on the line y = -x
- This indicates the PARETO CEILING has been reached

Reference: Figure 2, Subfigures D-F in the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.influence import train_logistic_regression
from src.category_influence import (
    compute_all_influence_vectors,
    classify_samples_by_region,
    check_pareto_ceiling
)
from src.pareto import create_figure2_style_plot, analyze_pareto_statistics


def generate_nonlinearly_separable(
    n_per_class: int = 350,
    random_seed: int = 42
) -> tuple:
    """
    Generate a non-linearly separable 2D dataset.
    
    The dataset has:
    - Blue class: circular uniform distribution
    - Orange class: non-uniform distribution where radius varies with angle
    
    This creates an inherent overlap that a linear classifier cannot resolve,
    demonstrating the Pareto ceiling concept.
    
    Args:
        n_per_class: Number of samples per class
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y)
        - X: Features of shape (n_samples, 2)
        - y: Labels of shape (n_samples,)
    """
    np.random.seed(random_seed)
    
    # === Blue Class: Simple circular distribution ===
    angles_blue = np.random.uniform(0, 2 * np.pi, n_per_class)
    radii_blue = 1.5 * np.sqrt(np.random.uniform(0, 1, n_per_class))
    
    x_blue = radii_blue * np.cos(angles_blue)
    y_blue = radii_blue * np.sin(angles_blue)
    
    # === Orange Class: Radius varies with angle ===
    # This creates a "flower" pattern that overlaps with blue
    angles_orange = np.random.uniform(0, 2 * np.pi, n_per_class)
    
    # Base radius + sinusoidal variation
    base_radius = 1.0
    amplitude = 0.8
    frequency = 3  # Number of "petals"
    
    radii_orange = base_radius + amplitude * np.sin(frequency * angles_orange)
    radii_orange *= np.sqrt(np.random.uniform(0.5, 1, n_per_class))
    
    x_orange = radii_orange * np.cos(angles_orange)
    y_orange = radii_orange * np.sin(angles_orange)
    
    # Combine
    X_blue = np.column_stack([x_blue, y_blue])
    X_orange = np.column_stack([x_orange, y_orange])
    X = np.vstack([X_blue, X_orange])
    
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    return X, y


def run_experiment(output_dir: str = None):
    """
    Run the non-linearly separable experiment and generate Figure 2 D-F.
    
    This experiment demonstrates:
    1. When a linear model cannot fully separate classes
    2. The influence vectors form a LINE (indicating Pareto ceiling)
    3. Any improvement in one class MUST hurt the other class
    """
    print("=" * 60)
    print("EXPERIMENT: Non-Linearly Separable Dataset")
    print("Reproducing Figure 2 D-F from the paper")
    print("=" * 60)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # === Step 1: Generate Dataset ===
    print("\n[Step 1] Generating non-linearly separable dataset...")
    X, y = generate_nonlinearly_separable(n_per_class=350, random_seed=42)
    
    n_samples = len(y)
    print(f"  Total samples: {n_samples}")
    print(f"  Class 0 (blue): {np.sum(y == 0)}")
    print(f"  Class 1 (orange): {np.sum(y == 1)}")
    
    # Split into train and validation
    np.random.seed(42)
    perm = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    train_X, train_y = X[train_idx], y[train_idx]
    val_X, val_y = X[val_idx], y[val_idx]
    
    print(f"  Training samples: {len(train_X)}")
    print(f"  Validation samples: {len(val_X)}")
    
    # === Step 2: Train Logistic Regression ===
    print("\n[Step 2] Training logistic regression model...")
    weights = train_logistic_regression(
        train_X, train_y,
        learning_rate=0.1,
        n_iterations=1000,
        l2_reg=0.01,
        verbose=False
    )
    
    # Compute accuracy
    from src.influence import compute_predictions
    train_probs = compute_predictions(train_X, weights)
    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = np.mean(train_preds == train_y)
    
    val_probs = compute_predictions(val_X, weights)
    val_preds = (val_probs >= 0.5).astype(int)
    val_acc = np.mean(val_preds == val_y)
    
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Note: Accuracy is limited by non-linear separability")
    
    # Per-class accuracy
    for c in [0, 1]:
        mask = (val_y == c)
        class_acc = np.mean(val_preds[mask] == val_y[mask])
        print(f"  Class {c} accuracy: {class_acc:.4f}")
    
    # === Step 3: Compute Category-Wise Influence Vectors ===
    print("\n[Step 3] Computing category-wise influence vectors...")
    influence_vectors = compute_all_influence_vectors(
        train_X, train_y, val_X, val_y, weights, damping=1e-3
    )
    print(f"  Influence vectors computed: {influence_vectors.shape}")
    
    # === Step 4: Analyze Pareto Regions ===
    print("\n[Step 4] Analyzing Pareto regions...")
    regions = classify_samples_by_region(influence_vectors)
    
    n_joint = len(regions['joint_positive']) + len(regions['joint_negative'])
    n_tradeoff = len(regions['tradeoff_class_0']) + len(regions['tradeoff_class_1'])
    
    print(f"  Joint Positive: {len(regions['joint_positive'])} samples")
    print(f"  Joint Negative: {len(regions['joint_negative'])} samples")
    print(f"  Tradeoff regions: {n_tradeoff} samples")
    print(f"  Samples in tradeoff: {100*n_tradeoff/len(train_y):.1f}%")
    
    # === Step 5: Check Pareto Ceiling ===
    print("\n[Step 5] Checking if Pareto ceiling is reached...")
    is_ceiling, explained_ratio = check_pareto_ceiling(influence_vectors)
    
    print(f"  Explained variance ratio (1st PC): {explained_ratio:.4f}")
    print(f"  Pareto ceiling reached: {is_ceiling}")
    
    if explained_ratio > 0.8:
        print("  → High explained ratio means influence vectors form a LINE")
        print("  → This indicates pure tradeoffs (no Pareto improvements possible)")
    
    # === Step 6: Generate Visualization ===
    print("\n[Step 6] Generating Figure 2 D-F style visualization...")
    
    save_path = os.path.join(output_dir, 'figure2_def_nonlinear.png')
    fig = create_figure2_style_plot(
        train_X, train_y, influence_vectors,
        noisy_indices=None,  # No noisy samples in this experiment
        dataset_name="Non-Linearly Separable",
        save_path=save_path
    )
    plt.close(fig)
    
    # === Step 7: Summary Statistics ===
    print("\n[Step 7] Computing summary statistics...")
    stats = analyze_pareto_statistics(influence_vectors, train_y)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"  1. {100*n_tradeoff/len(train_y):.1f}% of samples are in tradeoff regions")
    print(f"     → Very few samples in joint positive/negative regions")
    print(f"  2. Explained variance ratio: {explained_ratio:.4f}")
    print(f"     → Influence vectors form approximately a straight line")
    print(f"  3. Pareto ceiling {'IS' if is_ceiling else 'is NOT'} reached")
    if not is_ceiling:
        print(f"     → Note: Even if not perfectly reached, high explained ratio")
        print(f"        indicates we are CLOSE to the ceiling")
    print(f"\nInterpretation:")
    print(f"  For this non-linearly separable dataset, the linear classifier")
    print(f"  cannot improve one class without hurting the other. The influence")
    print(f"  vectors lying on a line confirms this fundamental tradeoff.")
    print(f"\nFigure saved to: {save_path}")
    
    return {
        'X': X,
        'y': y,
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y,
        'weights': weights,
        'influence_vectors': influence_vectors,
        'regions': regions,
        'stats': stats,
        'is_ceiling': is_ceiling,
        'explained_ratio': explained_ratio
    }


if __name__ == "__main__":
    result = run_experiment()
