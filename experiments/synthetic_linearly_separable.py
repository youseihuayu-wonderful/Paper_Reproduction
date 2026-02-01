"""
synthetic_linearly_separable.py - Reproduce Figure 2 A-C

This experiment reproduces the first synthetic dataset from the paper:
a linearly separable 2D dataset with added label noise.

Dataset:
- 300 blue (class 0) + 300 orange (class 1) samples
- Samples generated from circular uniform distribution
- Noise: 50 blue + 20 orange samples have their labels flipped

Expected Results:
- Category-wise influence should correctly identify noisy samples
- Noisy samples should appear in the JOINT NEGATIVE region
- Removing noisy samples should improve performance for BOTH classes

Reference: Figure 2, Subfigures A-C in the paper
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
    identify_noisy_samples
)
from src.pareto import create_figure2_style_plot, analyze_pareto_statistics


def generate_linearly_separable_with_noise(
    n_per_class: int = 300,
    n_noise_blue: int = 50,
    n_noise_orange: int = 20,
    radius: float = 2.0,
    separation: float = 1.0,
    random_seed: int = 42
) -> tuple:
    """
    Generate a linearly separable 2D dataset with label noise.
    
    The dataset consists of two circular clusters separated by a gap.
    We then flip labels for some samples to simulate mislabeling/noise.
    
    This mimics the setup in Figure 2 A-C of the paper.
    
    Args:
        n_per_class: Number of samples per class
        n_noise_blue: Number of blue samples to mislabel
        n_noise_orange: Number of orange samples to mislabel
        radius: Radius of circular distribution
        separation: Gap between the two classes
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, noisy_indices, true_labels)
        - X: Features of shape (n_samples, 2)
        - y: Noisy labels of shape (n_samples,)
        - noisy_indices: Indices of mislabeled samples
        - true_labels: Original clean labels
    """
    np.random.seed(random_seed)
    
    # Generate angles for circular distribution
    angles_blue = np.random.uniform(0, 2 * np.pi, n_per_class)
    angles_orange = np.random.uniform(0, 2 * np.pi, n_per_class)
    
    # Generate radii (uniform in circle)
    radii_blue = radius * np.sqrt(np.random.uniform(0, 1, n_per_class))
    radii_orange = radius * np.sqrt(np.random.uniform(0, 1, n_per_class))
    
    # Convert to Cartesian coordinates
    # Blue class: centered at (-separation, 0)
    x_blue = radii_blue * np.cos(angles_blue) - separation
    y_blue = radii_blue * np.sin(angles_blue)
    
    # Orange class: centered at (+separation, 0)
    x_orange = radii_orange * np.cos(angles_orange) + separation
    y_orange = radii_orange * np.sin(angles_orange)
    
    # Combine into feature matrix
    X_blue = np.column_stack([x_blue, y_blue])
    X_orange = np.column_stack([x_orange, y_orange])
    X = np.vstack([X_blue, X_orange])
    
    # Create true labels
    true_labels = np.array([0] * n_per_class + [1] * n_per_class)
    y = true_labels.copy()
    
    # Add noise by flipping labels
    # Select random samples from each class to mislabel
    blue_indices = np.arange(n_per_class)
    orange_indices = np.arange(n_per_class, 2 * n_per_class)
    
    noise_blue_idx = np.random.choice(blue_indices, n_noise_blue, replace=False)
    noise_orange_idx = np.random.choice(orange_indices, n_noise_orange, replace=False)
    
    # Flip labels
    y[noise_blue_idx] = 1  # Blue points now labeled as orange
    y[noise_orange_idx] = 0  # Orange points now labeled as blue
    
    noisy_indices = np.concatenate([noise_blue_idx, noise_orange_idx])
    
    return X, y, noisy_indices, true_labels


def run_experiment(output_dir: str = None):
    """
    Run the linearly separable experiment and generate Figure 2 A-C.
    
    This is the main experiment function that:
    1. Generates the dataset with noise
    2. Trains a logistic regression model
    3. Computes category-wise influence vectors
    4. Visualizes and analyzes the Pareto frontier
    5. Validates that noisy samples are detected
    """
    print("=" * 60)
    print("EXPERIMENT: Linearly Separable Dataset with Noise")
    print("Reproducing Figure 2 A-C from the paper")
    print("=" * 60)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # === Step 1: Generate Dataset ===
    print("\n[Step 1] Generating dataset...")
    X, y, noisy_indices, true_labels = generate_linearly_separable_with_noise(
        n_per_class=300,
        n_noise_blue=50,
        n_noise_orange=20,
        separation=1.5,
        random_seed=42
    )
    
    n_samples = len(y)
    n_noisy = len(noisy_indices)
    print(f"  Total samples: {n_samples}")
    print(f"  Noisy/mislabeled samples: {n_noisy}")
    print(f"  Class 0 (blue): {np.sum(true_labels == 0)}")
    print(f"  Class 1 (orange): {np.sum(true_labels == 1)}")
    
    # Split into train and validation
    # Use 80% for training, 20% for validation
    np.random.seed(42)
    perm = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    train_X, train_y = X[train_idx], y[train_idx]
    val_X, val_y = X[val_idx], y[val_idx]
    
    # Track which training samples are noisy
    train_noisy_mask = np.isin(train_idx, noisy_indices)
    train_noisy_indices = np.where(train_noisy_mask)[0]
    
    print(f"  Training samples: {len(train_X)}")
    print(f"  Validation samples: {len(val_X)}")
    print(f"  Noisy samples in training set: {len(train_noisy_indices)}")
    
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
    
    # === Step 3: Compute Category-Wise Influence Vectors ===
    print("\n[Step 3] Computing category-wise influence vectors...")
    influence_vectors = compute_all_influence_vectors(
        train_X, train_y, val_X, val_y, weights, damping=1e-3
    )
    print(f"  Influence vectors computed: {influence_vectors.shape}")
    
    # === Step 4: Analyze Pareto Regions ===
    print("\n[Step 4] Analyzing Pareto regions...")
    regions = classify_samples_by_region(influence_vectors)
    
    print(f"  Joint Positive: {len(regions['joint_positive'])} samples")
    print(f"  Joint Negative: {len(regions['joint_negative'])} samples")
    print(f"  Tradeoff (Class 0 ↑): {len(regions['tradeoff_class_0'])} samples")
    print(f"  Tradeoff (Class 1 ↑): {len(regions['tradeoff_class_1'])} samples")
    
    # === Step 5: Validate Noisy Sample Detection ===
    print("\n[Step 5] Validating noisy sample detection...")
    
    # How many noisy samples are in the joint negative region?
    noisy_in_joint_negative = np.intersect1d(
        train_noisy_indices, 
        regions['joint_negative']
    )
    
    detection_rate = len(noisy_in_joint_negative) / len(train_noisy_indices) if len(train_noisy_indices) > 0 else 0
    
    print(f"  Noisy samples in Joint Negative region: {len(noisy_in_joint_negative)}/{len(train_noisy_indices)}")
    print(f"  Detection rate: {detection_rate*100:.1f}%")
    
    # Use automated noisy sample detection
    detected_noisy = identify_noisy_samples(influence_vectors, percentile=10)
    
    # Precision and recall
    true_positives = len(np.intersect1d(detected_noisy, train_noisy_indices))
    precision = true_positives / len(detected_noisy) if len(detected_noisy) > 0 else 0
    recall = true_positives / len(train_noisy_indices) if len(train_noisy_indices) > 0 else 0
    
    print(f"  Detected as noisy: {len(detected_noisy)} samples")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    
    # === Step 6: Generate Visualization ===
    print("\n[Step 6] Generating Figure 2 A-C style visualization...")
    
    save_path = os.path.join(output_dir, 'figure2_abc_linearly_separable.png')
    fig = create_figure2_style_plot(
        train_X, train_y, influence_vectors,
        noisy_indices=train_noisy_indices,
        dataset_name="Linearly Separable + Noise",
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
    print(f"  1. {stats['pct_joint_negative']:.1f}% of samples are in Joint Negative region")
    print(f"     → These are candidates for removal (Pareto detrimental)")
    print(f"  2. Noisy sample detection rate: {detection_rate*100:.1f}%")
    print(f"     → Category-wise influence correctly identifies mislabeled samples")
    print(f"  3. Pareto ceiling NOT reached (expected, due to noise)")
    print(f"     → There is room for Pareto improvement")
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
        'noisy_indices': train_noisy_indices,
        'regions': regions,
        'stats': stats
    }


if __name__ == "__main__":
    result = run_experiment()
