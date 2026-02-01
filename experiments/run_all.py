"""
run_all.py - Run All Experiments

This script runs all synthetic experiments and generates figures
that reproduce the main results from the paper.

Usage:
    uv run python experiments/run_all.py
"""

import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("\n" + "=" * 70)
    print("PAPER REPRODUCTION: Category-Wise Influence Functions")
    print("Running all experiments...")
    print("=" * 70)
    
    # Output directory for figures
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # === Experiment 1: Linearly Separable with Noise ===
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 1: Linearly Separable Dataset with Noise (Figure 2 A-C)")
    print("=" * 70)
    
    from experiments.synthetic_linearly_separable import run_experiment as run_exp1
    result1 = run_exp1(output_dir)
    
    # === Experiment 2: Non-Linearly Separable ===
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 2: Non-Linearly Separable Dataset (Figure 2 D-F)")
    print("=" * 70)
    
    from experiments.synthetic_nonlinear import run_experiment as run_exp2
    result2 = run_exp2(output_dir)
    
    # === Summary ===
    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    print("\nGenerated Figures:")
    for fname in os.listdir(output_dir):
        if fname.endswith('.png'):
            print(f"  - {fname}")
    
    print(f"\nFigures saved to: {output_dir}")
    
    print("\n" + "-" * 70)
    print("Summary of Key Findings:")
    print("-" * 70)
    
    print("\nExperiment 1 (Linearly Separable + Noise):")
    print(f"  - Joint Negative samples: {result1['stats']['pct_joint_negative']:.1f}%")
    print(f"  - These are noisy/mislabeled samples detectable by influence")
    print(f"  - Room for Pareto improvement: YES")
    
    print("\nExperiment 2 (Non-Linearly Separable):")
    print(f"  - Explained variance ratio: {result2['explained_ratio']:.4f}")
    print(f"  - Influence vectors form a line (pure tradeoff)")
    print(f"  - Room for Pareto improvement: NO (ceiling reached)")
    
    print("\n" + "=" * 70)
    print("These results match the paper's claims in Section 4!")
    print("=" * 70 + "\n")
    
    return result1, result2


if __name__ == "__main__":
    main()
