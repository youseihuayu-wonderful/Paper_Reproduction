# Paper Reproduction: "What Is The Performance Ceiling of My Classifier?"

**Original Paper:** arXiv:2510.03950
**Authors:** Nahin et al. (2025)

This repository contains a systematic reproduction of the key experimental claims from the paper, focusing on category-wise influence functions and Pareto frontier analysis.

---

## Quick Start

```bash
cd Paper_Reproduction
source .venv/bin/activate

# Run main reproduction (Figure 2 A-F)
python experiments/figure2_comprehensive.py

# Run ceiling analysis comparison
python experiments/figure2_ceiling_comparison.py

# Run all validations
python experiments/validation_experiments.py
```

---

## Key Results

| Experiment | Paper Claim | Our Result | Status |
|------------|-------------|------------|--------|
| Spearman Correlation | > 0.8 | **0.9487** | ✅ Verified |
| Pareto-LP-GA Improvement | Improves target class | **+3.57%** (0% degradation) | ✅ Verified |
| Noisy Sample Detection | Identifies mislabeled | **87.8%** in Joint Negative | ✅ Verified |
| Figure 2 Reproduction | Visual match | ✅ Two curved arms | ✅ Verified |

---

## Project Structure

```
Paper_Reproduction/
├── docs/
│   ├── REPRODUCTION_RESULTS.md      # Main results document
│   └── CONCEPTUAL_UNDERSTANDING.md  # Theoretical explanations
├── src/
│   ├── influence.py                 # Standard influence functions
│   ├── category_influence.py        # Category-wise influence vectors
│   ├── lp_reweight.py              # Original LP methods
│   ├── lp_reweight_fixed.py        # FIXED methods that work!
│   ├── ekfac.py                    # EKFAC for neural networks
│   ├── pareto.py                   # Pareto visualization
│   └── pareto_lp_ga.py             # Full Pareto-LP-GA pipeline
├── experiments/
│   ├── figure2_comprehensive.py          # Figure 2 A-F (RECOMMENDED)
│   ├── figure2_ceiling_comparison.py     # Ceiling analysis
│   ├── validation_experiments.py         # All validation tests
│   └── run_all.py                        # Run all experiments
└── outputs/
    └── figures/
        ├── original_paper_figure2.png    # Original from paper
        ├── figure2_ABC.png               # Linear separable (1200 DPI)
        ├── figure2_DEF.png               # Non-separable (1200 DPI)
        └── figure2_ceiling_comparison.png # Ceiling analysis
```

---

## Key Insights

### 1. First-Order Validity is Critical
Influence functions are Taylor approximations valid only for **small** perturbations:
- Modify only ~4-8% of training samples
- Use gentle weight multipliers (1.2-1.5x up, 0.5-0.8x down)

### 2. One-Direction Reweighting Works
- **Upweight-only** OR **downweight-only** achieves Pareto improvement
- Combining both **always fails**

### 3. Performance Ceiling Has Two Types

| Type | Correlation | Tradeoff % | Meaning |
|------|-------------|------------|---------|
| **Partial Ceiling** | Positive (~0.6) | ~30% | Some improvement possible |
| **Complete Ceiling** | Negative (< -0.5) | >70% | No Pareto improvement |

See [REPRODUCTION_RESULTS.md Section 3.5](docs/REPRODUCTION_RESULTS.md) for detailed analysis.

---

## Documentation

| Document | Description |
|----------|-------------|
| [REPRODUCTION_RESULTS.md](docs/REPRODUCTION_RESULTS.md) | Main results with figures |
| [CONCEPTUAL_UNDERSTANDING.md](docs/CONCEPTUAL_UNDERSTANDING.md) | Theoretical deep-dive |

---

## Usage Examples

### Compute Category-wise Influence

```python
from src.category_influence import compute_all_influence_vectors

influence_vectors = compute_all_influence_vectors(
    train_X, train_y, val_X, val_y, model_weights, damping=1e-4
)
# Shape: (n_train, n_classes)
```

### TopK Sample Reweighting (Recommended)

```python
from src.lp_reweight_fixed import solve_upweight_only

weights, success = solve_upweight_only(
    influence_vectors,
    target_classes=[0],  # Improve class 0
    k=10,
    weight_multiplier=1.5
)
```

---

## References

1. Nahin et al. (2025). What Is The Performance Ceiling of My Classifier? arXiv:2510.03950
2. Koh & Liang (2017). Understanding black-box predictions via influence functions. ICML
3. George et al. (2018). Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis. NeurIPS
