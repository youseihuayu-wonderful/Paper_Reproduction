# Paper Reproduction: Category-Wise Influence Functions

**Paper:** "What Is The Performance Ceiling of My Classifier?" (arXiv:2510.03950)

This project reproduces and analyzes the key concepts from the paper on category-wise influence functions for Pareto frontier analysis.

---

## Reproduction Status

| Component | Status | Notes |
|-----------|--------|-------|
| Influence Computation | ✅ Verified | Exact Hessian for logistic regression |
| Category-wise Influence | ✅ Verified | K-dimensional vectors |
| Spearman Correlation | ✅ Verified | 0.9487 > 0.8 threshold |
| Pareto-LP-GA | ✅ Working | +3.57% with TopK approach |
| EKFAC Implementation | ✅ Completed | For neural networks |
| Course Correction | 📝 Documented | Requires non-convex model |
| CIFAR-10/ResNet | 📝 Documented | Resource constraints |

---

## Quick Start

```bash
cd Paper_Reproduction

# Run validation experiments
.venv/bin/python3 experiments/validation_experiments.py
```

**Expected Output:**
```
1. SPEARMAN CORRELATION: ✅ VERIFIED (0.9487)
2. PARETO-LP-GA: ✅ WORKING (+3.57%)
3. COURSE CORRECTION: ⚠️ N/A (convex model)
```

---

## Project Structure

```
Paper_Reproduction/
├── README.md                  # This file
├── src/                       # Core implementation
│   ├── influence.py           # Standard influence functions
│   ├── category_influence.py  # Category-wise influence vectors
│   ├── lp_reweight.py         # LP, TopK, entropy reweighting
│   ├── ekfac.py               # EKFAC for neural networks
│   ├── ga_search.py           # Genetic algorithm for threshold search
│   ├── pareto.py              # Pareto visualization
│   └── pareto_lp_ga.py        # Main Pareto-LP-GA pipeline
├── experiments/               # Reproduction experiments
│   ├── validation_experiments.py  # Main validation suite
│   ├── synthetic_linearly_separable.py
│   ├── synthetic_nonlinear.py
│   └── run_all.py
├── docs/                      # Documentation
│   ├── CONCEPTUAL_UNDERSTANDING.md    # Theoretical deep-dive
│   ├── reproducibility_analysis_and_plan.md  # Analysis summary
│   ├── CHECKPOINT_2026_01_30.md       # Session 1 notes
│   └── CHECKPOINT_2026_01_30_v2.md    # Session 2 notes
├── outputs/                   # Generated outputs
│   └── figures/               # Visualization outputs
└── tests/                     # Unit tests
```

---

## Key Findings

### 1. LP Produces Binary Weights (Critical Discovery)

Pure Linear Programming produces binary weights {0, 1} due to the fundamental theorem of LP. This violates the first-order approximation assumption of influence functions.

**Solution:** Use TopK upweighting instead of LP.

### 2. First-Order Approximation is Fragile

Influence functions predict loss changes for **small** perturbations only. Binary weights (removing 50% of samples) cause predictions to be wildly inaccurate.

### 3. Course Correction Requires Non-Convex Models

Logistic regression (convex) always converges to the same global optimum. Course Correction only works with neural networks that have multiple local minima.

---

## Documentation

| Document | Description |
|----------|-------------|
| [CONCEPTUAL_UNDERSTANDING.md](docs/CONCEPTUAL_UNDERSTANDING.md) | Complete theoretical analysis of the paper |
| [reproducibility_analysis_and_plan.md](docs/reproducibility_analysis_and_plan.md) | Reproduction status and findings |
| [CHECKPOINT_2026_01_30_v2.md](docs/CHECKPOINT_2026_01_30_v2.md) | Detailed session 2 notes |

---

## Usage Examples

### Compute Category-wise Influence

```python
from src.category_influence import compute_all_influence_vectors

influence_vectors = compute_all_influence_vectors(
    train_X, train_y, val_X, val_y, model_weights, damping=1e-3
)
# Shape: (n_train, n_classes)
```

### TopK Sample Reweighting (Recommended)

```python
from src.lp_reweight import solve_topk_weights

weights, success = solve_topk_weights(
    influence_vectors,
    target_classes=[0],  # Improve class 0
    top_k=10,
    weight_multiplier=1.5
)
```

### EKFAC for Neural Networks

```python
from src.ekfac import EKFACInfluence

ekfac = EKFACInfluence(model, damping=0.01)
ekfac.compute_factors(train_loader)
influences = ekfac.compute_all_influences(train_loader, val_loader)
```

---

## References

1. Koh & Liang (2017). Understanding black-box predictions via influence functions. ICML.
2. Nahin et al. (2025). What Is The Performance Ceiling of My Classifier? arXiv:2510.03950.
