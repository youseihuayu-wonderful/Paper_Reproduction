# Paper Reproduction: What Is The Performance Ceiling of My Classifier?

[![Paper](https://img.shields.io/badge/arXiv-2510.03950-b31b1b.svg)](https://arxiv.org/abs/2510.03950)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A systematic reproduction of the key experimental claims from **Nahin et al. (2025)** - *"What Is The Performance Ceiling of My Classifier?"*

---

## What I Did

1. **Implemented Influence Functions from Scratch**
   - Derived and coded the mathematical formula: $\mathcal{I}(z, V) = \nabla L_{val}^\top H^{-1} \nabla \ell_{train}$
   - Built Hessian computation and matrix inversion with numerical stability (damping)

2. **Extended to Category-wise Influence Vectors**
   - Instead of scalar influence, computed per-class influence vectors
   - Enabled analysis of how each training sample affects different classes

3. **Reproduced Figure 2 (Pareto Frontier)**
   - Generated the characteristic "two curved arms" pattern
   - Verified Spearman correlation > 0.94 between predicted and actual improvements

4. **Discovered Why LP Fails & Fixed It**
   - Found that Linear Programming produces binary weights, violating first-order approximation
   - Developed TopK reweighting solution that actually works

5. **Validated All Key Claims**
   - Noisy sample detection: 87.8% accuracy
   - Pareto improvement: +3.57% on target class with 0% degradation

---

## What I Learned

| Category | Skills Gained |
|----------|---------------|
| **Math** | Influence functions, Hessian matrices, Taylor approximation, implicit differentiation |
| **ML Theory** | Sample reweighting, Pareto optimality, performance ceiling detection |
| **Optimization** | Linear programming, convex optimization, why LP produces vertex solutions |
| **Coding** | NumPy vectorization, numerical stability, modular code design |
| **Research** | Reading papers critically, reproducing experiments, debugging theoretical implementations |

### Key Insight
> The paper's LP method fails because influence functions are **first-order approximations** valid only for small perturbations. Modifying 50% of samples (as LP does) breaks this assumption. The fix: modify only 4-8% of samples with gentle weight changes.

---

## Overview

This project implements **category-wise influence functions** and **Pareto frontier analysis** to answer a fundamental question in machine learning:

> *Can my classifier still be improved, or has it reached its theoretical performance limit?*

### Key Results

| Claim | Paper | Our Result | Status |
|-------|-------|------------|--------|
| Spearman Correlation | > 0.8 | **0.9487** | ✅ Verified |
| Pareto-LP-GA Improvement | Improves target class | **+3.57%** | ✅ Verified |
| Noisy Sample Detection | Identifies mislabeled data | **87.8%** accuracy | ✅ Verified |
| Figure 2 Reproduction | Two curved arms pattern | Visual match | ✅ Verified |

---

## Core Concepts

### 1. Category-wise Influence Vectors
Instead of scalar influence scores, we compute influence **per class**:

$$P(z) = [P^0(z), P^1(z), ..., P^{K-1}(z)] \in \mathbb{R}^K$$

This reveals how each training sample affects different classes differently.

### 2. Pareto Frontier Analysis
Samples are categorized into four regions:
- **Joint Positive**: Helps all classes → upweight these
- **Joint Negative**: Hurts all classes → downweight these
- **Tradeoff Regions**: Improving one class hurts another

### 3. Performance Ceiling Detection
When all samples fall in tradeoff regions, the classifier has reached its **performance ceiling** - no Pareto improvement is possible.

---

## Technical Implementation

### Influence Function Formula

</>LaTeX
$$\mathcal{I}(z, V_k) = \sum_{z' \in V_k} \nabla_\theta \ell(z'; \theta^*)^\top H^{-1} \nabla_\theta \ell(z; \theta^*)$$

### Project Structure

```
Paper_Reproduction/
├── src/
│   ├── influence.py           # Standard influence functions (Koh & Liang, 2017)
│   ├── category_influence.py  # Category-wise influence vectors
│   ├── lp_reweight_fixed.py   # Working TopK reweighting methods
│   ├── pareto.py              # Pareto frontier visualization
│   └── pareto_lp_ga.py        # Full Pareto-LP-GA pipeline
├── experiments/
│   ├── figure2_comprehensive.py     # Figure 2 A-F reproduction
│   ├── figure2_ceiling_comparison.py # Ceiling analysis
│   └── validation_experiments.py     # All validation tests
├── docs/
│   ├── REPRODUCTION_RESULTS.md      # Main results document
│   └── CONCEPTUAL_UNDERSTANDING.md  # Theoretical explanations
└── outputs/figures/                  # Generated visualizations
```

---

## Key Findings

### Why LP Fails (Original Insight)
The paper's LP formulation produces **binary weights** modifying ~50% of samples, violating the first-order Taylor approximation validity.

### Why TopK Works (Our Solution)
- Modify only **4-8%** of training samples
- Use gentle weight multipliers (**1.2-1.5x**)
- Apply **upweight-only OR downweight-only**, not both

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/youseihuayu-wonderful/Paper_Reproduction.git
cd Paper_Reproduction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib scikit-learn

# Run main reproduction experiment
python experiments/figure2_comprehensive.py

# Run ceiling analysis
python experiments/figure2_ceiling_comparison.py
```

---

## Usage Example

```python
from src.category_influence import compute_all_influence_vectors
from src.lp_reweight_fixed import solve_upweight_only

# Compute category-wise influence vectors
influence_vectors = compute_all_influence_vectors(
    train_X, train_y, val_X, val_y, model_weights, damping=1e-4
)
# Shape: (n_train, n_classes)

# TopK sample reweighting
weights, success = solve_upweight_only(
    influence_vectors,
    target_classes=[0],  # Improve class 0
    k=10,
    weight_multiplier=1.5
)
```

---

## Results Visualization

The reproduction successfully generates the characteristic "two curved arms" pattern from Figure 2 of the original paper, validating the category-wise influence function implementation.

<details>
<summary>View Figure 2 Reproduction</summary>

The influence space shows:
- **X-axis**: Influence on Class 0
- **Y-axis**: Influence on Class 1
- **Four quadrants**: Joint Positive, Joint Negative, and two Tradeoff regions

</details>

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Mathematics** | Influence functions, Hessian computation, Taylor approximations |
| **Optimization** | Linear programming, convex optimization, Pareto analysis |
| **Machine Learning** | Logistic regression, sample reweighting, performance ceiling detection |
| **Software Engineering** | Modular code design, comprehensive documentation, reproducible research |

---

## References

1. Nahin et al. (2025). *What Is The Performance Ceiling of My Classifier?* [arXiv:2510.03950](https://arxiv.org/abs/2510.03950)
2. Koh & Liang (2017). *Understanding Black-box Predictions via Influence Functions.* ICML
3. George et al. (2018). *Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis.* NeurIPS

---

## Author

**Shihua Yu**
MS Computer Engineering, NYU Tandon School of Engineering
GitHub: [@youseihuayu-wonderful](https://github.com/youseihuayu-wonderful)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
