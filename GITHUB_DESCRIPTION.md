# GitHub Repository Description

## Short Description (For GitHub "About" Section)
```
Reproduction of "What Is The Performance Ceiling of My Classifier?" (arXiv:2510.03950) - Implementing category-wise influence functions and Pareto frontier analysis to detect classifier performance limits.
```

---

## Full README Description

# Paper Reproduction: What Is The Performance Ceiling of My Classifier?

[![Paper](https://img.shields.io/badge/arXiv-2510.03950-b31b1b.svg)](https://arxiv.org/abs/2510.03950)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A systematic reproduction of the key experimental claims from **Nahin et al. (2025)** - *"What Is The Performance Ceiling of My Classifier?"*

## Overview

This project implements **category-wise influence functions** and **Pareto frontier analysis** to answer a fundamental question in machine learning:

> *Can my classifier still be improved, or has it reached its theoretical performance limit?*

### Key Contributions Reproduced

| Claim | Paper | Our Result | Status |
|-------|-------|------------|--------|
| Spearman Correlation | > 0.8 | **0.9487** | Verified |
| Pareto-LP-GA Improvement | Improves target class | **+3.57%** | Verified |
| Noisy Sample Detection | Identifies mislabeled data | **87.8%** accuracy | Verified |
| Figure 2 Reproduction | Two curved arms pattern | Visual match | Verified |

---

## Core Concepts

### 1. Category-wise Influence Vectors
Instead of scalar influence scores, we compute influence **per class**:

$$P(z) = [P^0(z), P^1(z), ..., P^{K-1}(z)] \in \mathbb{R}^K$$

This reveals how each training sample affects different classes differently.

### 2. Pareto Frontier Analysis
Samples are categorized into four regions:
- **Joint Positive**: Helps all classes (upweight these)
- **Joint Negative**: Hurts all classes (downweight these)
- **Tradeoff Regions**: Improving one class hurts another

### 3. Performance Ceiling Detection
When all samples fall in tradeoff regions, the classifier has reached its **performance ceiling** - no Pareto improvement is possible.

---

## Technical Implementation

### Influence Function Formula
$$\mathcal{I}(z, V_k) = \sum_{z' \in V_k} \nabla_\theta \ell(z'; \theta^*)^\top H^{-1} \nabla_\theta \ell(z; \theta^*)$$

### Key Files
```
src/
├── influence.py           # Standard influence functions (Koh & Liang, 2017)
├── category_influence.py  # Category-wise influence vectors
├── lp_reweight_fixed.py   # Working TopK reweighting methods
├── pareto.py              # Pareto frontier visualization
└── pareto_lp_ga.py        # Full Pareto-LP-GA pipeline
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

# Install dependencies
pip install -r requirements.txt

# Run main reproduction experiment
python experiments/figure2_comprehensive.py
```

---

## Results Visualization

The reproduction successfully generates the characteristic "two curved arms" pattern from Figure 2 of the original paper, validating the category-wise influence function implementation.

---

## Skills Demonstrated

- **Mathematical Foundations**: Influence functions, Hessian computation, Taylor approximations
- **Optimization**: Linear programming, convex optimization
- **Machine Learning**: Logistic regression, sample reweighting, Pareto analysis
- **Software Engineering**: Modular code design, comprehensive documentation

---

## References

1. Nahin et al. (2025). *What Is The Performance Ceiling of My Classifier?* arXiv:2510.03950
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
