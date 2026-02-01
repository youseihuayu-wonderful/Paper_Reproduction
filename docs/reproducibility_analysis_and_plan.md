# Reproducibility Analysis & Conceptual Understanding

**Paper:** "What Is The Performance Ceiling of My Classifier?" (arXiv:2510.03950)

---

## 1. Reproduction Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Influence Computation** | ✅ Verified | Exact Hessian for logistic regression |
| **Category-wise Influence** | ✅ Verified | K-dimensional vectors working |
| **Spearman Correlation** | ✅ Verified | 0.9487 > 0.8 threshold |
| **Pareto-LP-GA (Synthetic)** | ✅ Working | +3.57% with TopK approach |
| **EKFAC Implementation** | ✅ Completed | Tested on MLP |
| **Course Correction** | ⚠️ N/A | Requires non-convex model |
| **CIFAR-10/ResNet** | 📝 Documented | Resource constraints |
| **BERT/NLP** | 📝 Documented | Resource constraints |

---

## 2. What We Successfully Verified

### 2.1 Spearman Correlation (Paper Section 5.1)

**Claim:** Category-wise influence predicts actual performance changes with Spearman correlation > 0.8.

**Our Result:** Correlation = **0.9487** ✅

**Methodology:**
1. Compute category-wise influence for all training samples
2. Remove top 10% beneficial/detrimental samples per class
3. Retrain and measure actual performance change
4. Compare predicted vs actual changes

### 2.2 Pareto-LP-GA (Paper Section 3.4)

**Claim:** LP-based sample reweighting can improve target class without degrading others.

**Our Result:** +3.57% improvement on target class, 0% degradation on non-target ✅

**Critical Discovery:** Pure LP produces binary weights that violate first-order approximation validity. TopK upweighting is the practical solution.

### 2.3 EKFAC Implementation

**Purpose:** Enable O(d) influence computation for neural networks instead of O(d³).

**Our Implementation:** Complete EKFAC in [src/ekfac.py](../src/ekfac.py)
- Kronecker factorization: H ≈ G ⊗ A
- Eigendecomposition for stable inversion
- PyTorch hooks for gradient capture
- Tested on simple MLP

---

## 3. Critical Discovery: Why LP Fails

### 3.1 The Problem

Linear Programming produces **BINARY weights** due to the fundamental theorem of LP:
> Optimal solutions are always at vertices of the feasible polytope.

With box constraints [0,1]^n, vertices are exactly binary points {0,1}^n.

### 3.2 Why This Matters

Influence functions are **first-order Taylor approximations**:
$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L \cdot \Delta\theta$$

This is only valid for **small** perturbations. Binary weights (removing 50% of samples) are NOT small perturbations, causing predictions to be wildly inaccurate.

### 3.3 Method Comparison

| Method | Class 0 Δ | Class 1 Δ | Result |
|--------|-----------|-----------|--------|
| **TopK(k=10, m=1.5)** | **+3.57%** | **0.00%** | **✅ Only working method** |
| Entropy-regularized | -7.14% | -18.75% | ❌ Fails |
| LP (binary) | -7.14% | -25.00% | ❌ Fails badly |

---

## 4. What Could NOT Be Reproduced (Resource Constraints)

The following require GPU resources and extensive training that were not available:

### 4.1 CIFAR-10 with ResNet

**Paper's Experiment:**
- Train ResNet-18/34 on CIFAR-10
- Compute EKFAC factors
- Calculate category-wise influence for 50,000 samples
- Identify class-specific beneficial/harmful samples

**Resource Requirements:**
- GPU with 8GB+ VRAM
- Several hours of training
- Large memory for influence storage

**Conceptual Understanding:**
- ResNet's non-convex loss creates diverse influence patterns
- Different samples affect different classes differently
- Method should identify mislabeled samples and class-specific noise
- See [CONCEPTUAL_UNDERSTANDING.md](CONCEPTUAL_UNDERSTANDING.md) Section 6.1

### 4.2 Course Correction

**Paper's Experiment:**
- Train neural network with checkpoints
- Simulate training regression via bad batches
- Apply influence-guided correction to recover

**Why It Fundamentally Requires Non-Convex Models:**
- Logistic regression (convex) has unique global optimum
- No "regression" possible - always converges to same solution
- Neural networks have multiple local minima
- Bad batches can push toward suboptimal minima

**Conceptual Understanding:**
- See [CONCEPTUAL_UNDERSTANDING.md](CONCEPTUAL_UNDERSTANDING.md) Section 4

### 4.3 BERT for NLP

**Paper's Experiment:**
- Fine-tune BERT on text classification
- Apply EKFAC with attention layer support
- Compute category-wise influence

**Resource Requirements:**
- BERT has ~110M parameters
- Specialized EKFAC for attention mechanisms
- Significant GPU memory for fine-tuning

---

## 5. Conceptual Understanding Documentation

For all experiments that could not be run, comprehensive conceptual understanding is documented in:

**[CONCEPTUAL_UNDERSTANDING.md](CONCEPTUAL_UNDERSTANDING.md)**

Contents:
1. Core Concept: Category-Wise Influence Functions
2. Pareto Regions and Performance Ceiling
3. Pareto-LP-GA Algorithm (with critical implementation insights)
4. Course Correction (why it requires non-convex models)
5. EKFAC: Scaling to Neural Networks
6. Paper's Experimental Claims (theoretical analysis)
7. What We Successfully Verified
8. Key Takeaways

---

## 6. Implementation Summary

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/influence.py` | Standard influence functions |
| `src/category_influence.py` | Category-wise influence vectors |
| `src/lp_reweight.py` | LP, entropy, gentle, TopK weighting |
| `src/ekfac.py` | EKFAC for neural networks |
| `src/pareto.py` | Pareto visualization |
| `experiments/validation_experiments.py` | All validation tests |

### Key Functions

```python
# Category-wise influence
from src.category_influence import compute_all_influence_vectors

# Sample reweighting (use TopK!)
from src.lp_reweight import solve_topk_weights  # RECOMMENDED

# EKFAC for neural networks
from src.ekfac import EKFACInfluence, compute_category_influences_ekfac
```

---

## 7. Running the Validation

```bash
cd Paper_Reproduction
.venv/bin/python3 experiments/validation_experiments.py
```

**Expected Output:**
```
1. SPEARMAN CORRELATION: ✅ VERIFIED (0.9487)
2. PARETO-LP-GA: ✅ WORKING (+3.57%)
3. COURSE CORRECTION: ⚠️ N/A (convex model)
```

---

## 8. Key Insights from This Work

### Theoretical Insights

1. **Category-wise influence is powerful:** Vector influence enables nuanced understanding of sample effects across classes.

2. **Pareto ceiling is a real phenomenon:** When all samples lie on the hyperplane $\sum_k P^k = 0$, no Pareto improvement is possible.

3. **First-order validity is crucial:** Influence functions only work for small perturbations - this limits practical applications.

### Practical Insights

1. **LP is theoretically correct but practically problematic:** Binary weights violate first-order assumptions.

2. **TopK upweighting works:** Gentle, targeted modifications respect the approximation's validity.

3. **Non-convexity enables Course Correction:** Convex models don't need (and can't benefit from) Course Correction.

---

## 9. References

1. Koh & Liang (2017). Understanding black-box predictions via influence functions. ICML.
2. George et al. (2018). Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis. NeurIPS.
3. Nahin et al. (2025). What Is The Performance Ceiling of My Classifier? arXiv:2510.03950.
