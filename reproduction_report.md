# Paper Reproduction: "What Is The Performance Ceiling of My Classifier?"

## Overview

This document reproduces the key results from Nahin et al. (2025) "What Is The Performance Ceiling of My Classifier?" (arXiv:2510.03950).

**Core Question**: Given a trained classifier, can we determine if its performance can still be improved, or has it reached its theoretical limit?

---

## Part 1: Theoretical Foundation

### 1.1 Standard Influence Functions (Koh & Liang, 2017)

**Goal**: Measure how each training sample affects model predictions.

#### Setup
- Training set: $\mathcal{D}_{train} = \{z_1, z_2, ..., z_n\}$ where $z_i = (x_i, y_i)$
- Validation set: $V = \{z'_1, z'_2, ..., z'_m\}$
- Model parameters: $\theta \in \mathbb{R}^d$
- Loss function: $\ell(z; \theta)$

#### Original Training Problem
$$\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \ell(z_i; \theta)$$

#### Perturbed Problem (upweight sample $z$ by $\epsilon$)
$$\theta^*_\epsilon = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \ell(z_i; \theta) + \epsilon \cdot \ell(z; \theta)$$

#### Key Derivation

**Step 1**: At optimum, gradient = 0
$$\nabla_\theta \left[ \frac{1}{n} \sum_i \ell(z_i; \theta^*_\epsilon) + \epsilon \cdot \ell(z; \theta^*_\epsilon) \right] = 0$$

**Step 2**: Implicit differentiation (take derivative w.r.t. $\epsilon$ at $\epsilon=0$)
$$\frac{d\theta^*_\epsilon}{d\epsilon}\bigg|_{\epsilon=0} = -H^{-1} \nabla_\theta \ell(z; \theta^*)$$

where $H = \frac{1}{n} \sum_i \nabla^2_\theta \ell(z_i; \theta^*)$ is the Hessian.

**Step 3**: First-order approximation
$$\theta^*_\epsilon \approx \theta^* - \epsilon \cdot H^{-1} \nabla_\theta \ell(z; \theta^*)$$

**Step 4**: Effect on validation loss
$$\Delta L_{val} \approx \nabla L_{val}(\theta^*)^\top \cdot (\theta^*_\epsilon - \theta^*) = -\epsilon \cdot \underbrace{\nabla L_{val}(\theta^*)^\top H^{-1} \nabla_\theta \ell(z; \theta^*)}_{\mathcal{I}(z, V)}$$

#### Influence Function Definition
$$\boxed{\mathcal{I}(z, V) = \sum_{z' \in V} \nabla_\theta \ell(z'; \theta^*)^\top H^{-1} \nabla_\theta \ell(z; \theta^*)}$$

**Interpretation**:
- $\mathcal{I}(z, V) > 0$: Sample $z$ is **beneficial** (removing it increases validation loss)
- $\mathcal{I}(z, V) < 0$: Sample $z$ is **detrimental** (removing it decreases validation loss)

---

### 1.2 Category-Wise Influence Vectors (Paper's Innovation)

**Key Insight**: Instead of scalar influence, compute influence **per class**.

#### Definition
For $K$ classes, each training sample $z$ has an influence **vector**:
$$P(z) = [P^0(z), P^1(z), ..., P^{K-1}(z)] \in \mathbb{R}^K$$

where:
$$P^k(z) = \mathcal{I}(z, V_k) = \sum_{z' \in V_k} \nabla_\theta \ell(z'; \theta^*)^\top H^{-1} \nabla_\theta \ell(z; \theta^*)$$

and $V_k$ is the validation samples of class $k$.

#### Why This Matters
A sample can have **different effects on different classes**:

| Sample | $P^0(z)$ | $P^1(z)$ | Interpretation |
|--------|----------|----------|----------------|
| $z_1$  | +0.5     | +0.3     | Helps both classes (Joint Positive) |
| $z_2$  | -0.4     | -0.2     | Hurts both classes (Joint Negative) |
| $z_3$  | +0.6     | -0.4     | Tradeoff: helps class 0, hurts class 1 |
| $z_4$  | -0.3     | +0.7     | Tradeoff: hurts class 0, helps class 1 |

---

### 1.3 Pareto Frontier Analysis

#### Four Regions (2-class case)

```
        P¹(z)
          ↑
          │    Tradeoff          Joint
          │    (helps 1,         Positive
          │     hurts 0)         (helps both)
          │         II     │     I
          │                │
    ──────┼────────────────┼──────────→ P⁰(z)
          │                │
          │        III     │    IV
          │    Joint           Tradeoff
          │    Negative        (helps 0,
          │    (hurts both)     hurts 1)
          │
```

#### Pareto Improvement Conditions

**Pareto Improvement Possible** when:
- Region I (Joint Positive) is non-empty → Upweight these samples
- Region III (Joint Negative) is non-empty → Downweight these samples

**Performance Ceiling Reached** when:
- All samples in tradeoff regions (II and IV)
- Points lie on line $P^0 + P^1 = 0$
- Any improvement in one class must hurt another

---

### 1.4 Reweighting Optimization

#### Paper's LP Formulation
$$\max_w \sum_{k \in \text{target}} \sum_i P^k(z_i) \cdot w_i$$

Subject to:
- $\sum_i P^k(z_i) \cdot w_i \geq \alpha_k \cdot \sum_i P^k(z_i)$ for non-target classes
- $0 \leq w_i \leq 1$

#### Why LP Fails (Our Finding)
LP produces **binary weights** $w_i \in \{0, 1\}$, modifying ~50% of samples.

The influence function is a **first-order Taylor approximation**:
$$\Delta L \approx \epsilon \cdot \mathcal{I}(z, V)$$

This is only valid for **small** $\epsilon$. When $\epsilon = \pm 1$ for many samples, the approximation breaks down.

#### TopK Solution (What Actually Works)
Instead of LP, use simple top-k selection:
1. Sort samples by $P^{target}(z_i)$
2. Upweight only top-k samples (k ≈ 4% of training set)
3. Use gentle multiplier (1.5x instead of 0 or ∞)

---

## Part 2: Implementation

### 2.1 Logistic Regression (Binary Classification)

For logistic regression with sigmoid $\sigma(x) = 1/(1+e^{-x})$:

**Prediction**: $p_i = \sigma(w^\top x_i)$

**Loss**: $\ell(z_i; w) = -[y_i \log p_i + (1-y_i) \log(1-p_i)]$

**Gradient**: $\nabla_w \ell(z_i; w) = (p_i - y_i) \cdot x_i$

**Hessian**: $H = \frac{1}{n} \sum_i p_i(1-p_i) \cdot x_i x_i^\top$

### 2.2 Algorithm: Compute Influence Vectors

```python
def compute_influence_vectors(train_X, train_y, val_X, val_y, weights, damping=1e-4):
    """
    Compute category-wise influence vectors for all training samples.

    Returns: influence_vectors of shape (n_train, n_classes)
    """
    # Step 1: Compute Hessian
    probs = sigmoid(train_X @ weights)
    variance = probs * (1 - probs)
    H = (1/n) * (train_X.T @ diag(variance) @ train_X)

    # Step 2: Add damping and invert
    H_inv = inv(H + damping * I)

    # Step 3: For each training sample, compute influence on each class
    for j in range(n_train):
        grad_train_j = (probs[j] - train_y[j]) * train_X[j]
        influence_direction = H_inv @ grad_train_j

        for k in range(n_classes):
            val_k = val_X[val_y == k]
            for val_sample in val_k:
                grad_val = (pred - true) * val_sample
                influence_vectors[j, k] += grad_val @ influence_direction

    return influence_vectors
```

### 2.3 Algorithm: TopK Reweighting

```python
def topk_reweight(influence_vectors, target_class, k=10, multiplier=1.5):
    """
    Upweight top-k most beneficial samples for target class.
    """
    # Get influence scores for target class
    scores = influence_vectors[:, target_class]

    # Find top-k indices (highest positive influence)
    top_k_idx = argsort(scores)[-k:]

    # Create weights
    weights = ones(n_samples)
    weights[top_k_idx] = multiplier

    return weights
```

---

## Part 3: Experiments

### 3.1 Synthetic Data Experiments

#### Experiment A: Linearly Separable Data
- Two Gaussian clusters with separation
- Verify influence function correctness
- Test Pareto analysis

#### Experiment B: Overlapping Classes
- Reduced separation → more tradeoff samples
- Verify ceiling detection

#### Experiment C: Noisy Labels
- Inject label noise
- Verify joint negative detection

### 3.2 Method Comparison

| Method | Modifies | Weight Range | Works? |
|--------|----------|--------------|--------|
| Pure LP | ~50% | {0, 1} | ❌ |
| Entropy LP | ~50% | (0, 1) | ❌ |
| TopK Upweight | ~4% | {1.0, 1.5} | ✅ |
| TopK Downweight | ~4% | {0.7, 1.0} | ✅ |

### 3.3 DI vs CC Experiments

#### Direct Improvement (DI) - Epoch 10→11
- Applied early when model is learning
- Upweight beneficial samples for target class
- Goal: Proactively boost performance

#### Curse Correction (CC) - Epoch 15→16
- Applied later when errors accumulated
- Remove/downweight harmful samples
- Goal: Fix "curse" of error amplification

---

## Part 4: Results

(To be filled with experimental results)

### 4.1 Synthetic Data Results

### 4.2 LP vs TopK Comparison

### 4.3 Table 1 Reproduction (DI and CC)

### 4.4 Performance Ceiling Analysis

---

## Part 5: Key Findings

### 5.1 Why LP Fails
1. LP produces binary weights (vertex solutions)
2. Modifies too many samples (~50%)
3. Violates first-order approximation validity
4. Combined up+down weighting fails empirically

### 5.2 Why TopK Works
1. Modifies few samples (~4%)
2. Uses gentle weight changes (1.5x)
3. Stays within first-order validity
4. Single-direction (up OR down) is stable

### 5.3 Practical Recommendations
1. Use TopK instead of LP for reweighting
2. Modify ≤ 10% of samples
3. Use multipliers in range [0.5, 1.5]
4. Choose upweight-only OR downweight-only, not both

---

## Appendix A: Mathematical Details

### A.1 Hessian Derivation for Logistic Regression

### A.2 Influence Function for Multi-class (Softmax)

### A.3 Connection to Leave-One-Out Retraining

---

## References

1. Nahin et al. (2025). "What Is The Performance Ceiling of My Classifier?" arXiv:2510.03950
2. Koh & Liang (2017). "Understanding Black-box Predictions via Influence Functions." ICML.
