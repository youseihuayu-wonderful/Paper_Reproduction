"""
ekfac.py - Eigenvalue-corrected Kronecker-Factored Approximate Curvature

This module implements EKFAC for efficient influence function computation
in neural networks. Standard influence functions require inverting the
Hessian matrix H ∈ R^{d×d} where d is the number of parameters, which is
O(d³) and infeasible for modern neural networks.

EKFAC approximates the Hessian using Kronecker factorization:
    H ≈ G ⊗ A

Where:
    - A: Activation covariance (input to layer)
    - G: Gradient covariance (backpropagated gradients)
    - ⊗: Kronecker product

This allows O(d) influence computation instead of O(d³).

Key Equations:
    For a layer with weight W ∈ R^{out × in}:
    - A = E[a aᵀ] where a is the input activation
    - G = E[g gᵀ] where g is the gradient w.r.t. pre-activation
    - H_layer ≈ G ⊗ A

    Influence computation:
    - H⁻¹v ≈ (G⁻¹ ⊗ A⁻¹)v
    - Using vec(XYZ) = (Zᵀ ⊗ X)vec(Y), this is efficient

Reference:
    George et al. (2018). "Fast Approximate Natural Gradient Descent
    in a Kronecker-factored Eigenbasis"

    Grosse & Martens (2016). "A Kronecker-factored approximate Fisher
    for convolution layers"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict


class EKFACInfluence:
    """
    EKFAC-based influence function computation for neural networks.

    This class registers hooks on a PyTorch model to capture activations
    and gradients, computes Kronecker-factored approximations of the
    Fisher information matrix, and uses these to efficiently compute
    influence scores.

    Usage:
    ```python
    model = MyModel()
    ekfac = EKFACInfluence(model, damping=0.01)

    # Compute factors from training data
    ekfac.compute_factors(train_loader)

    # Compute influence of training samples on validation
    influences = ekfac.compute_influences(train_loader, val_loader)
    ```
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 0.01,
        layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d)
    ):
        """
        Initialize EKFAC influence computation.

        Args:
            model: PyTorch model to analyze
            damping: Regularization for matrix inversion (default: 0.01)
            layer_types: Types of layers to track (default: Linear, Conv2d)
        """
        self.model = model
        self.damping = damping
        self.layer_types = layer_types

        # Storage for Kronecker factors
        self.A_factors: Dict[str, torch.Tensor] = {}  # Activation covariance
        self.G_factors: Dict[str, torch.Tensor] = {}  # Gradient covariance

        # Storage for eigendecompositions
        self.A_eigvals: Dict[str, torch.Tensor] = {}
        self.A_eigvecs: Dict[str, torch.Tensor] = {}
        self.G_eigvals: Dict[str, torch.Tensor] = {}
        self.G_eigvecs: Dict[str, torch.Tensor] = {}

        # Temporary storage for forward/backward hooks
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}

        # Hooks
        self._hooks: List = []

        # Get tracked layers
        self.tracked_layers = self._get_tracked_layers()

    def _get_tracked_layers(self) -> Dict[str, nn.Module]:
        """Get all layers of tracked types."""
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types):
                layers[name] = module
        return layers

    def _save_activation(self, name: str):
        """Create hook to save activation."""
        def hook(module, input, output):
            # For Linear: input is (batch, in_features)
            # For Conv2d: input is (batch, channels, h, w)
            inp = input[0].detach()

            if isinstance(module, nn.Conv2d):
                # Unfold convolution input to matrix form
                # Shape: (batch * h_out * w_out, in_channels * kernel_h * kernel_w)
                inp = self._unfold_conv_input(inp, module)
            else:
                # Linear layer: add bias term
                # Shape: (batch, in_features + 1)
                ones = torch.ones(inp.shape[0], 1, device=inp.device)
                inp = torch.cat([inp, ones], dim=1)

            self._activations[name] = inp
        return hook

    def _save_gradient(self, name: str):
        """Create hook to save gradient w.r.t. pre-activation."""
        def hook(module, grad_input, grad_output):
            # grad_output is gradient w.r.t. the output of this layer
            grad = grad_output[0].detach()

            if isinstance(module, nn.Conv2d):
                # Reshape gradient for conv layer
                # Shape: (batch * h_out * w_out, out_channels)
                grad = grad.permute(0, 2, 3, 1).reshape(-1, grad.shape[1])
            # else: Linear layer gradient is already (batch, out_features)

            self._gradients[name] = grad
        return hook

    def _unfold_conv_input(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        """
        Unfold convolution input to matrix form.

        For a conv layer with input (N, C_in, H, W) and kernel (C_out, C_in, K_h, K_w),
        we unfold the input to shape (N * H_out * W_out, C_in * K_h * K_w).

        This allows treating conv as matrix multiplication.
        """
        # Use unfold to extract patches
        x_unfold = torch.nn.functional.unfold(
            x,
            kernel_size=conv.kernel_size,
            padding=conv.padding,
            stride=conv.stride
        )
        # Shape: (batch, C_in * K_h * K_w, H_out * W_out)
        x_unfold = x_unfold.permute(0, 2, 1)
        # Shape: (batch, H_out * W_out, C_in * K_h * K_w)
        x_unfold = x_unfold.reshape(-1, x_unfold.shape[-1])
        # Shape: (batch * H_out * W_out, C_in * K_h * K_w)

        # Add bias term
        ones = torch.ones(x_unfold.shape[0], 1, device=x_unfold.device)
        x_unfold = torch.cat([x_unfold, ones], dim=1)

        return x_unfold

    def register_hooks(self):
        """Register forward and backward hooks on tracked layers."""
        self.remove_hooks()  # Remove any existing hooks

        for name, module in self.tracked_layers.items():
            # Forward hook for activations
            fwd_hook = module.register_forward_hook(self._save_activation(name))
            self._hooks.append(fwd_hook)

            # Backward hook for gradients
            bwd_hook = module.register_full_backward_hook(self._save_gradient(name))
            self._hooks.append(bwd_hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def compute_factors(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Optional[Callable] = None,
        device: str = 'cpu'
    ):
        """
        Compute Kronecker factors A and G from data.

        This performs a single pass through the dataloader, accumulating
        the empirical covariance matrices.

        Args:
            dataloader: DataLoader providing (input, target) batches
            loss_fn: Loss function. If None, uses cross-entropy.
            device: Device to compute on ('cpu' or 'cuda')
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.to(device)
        self.model.eval()
        self.register_hooks()

        # Initialize accumulators
        A_sum = {name: None for name in self.tracked_layers}
        G_sum = {name: None for name in self.tracked_layers}
        n_samples = 0

        try:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.shape[0]
                n_samples += batch_size

                # Forward pass
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

                # Backward pass
                loss.backward()

                # Accumulate factors
                for name in self.tracked_layers:
                    a = self._activations[name]  # (n, d_in)
                    g = self._gradients[name]     # (n, d_out)

                    # Compute outer products
                    # A_i = a_i a_i^T, G_i = g_i g_i^T
                    A_batch = torch.einsum('ni,nj->ij', a, a)  # (d_in, d_in)
                    G_batch = torch.einsum('ni,nj->ij', g, g)  # (d_out, d_out)

                    if A_sum[name] is None:
                        A_sum[name] = A_batch
                        G_sum[name] = G_batch
                    else:
                        A_sum[name] += A_batch
                        G_sum[name] += G_batch

            # Normalize to get empirical covariance
            for name in self.tracked_layers:
                self.A_factors[name] = A_sum[name] / n_samples
                self.G_factors[name] = G_sum[name] / n_samples

            # Compute eigendecompositions for efficient inversion
            self._compute_eigendecompositions()

        finally:
            self.remove_hooks()

    def _compute_eigendecompositions(self):
        """Compute eigendecomposition of A and G factors."""
        for name in self.tracked_layers:
            A = self.A_factors[name]
            G = self.G_factors[name]

            # Eigendecomposition: A = V_A D_A V_A^T
            eigvals_A, eigvecs_A = torch.linalg.eigh(A)
            eigvals_G, eigvecs_G = torch.linalg.eigh(G)

            self.A_eigvals[name] = eigvals_A
            self.A_eigvecs[name] = eigvecs_A
            self.G_eigvals[name] = eigvals_G
            self.G_eigvecs[name] = eigvecs_G

    def compute_inverse_hvp(
        self,
        v: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute H^{-1} v using Kronecker factorization.

        For each layer:
            H^{-1} v = (G^{-1} ⊗ A^{-1}) v

        Using eigendecomposition for stable inversion:
            A^{-1} = V_A (D_A + λI)^{-1} V_A^T

        Args:
            v: Dictionary mapping layer name -> gradient vector

        Returns:
            Dictionary mapping layer name -> H^{-1} v
        """
        result = {}

        for name in self.tracked_layers:
            if name not in v:
                continue

            v_layer = v[name]

            # Get eigendecompositions
            eigvals_A = self.A_eigvals[name]
            eigvecs_A = self.A_eigvecs[name]
            eigvals_G = self.G_eigvals[name]
            eigvecs_G = self.G_eigvecs[name]

            # Dimensions from eigendecomposition (includes bias)
            d_out = len(eigvals_G)
            d_in_with_bias = len(eigvals_A)

            # v_layer is weight gradient (d_out, d_in) without bias
            # We need to pad it to include bias dimension
            if v_layer.dim() == 1:
                V = v_layer.reshape(d_out, -1)
            else:
                V = v_layer

            # Pad with zeros for bias if needed
            d_in_actual = V.shape[1]
            if d_in_actual < d_in_with_bias:
                # Add column for bias (gradient of bias is stored separately)
                # For now, just pad with small values
                padding = torch.zeros(d_out, d_in_with_bias - d_in_actual,
                                      device=V.device, dtype=V.dtype)
                V = torch.cat([V, padding], dim=1)

            # Transform to eigenbasis: V' = U_G^T V U_A
            V_eigen = eigvecs_G.T @ V @ eigvecs_A

            # Apply inverse in eigenbasis:
            # (H^{-1} V')_ij = V'_ij / (λ_G_i * λ_A_j + damping)
            damped_eigvals = torch.outer(eigvals_G, eigvals_A) + self.damping
            V_inv = V_eigen / damped_eigvals

            # Transform back: V'' = U_G V_inv U_A^T
            V_result = eigvecs_G @ V_inv @ eigvecs_A.T

            # Return only the weight gradient part (without bias padding)
            result[name] = V_result[:, :d_in_actual]

        return result

    def compute_sample_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient of loss w.r.t. each tracked layer's parameters
        for a single sample.

        Args:
            x: Input sample (should have batch dimension)
            y: Target label (should have batch dimension for classification)
            loss_fn: Loss function

        Returns:
            Dictionary mapping layer name -> parameter gradient
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Ensure batch dimension for input
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure target has correct shape for cross entropy
        # CrossEntropyLoss expects target shape (N,) for class indices
        if y.dim() == 0:
            y = y.unsqueeze(0)
        elif y.dim() == 2:
            y = y.squeeze(1)

        self.model.zero_grad()
        output = self.model(x)
        loss = loss_fn(output, y)
        loss.backward()

        gradients = {}
        for name, module in self.tracked_layers.items():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                gradients[name] = module.weight.grad.clone()

        return gradients

    def compute_influence_score(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> float:
        """
        Compute influence of a single training sample on validation loss.

        Influence = -∇L_val^T H^{-1} ∇L_train

        Args:
            train_x: Training sample input
            train_y: Training sample target
            val_x: Validation sample(s) input
            val_y: Validation sample(s) target
            loss_fn: Loss function

        Returns:
            Scalar influence score
        """
        # Compute training gradient
        grad_train = self.compute_sample_gradient(train_x, train_y, loss_fn)

        # Compute H^{-1} @ grad_train
        hvp = self.compute_inverse_hvp(grad_train)

        # Compute validation gradient
        grad_val = self.compute_sample_gradient(val_x, val_y, loss_fn)

        # Dot product
        influence = 0.0
        for name in self.tracked_layers:
            if name in hvp and name in grad_val:
                influence += torch.sum(grad_val[name] * hvp[name]).item()

        return influence

    def compute_all_influences(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cpu',
        loss_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Compute influence scores for all training samples.

        Args:
            train_loader: DataLoader for training data (batch_size should be 1
                          or we compute per-sample gradients)
            val_loader: DataLoader for validation data
            device: Device for computation
            loss_fn: Loss function

        Returns:
            Array of influence scores, shape (n_train,)
        """
        self.model.to(device)
        self.model.eval()

        # First, compute sum of validation gradients (per-sample)
        val_grad_sum = {name: None for name in self.tracked_layers}

        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)

            # Process each validation sample
            for i in range(val_x.shape[0]):
                grad = self.compute_sample_gradient(
                    val_x[i:i+1], val_y[i:i+1], loss_fn
                )

                for name in self.tracked_layers:
                    if name in grad:
                        if val_grad_sum[name] is None:
                            val_grad_sum[name] = grad[name].clone()
                        else:
                            val_grad_sum[name] += grad[name]

        # Compute H^{-1} @ val_grad
        hvp_val = self.compute_inverse_hvp(val_grad_sum)

        # Compute influence for each training sample
        influences = []

        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            # Handle batch
            for i in range(train_x.shape[0]):
                x_i = train_x[i:i+1]
                y_i = train_y[i:i+1]

                grad_train = self.compute_sample_gradient(x_i, y_i, loss_fn)

                # Influence = grad_train^T @ H^{-1} @ grad_val
                influence = 0.0
                for name in self.tracked_layers:
                    if name in grad_train and name in hvp_val:
                        influence += torch.sum(
                            grad_train[name] * hvp_val[name]
                        ).item()

                influences.append(influence)

        return np.array(influences)


def compute_category_influences_ekfac(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_classes: int,
    damping: float = 0.01,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Compute category-wise influence vectors using EKFAC.

    This is the main entry point for neural network influence computation.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (will be split by class)
        n_classes: Number of classes
        damping: EKFAC damping parameter
        device: Computation device

    Returns:
        Influence vectors of shape (n_train, n_classes)
    """
    ekfac = EKFACInfluence(model, damping=damping)

    # Compute factors from training data
    ekfac.compute_factors(train_loader, device=device)

    # Split validation by class
    val_by_class = defaultdict(list)
    for x, y in val_loader:
        for i in range(x.shape[0]):
            val_by_class[y[i].item()].append((x[i], y[i]))

    # Count training samples
    n_train = sum(x.shape[0] for x, _ in train_loader)
    influence_vectors = np.zeros((n_train, n_classes))

    # Compute influence on each class
    for class_k in range(n_classes):
        print(f"Computing influences for class {class_k}...")

        # Create loader for this class
        class_data = val_by_class[class_k]
        if len(class_data) == 0:
            continue

        # Stack class data
        class_x = torch.stack([x for x, _ in class_data])
        class_y = torch.stack([y for _, y in class_data])

        # Compute val gradient sum for this class
        model.to(device)
        val_grad_sum = {name: None for name in ekfac.tracked_layers}

        for i in range(class_x.shape[0]):
            grad = ekfac.compute_sample_gradient(
                class_x[i:i+1].to(device),
                class_y[i:i+1].to(device)
            )
            for name in ekfac.tracked_layers:
                if name in grad:
                    if val_grad_sum[name] is None:
                        val_grad_sum[name] = grad[name]
                    else:
                        val_grad_sum[name] += grad[name]

        # Compute H^{-1} @ val_grad
        hvp_val = ekfac.compute_inverse_hvp(val_grad_sum)

        # Compute influence for each training sample
        train_idx = 0
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            for i in range(train_x.shape[0]):
                grad_train = ekfac.compute_sample_gradient(
                    train_x[i:i+1], train_y[i:i+1]
                )

                influence = 0.0
                for name in ekfac.tracked_layers:
                    if name in grad_train and name in hvp_val:
                        influence += torch.sum(
                            grad_train[name] * hvp_val[name]
                        ).item()

                influence_vectors[train_idx, class_k] = influence
                train_idx += 1

    return influence_vectors
