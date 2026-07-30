"""Finite-checked Charbonnier reconstruction loss from the verified baseline."""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Compute ``sqrt((prediction - target)^2 + epsilon^2)``."""

    def __init__(self, epsilon: float = 1.0e-3, reduction: str = "mean") -> None:
        super().__init__()
        if not isinstance(epsilon, (float, int)) or epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        self.epsilon = float(epsilon)
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got {prediction.shape} "
                f"and {target.shape}."
            )
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("Charbonnier loss requires floating-point tensors.")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Charbonnier loss inputs must contain only finite values.")
        elementwise = torch.sqrt(
            torch.square(prediction - target) + self.epsilon * self.epsilon
        )
        if not torch.isfinite(elementwise).all():
            raise FloatingPointError("Charbonnier loss produced a non-finite value.")
        if self.reduction == "mean":
            return elementwise.mean()
        if self.reduction == "sum":
            return elementwise.sum()
        return elementwise
