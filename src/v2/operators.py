"""Lightweight deterministic RGB-domain operators for the v2 order study."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


def _positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    return value


def _positive_finite(name: str, value: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
    return result


def _open_unit(name: str, value: float) -> float:
    result = _positive_finite(name, value)
    if result >= 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return result


def _validate_rgb_tensor(value: torch.Tensor, *, operator: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{operator} input must be a torch.Tensor.")
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(
            f"{operator} requires [B,3,H,W], got {tuple(value.shape)}."
        )
    if value.shape[0] < 1 or value.shape[2] < 1 or value.shape[3] < 1:
        raise ValueError(f"{operator} requires positive B, H, and W.")
    if not value.is_floating_point():
        raise TypeError(f"{operator} requires a floating-point tensor.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{operator} input contains NaN or Inf.")


def _require_finite_output(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} produced NaN or Inf.")


class ColorCorrectionOperator(nn.Module):
    """Predict one bounded affine RGB transform per input image."""

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        matrix_scale: float = 0.10,
        bias_scale: float = 0.05,
    ) -> None:
        super().__init__()
        hidden_channels = _positive_integer("hidden_channels", hidden_channels)
        self.matrix_scale = _positive_finite("matrix_scale", matrix_scale)
        self.bias_scale = _positive_finite("bias_scale", bias_scale)
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.parameter_head = nn.Conv2d(hidden_channels, 12, kernel_size=1)
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)

    def predict_parameters(
        self, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded matrix ``M`` and bias ``b`` used by ``forward``."""

        _validate_rgb_tensor(value, operator="ColorCorrectionOperator")
        parameters = self.parameter_head(self.features(value)).flatten(1)
        delta_matrix = parameters[:, :9].reshape(-1, 3, 3)
        delta_bias = parameters[:, 9:].reshape(-1, 3, 1, 1)
        identity = torch.eye(
            3, dtype=parameters.dtype, device=parameters.device
        ).unsqueeze(0)
        matrix = identity + self.matrix_scale * torch.tanh(delta_matrix)
        bias = self.bias_scale * torch.tanh(delta_bias)
        _require_finite_output("ColorCorrectionOperator matrix", matrix)
        _require_finite_output("ColorCorrectionOperator bias", bias)
        return matrix, bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        matrix, bias = self.predict_parameters(value)
        output = torch.einsum("bij,bjhw->bihw", matrix, value) + bias
        if output.shape != value.shape:
            raise RuntimeError("ColorCorrectionOperator changed the tensor shape.")
        _require_finite_output("ColorCorrectionOperator", output)
        return output


class ScatteringRemovalOperator(nn.Module):
    """Apply a bounded inverse-scattering residual in the RGB input domain."""

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        t_min: float = 0.20,
        eps: float = 1.0e-6,
        residual_max: float = 0.10,
        initial_t: float = 0.95,
        initial_A: float = 0.50,
    ) -> None:
        super().__init__()
        hidden_channels = _positive_integer("hidden_channels", hidden_channels)
        self.t_min = _open_unit("t_min", t_min)
        self.initial_t = _open_unit("initial_t", initial_t)
        self.initial_A = _open_unit("initial_A", initial_A)
        if not self.t_min < self.initial_t:
            raise ValueError("ScatteringRemovalOperator requires t_min < initial_t.")
        self.eps = _positive_finite("eps", eps)
        self.residual_max = _positive_finite("residual_max", residual_max)

        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.transmission_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.atmospheric_pool = nn.AdaptiveAvgPool2d(1)
        self.atmospheric_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)

        normalized_t = (self.initial_t - self.t_min) / (1.0 - self.t_min)
        transmission_logit = self._finite_logit("initial_t", normalized_t)
        atmospheric_logit = self._finite_logit("initial_A", self.initial_A)
        nn.init.zeros_(self.transmission_head.weight)
        nn.init.constant_(self.transmission_head.bias, transmission_logit)
        nn.init.zeros_(self.atmospheric_head.weight)
        nn.init.constant_(self.atmospheric_head.bias, atmospheric_logit)

    @staticmethod
    def _finite_logit(name: str, probability: float) -> float:
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError(
                f"{name} produces an invalid logit probability {probability!r}."
            )
        result = math.log(probability / (1.0 - probability))
        if not math.isfinite(result):
            raise ValueError(f"{name} produces a non-finite initialization logit.")
        return result

    def forward_with_diagnostics(
        self, value: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the enhanced image plus transmission and atmospheric light."""

        _validate_rgb_tensor(value, operator="ScatteringRemovalOperator")
        features = self.stem(value)
        transmission = self.t_min + (1.0 - self.t_min) * torch.sigmoid(
            self.transmission_head(features)
        )
        atmospheric_light = torch.sigmoid(
            self.atmospheric_head(self.atmospheric_pool(features))
        )
        raw = (
            value - atmospheric_light * (1.0 - transmission)
        ) / (transmission + self.eps)
        output = value + self.residual_max * torch.tanh(
            (raw - value) / self.residual_max
        )
        if output.shape != value.shape:
            raise RuntimeError("ScatteringRemovalOperator changed the tensor shape.")
        for name, tensor in (
            ("ScatteringRemovalOperator transmission", transmission),
            ("ScatteringRemovalOperator atmospheric_light", atmospheric_light),
            ("ScatteringRemovalOperator", output),
        ):
            _require_finite_output(name, tensor)
        return output, {
            "transmission": transmission,
            "atmospheric_light": atmospheric_light,
        }

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_diagnostics(value)
        return output
