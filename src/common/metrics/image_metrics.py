"""Per-image and batch RGB PSNR/SSIM with the verified UIE3 semantics."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as functional


def _validate_metric_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    crop_border: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {prediction.shape} "
            f"and {target.shape}."
        )
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(
            f"RGB metrics require [B,3,H,W], got shape {tuple(prediction.shape)}."
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("RGB metrics require floating-point tensors.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RGB metric inputs must contain only finite values.")
    if float(data_range) != 1.0:
        raise ValueError(f"v1 fixes data_range=1.0, got {data_range!r}.")
    if crop_border != 0:
        raise ValueError(f"v1 fixes crop_border=0, got {crop_border!r}.")
    if torch.any(target < 0.0) or torch.any(target > 1.0):
        raise ValueError("Metric target must already lie in [0,1].")
    return prediction.clamp(0.0, 1.0), target


def rgb_psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    crop_border: int = 0,
) -> torch.Tensor:
    """Return joint-RGB PSNR per image; identical images yield positive infinity."""

    prediction, target = _validate_metric_inputs(
        prediction, target, data_range=data_range, crop_border=crop_border
    )
    mse = torch.mean(torch.square(prediction - target), dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((data_range * data_range) / mse)
    return torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)


def rgb_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    crop_border: int = 0,
) -> torch.Tensor:
    """Return the mean of independently computed per-image RGB PSNR."""

    return rgb_psnr_per_image(
        prediction, target, data_range=data_range, crop_border=crop_border
    ).mean()


def _gaussian_window(
    window_size: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    coordinates = coordinates - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    return window_2d.view(1, 1, window_size, window_size).expand(3, 1, -1, -1)


def rgb_ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    crop_border: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return valid-window SSIM per image, averaged spatially and across RGB."""

    prediction, target = _validate_metric_inputs(
        prediction, target, data_range=data_range, crop_border=crop_border
    )
    if window_size != 11:
        raise ValueError(f"v1 fixes SSIM window_size=11, got {window_size}.")
    if not math.isclose(float(sigma), 1.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"v1 fixes SSIM sigma=1.5, got {sigma!r}.")
    height, width = prediction.shape[-2:]
    if height < window_size or width < window_size:
        raise ValueError(
            f"SSIM requires H and W >= {window_size}, got {height}x{width}."
        )
    window = _gaussian_window(
        window_size, sigma, dtype=prediction.dtype, device=prediction.device
    )
    mu_prediction = functional.conv2d(prediction, window, groups=3)
    mu_target = functional.conv2d(target, window, groups=3)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction = (
        functional.conv2d(prediction.square(), window, groups=3) - mu_prediction_sq
    )
    sigma_target = (
        functional.conv2d(target.square(), window, groups=3) - mu_target_sq
    )
    sigma_cross = (
        functional.conv2d(prediction * target, window, groups=3) - mu_product
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction + sigma_target + c2
    )
    ssim_map = numerator / denominator
    if not torch.isfinite(ssim_map).all():
        raise FloatingPointError("SSIM produced a non-finite value.")
    return ssim_map.mean(dim=(1, 2, 3))


def rgb_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    crop_border: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return the mean of independently computed per-image RGB SSIM."""

    return rgb_ssim_per_image(
        prediction,
        target,
        data_range=data_range,
        crop_border=crop_border,
        window_size=window_size,
        sigma=sigma,
    ).mean()
