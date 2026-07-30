from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.common.metrics.image_metrics import (  # noqa: E402
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)


def test_psnr_and_ssim_are_finite_for_distinct_valid_images() -> None:
    target = torch.zeros(2, 3, 16, 17)
    prediction = torch.full_like(target, 0.25)
    psnr = rgb_psnr_per_image(prediction, target)
    ssim = rgb_ssim_per_image(prediction, target)
    assert torch.isfinite(psnr).all()
    assert torch.isfinite(ssim).all()


def test_prediction_is_clamped_only_for_metrics() -> None:
    prediction = torch.full((1, 3, 16, 16), 2.0)
    original = prediction.clone()
    target = torch.zeros_like(prediction)
    result = rgb_psnr_per_image(prediction, target)
    assert float(result[0]) == pytest.approx(0.0, abs=1.0e-6)
    assert torch.equal(prediction, original)


@pytest.mark.parametrize(
    "prediction,target",
    [
        (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 17)),
        (torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16)),
        (torch.zeros(3, 16, 16), torch.zeros(3, 16, 16)),
    ],
)
def test_metric_shape_validation(prediction, target) -> None:
    with pytest.raises(ValueError):
        rgb_psnr_per_image(prediction, target)
