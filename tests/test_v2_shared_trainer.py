from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.v1.loss import CharbonnierLoss  # noqa: E402
from src.v2.trainer import SharedOrderTrainer  # noqa: E402


class ToySharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.5))
        self.cs = torch.nn.Parameter(torch.tensor(0.25))
        self.sc = torch.nn.Parameter(torch.tensor(-0.25))

    def forward_color_then_scatter(self, value):
        return value * self.shared + self.cs

    def forward_scatter_then_color(self, value):
        return value * self.shared + self.sc


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _batch():
    return {
        "input": torch.full((1, 3, 12, 13), 0.4),
        "target": torch.full((1, 3, 12, 13), 0.1),
        "sample_id": ["sample"],
        "input_relative_path": ["input.png"],
        "gt_relative_path": ["gt.png"],
    }


def _trainer():
    model = ToySharedModel()
    optimizer = CountingSGD(model.parameters())
    trainer = SharedOrderTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=CharbonnierLoss(),
        device=torch.device("cpu"),
        amp=False,
        fail_on_nonfinite=True,
        gradient_clip_norm=None,
        metrics_config={
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
    )
    return trainer, model, optimizer


def test_shared_train_step_has_one_optimizer_step_and_two_gradients(monkeypatch) -> None:
    trainer, model, optimizer = _trainer()
    update_calls = 0
    original_update = trainer.scaler.update

    def counted_update(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        return original_update(*args, **kwargs)

    monkeypatch.setattr(trainer.scaler, "update", counted_update)
    result = trainer.train_step(_batch())
    assert optimizer.zero_calls == 1
    assert optimizer.step_calls == 1
    assert update_calls == 1
    assert trainer.global_step == 1
    assert result["loss_joint"] == pytest.approx(
        0.5 * result["loss_cs"] + 0.5 * result["loss_sc"]
    )
    assert model.cs.grad is not None and torch.isfinite(model.cs.grad)
    assert model.sc.grad is not None and torch.isfinite(model.sc.grad)
    assert model.shared.grad is not None and torch.isfinite(model.shared.grad)


def test_shared_validation_reports_both_paths_and_mean_metric() -> None:
    trainer, _, _ = _trainer()
    result = trainer.validate([_batch()])
    assert result["num_samples"] == 1
    assert result["mean_path_psnr"] == pytest.approx(
        0.5 * result["psnr_cs"] + 0.5 * result["psnr_sc"]
    )
    assert result["mean_path_ssim"] == pytest.approx(
        0.5 * result["ssim_cs"] + 0.5 * result["ssim_sc"]
    )
    assert result["joint_val_loss"] == pytest.approx(
        0.5 * result["val_loss_cs"] + 0.5 * result["val_loss_sc"]
    )
    assert len(result["per_image_cs"]) == 1
    assert len(result["per_image_sc"]) == 1
    assert len(result["comparison"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_shared_cuda_amp_uses_one_finite_optimizer_step() -> None:
    model = ToySharedModel()
    optimizer = CountingSGD(model.parameters())
    trainer = SharedOrderTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=CharbonnierLoss(),
        device=torch.device("cuda"),
        amp=True,
        fail_on_nonfinite=True,
        gradient_clip_norm=None,
        metrics_config={
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
    )
    result = trainer.train_step(_batch())
    assert optimizer.step_calls == 1
    assert result["optimizer_step_applied"] is True
    assert result["loss_joint"] == pytest.approx(
        0.5 * result["loss_cs"] + 0.5 * result["loss_sc"]
    )
