"""Single-device v1 trainer with validated AMP-overflow step accounting."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Mapping

import torch

from src.common.metrics.image_metrics import (
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )
    return torch.cuda.amp.autocast(enabled=True)


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _string_batch(value: Any, batch_size: int, field_name: str) -> List[str]:
    values = [value] if isinstance(value, str) else [str(item) for item in value]
    if len(values) != batch_size:
        raise ValueError(
            f"Batch field {field_name!r} has {len(values)} values for "
            f"batch size {batch_size}."
        )
    return values


class V1Trainer:
    """Train NAFNet-small without any wrapper residual or output activation."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        device: torch.device,
        amp: bool,
        fail_on_nonfinite: bool,
        gradient_clip_norm: float | None,
        metrics_config: Mapping[str, Any],
        scheduler: Any = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp_enabled)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0:
            raise ValueError("gradient_clip_norm must be positive or None.")
        self.gradient_clip_norm = gradient_clip_norm
        self.metrics_config = dict(metrics_config)
        self.global_step = 0

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        self.model.train()
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        require_finite_tensor("training input", inputs)
        require_finite_tensor("training target", targets)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.amp_enabled):
            predictions = self.model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Training prediction/target shape mismatch: "
                    f"{predictions.shape} versus {targets.shape}."
                )
            require_finite_tensor("training prediction", predictions)
            loss = self.loss_function(predictions, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Training loss must be a finite scalar.")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradients_finite = gradients_are_finite(self.model)
        if not gradients_finite and not self.amp_enabled:
            raise FloatingPointError("A training gradient contains NaN or Inf.")
        if gradients_finite and self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.gradient_clip_norm),
                error_if_nonfinite=True,
            )

        amp_scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        amp_scale_after = float(self.scaler.get_scale())
        amp_overflow = self.amp_enabled and amp_scale_after < amp_scale_before
        optimizer_step_applied = not amp_overflow
        if optimizer_step_applied:
            self.global_step += 1
        return {
            "loss": float(loss.detach().cpu()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": self.global_step,
            "optimizer_step_applied": optimizer_step_applied,
            "amp_overflow_detected": amp_overflow,
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": amp_scale_after,
        }

    def train_epoch(
        self, data_loader: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        losses: List[float] = []
        applied_steps = 0
        skipped_steps = 0
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        amp_scale = float(self.scaler.get_scale())
        for batch in data_loader:
            result = self.train_step(batch)
            losses.append(result["loss"])
            learning_rate = result["learning_rate"]
            amp_scale = result["amp_scale_after"]
            if result["optimizer_step_applied"]:
                applied_steps += 1
            else:
                skipped_steps += 1
        if not losses:
            raise ValueError("Training DataLoader produced no batches.")
        train_loss = sum(losses) / len(losses)
        if self.fail_on_nonfinite and not math.isfinite(train_loss):
            raise FloatingPointError("Mean training loss is non-finite.")
        return {
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "amp_scale": amp_scale,
            "optimizer_applied_steps": applied_steps,
            "skipped_amp_overflow_steps": skipped_steps,
        }

    @torch.no_grad()
    def validate(
        self, data_loader: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        records: List[Dict[str, Any]] = []
        losses: List[float] = []
        try:
            for batch in data_loader:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                require_finite_tensor("validation input", inputs)
                require_finite_tensor("validation target", targets)
                predictions = self.model(inputs)
                if predictions.shape != targets.shape:
                    raise ValueError(
                        "Validation prediction/target shape mismatch: "
                        f"{predictions.shape} versus {targets.shape}."
                    )
                require_finite_tensor("validation prediction", predictions)
                loss = self.loss_function(predictions, targets)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Validation loss must be a finite scalar."
                    )
                psnr_values = rgb_psnr_per_image(
                    predictions,
                    targets,
                    data_range=float(self.metrics_config["data_range"]),
                    crop_border=int(self.metrics_config["crop_border"]),
                )
                ssim_values = rgb_ssim_per_image(
                    predictions,
                    targets,
                    data_range=float(self.metrics_config["data_range"]),
                    crop_border=int(self.metrics_config["crop_border"]),
                    window_size=int(self.metrics_config["ssim_window_size"]),
                    sigma=float(self.metrics_config["ssim_sigma"]),
                )
                batch_size = predictions.shape[0]
                sample_ids = _string_batch(
                    batch["sample_id"], batch_size, "sample_id"
                )
                input_paths = _string_batch(
                    batch["input_relative_path"],
                    batch_size,
                    "input_relative_path",
                )
                gt_paths = _string_batch(
                    batch["gt_relative_path"], batch_size, "gt_relative_path"
                )
                losses.extend(
                    [float(loss.detach().cpu()) for _ in range(batch_size)]
                )
                for index in range(batch_size):
                    records.append(
                        {
                            "sample_id": sample_ids[index],
                            "input_relative_path": input_paths[index],
                            "gt_relative_path": gt_paths[index],
                            "psnr_rgb": float(psnr_values[index].detach().cpu()),
                            "ssim_rgb": float(ssim_values[index].detach().cpu()),
                        }
                    )
        finally:
            self.model.train(was_training)
        if not records:
            raise ValueError("Validation DataLoader produced no samples.")
        result = {
            "num_samples": len(records),
            "val_loss": sum(losses) / len(losses),
            "psnr_rgb": sum(row["psnr_rgb"] for row in records) / len(records),
            "ssim_rgb": sum(row["ssim_rgb"] for row in records) / len(records),
            "per_image": records,
        }
        if self.fail_on_nonfinite and not all(
            math.isfinite(float(result[key]))
            for key in ("val_loss", "psnr_rgb", "ssim_rgb")
        ):
            raise FloatingPointError("Validation aggregate contains NaN or Inf.")
        return result

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()
