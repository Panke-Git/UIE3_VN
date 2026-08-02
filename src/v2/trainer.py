"""Memory-conscious shared-weight trainer for v2 order diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from src.common.metrics.image_metrics import (
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)
from src.v2.order_comparison import WINNER_TOLERANCE, winner
from src.v1.trainer import (
    ProgressCallback,
    _autocast_context,
    _loader_length,
    _make_grad_scaler,
    _string_batch,
    gradients_are_finite,
    require_finite_tensor,
)


class SharedOrderTrainer:
    """Train two orders through one shared C, one shared S, and one backbone."""

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
        color_then_scatter_loss_weight: float = 0.5,
        scatter_then_color_loss_weight: float = 0.5,
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
        self.color_then_scatter_loss_weight = float(
            color_then_scatter_loss_weight
        )
        self.scatter_then_color_loss_weight = float(
            scatter_then_color_loss_weight
        )
        if (
            self.color_then_scatter_loss_weight != 0.5
            or self.scatter_then_color_loss_weight != 0.5
        ):
            raise ValueError("SharedOrderTrainer requires two loss weights of 0.5.")
        self.global_step = 0

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Run two fresh graphs, accumulate gradients, then take one step."""

        self.model.train()
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        require_finite_tensor("shared training input", inputs)
        require_finite_tensor("shared training target", targets)
        self.optimizer.zero_grad(set_to_none=True)

        with _autocast_context(self.amp_enabled):
            prediction_cs = self.model.forward_color_then_scatter(inputs)
            if prediction_cs.shape != targets.shape:
                raise ValueError("Shared CS prediction/target shape mismatch.")
            require_finite_tensor("shared CS training prediction", prediction_cs)
            loss_cs = self.loss_function(prediction_cs, targets)
            weighted_cs = self.color_then_scatter_loss_weight * loss_cs
        if loss_cs.ndim != 0 or not torch.isfinite(loss_cs):
            raise FloatingPointError("Shared CS loss must be a finite scalar.")
        self.scaler.scale(weighted_cs).backward()
        del prediction_cs, weighted_cs

        with _autocast_context(self.amp_enabled):
            prediction_sc = self.model.forward_scatter_then_color(inputs)
            if prediction_sc.shape != targets.shape:
                raise ValueError("Shared SC prediction/target shape mismatch.")
            require_finite_tensor("shared SC training prediction", prediction_sc)
            loss_sc = self.loss_function(prediction_sc, targets)
            weighted_sc = self.scatter_then_color_loss_weight * loss_sc
        if loss_sc.ndim != 0 or not torch.isfinite(loss_sc):
            raise FloatingPointError("Shared SC loss must be a finite scalar.")
        self.scaler.scale(weighted_sc).backward()

        self.scaler.unscale_(self.optimizer)
        gradients_finite = gradients_are_finite(self.model)
        if not gradients_finite and not self.amp_enabled:
            raise FloatingPointError("A shared training gradient contains NaN or Inf.")
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

        loss_cs_value = float(loss_cs.detach().cpu())
        loss_sc_value = float(loss_sc.detach().cpu())
        joint_value = (
            self.color_then_scatter_loss_weight * loss_cs_value
            + self.scatter_then_color_loss_weight * loss_sc_value
        )
        if not all(math.isfinite(value) for value in (loss_cs_value, loss_sc_value, joint_value)):
            raise FloatingPointError("Shared training losses are non-finite.")
        return {
            "loss": joint_value,
            "loss_joint": joint_value,
            "loss_cs": loss_cs_value,
            "loss_sc": loss_sc_value,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": self.global_step,
            "optimizer_step_applied": optimizer_step_applied,
            "amp_overflow_detected": amp_overflow,
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": amp_scale_after,
        }

    def train_epoch(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        joint_losses: List[float] = []
        cs_losses: List[float] = []
        sc_losses: List[float] = []
        applied_steps = 0
        skipped_steps = 0
        total_batches = _loader_length(data_loader)
        for batch_index, batch in enumerate(data_loader, start=1):
            result = self.train_step(batch)
            joint_losses.append(result["loss_joint"])
            cs_losses.append(result["loss_cs"])
            sc_losses.append(result["loss_sc"])
            applied_steps += int(result["optimizer_step_applied"])
            skipped_steps += int(not result["optimizer_step_applied"])
            if progress_callback is not None:
                progress_callback(batch_index, total_batches, result)
        if not joint_losses:
            raise ValueError("Shared training DataLoader produced no batches.")
        result = {
            "train_loss": sum(joint_losses) / len(joint_losses),
            "train_loss_joint": sum(joint_losses) / len(joint_losses),
            "train_loss_cs": sum(cs_losses) / len(cs_losses),
            "train_loss_sc": sum(sc_losses) / len(sc_losses),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "amp_scale": float(self.scaler.get_scale()),
            "optimizer_applied_steps": applied_steps,
            "skipped_amp_overflow_steps": skipped_steps,
        }
        if self.fail_on_nonfinite and not all(
            math.isfinite(float(result[key]))
            for key in ("train_loss_joint", "train_loss_cs", "train_loss_sc")
        ):
            raise FloatingPointError("Shared epoch loss is non-finite.")
        return result

    def _path_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        psnr = rgb_psnr_per_image(
            prediction,
            target,
            data_range=float(self.metrics_config["data_range"]),
            crop_border=int(self.metrics_config["crop_border"]),
        )
        ssim = rgb_ssim_per_image(
            prediction,
            target,
            data_range=float(self.metrics_config["data_range"]),
            crop_border=int(self.metrics_config["crop_border"]),
            window_size=int(self.metrics_config["ssim_window_size"]),
            sigma=float(self.metrics_config["ssim_sigma"]),
        )
        return psnr, ssim

    @torch.no_grad()
    def validate(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        records_cs: List[Dict[str, Any]] = []
        records_sc: List[Dict[str, Any]] = []
        comparisons: List[Dict[str, Any]] = []
        losses_cs: List[float] = []
        losses_sc: List[float] = []
        try:
            total_batches = _loader_length(data_loader)
            for batch_index, batch in enumerate(data_loader, start=1):
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                require_finite_tensor("shared validation input", inputs)
                require_finite_tensor("shared validation target", targets)

                prediction_cs = self.model.forward_color_then_scatter(inputs)
                require_finite_tensor("shared CS validation prediction", prediction_cs)
                loss_cs = self.loss_function(prediction_cs, targets)
                psnr_cs, ssim_cs = self._path_metrics(prediction_cs, targets)
                del prediction_cs

                prediction_sc = self.model.forward_scatter_then_color(inputs)
                require_finite_tensor("shared SC validation prediction", prediction_sc)
                loss_sc = self.loss_function(prediction_sc, targets)
                psnr_sc, ssim_sc = self._path_metrics(prediction_sc, targets)
                del prediction_sc

                if not torch.isfinite(loss_cs) or not torch.isfinite(loss_sc):
                    raise FloatingPointError("Shared validation loss is non-finite.")
                batch_size = targets.shape[0]
                sample_ids = _string_batch(batch["sample_id"], batch_size, "sample_id")
                input_paths = _string_batch(
                    batch["input_relative_path"], batch_size, "input_relative_path"
                )
                gt_paths = _string_batch(
                    batch["gt_relative_path"], batch_size, "gt_relative_path"
                )
                loss_cs_value = float(loss_cs.detach().cpu())
                loss_sc_value = float(loss_sc.detach().cpu())
                losses_cs.extend([loss_cs_value] * batch_size)
                losses_sc.extend([loss_sc_value] * batch_size)
                for index in range(batch_size):
                    cs_psnr = float(psnr_cs[index].detach().cpu())
                    sc_psnr = float(psnr_sc[index].detach().cpu())
                    cs_ssim = float(ssim_cs[index].detach().cpu())
                    sc_ssim = float(ssim_sc[index].detach().cpu())
                    common = {
                        "sample_id": sample_ids[index],
                        "input_relative_path": input_paths[index],
                        "gt_relative_path": gt_paths[index],
                    }
                    records_cs.append(
                        {**common, "psnr_rgb": cs_psnr, "ssim_rgb": cs_ssim}
                    )
                    records_sc.append(
                        {**common, "psnr_rgb": sc_psnr, "ssim_rgb": sc_ssim}
                    )
                    delta_psnr = cs_psnr - sc_psnr
                    delta_ssim = cs_ssim - sc_ssim
                    comparisons.append(
                        {
                            **common,
                            "psnr_color_then_scatter": cs_psnr,
                            "psnr_scatter_then_color": sc_psnr,
                            "delta_psnr_cs_minus_sc": delta_psnr,
                            "ssim_color_then_scatter": cs_ssim,
                            "ssim_scatter_then_color": sc_ssim,
                            "delta_ssim_cs_minus_sc": delta_ssim,
                            "winner_psnr": winner(delta_psnr),
                            "winner_ssim": winner(delta_ssim),
                        }
                    )
                if progress_callback is not None:
                    progress_callback(
                        batch_index,
                        total_batches,
                        {
                            "loss_cs": loss_cs_value,
                            "loss_sc": loss_sc_value,
                            "processed_samples": len(comparisons),
                            "psnr_cs": float(psnr_cs.mean().detach().cpu()),
                            "psnr_sc": float(psnr_sc.mean().detach().cpu()),
                        },
                    )
        finally:
            self.model.train(was_training)
        if not comparisons:
            raise ValueError("Shared validation DataLoader produced no samples.")

        count = len(comparisons)
        val_loss_cs = sum(losses_cs) / len(losses_cs)
        val_loss_sc = sum(losses_sc) / len(losses_sc)
        psnr_cs_mean = sum(row["psnr_rgb"] for row in records_cs) / count
        psnr_sc_mean = sum(row["psnr_rgb"] for row in records_sc) / count
        ssim_cs_mean = sum(row["ssim_rgb"] for row in records_cs) / count
        ssim_sc_mean = sum(row["ssim_rgb"] for row in records_sc) / count
        joint_val_loss = 0.5 * val_loss_cs + 0.5 * val_loss_sc
        mean_path_psnr = 0.5 * psnr_cs_mean + 0.5 * psnr_sc_mean
        mean_path_ssim = 0.5 * ssim_cs_mean + 0.5 * ssim_sc_mean
        result: Dict[str, Any] = {
            "num_samples": count,
            "val_loss": joint_val_loss,
            "val_loss_cs": val_loss_cs,
            "val_loss_sc": val_loss_sc,
            "joint_val_loss": joint_val_loss,
            "psnr_rgb": mean_path_psnr,
            "ssim_rgb": mean_path_ssim,
            "psnr_cs": psnr_cs_mean,
            "psnr_sc": psnr_sc_mean,
            "ssim_cs": ssim_cs_mean,
            "ssim_sc": ssim_sc_mean,
            "mean_path_psnr": mean_path_psnr,
            "mean_path_ssim": mean_path_ssim,
            "mean_delta_psnr_cs_minus_sc": sum(
                row["delta_psnr_cs_minus_sc"] for row in comparisons
            )
            / count,
            "mean_delta_ssim_cs_minus_sc": sum(
                row["delta_ssim_cs_minus_sc"] for row in comparisons
            )
            / count,
            "color_then_scatter_psnr_win_count": sum(
                row["winner_psnr"] == "color_then_scatter" for row in comparisons
            ),
            "scatter_then_color_psnr_win_count": sum(
                row["winner_psnr"] == "scatter_then_color" for row in comparisons
            ),
            "psnr_tie_count": sum(row["winner_psnr"] == "tie" for row in comparisons),
            "winner_tolerance": WINNER_TOLERANCE,
            "per_image_cs": records_cs,
            "per_image_sc": records_sc,
            "comparison": comparisons,
        }
        if self.fail_on_nonfinite and not all(
            math.isfinite(float(result[key]))
            for key in (
                "val_loss_cs",
                "val_loss_sc",
                "joint_val_loss",
                "psnr_cs",
                "psnr_sc",
                "ssim_cs",
                "ssim_sc",
                "mean_path_psnr",
                "mean_path_ssim",
            )
        ):
            raise FloatingPointError("Shared validation aggregate contains NaN or Inf.")
        return result

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()
