"""Train one resolved v2 variant and test its best validation checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from src.common.experiment.checkpoint import (
    BestTracker,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    tracker_from_checkpoint,
)
from src.common.experiment.experiment import (
    RunPaths,
    collect_run_info,
    create_experiment,
    finish_run_info,
    git_state,
    run_training_and_auto_test,
    utc_now,
)
from src.common.experiment.logging_utils import (
    atomic_write_csv,
    atomic_write_json,
    build_logger,
)
from src.v1.train_v1 import (
    _progress_log_due,
    _resolve_resume_path,
    _validation_due,
)
from src.v2.config import (
    VALID_VARIANTS,
    V2_CONFIG_PATH,
    load_v2_config,
    resolve_v2_config_path,
)


METRIC_FIELDS = (
    "sample_id",
    "input_relative_path",
    "gt_relative_path",
    "psnr_rgb",
    "ssim_rgb",
)
ORDER_COMPARISON_FIELDS = (
    "sample_id",
    "input_relative_path",
    "gt_relative_path",
    "psnr_color_then_scatter",
    "psnr_scatter_then_color",
    "delta_psnr_cs_minus_sc",
    "ssim_color_then_scatter",
    "ssim_scatter_then_color",
    "delta_ssim_cs_minus_sc",
    "winner_psnr",
    "winner_ssim",
)


def resolve_training_config(
    config_path: Path | str,
    *,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the final validated config saved by a training invocation."""

    return load_v2_config(
        config_path,
        entry_point="train_v2",
        variant=variant,
        seed=seed,
    )


def _normalized_resume_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized["training"]["resume"] = None
    normalized["test"]["run_dir"] = None
    normalized["test"]["allow_overwrite"] = False
    return normalized


def require_v2_resume_config_match(
    current_config: Mapping[str, Any],
    saved_config: Mapping[str, Any],
    *,
    context: str = "Resume checkpoint",
) -> None:
    if _normalized_resume_config(current_config) != _normalized_resume_config(
        saved_config
    ):
        raise ValueError(
            f"{context} config does not match v2 semantics after ignoring "
            "training.resume, test.run_dir, and test.allow_overwrite."
        )


def _copy_resume_artifacts(
    resume_path: Path,
    paths: RunPaths,
    *,
    through_epoch: int,
    save_metrics_history: bool,
) -> list[dict[str, Any]]:
    """Copy history and independent best files from an ordinary run layout."""

    if resume_path.parent.name not in {"checkpoint", "best"}:
        return []
    source_run = resume_path.parent.parent
    history: list[dict[str, Any]] = []
    history_path = source_run / "log/metrics_history.json"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"Resume metrics history is invalid: {history_path}")
        history = [
            row for row in value if int(row.get("epoch", -1)) <= through_epoch
        ]
        if not history or int(history[-1].get("epoch", -1)) != through_epoch:
            raise ValueError(
                "Resume history does not contain the checkpoint's completed epoch."
            )
        if save_metrics_history:
            atomic_write_json(paths.log / "metrics_history.json", history)
    source_best = source_run / "best"
    if source_best.is_dir():
        for stem in ("best_psnr", "best_ssim", "best_loss"):
            for suffix in (".pt", ".json"):
                source = source_best / f"{stem}{suffix}"
                if source.is_file():
                    shutil.copy2(source, paths.best / source.name)
    source_result = source_run / "result"
    if source_result.is_dir():
        for source in source_result.glob("best_psnr_validation_*"):
            if source.is_file():
                shutil.copy2(source, paths.result / source.name)
    return history


def _ensure_resumed_best_files(
    *,
    paths: RunPaths,
    checkpoint: Mapping[str, Any],
    torch_module: Any,
    shared: bool,
) -> None:
    specifications = (
        (
            "best_psnr",
            "validation_mean_path_psnr" if shared else "validation_psnr",
            "psnr",
            float(checkpoint["psnr"]),
            float(checkpoint["best_validation_psnr"]),
        ),
        (
            "best_ssim",
            "validation_mean_path_ssim" if shared else "validation_ssim",
            "ssim",
            float(checkpoint["ssim"]),
            float(checkpoint["best_validation_ssim"]),
        ),
        (
            "best_loss",
            "validation_joint_loss" if shared else "validation_loss",
            "val_loss",
            float(checkpoint["val_loss"]),
            float(checkpoint["best_validation_loss"]),
        ),
    )
    for stem, selection_metric, field, current, best in specifications:
        destination = paths.best / f"{stem}.pt"
        sidecar_path = destination.with_suffix(".json")
        if destination.is_file() and sidecar_path.is_file():
            with sidecar_path.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            if math.isclose(float(sidecar[field]), best, rel_tol=0.0, abs_tol=0.0):
                continue
            raise ValueError(f"Copied {stem} does not match historical best value.")
        if math.isclose(current, best, rel_tol=0.0, abs_tol=0.0):
            save_checkpoint(
                destination,
                payload=checkpoint,
                torch_module=torch_module,
                selection_metric=selection_metric,
            )
            continue
        raise ValueError(
            f"Resume cannot recover historical {stem}; keep the source best/ "
            "directory next to the resume checkpoint."
        )


def _save_epoch_checkpoints(
    *,
    paths: RunPaths,
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    updates: Mapping[str, bool],
    epoch: int,
    torch_module: Any,
    shared: bool,
) -> None:
    checkpoint = config["checkpoint"]
    metric_names = {
        "psnr": "validation_mean_path_psnr" if shared else "validation_psnr",
        "ssim": "validation_mean_path_ssim" if shared else "validation_ssim",
        "val_loss": "validation_joint_loss" if shared else "validation_loss",
    }
    if updates["psnr"] and checkpoint["save_best_psnr"]:
        save_checkpoint(
            paths.best / "best_psnr.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric=metric_names["psnr"],
        )
    if updates["ssim"] and checkpoint["save_best_ssim"]:
        save_checkpoint(
            paths.best / "best_ssim.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric=metric_names["ssim"],
        )
    if updates["val_loss"] and checkpoint["save_best_val_loss"]:
        save_checkpoint(
            paths.best / "best_loss.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric=metric_names["val_loss"],
        )
    if checkpoint["save_last"]:
        save_checkpoint(
            paths.checkpoint / "last.pt",
            payload=payload,
            torch_module=torch_module,
        )
    if checkpoint["save_periodic"] and (
        (epoch + 1) % int(config["training"]["save_every"]) == 0
    ):
        save_checkpoint(
            paths.checkpoint / f"epoch_{epoch + 1:04d}.pt",
            payload=payload,
            torch_module=torch_module,
        )


def _write_validation_outputs(
    *,
    paths: RunPaths,
    validation: Mapping[str, Any],
    tracker: BestTracker,
    epoch: int,
    global_step: int,
    shared: bool,
    filename_prefix: str = "",
    checkpoint_selection_source: Optional[str] = None,
) -> None:
    summary_name = f"{filename_prefix}validation_summary.json"
    if not shared:
        atomic_write_csv(
            paths.result / f"{filename_prefix}validation_metrics.csv",
            validation["per_image"],
            METRIC_FIELDS,
        )
        summary = {
            "epoch": epoch,
            "global_step": global_step,
            "num_samples": validation["num_samples"],
            "val_loss": validation["val_loss"],
            "mean_psnr_rgb": validation["psnr_rgb"],
            "mean_ssim_rgb": validation["ssim_rgb"],
            "best_validation_psnr": tracker.psnr,
            "best_validation_ssim": tracker.ssim,
            "best_validation_loss": tracker.val_loss,
        }
        if checkpoint_selection_source is not None:
            summary.update(
                {
                    "checkpoint": "best_psnr.pt",
                    "checkpoint_selection_source": checkpoint_selection_source,
                }
            )
        atomic_write_json(
            paths.result / summary_name,
            summary,
        )
        return
    atomic_write_csv(
        paths.result
        / f"{filename_prefix}validation_metrics_color_then_scatter.csv",
        validation["per_image_cs"],
        METRIC_FIELDS,
    )
    atomic_write_csv(
        paths.result
        / f"{filename_prefix}validation_metrics_scatter_then_color.csv",
        validation["per_image_sc"],
        METRIC_FIELDS,
    )
    atomic_write_csv(
        paths.result / f"{filename_prefix}validation_order_comparison.csv",
        validation["comparison"],
        ORDER_COMPARISON_FIELDS,
    )
    summary = {
        "epoch": epoch,
        "global_step": global_step,
        "num_samples": validation["num_samples"],
        "mean_psnr_color_then_scatter": validation["psnr_cs"],
        "mean_psnr_scatter_then_color": validation["psnr_sc"],
        "mean_ssim_color_then_scatter": validation["ssim_cs"],
        "mean_ssim_scatter_then_color": validation["ssim_sc"],
        "mean_delta_psnr_cs_minus_sc": validation[
            "mean_delta_psnr_cs_minus_sc"
        ],
        "mean_delta_ssim_cs_minus_sc": validation[
            "mean_delta_ssim_cs_minus_sc"
        ],
        "mean_path_psnr": validation["mean_path_psnr"],
        "mean_path_ssim": validation["mean_path_ssim"],
        "joint_val_loss": validation["joint_val_loss"],
        "val_loss_color_then_scatter": validation["val_loss_cs"],
        "val_loss_scatter_then_color": validation["val_loss_sc"],
        "color_then_scatter_psnr_win_count": validation[
            "color_then_scatter_psnr_win_count"
        ],
        "scatter_then_color_psnr_win_count": validation[
            "scatter_then_color_psnr_win_count"
        ],
        "psnr_tie_count": validation["psnr_tie_count"],
        "winner_tolerance": validation["winner_tolerance"],
        "best_validation_mean_path_psnr": tracker.psnr,
        "best_validation_mean_path_ssim": tracker.ssim,
        "best_validation_joint_loss": tracker.val_loss,
    }
    if checkpoint_selection_source is not None:
        summary.update(
            {
                "checkpoint": "best_psnr.pt",
                "checkpoint_selection_source": checkpoint_selection_source,
            }
        )
    atomic_write_json(
        paths.result / summary_name,
        summary,
    )


def _write_best_psnr_validation_snapshot(
    *,
    paths: RunPaths,
    validation: Mapping[str, Any],
    tracker: BestTracker,
    epoch: int,
    global_step: int,
    shared: bool,
    updates: Mapping[str, bool],
    save_best_psnr: bool,
) -> bool:
    """Atomically snapshot the validation object that selected best_psnr.pt."""

    if not updates["psnr"] or not save_best_psnr:
        return False
    selected_psnr = (
        float(validation["mean_path_psnr"])
        if shared
        else float(validation["psnr_rgb"])
    )
    if not math.isclose(
        selected_psnr, float(tracker.psnr), rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError(
            "Best-PSNR validation snapshot does not match the tracker value."
        )
    _write_validation_outputs(
        paths=paths,
        validation=validation,
        tracker=tracker,
        epoch=epoch,
        global_step=global_step,
        shared=shared,
        filename_prefix="best_psnr_",
        checkpoint_selection_source=(
            "validation_mean_path_psnr" if shared else "validation_psnr"
        ),
    )
    return True


def _extend_shared_checkpoint(
    payload: Dict[str, Any],
    *,
    train_result: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    payload.update(
        {
            "train_loss_joint": float(train_result["train_loss_joint"]),
            "train_loss_cs": float(train_result["train_loss_cs"]),
            "train_loss_sc": float(train_result["train_loss_sc"]),
            "validation_psnr_cs": float(validation["psnr_cs"]),
            "validation_psnr_sc": float(validation["psnr_sc"]),
            "validation_ssim_cs": float(validation["ssim_cs"]),
            "validation_ssim_sc": float(validation["ssim_sc"]),
            "validation_loss_cs": float(validation["val_loss_cs"]),
            "validation_loss_sc": float(validation["val_loss_sc"]),
            "validation_mean_path_psnr": float(validation["mean_path_psnr"]),
            "validation_mean_path_ssim": float(validation["mean_path_ssim"]),
        }
    )


def run(
    config_path: Path = V2_CONFIG_PATH,
    *,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> Path:
    resolved_config_path = resolve_v2_config_path(config_path)
    config = resolve_training_config(
        resolved_config_path, variant=variant, seed=seed
    )
    paths = create_experiment(config)
    start_monotonic = time.monotonic()
    run_info: MutableMapping[str, Any] = {
        "experiment_name": config["experiment"]["name"],
        "version": "v2",
        "variant": config["experiment"]["variant"],
        "seed": config["experiment"]["seed"],
        "source_config_path": str(resolved_config_path),
        "run_directory": str(paths.root),
        "start_time": utc_now(),
        "end_time": None,
    }
    atomic_write_json(paths.root / "run_info.json", run_info)
    train_logger = build_logger(
        f"uie3_v2_train_{paths.root.name}",
        paths.log / "train.log",
        console=bool(config["logging"]["console"]),
        file_enabled=bool(config["logging"]["save_train_log"]),
    )
    val_logger = build_logger(
        f"uie3_v2_val_{paths.root.name}",
        paths.log / "val.log",
        console=bool(config["logging"]["console"]),
        file_enabled=bool(config["logging"]["save_validation_log"]),
    )
    progress: MutableMapping[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "validation": "PENDING",
        "stage": "training",
    }

    def train_all_epochs() -> Dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch must be installed for v2 training.") from exc
        from src.common.data.dataloader import build_dataloaders
        from src.common.experiment.seed import set_global_seed
        from src.v1.loss import CharbonnierLoss
        from src.v1.trainer import V1Trainer
        from src.v2.model import build_v2_model
        from src.v2.trainer import SharedOrderTrainer

        experiment = config["experiment"]
        shared = experiment["variant"] == "shared_order_diagnostic"
        seed_value = int(experiment["seed"])
        set_global_seed(
            seed_value,
            deterministic=bool(config["training"]["deterministic"]),
        )
        model = build_v2_model(
            variant=experiment["variant"], model_config=config["model"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
        )
        criterion = CharbonnierLoss(
            epsilon=float(config["loss"]["epsilon"]), reduction="mean"
        )
        common_trainer = {
            "model": model,
            "optimizer": optimizer,
            "loss_function": criterion,
            "device": device,
            "amp": bool(config["training"]["amp"]),
            "fail_on_nonfinite": bool(config["training"]["fail_on_nonfinite"]),
            "gradient_clip_norm": config["training"]["gradient_clip_norm"],
            "metrics_config": config["metrics"],
            "scheduler": None,
        }
        if shared:
            trainer = SharedOrderTrainer(
                **common_trainer,
                color_then_scatter_loss_weight=float(
                    config["order_study"]["color_then_scatter_loss_weight"]
                ),
                scatter_then_color_loss_weight=float(
                    config["order_study"]["scatter_then_color_loss_weight"]
                ),
            )
        else:
            trainer = V1Trainer(**common_trainer)
        loaders = build_dataloaders(config)
        detailed_info = collect_run_info(
            config,
            paths,
            source_config_path=resolved_config_path,
            model=model,
            torch_module=torch,
            command=sys.argv,
        )
        detailed_info["variant"] = experiment["variant"]
        detailed_info["start_time"] = run_info["start_time"]
        run_info.update(detailed_info)
        atomic_write_json(paths.root / "run_info.json", run_info)
        train_logger.info("Run directory: %s", paths.root)
        train_logger.info("Variant: %s", experiment["variant"])
        train_logger.info("Device: %s", device)
        train_logger.info("Parameters: %s", run_info["parameter_count"])

        start_epoch = 0
        tracker = BestTracker()
        history: list[dict[str, Any]] = []
        resume_path = _resolve_resume_path(config["training"]["resume"])
        if resume_path is not None:
            checkpoint = load_checkpoint(
                resume_path,
                model=model,
                torch_module=torch,
                optimizer=optimizer,
                scheduler=None,
                scaler=trainer.scaler,
                map_location="cpu",
                restore_training_state=True,
                restore_rng=True,
            )
            require_v2_resume_config_match(config, checkpoint["config"])
            start_epoch = int(checkpoint["epoch"]) + 1
            trainer.global_step = int(checkpoint["global_step"])
            tracker = tracker_from_checkpoint(checkpoint)
            history = _copy_resume_artifacts(
                resume_path,
                paths,
                through_epoch=start_epoch - 1,
                save_metrics_history=bool(
                    config["logging"]["save_metrics_history_json"]
                ),
            )
            _ensure_resumed_best_files(
                paths=paths,
                checkpoint=checkpoint,
                torch_module=torch,
                shared=shared,
            )
            train_logger.info(
                "Resumed %s at epoch=%d global_step=%d",
                resume_path,
                start_epoch + 1,
                trainer.global_step,
            )

        total_epochs = int(config["training"]["epochs"])
        validate_every = int(config["training"]["validate_every"])
        log_every_steps = int(config["logging"]["log_every_steps"])
        if start_epoch >= total_epochs:
            raise ValueError("Resume checkpoint already reaches configured epochs.")
        git_commit = git_state()["commit"]
        for epoch in range(start_epoch, total_epochs):
            epoch_started = time.monotonic()
            progress.update(
                {
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                    "stage": "training",
                }
            )
            if getattr(loaders["train"], "generator", None) is not None:
                loaders["train"].generator.manual_seed(seed_value + epoch)

            def train_progress(
                batch_index: int,
                total_batches: Optional[int],
                result: Mapping[str, Any],
            ) -> None:
                if not _progress_log_due(
                    batch_index, total_batches, log_every_steps
                ):
                    return
                if shared:
                    train_logger.info(
                        "epoch=%d/%d train_batch=%d/%s joint=%.8f cs=%.8f sc=%.8f",
                        epoch + 1,
                        total_epochs,
                        batch_index,
                        total_batches or "?",
                        result["loss_joint"],
                        result["loss_cs"],
                        result["loss_sc"],
                    )
                else:
                    train_logger.info(
                        "epoch=%d/%d train_batch=%d/%s loss=%.8f",
                        epoch + 1,
                        total_epochs,
                        batch_index,
                        total_batches or "?",
                        result["loss"],
                    )

            train_result = trainer.train_epoch(
                loaders["train"],
                progress_callback=train_progress if log_every_steps > 0 else None,
            )
            progress["global_step"] = trainer.global_step
            learning_rate = float(optimizer.param_groups[0]["lr"])
            if shared:
                train_logger.info(
                    "epoch=%d/%d train_loss_joint=%.8f train_loss_cs=%.8f "
                    "train_loss_sc=%.8f global_step=%d lr=%.8g",
                    epoch + 1,
                    total_epochs,
                    train_result["train_loss_joint"],
                    train_result["train_loss_cs"],
                    train_result["train_loss_sc"],
                    trainer.global_step,
                    learning_rate,
                )
            else:
                train_logger.info(
                    "epoch=%d/%d train_loss=%.8f global_step=%d lr=%.8g",
                    epoch + 1,
                    total_epochs,
                    train_result["train_loss"],
                    trainer.global_step,
                    learning_rate,
                )

            if not _validation_due(epoch, total_epochs, validate_every):
                history.append(
                    {
                        "epoch": epoch,
                        "global_step": trainer.global_step,
                        "train_loss": train_result["train_loss"],
                        "val_loss": None,
                        "psnr": None,
                        "ssim": None,
                        "learning_rate": learning_rate,
                        "epoch_time_seconds": time.monotonic() - epoch_started,
                    }
                )
                if config["logging"]["save_metrics_history_json"]:
                    atomic_write_json(paths.log / "metrics_history.json", history)
                trainer.step_scheduler()
                continue

            progress.update({"validation": "RUNNING", "stage": "validation"})
            validation = trainer.validate(loaders["validation"])
            progress["validation"] = "COMPLETED"
            updates = tracker.consider(
                psnr=validation["psnr_rgb"],
                ssim=validation["ssim_rgb"],
                val_loss=validation["val_loss"],
                source="validation",
            )
            epoch_record = {
                "epoch": epoch,
                "global_step": trainer.global_step,
                "train_loss": train_result["train_loss"],
                "val_loss": validation["val_loss"],
                "psnr": validation["psnr_rgb"],
                "ssim": validation["ssim_rgb"],
                "learning_rate": learning_rate,
                "epoch_time_seconds": time.monotonic() - epoch_started,
            }
            if shared:
                epoch_record.update(
                    {
                        "train_loss_cs": train_result["train_loss_cs"],
                        "train_loss_sc": train_result["train_loss_sc"],
                        "validation_psnr_cs": validation["psnr_cs"],
                        "validation_psnr_sc": validation["psnr_sc"],
                        "validation_ssim_cs": validation["ssim_cs"],
                        "validation_ssim_sc": validation["ssim_sc"],
                    }
                )
            history.append(epoch_record)
            if config["logging"]["save_metrics_history_json"]:
                atomic_write_json(paths.log / "metrics_history.json", history)
            _write_validation_outputs(
                paths=paths,
                validation=validation,
                tracker=tracker,
                epoch=epoch,
                global_step=trainer.global_step,
                shared=shared,
            )
            if shared:
                val_logger.info(
                    "epoch=%d/%d joint_val_loss=%.8f mean_path_psnr=%.8f "
                    "mean_path_ssim=%.8f psnr_cs=%.8f psnr_sc=%.8f",
                    epoch + 1,
                    total_epochs,
                    validation["joint_val_loss"],
                    validation["mean_path_psnr"],
                    validation["mean_path_ssim"],
                    validation["psnr_cs"],
                    validation["psnr_sc"],
                )
            else:
                val_logger.info(
                    "epoch=%d/%d val_loss=%.8f psnr=%.8f ssim=%.8f",
                    epoch + 1,
                    total_epochs,
                    validation["val_loss"],
                    validation["psnr_rgb"],
                    validation["ssim_rgb"],
                )
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
                global_step=trainer.global_step,
                config=config,
                seed=seed_value,
                train_loss=train_result["train_loss"],
                val_loss=validation["val_loss"],
                psnr=validation["psnr_rgb"],
                ssim=validation["ssim_rgb"],
                learning_rate=learning_rate,
                best_tracker=tracker,
                torch_module=torch,
                git_commit=git_commit,
            )
            if shared:
                _extend_shared_checkpoint(
                    payload,
                    train_result=train_result,
                    validation=validation,
                )
            _write_best_psnr_validation_snapshot(
                paths=paths,
                validation=validation,
                tracker=tracker,
                epoch=epoch,
                global_step=trainer.global_step,
                shared=shared,
                updates=updates,
                save_best_psnr=bool(
                    config["checkpoint"]["save_best_psnr"]
                ),
            )
            _save_epoch_checkpoints(
                paths=paths,
                config=config,
                payload=payload,
                updates=updates,
                epoch=epoch,
                torch_module=torch,
                shared=shared,
            )
            trainer.step_scheduler()
            atomic_write_json(
                paths.root / "status.json",
                {
                    "overall": "RUNNING",
                    "training": "RUNNING",
                    "validation": "COMPLETED",
                    "test": "PENDING",
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                },
            )
        return {
            "last_epoch": progress["epoch"],
            "global_step": trainer.global_step,
            "best_validation_psnr": tracker.psnr,
        }

    def auto_test(checkpoint_key: str) -> Dict[str, Any]:
        from src.v2.test_v2 import execute_test

        test_logger = build_logger(
            f"uie3_v2_auto_test_{paths.root.name}",
            paths.log / "test.log",
            console=bool(config["logging"]["console"]),
            file_enabled=bool(config["logging"]["save_test_log"]),
        )
        return execute_test(
            config,
            run_dir=paths.root,
            checkpoint_key=checkpoint_key,
            allow_overwrite=False,
            logger=test_logger,
        )

    try:
        run_training_and_auto_test(
            training_callable=train_all_epochs,
            test_callable=auto_test,
            status_path=paths.root / "status.json",
            error_path=paths.root / "error.json",
            progress=progress,
            auto_test=bool(config["test"]["auto_run_after_training"]),
        )
    except Exception:
        train_logger.exception("v2 run failed")
        raise
    finally:
        finish_run_info(run_info, start_monotonic=start_monotonic)
        atomic_write_json(paths.root / "run_info.json", run_info)
    return paths.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V2_CONFIG_PATH)
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.config, variant=args.variant, seed=args.seed)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
