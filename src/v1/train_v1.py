"""Train v1, validate every epoch, then test best validation PSNR exactly once."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import platform
import shlex
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
from src.common.experiment.config import (
    PROJECT_ROOT,
    V1_CONFIG_PATH,
    load_v1_config,
    resolve_v1_config_path,
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


VALIDATION_FIELDS = (
    "sample_id",
    "input_relative_path",
    "gt_relative_path",
    "psnr_rgb",
    "ssim_rgb",
)


def _validation_due(epoch: int, total_epochs: int, validate_every: int) -> bool:
    """Validate on the configured cadence and always on the final epoch."""

    return (epoch + 1) % validate_every == 0 or epoch == total_epochs - 1


def _initial_run_info(
    config: Mapping[str, Any], paths: RunPaths, config_path: Path
) -> Dict[str, Any]:
    git = git_state()
    return {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"]["version"],
        "seed": config["experiment"]["seed"],
        "source_config_path": str(config_path),
        "run_directory": str(paths.root),
        "git_commit": git["commit"] if git["commit"] is not None else "uncommitted",
        "git_dirty": git["dirty"],
        "python_version": platform.python_version(),
        "pytorch_version": None,
        "cuda_runtime": None,
        "cuda_available": None,
        "cudnn": None,
        "gpu_name": None,
        "parameter_count": None,
        "trainable_parameter_count": None,
        "start_time": utc_now(),
        "end_time": None,
        "elapsed_time_seconds": None,
        "command": shlex.join(sys.argv),
        "dataset_root": config["data"]["root"],
        "manifest_paths": {
            "train": config["data"]["train_manifest"],
            "validation": config["data"]["validation_manifest"],
            "test": config["data"]["test_manifest"],
        },
    }


def _resolve_resume_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve(strict=False)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"training.resume does not exist: {resolved}")
    return resolved


def _normalized_resume_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized["training"]["resume"] = None
    normalized["test"]["run_dir"] = None
    normalized["test"]["allow_overwrite"] = False
    return normalized


def _require_resume_config_match(
    current_config: Mapping[str, Any], saved_config: Mapping[str, Any]
) -> None:
    if _normalized_resume_config(current_config) != _normalized_resume_config(
        saved_config
    ):
        raise ValueError(
            "Resume checkpoint config does not match v1 semantics after ignoring "
            "training.resume, test.run_dir, and test.allow_overwrite."
        )


def _copy_resume_history(
    resume_path: Path,
    paths: RunPaths,
    *,
    through_epoch: int,
    save_metrics_history: bool,
) -> list[dict[str, Any]]:
    """Carry forward history and independent best files into the unique new run."""

    if resume_path.parent.name not in {"checkpoint", "best"}:
        return []
    source_run = resume_path.parent.parent
    experiments_root = (PROJECT_ROOT / "experiments").resolve(strict=False)
    try:
        source_run.resolve(strict=False).relative_to(experiments_root)
    except ValueError:
        return []
    history_path = source_run / "log/metrics_history.json"
    history: list[dict[str, Any]] = []
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError(f"Resume metrics history is invalid: {history_path}")
        history = [
            item for item in value if int(item.get("epoch", -1)) <= through_epoch
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
            source_checkpoint = source_best / f"{stem}.pt"
            source_sidecar = source_best / f"{stem}.json"
            if not source_checkpoint.is_file() or not source_sidecar.is_file():
                continue
            with source_sidecar.open("r", encoding="utf-8") as handle:
                selection = json.load(handle)
            if int(selection.get("epoch", -1)) > through_epoch:
                continue
            shutil.copy2(source_checkpoint, paths.best / source_checkpoint.name)
            shutil.copy2(source_sidecar, paths.best / source_sidecar.name)
    return history


def _ensure_resumed_best_files(
    *,
    paths: RunPaths,
    checkpoint: Mapping[str, Any],
    torch_module: Any,
) -> None:
    """Require or reconstruct each historical best without inventing weights."""

    specifications = (
        (
            "best_psnr",
            "validation_psnr",
            "psnr",
            float(checkpoint["psnr"]),
            float(checkpoint["best_validation_psnr"]),
        ),
        (
            "best_ssim",
            "validation_ssim",
            "ssim",
            float(checkpoint["ssim"]),
            float(checkpoint["best_validation_ssim"]),
        ),
        (
            "best_loss",
            "validation_loss",
            "val_loss",
            float(checkpoint["val_loss"]),
            float(checkpoint["best_validation_loss"]),
        ),
    )
    for (
        stem,
        selection_metric,
        sidecar_field,
        current_value,
        best_value,
    ) in specifications:
        destination = paths.best / f"{stem}.pt"
        destination_sidecar = destination.with_suffix(".json")
        if destination.is_file() and destination_sidecar.is_file():
            with destination_sidecar.open("r", encoding="utf-8") as handle:
                selection = json.load(handle)
            if math.isclose(
                float(selection[sidecar_field]),
                best_value,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                continue
            raise ValueError(
                f"Copied {stem} sidecar does not match the resume checkpoint's "
                "historical best value."
            )
        if math.isclose(current_value, best_value, rel_tol=0.0, abs_tol=0.0):
            save_checkpoint(
                destination,
                payload=checkpoint,
                torch_module=torch_module,
                selection_metric=selection_metric,
            )
            continue
        raise ValueError(
            f"Resume cannot recover historical {stem} weights from {destination}; "
            "resume from the source run's latest last.pt with its best directory "
            "intact."
        )


def _resolve_device(torch_module: Any) -> Any:
    return torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def _save_epoch_checkpoints(
    *,
    paths: RunPaths,
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    updates: Mapping[str, bool],
    epoch: int,
    torch_module: Any,
) -> None:
    checkpoint_config = config["checkpoint"]
    if updates["psnr"] and checkpoint_config["save_best_psnr"]:
        save_checkpoint(
            paths.best / "best_psnr.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric="validation_psnr",
        )
    if updates["ssim"] and checkpoint_config["save_best_ssim"]:
        save_checkpoint(
            paths.best / "best_ssim.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric="validation_ssim",
        )
    if updates["val_loss"] and checkpoint_config["save_best_val_loss"]:
        save_checkpoint(
            paths.best / "best_loss.pt",
            payload=payload,
            torch_module=torch_module,
            selection_metric="validation_loss",
        )
    if checkpoint_config["save_last"]:
        save_checkpoint(
            paths.checkpoint / "last.pt",
            payload=payload,
            torch_module=torch_module,
        )
    save_every = int(config["training"]["save_every"])
    if (
        checkpoint_config["save_periodic"]
        and (epoch + 1) % save_every == 0
    ):
        save_checkpoint(
            paths.checkpoint / f"epoch_{epoch + 1:04d}.pt",
            payload=payload,
            torch_module=torch_module,
        )


def run(config_path: Path = V1_CONFIG_PATH) -> Path:
    resolved_config_path = resolve_v1_config_path(config_path)
    config = load_v1_config(resolved_config_path, entry_point="train_v1")
    paths = create_experiment(config)
    start_monotonic = time.monotonic()
    run_info = _initial_run_info(config, paths, resolved_config_path)
    atomic_write_json(paths.root / "run_info.json", run_info)
    train_logger = build_logger(
        f"uie3_v1_train_{paths.root.name}",
        paths.log / "train.log",
        console=bool(config["logging"]["console"]),
        file_enabled=bool(config["logging"]["save_train_log"]),
    )
    val_logger = build_logger(
        f"uie3_v1_val_{paths.root.name}",
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
            raise RuntimeError(
                "PyTorch must already be installed for v1 training."
            ) from exc
        from src.common.data.dataloader import build_dataloaders
        from src.common.experiment.seed import set_global_seed
        from src.v1.loss import CharbonnierLoss
        from src.v1.model import build_nafnet_small
        from src.v1.trainer import V1Trainer

        seed = int(config["experiment"]["seed"])
        set_global_seed(
            seed, deterministic=bool(config["training"]["deterministic"])
        )
        model = build_nafnet_small(**config["model"])
        device = _resolve_device(torch)
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
        )
        loss_function = CharbonnierLoss(
            epsilon=float(config["loss"]["epsilon"]), reduction="mean"
        )
        trainer = V1Trainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            amp=bool(config["training"]["amp"]),
            fail_on_nonfinite=bool(config["training"]["fail_on_nonfinite"]),
            gradient_clip_norm=config["training"]["gradient_clip_norm"],
            metrics_config=config["metrics"],
            scheduler=None,
        )
        loaders = build_dataloaders(config)
        detailed_info = collect_run_info(
            config,
            paths,
            source_config_path=resolved_config_path,
            model=model,
            torch_module=torch,
            command=sys.argv,
        )
        detailed_info["start_time"] = run_info["start_time"]
        run_info.update(detailed_info)
        atomic_write_json(paths.root / "run_info.json", run_info)
        train_logger.info("Run directory: %s", paths.root)
        train_logger.info("Device: %s", device)
        train_logger.info("Parameters: %s", run_info["parameter_count"])

        resume_path = _resolve_resume_path(config["training"]["resume"])
        start_epoch = 0
        tracker = BestTracker()
        history: list[dict[str, Any]] = []
        git_commit = git_state()["commit"]
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
            _require_resume_config_match(config, checkpoint["config"])
            start_epoch = int(checkpoint["epoch"]) + 1
            trainer.global_step = int(checkpoint["global_step"])
            tracker = tracker_from_checkpoint(checkpoint)
            history = _copy_resume_history(
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
            )
            train_logger.info(
                "Resumed checkpoint %s at next epoch %d and global_step %d",
                resume_path,
                start_epoch,
                trainer.global_step,
            )
        total_epochs = int(config["training"]["epochs"])
        validate_every = int(config["training"]["validate_every"])
        save_metrics_history = bool(
            config["logging"]["save_metrics_history_json"]
        )
        if start_epoch >= total_epochs:
            raise ValueError(
                f"Resume checkpoint epoch {start_epoch - 1} already reaches "
                f"training.epochs={total_epochs}."
            )

        for epoch in range(start_epoch, total_epochs):
            epoch_started = time.monotonic()
            progress["epoch"] = epoch
            progress["global_step"] = trainer.global_step
            progress["stage"] = "training"
            if getattr(loaders["train"], "generator", None) is not None:
                loaders["train"].generator.manual_seed(seed + epoch)
            train_result = trainer.train_epoch(loaders["train"])
            progress["global_step"] = trainer.global_step
            validation_due = _validation_due(
                epoch, total_epochs, validate_every
            )
            if not validation_due:
                learning_rate = float(optimizer.param_groups[0]["lr"])
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
                        "amp_scale": train_result["amp_scale"],
                        "optimizer_applied_steps": train_result[
                            "optimizer_applied_steps"
                        ],
                        "skipped_amp_overflow_steps": train_result[
                            "skipped_amp_overflow_steps"
                        ],
                    }
                )
                if save_metrics_history:
                    atomic_write_json(
                        paths.log / "metrics_history.json", history
                    )
                train_logger.info(
                    "epoch=%d global_step=%d train_loss=%.8f validation=skipped "
                    "optimizer_applied_steps=%d skipped_amp_overflow_steps=%d",
                    epoch,
                    trainer.global_step,
                    train_result["train_loss"],
                    train_result["optimizer_applied_steps"],
                    train_result["skipped_amp_overflow_steps"],
                )
                trainer.step_scheduler()
                atomic_write_json(
                    paths.root / "status.json",
                    {
                        "overall": "RUNNING",
                        "training": "RUNNING",
                        "validation": "PENDING",
                        "test": "PENDING",
                        "epoch": epoch,
                        "global_step": trainer.global_step,
                    },
                )
                continue
            progress["validation"] = "RUNNING"
            progress["stage"] = "validation"
            try:
                validation = trainer.validate(loaders["validation"])
            except Exception:
                val_logger.exception("Validation failed at epoch %d", epoch)
                raise
            progress["validation"] = "COMPLETED"
            updates = tracker.consider(
                psnr=validation["psnr_rgb"],
                ssim=validation["ssim_rgb"],
                val_loss=validation["val_loss"],
                source="validation",
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "global_step": trainer.global_step,
                "train_loss": train_result["train_loss"],
                "val_loss": validation["val_loss"],
                "psnr": validation["psnr_rgb"],
                "ssim": validation["ssim_rgb"],
                "learning_rate": learning_rate,
                "epoch_time_seconds": time.monotonic() - epoch_started,
                "amp_scale": train_result["amp_scale"],
                "optimizer_applied_steps": train_result[
                    "optimizer_applied_steps"
                ],
                "skipped_amp_overflow_steps": train_result[
                    "skipped_amp_overflow_steps"
                ],
            }
            history.append(epoch_record)
            if save_metrics_history:
                atomic_write_json(paths.log / "metrics_history.json", history)
            atomic_write_csv(
                paths.result / "validation_metrics.csv",
                validation["per_image"],
                VALIDATION_FIELDS,
            )
            atomic_write_json(
                paths.result / "validation_summary.json",
                {
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                    "num_samples": validation["num_samples"],
                    "val_loss": validation["val_loss"],
                    "mean_psnr_rgb": validation["psnr_rgb"],
                    "mean_ssim_rgb": validation["ssim_rgb"],
                    "best_validation_psnr": tracker.psnr,
                    "best_validation_ssim": tracker.ssim,
                    "best_validation_loss": tracker.val_loss,
                },
            )
            val_logger.info(
                "epoch=%d val_loss=%.8f psnr=%.8f ssim=%.8f",
                epoch,
                validation["val_loss"],
                validation["psnr_rgb"],
                validation["ssim_rgb"],
            )
            train_logger.info(
                "epoch=%d global_step=%d train_loss=%.8f "
                "optimizer_applied_steps=%d skipped_amp_overflow_steps=%d",
                epoch,
                trainer.global_step,
                train_result["train_loss"],
                train_result["optimizer_applied_steps"],
                train_result["skipped_amp_overflow_steps"],
            )
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
                global_step=trainer.global_step,
                config=config,
                seed=seed,
                train_loss=train_result["train_loss"],
                val_loss=validation["val_loss"],
                psnr=validation["psnr_rgb"],
                ssim=validation["ssim_rgb"],
                learning_rate=learning_rate,
                best_tracker=tracker,
                torch_module=torch,
                git_commit=git_commit,
            )
            _save_epoch_checkpoints(
                paths=paths,
                config=config,
                payload=payload,
                updates=updates,
                epoch=epoch,
                torch_module=torch,
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
        if progress["epoch"] != total_epochs - 1:
            raise RuntimeError("Training did not complete every configured epoch.")
        return {
            "last_epoch": progress["epoch"],
            "global_step": progress["global_step"],
            "best_validation_psnr": tracker.psnr,
        }

    def auto_test(checkpoint_key: str) -> Dict[str, Any]:
        from src.v1.test_v1 import execute_test

        test_logger = build_logger(
            f"uie3_v1_auto_test_{paths.root.name}",
            paths.log / "test.log",
            console=bool(config["logging"]["console"]),
            file_enabled=bool(config["logging"]["save_test_log"]),
        )
        try:
            return execute_test(
                config,
                run_dir=paths.root,
                checkpoint_key=checkpoint_key,
                allow_overwrite=False,
                logger=test_logger,
            )
        except Exception:
            test_logger.exception("Automatic test failed")
            raise

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
        train_logger.exception("v1 run failed")
        raise
    finally:
        finish_run_info(run_info, start_monotonic=start_monotonic)
        atomic_write_json(paths.root / "run_info.json", run_info)
    return paths.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=V1_CONFIG_PATH,
        help=f"v1 YAML configuration (default: {V1_CONFIG_PATH})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.config)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
