"""Independent best tracking and atomic, resume-complete checkpoint storage."""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .logging_utils import atomic_write_json


CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "schema_version",
    "epoch",
    "global_step",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "config",
    "seed",
    "train_loss",
    "val_loss",
    "psnr",
    "ssim",
    "learning_rate",
    "best_validation_psnr",
    "best_validation_ssim",
    "best_validation_loss",
    "python_random_state",
    "numpy_random_state",
    "torch_random_state",
    "cuda_random_state",
}


@dataclass
class BestTracker:
    """Keep three validation-only optima independently."""

    psnr: float = float("-inf")
    ssim: float = float("-inf")
    val_loss: float = float("inf")

    def consider(
        self,
        *,
        psnr: float,
        ssim: float,
        val_loss: float,
        source: str = "validation",
    ) -> Dict[str, bool]:
        if source != "validation":
            return {"psnr": False, "ssim": False, "val_loss": False}
        updates = {
            "psnr": float(psnr) > self.psnr,
            "ssim": float(ssim) > self.ssim,
            "val_loss": float(val_loss) < self.val_loss,
        }
        if updates["psnr"]:
            self.psnr = float(psnr)
        if updates["ssim"]:
            self.ssim = float(ssim)
        if updates["val_loss"]:
            self.val_loss = float(val_loss)
        return updates


def _state_dict_or_none(component: Optional[Any]) -> Optional[Dict[str, Any]]:
    return None if component is None else component.state_dict()


def build_checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Optional[Any],
    scaler: Any,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    seed: int,
    train_loss: float,
    val_loss: float,
    psnr: float,
    ssim: float,
    learning_rate: float,
    best_tracker: BestTracker,
    torch_module: Any,
    git_commit: Optional[str],
) -> Dict[str, Any]:
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer.")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a non-negative integer.")
    cuda_state = (
        torch_module.cuda.get_rng_state_all()
        if torch_module.cuda.is_available()
        else None
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "creation_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": git_commit,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": _state_dict_or_none(scheduler),
        "scaler_state_dict": _state_dict_or_none(scaler),
        "config": copy.deepcopy(dict(config)),
        "seed": seed,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "learning_rate": float(learning_rate),
        "best_validation_psnr": float(best_tracker.psnr),
        "best_validation_ssim": float(best_tracker.ssim),
        "best_validation_loss": float(best_tracker.val_loss),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch_module.get_rng_state(),
        "cuda_random_state": cuda_state,
    }


def atomic_torch_save(path: Path | str, payload: Mapping[str, Any], torch_module: Any) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch_module.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def save_checkpoint(
    path: Path | str,
    *,
    payload: Mapping[str, Any],
    torch_module: Any,
    selection_metric: Optional[str] = None,
) -> Path:
    destination = atomic_torch_save(path, payload, torch_module)
    if selection_metric is not None:
        sidecar = {
            "selection_metric": selection_metric,
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "train_loss": payload["train_loss"],
            "val_loss": payload["val_loss"],
            "psnr": payload["psnr"],
            "ssim": payload["ssim"],
            "learning_rate": payload["learning_rate"],
            "checkpoint": destination.name,
        }
        atomic_write_json(destination.with_suffix(".json"), sidecar)
    return destination


def _torch_load(path: Path, torch_module: Any, map_location: Any) -> Any:
    try:
        return torch_module.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def load_checkpoint(
    path: Path | str,
    *,
    model: Any,
    torch_module: Any,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Any = "cpu",
    restore_training_state: bool = False,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """Validate and load a checkpoint for inference or strict training resume."""

    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = _torch_load(checkpoint_path, torch_module, map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    missing = sorted(REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema_version={payload['schema_version']!r}."
        )
    if not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("model_state_dict must be a mapping.")
    model.load_state_dict(payload["model_state_dict"], strict=True)

    if restore_training_state:
        if optimizer is None or scaler is None:
            raise ValueError("optimizer and scaler are required for training resume.")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        saved_scheduler = payload["scheduler_state_dict"]
        if saved_scheduler is not None:
            if scheduler is None:
                raise ValueError(
                    "Checkpoint has scheduler state but no scheduler was supplied."
                )
            scheduler.load_state_dict(saved_scheduler)
        elif scheduler is not None:
            raise ValueError(
                "A scheduler was supplied but the checkpoint has no scheduler state."
            )
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch_module.set_rng_state(payload["torch_random_state"])
        cuda_state = payload["cuda_random_state"]
        if cuda_state is not None:
            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
            torch_module.cuda.set_rng_state_all(cuda_state)
    return dict(payload)


def tracker_from_checkpoint(payload: Mapping[str, Any]) -> BestTracker:
    return BestTracker(
        psnr=float(payload["best_validation_psnr"]),
        ssim=float(payload["best_validation_ssim"]),
        val_loss=float(payload["best_validation_loss"]),
    )
