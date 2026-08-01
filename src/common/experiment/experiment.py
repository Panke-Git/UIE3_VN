"""Unique run creation, metadata, status, and simple lifecycle orchestration."""

from __future__ import annotations

import os
import platform
import secrets
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from .config import PROJECT_ROOT, resolve_output_root
from .logging_utils import atomic_write_json


@dataclass(frozen=True)
class RunPaths:
    root: Path
    log: Path
    best: Path
    checkpoint: Path
    result: Path
    test_samples: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_experiment(config: Mapping[str, Any], *, now: datetime | None = None) -> RunPaths:
    """Create one timestamped run without ever reusing an existing directory."""

    moment = now or datetime.now()
    timestamp = moment.strftime("%Y%m%d_%H%M%S")
    output_root = resolve_output_root(
        config["experiment"]["output_root"], project_root=PROJECT_ROOT
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"{config['experiment']['version']}_{config['experiment']['name']}_{timestamp}"
    )
    run_dir = output_root / base_name
    if run_dir.exists():
        while True:
            candidate = output_root / f"{base_name}_{secrets.token_hex(3)}"
            if not candidate.exists():
                run_dir = candidate
                break
    run_dir.mkdir(parents=False, exist_ok=False)
    paths = RunPaths(
        root=run_dir,
        log=run_dir / "log",
        best=run_dir / "best",
        checkpoint=run_dir / "checkpoint",
        result=run_dir / "result",
        test_samples=run_dir / "result/test_samples",
    )
    for directory in (
        paths.log,
        paths.best,
        paths.checkpoint,
        paths.result,
        paths.test_samples,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(paths.root / "config.json", dict(config))
    return paths


def git_state(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return truthful local commit and dirty status, including an unborn branch."""

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit: Optional[str] = (
        commit_result.stdout.strip() if commit_result.returncode == 0 else None
    )
    status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty: Optional[bool] = (
        bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    )
    return {"commit": commit, "dirty": dirty}


def collect_run_info(
    config: Mapping[str, Any],
    paths: RunPaths,
    *,
    source_config_path: Path,
    model: Any,
    torch_module: Any,
    command: list[str],
) -> Dict[str, Any]:
    """Collect the requested environment and provenance fields."""

    git = git_state()
    cuda_version = getattr(torch_module.version, "cuda", None)
    cudnn_version = (
        torch_module.backends.cudnn.version()
        if hasattr(torch_module.backends, "cudnn")
        else None
    )
    cuda_available = bool(torch_module.cuda.is_available())
    gpu_name = torch_module.cuda.get_device_name(0) if cuda_available else None
    parameters = list(model.parameters())
    return {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"]["version"],
        "seed": config["experiment"]["seed"],
        "source_config_path": str(source_config_path),
        "run_directory": str(paths.root),
        "git_commit": git["commit"] if git["commit"] is not None else "uncommitted",
        "git_dirty": git["dirty"],
        "python_version": platform.python_version(),
        "pytorch_version": str(torch_module.__version__),
        "cuda_runtime": cuda_version,
        "cuda_available": cuda_available,
        "cudnn": cudnn_version,
        "gpu_name": gpu_name,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "start_time": utc_now(),
        "end_time": None,
        "elapsed_time_seconds": None,
        "command": shlex.join(command),
        "dataset_root": config["data"]["root"],
        "manifest_paths": {
            "train": config["data"]["train_manifest"],
            "validation": config["data"]["validation_manifest"],
            "test": config["data"]["test_manifest"],
        },
        "process_id": os.getpid(),
        "python_executable": sys.executable,
    }


def finish_run_info(
    run_info: MutableMapping[str, Any], *, start_monotonic: float
) -> Dict[str, Any]:
    run_info["end_time"] = utc_now()
    run_info["elapsed_time_seconds"] = time.monotonic() - start_monotonic
    return dict(run_info)


def write_error(
    path: Path,
    *,
    stage: str,
    error: BaseException,
    epoch: Optional[int],
    global_step: Optional[int],
) -> Dict[str, Any]:
    payload = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "time": utc_now(),
    }
    atomic_write_json(path, payload)
    return payload


def run_training_and_auto_test(
    *,
    training_callable: Callable[[], Any],
    test_callable: Callable[[str], Any],
    status_path: Path,
    error_path: Path,
    progress: MutableMapping[str, Any],
    auto_test: bool = True,
) -> Any:
    """Run test only after successful training and always select best PSNR."""

    atomic_write_json(
        status_path,
        {
            "overall": "RUNNING",
            "training": "RUNNING",
            "validation": "PENDING",
            "test": "PENDING",
            "epoch": 0,
        },
    )
    try:
        training_result = training_callable()
    except Exception as exc:
        atomic_write_json(
            status_path,
            {
                "overall": "FAILED",
                "training": "FAILED",
                "validation": progress.get("validation", "PENDING"),
                "test": "NOT_RUN",
                "last_epoch": progress.get("epoch"),
            },
        )
        write_error(
            error_path,
            stage=str(progress.get("stage", "training")),
            error=exc,
            epoch=progress.get("epoch"),
            global_step=progress.get("global_step"),
        )
        raise

    completed = {
        "overall": "RUNNING" if auto_test else "COMPLETED",
        "training": "COMPLETED",
        "validation": "COMPLETED",
        "test": "RUNNING" if auto_test else "NOT_RUN",
        "last_epoch": progress.get("epoch"),
    }
    atomic_write_json(status_path, completed)
    if not auto_test:
        return training_result
    try:
        test_result = test_callable("best_psnr")
    except Exception as exc:
        atomic_write_json(
            status_path,
            {
                "overall": "PARTIAL_FAILURE",
                "training": "COMPLETED",
                "validation": "COMPLETED",
                "test": "FAILED",
                "last_epoch": progress.get("epoch"),
            },
        )
        write_error(
            error_path,
            stage="test",
            error=exc,
            epoch=progress.get("epoch"),
            global_step=progress.get("global_step"),
        )
        raise
    atomic_write_json(
        status_path,
        {
            "overall": "COMPLETED",
            "training": "COMPLETED",
            "validation": "COMPLETED",
            "test": "COMPLETED",
            "last_epoch": progress.get("epoch"),
        },
    )
    return training_result, test_result
