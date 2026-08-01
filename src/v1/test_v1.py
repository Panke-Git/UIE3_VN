"""Evaluate exactly one run's best-validation-PSNR checkpoint on full test."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from src.common.experiment.config import (
    PROJECT_ROOT,
    V1_CONFIG_PATH,
    load_v1_config,
    resolve_v1_config_path,
    validate_v1_config,
)
from src.common.experiment.experiment import write_error
from src.common.experiment.logging_utils import (
    atomic_write_csv,
    atomic_write_json,
    build_logger,
)
from src.common.experiment.visualization import (
    save_test_visualization,
    select_visualization_records,
)


TEST_FIELDS = (
    "sample_id",
    "input_relative_path",
    "gt_relative_path",
    "psnr_rgb",
    "ssim_rgb",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_device(torch_module: Any) -> Any:
    return torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def _resolve_run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve(strict=False)
    )
    experiments_root = (PROJECT_ROOT / "experiments").resolve(strict=False)
    try:
        resolved.relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError(
            f"test.run_dir must be inside {experiments_root}; got {resolved}."
        ) from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"test.run_dir is not a directory: {resolved}")
    return resolved


def ensure_test_outputs_available(result_dir: Path, *, allow_overwrite: bool) -> None:
    """Reject any existing complete/partial test output unless explicitly allowed."""

    candidates = (
        result_dir / "test_metrics.csv",
        result_dir / "test_summary.json",
        result_dir / "test_visualization_samples.json",
        result_dir / "test_grid_10x3.png",
    )
    sample_dir = result_dir / "test_samples"
    occupied = [path for path in candidates if path.exists()]
    if sample_dir.exists() and any(sample_dir.iterdir()):
        occupied.append(sample_dir)
    if occupied and not allow_overwrite:
        raise FileExistsError(
            "Test output already exists and test.allow_overwrite=false: "
            + ", ".join(str(path) for path in occupied)
        )


def _tensor_to_rgb_image(tensor: Any) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    uint8 = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def _single_string(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    values = list(value)
    if len(values) != 1:
        raise ValueError(f"Test field {field!r} must contain exactly one value.")
    return str(values[0])


def execute_test(
    saved_config: Mapping[str, Any],
    *,
    run_dir: Path,
    checkpoint_key: str,
    allow_overwrite: bool,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run read-only inference and metrics; never create training state."""

    if checkpoint_key != "best_psnr":
        raise ValueError("v1 test only permits checkpoint='best_psnr'.")
    config = validate_v1_config(dict(saved_config))
    result_dir = run_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    ensure_test_outputs_available(result_dir, allow_overwrite=allow_overwrite)
    checkpoint_path = run_dir / "best/best_psnr.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Formal best-validation-PSNR checkpoint is missing: {checkpoint_path}"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch must already be installed for v1 test execution."
        ) from exc
    from src.common.data.dataloader import build_test_dataloader
    from src.common.experiment.checkpoint import load_checkpoint
    from src.common.metrics.image_metrics import (
        rgb_psnr_per_image,
        rgb_ssim_per_image,
    )
    from src.v1.model import build_nafnet_small
    from src.v1.trainer import require_finite_tensor

    device = _resolve_device(torch)
    model = build_nafnet_small(**config["model"])
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        torch_module=torch,
        map_location="cpu",
        restore_training_state=False,
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    loader = build_test_dataloader(config)

    visualization_config = config["test"]["visualization"]
    candidate_records = [
        {
            "sample_id": entry.sample_id,
            "input_relative_path": entry.input_relative_path,
            "gt_relative_path": entry.gt_relative_path,
        }
        for entry in loader.dataset.entries
    ]
    chosen_records = select_visualization_records(
        candidate_records,
        num_samples=int(visualization_config["num_samples"]),
        random_seed=int(visualization_config["random_seed"]),
    )
    chosen_ids = {record["sample_id"] for record in chosen_records}
    visual_by_id: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    start_time = _utc_now()
    all_images_dir = result_dir / "test_all_enhanced"
    if config["test"]["save_all_enhanced_images"]:
        all_images_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            require_finite_tensor("test input", inputs)
            require_finite_tensor("test target", targets)
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Test prediction/target shape mismatch: {predictions.shape} "
                    f"versus {targets.shape}."
                )
            require_finite_tensor("test prediction", predictions)
            psnr_values = rgb_psnr_per_image(
                predictions,
                targets,
                data_range=float(config["metrics"]["data_range"]),
                crop_border=int(config["metrics"]["crop_border"]),
            )
            ssim_values = rgb_ssim_per_image(
                predictions,
                targets,
                data_range=float(config["metrics"]["data_range"]),
                crop_border=int(config["metrics"]["crop_border"]),
                window_size=int(config["metrics"]["ssim_window_size"]),
                sigma=float(config["metrics"]["ssim_sigma"]),
            )
            if predictions.shape[0] != 1:
                raise ValueError("The v1 test DataLoader must use batch_size=1.")
            sample_id = _single_string(batch["sample_id"], "sample_id")
            input_path = _single_string(
                batch["input_relative_path"], "input_relative_path"
            )
            gt_path = _single_string(batch["gt_relative_path"], "gt_relative_path")
            record = {
                "sample_id": sample_id,
                "input_relative_path": input_path,
                "gt_relative_path": gt_path,
                "psnr_rgb": float(psnr_values[0].detach().cpu()),
                "ssim_rgb": float(ssim_values[0].detach().cpu()),
            }
            records.append(record)
            if sample_id in chosen_ids:
                visual_by_id[sample_id] = {
                    **record,
                    "input": _tensor_to_rgb_image(inputs[0]),
                    "enhanced": _tensor_to_rgb_image(predictions[0]),
                    "gt": _tensor_to_rgb_image(targets[0]),
                }
            if config["test"]["save_all_enhanced_images"]:
                _tensor_to_rgb_image(predictions[0]).save(
                    all_images_dir / f"{sample_id}_enhanced.png", format="PNG"
                )

    if not records:
        raise ValueError("Full test DataLoader produced no samples.")
    atomic_write_csv(result_dir / "test_metrics.csv", records, TEST_FIELDS)
    values = [
        float(record[key])
        for record in records
        for key in ("psnr_rgb", "ssim_rgb")
    ]
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection_source": "formal_validation_psnr",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "num_samples": len(records),
        "mean_psnr_rgb": sum(row["psnr_rgb"] for row in records) / len(records),
        "mean_ssim_rgb": sum(row["ssim_rgb"] for row in records) / len(records),
        "all_metrics_finite": all(math.isfinite(value) for value in values),
        "test_start_time": start_time,
        "test_end_time": _utc_now(),
    }
    atomic_write_json(result_dir / "test_summary.json", summary)
    if visualization_config["enabled"]:
        missing = chosen_ids - set(visual_by_id)
        if missing:
            raise RuntimeError(
                f"Selected visualization samples were not evaluated: {sorted(missing)}"
            )
        selected_visuals = [
            visual_by_id[record["sample_id"]] for record in chosen_records
        ]
        save_test_visualization(
            selected_visuals, visualization_config, result_dir
        )
    if logger is not None:
        logger.info(
            "Test completed: samples=%d mean_psnr_rgb=%.8f mean_ssim_rgb=%.8f",
            summary["num_samples"],
            summary["mean_psnr_rgb"],
            summary["mean_ssim_rgb"],
        )
    return summary


def run_standalone(config_path: Path = V1_CONFIG_PATH) -> Dict[str, Any]:
    resolved_config_path = resolve_v1_config_path(config_path)
    current_config = load_v1_config(
        resolved_config_path, entry_point="test_v1"
    )
    run_dir_value = current_config["test"]["run_dir"]
    if run_dir_value is None:
        raise ValueError(
            "Set test.run_dir in the selected v1 configuration before standalone "
            "test."
        )
    run_dir = _resolve_run_dir(run_dir_value)
    saved_config_path = run_dir / "config.json"
    if not saved_config_path.is_file():
        raise FileNotFoundError(
            f"Experiment config snapshot is missing: {saved_config_path}"
        )
    with saved_config_path.open("r", encoding="utf-8") as handle:
        saved_config = validate_v1_config(json.load(handle))
    logger = build_logger(
        f"uie3_v1_test_{run_dir.name}",
        run_dir / "log/test.log",
        console=bool(saved_config["logging"]["console"]),
    )
    status_path = run_dir / "status.json"
    try:
        result = execute_test(
            saved_config,
            run_dir=run_dir,
            checkpoint_key=current_config["test"]["checkpoint"],
            allow_overwrite=bool(current_config["test"]["allow_overwrite"]),
            logger=logger,
        )
    except Exception as exc:
        write_error(
            run_dir / "error.json",
            stage="test",
            error=exc,
            epoch=None,
            global_step=None,
        )
        atomic_write_json(
            status_path,
            {
                "overall": "PARTIAL_FAILURE",
                "training": "COMPLETED",
                "validation": "COMPLETED",
                "test": "FAILED",
            },
        )
        logger.exception("Standalone test failed")
        raise
    atomic_write_json(
        status_path,
        {
            "overall": "COMPLETED",
            "training": "COMPLETED",
            "validation": "COMPLETED",
            "test": "COMPLETED",
        },
    )
    return result


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
        run_standalone(args.config)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
