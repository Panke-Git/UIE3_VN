"""Evaluate one v2 run using only its best validation-PSNR checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.common.experiment.experiment import write_error
from src.common.experiment.logging_utils import atomic_write_csv, atomic_write_json, build_logger
from src.common.experiment.visualization import save_test_visualization, select_visualization_records
from src.v1.test_v1 import _resize_output_image, _single_string, _tensor_to_rgb_image
from src.v2.config import V2_CONFIG_PATH, load_v2_config, validate_v2_config
from src.v2.order_comparison import WINNER_TOLERANCE, winner
from src.v2.train_v2 import (
    METRIC_FIELDS,
    ORDER_COMPARISON_FIELDS,
    require_v2_resume_config_match,
)
from src.v2.visualization import save_shared_test_visualization


CHECKPOINT_MATCH_TOLERANCE = 1.0e-12
_SIDECAR_FLOAT_FIELDS = (
    "train_loss",
    "val_loss",
    "psnr",
    "ssim",
    "learning_rate",
)
_SHARED_PROVENANCE_FIELDS = (
    "validation_psnr_cs",
    "validation_psnr_sc",
    "validation_ssim_cs",
    "validation_ssim_sc",
    "validation_loss_cs",
    "validation_loss_sc",
    "validation_mean_path_psnr",
    "validation_mean_path_ssim",
)


def _finite_number(mapping: Mapping[str, Any], field: str, owner: str) -> float:
    value = mapping.get(field)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{owner}.{field} must be a finite number, got {value!r}.")
    return float(value)


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=CHECKPOINT_MATCH_TOLERANCE,
        abs_tol=CHECKPOINT_MATCH_TOLERANCE,
    ):
        raise ValueError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}."
        )


def require_v2_test_checkpoint_match(
    *,
    run_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    shared: bool,
) -> None:
    """Reject a checkpoint that is not the run's recorded best-PSNR model."""

    validated_run = validate_v2_config(
        dict(run_config), entry_point="test_v2 run config"
    )
    expected_shared = (
        validated_run["experiment"]["variant"]
        == "shared_order_diagnostic"
    )
    if bool(shared) != expected_shared:
        raise ValueError(
            "Shared checkpoint validation mode does not match run variant."
        )
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint.config must be a V2 config mapping.")
    validated_checkpoint = validate_v2_config(
        dict(checkpoint_config), entry_point="test_v2 checkpoint config"
    )
    require_v2_resume_config_match(
        validated_run,
        validated_checkpoint,
        context="Test checkpoint",
    )

    checkpoint_seed = checkpoint.get("seed")
    run_seed = validated_run["experiment"]["seed"]
    if type(checkpoint_seed) is not int or checkpoint_seed != run_seed:
        raise ValueError(
            "Checkpoint seed does not match run config: "
            f"checkpoint={checkpoint_seed!r}, run={run_seed!r}."
        )
    if not isinstance(sidecar, Mapping):
        raise ValueError("best_psnr.json root must be a mapping.")
    expected_metric = (
        "validation_mean_path_psnr" if shared else "validation_psnr"
    )
    if sidecar.get("selection_metric") != expected_metric:
        raise ValueError(
            "best_psnr.json selection_metric mismatch: "
            f"expected {expected_metric!r}, got "
            f"{sidecar.get('selection_metric')!r}."
        )
    if sidecar.get("checkpoint") != "best_psnr.pt":
        raise ValueError(
            "best_psnr.json checkpoint must equal 'best_psnr.pt', got "
            f"{sidecar.get('checkpoint')!r}."
        )
    for field in ("epoch", "global_step"):
        sidecar_value = sidecar.get(field)
        checkpoint_value = checkpoint.get(field)
        if (
            type(sidecar_value) is not int
            or type(checkpoint_value) is not int
            or sidecar_value != checkpoint_value
        ):
            raise ValueError(
                f"best_psnr.json {field} does not match checkpoint payload: "
                f"sidecar={sidecar_value!r}, checkpoint={checkpoint_value!r}."
            )
    for field in _SIDECAR_FLOAT_FIELDS:
        sidecar_value = _finite_number(sidecar, field, "best_psnr.json")
        checkpoint_value = _finite_number(checkpoint, field, "checkpoint")
        _require_close(
            f"best_psnr.json {field}", sidecar_value, checkpoint_value
        )

    if not shared:
        return
    missing = sorted(set(_SHARED_PROVENANCE_FIELDS) - set(checkpoint))
    if missing:
        raise ValueError(
            f"Shared checkpoint is missing provenance fields: {missing}."
        )
    values = {
        field: _finite_number(checkpoint, field, "checkpoint")
        for field in _SHARED_PROVENANCE_FIELDS
    }
    checkpoint_psnr = _finite_number(checkpoint, "psnr", "checkpoint")
    checkpoint_ssim = _finite_number(checkpoint, "ssim", "checkpoint")
    checkpoint_val_loss = _finite_number(checkpoint, "val_loss", "checkpoint")
    _require_close(
        "checkpoint psnr/validation_mean_path_psnr",
        checkpoint_psnr,
        values["validation_mean_path_psnr"],
    )
    _require_close(
        "checkpoint ssim/validation_mean_path_ssim",
        checkpoint_ssim,
        values["validation_mean_path_ssim"],
    )
    _require_close(
        "checkpoint joint validation loss",
        checkpoint_val_loss,
        0.5 * values["validation_loss_cs"]
        + 0.5 * values["validation_loss_sc"],
    )
    _require_close(
        "checkpoint validation mean-path PSNR",
        values["validation_mean_path_psnr"],
        0.5 * values["validation_psnr_cs"]
        + 0.5 * values["validation_psnr_sc"],
    )
    _require_close(
        "checkpoint validation mean-path SSIM",
        values["validation_mean_path_ssim"],
        0.5 * values["validation_ssim_cs"]
        + 0.5 * values["validation_ssim_sc"],
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_run_dir(value: str | Path) -> Path:
    resolved = Path(value).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"test run directory does not exist: {resolved}")
    return resolved


def ensure_v2_test_outputs_available(
    result_dir: Path, *, allow_overwrite: bool
) -> None:
    """Reject complete or partial v2 test output unless replacement is explicit."""

    filenames = (
        "test_metrics.csv",
        "test_metrics_color_then_scatter.csv",
        "test_metrics_scatter_then_color.csv",
        "test_order_comparison.csv",
        "test_summary.json",
        "test_visualization_samples.json",
    )
    occupied = [result_dir / name for name in filenames if (result_dir / name).exists()]
    occupied.extend(sorted(result_dir.glob("test_grid_*x*.png")))
    for directory_name in ("test_samples", "test_all_enhanced"):
        directory = result_dir / directory_name
        if directory.exists() and any(directory.rglob("*")):
            occupied.append(directory)
    if occupied and not allow_overwrite:
        raise FileExistsError(
            "Test output already exists and test.allow_overwrite=false: "
            + ", ".join(str(path) for path in occupied)
        )


def _path_metrics(
    prediction: Any, target: Any, metrics: Mapping[str, Any]
) -> tuple[Any, Any]:
    from src.common.metrics.image_metrics import rgb_psnr_per_image, rgb_ssim_per_image

    psnr = rgb_psnr_per_image(
        prediction,
        target,
        data_range=float(metrics["data_range"]),
        crop_border=int(metrics["crop_border"]),
    )
    ssim = rgb_ssim_per_image(
        prediction,
        target,
        data_range=float(metrics["data_range"]),
        crop_border=int(metrics["crop_border"]),
        window_size=int(metrics["ssim_window_size"]),
        sigma=float(metrics["ssim_sigma"]),
    )
    return psnr, ssim


def _comparison_summary(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(records)
    cs_wins = sum(row["winner_psnr"] == "color_then_scatter" for row in records)
    sc_wins = sum(row["winner_psnr"] == "scatter_then_color" for row in records)
    ties = sum(row["winner_psnr"] == "tie" for row in records)
    return {
        "mean_psnr_color_then_scatter": sum(
            float(row["psnr_color_then_scatter"]) for row in records
        )
        / count,
        "mean_psnr_scatter_then_color": sum(
            float(row["psnr_scatter_then_color"]) for row in records
        )
        / count,
        "mean_ssim_color_then_scatter": sum(
            float(row["ssim_color_then_scatter"]) for row in records
        )
        / count,
        "mean_ssim_scatter_then_color": sum(
            float(row["ssim_scatter_then_color"]) for row in records
        )
        / count,
        "mean_delta_psnr_cs_minus_sc": sum(
            float(row["delta_psnr_cs_minus_sc"]) for row in records
        )
        / count,
        "mean_delta_ssim_cs_minus_sc": sum(
            float(row["delta_ssim_cs_minus_sc"]) for row in records
        )
        / count,
        "color_then_scatter_psnr_win_count": cs_wins,
        "scatter_then_color_psnr_win_count": sc_wins,
        "psnr_tie_count": ties,
        "color_then_scatter_psnr_win_rate": cs_wins / count,
        "scatter_then_color_psnr_win_rate": sc_wins / count,
        "winner_tolerance": WINNER_TOLERANCE,
    }


def execute_test(
    saved_config: Mapping[str, Any],
    *,
    run_dir: Path,
    checkpoint_key: str,
    allow_overwrite: bool,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run full-resolution v2 inference; resize only saved visualization files."""

    if checkpoint_key != "best_psnr":
        raise ValueError("v2 test only permits checkpoint='best_psnr'.")
    config = validate_v2_config(dict(saved_config), entry_point="test_v2")
    variant = str(config["experiment"]["variant"])
    shared = variant == "shared_order_diagnostic"
    result_dir = run_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    ensure_v2_test_outputs_available(result_dir, allow_overwrite=allow_overwrite)
    checkpoint_path = run_dir / "best/best_psnr.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best-validation-PSNR checkpoint is missing: {checkpoint_path}"
        )
    sidecar_path = run_dir / "best/best_psnr.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Best-validation-PSNR sidecar is missing: {sidecar_path}"
        )
    with sidecar_path.open("r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    if not isinstance(sidecar, Mapping):
        raise ValueError("best_psnr.json root must be a mapping.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch must be installed for v2 test execution.") from exc
    from src.common.data.dataloader import build_test_dataloader
    from src.common.experiment.checkpoint import load_checkpoint
    from src.v1.trainer import require_finite_tensor
    from src.v2.model import build_v2_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v2_model(variant=variant, model_config=config["model"])
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        torch_module=torch,
        map_location="cpu",
        restore_training_state=False,
        restore_rng=False,
    )
    require_v2_test_checkpoint_match(
        run_config=config,
        checkpoint=checkpoint,
        sidecar=sidecar,
        shared=shared,
    )
    model.to(device).eval()
    loader = build_test_dataloader(config)
    visualization = config["test"]["visualization"]
    candidates = [
        {
            "sample_id": entry.sample_id,
            "input_relative_path": entry.input_relative_path,
            "gt_relative_path": entry.gt_relative_path,
        }
        for entry in loader.dataset.entries
    ]
    chosen = select_visualization_records(
        candidates,
        num_samples=int(visualization["num_samples"]),
        random_seed=int(visualization["random_seed"]),
    )
    chosen_ids = {row["sample_id"] for row in chosen}
    visuals: Dict[str, Dict[str, Any]] = {}
    output_size = config["test"]["output_size"]
    start_time = _utc_now()
    normal_records: List[Dict[str, Any]] = []
    cs_records: List[Dict[str, Any]] = []
    sc_records: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    all_root = result_dir / "test_all_enhanced"
    if config["test"]["save_all_enhanced_images"]:
        if shared:
            (all_root / "color_then_scatter").mkdir(parents=True, exist_ok=True)
            (all_root / "scatter_then_color").mkdir(parents=True, exist_ok=True)
        else:
            all_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            require_finite_tensor("v2 test input", inputs)
            require_finite_tensor("v2 test target", targets)
            if inputs.shape[0] != 1:
                raise ValueError("The v2 test DataLoader must use batch_size=1.")
            sample_id = _single_string(batch["sample_id"], "sample_id")
            input_path = _single_string(
                batch["input_relative_path"], "input_relative_path"
            )
            gt_path = _single_string(batch["gt_relative_path"], "gt_relative_path")
            common = {
                "sample_id": sample_id,
                "input_relative_path": input_path,
                "gt_relative_path": gt_path,
            }
            if not shared:
                prediction = model(inputs)
                require_finite_tensor("v2 test prediction", prediction)
                psnr, ssim = _path_metrics(prediction, targets, config["metrics"])
                row = {
                    **common,
                    "psnr_rgb": float(psnr[0].detach().cpu()),
                    "ssim_rgb": float(ssim[0].detach().cpu()),
                }
                normal_records.append(row)
                image = _resize_output_image(
                    _tensor_to_rgb_image(prediction[0]), output_size
                )
                if config["test"]["save_all_enhanced_images"]:
                    image.save(all_root / f"{sample_id}_enhanced.png", format="PNG")
                if sample_id in chosen_ids:
                    visuals[sample_id] = {
                        **common,
                        "input": _resize_output_image(
                            _tensor_to_rgb_image(inputs[0]), output_size
                        ),
                        "enhanced": image,
                        "gt": _resize_output_image(
                            _tensor_to_rgb_image(targets[0]), output_size
                        ),
                    }
                continue

            prediction_cs = model.forward_color_then_scatter(inputs)
            require_finite_tensor("shared CS test prediction", prediction_cs)
            psnr_cs, ssim_cs = _path_metrics(
                prediction_cs, targets, config["metrics"]
            )
            image_cs = _resize_output_image(
                _tensor_to_rgb_image(prediction_cs[0]), output_size
            )
            del prediction_cs
            prediction_sc = model.forward_scatter_then_color(inputs)
            require_finite_tensor("shared SC test prediction", prediction_sc)
            psnr_sc, ssim_sc = _path_metrics(
                prediction_sc, targets, config["metrics"]
            )
            image_sc = _resize_output_image(
                _tensor_to_rgb_image(prediction_sc[0]), output_size
            )
            del prediction_sc
            cs_psnr = float(psnr_cs[0].detach().cpu())
            sc_psnr = float(psnr_sc[0].detach().cpu())
            cs_ssim = float(ssim_cs[0].detach().cpu())
            sc_ssim = float(ssim_sc[0].detach().cpu())
            cs_records.append({**common, "psnr_rgb": cs_psnr, "ssim_rgb": cs_ssim})
            sc_records.append({**common, "psnr_rgb": sc_psnr, "ssim_rgb": sc_ssim})
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
            if config["test"]["save_all_enhanced_images"]:
                image_cs.save(
                    all_root / "color_then_scatter" / f"{sample_id}_enhanced.png",
                    format="PNG",
                )
                image_sc.save(
                    all_root / "scatter_then_color" / f"{sample_id}_enhanced.png",
                    format="PNG",
                )
            if sample_id in chosen_ids:
                visuals[sample_id] = {
                    **common,
                    "input": _resize_output_image(
                        _tensor_to_rgb_image(inputs[0]), output_size
                    ),
                    "color_then_scatter": image_cs,
                    "scatter_then_color": image_sc,
                    "gt": _resize_output_image(
                        _tensor_to_rgb_image(targets[0]), output_size
                    ),
                }

    if not normal_records and not comparisons:
        raise ValueError("Full v2 test DataLoader produced no samples.")
    if not shared:
        atomic_write_csv(result_dir / "test_metrics.csv", normal_records, METRIC_FIELDS)
        metric_values = [
            float(row[key])
            for row in normal_records
            for key in ("psnr_rgb", "ssim_rgb")
        ]
        summary: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection_source": "formal_validation_psnr",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "num_samples": len(normal_records),
            "mean_psnr_rgb": sum(row["psnr_rgb"] for row in normal_records)
            / len(normal_records),
            "mean_ssim_rgb": sum(row["ssim_rgb"] for row in normal_records)
            / len(normal_records),
            "all_metrics_finite": all(math.isfinite(value) for value in metric_values),
        }
    else:
        atomic_write_csv(
            result_dir / "test_metrics_color_then_scatter.csv",
            cs_records,
            METRIC_FIELDS,
        )
        atomic_write_csv(
            result_dir / "test_metrics_scatter_then_color.csv",
            sc_records,
            METRIC_FIELDS,
        )
        atomic_write_csv(
            result_dir / "test_order_comparison.csv",
            comparisons,
            ORDER_COMPARISON_FIELDS,
        )
        comparison_summary = _comparison_summary(comparisons)
        metric_values = [
            float(row[key])
            for row in comparisons
            for key in (
                "psnr_color_then_scatter",
                "psnr_scatter_then_color",
                "ssim_color_then_scatter",
                "ssim_scatter_then_color",
                "delta_psnr_cs_minus_sc",
                "delta_ssim_cs_minus_sc",
            )
        ]
        summary = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection_source": "validation_mean_path_psnr",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "num_samples": len(comparisons),
            **comparison_summary,
            "all_metrics_finite": all(math.isfinite(value) for value in metric_values),
        }
    summary.update(
        {
            "saved_output_size": (
                None
                if output_size is None
                else [int(output_size), int(output_size)]
            ),
            "test_start_time": start_time,
            "test_end_time": _utc_now(),
        }
    )
    atomic_write_json(result_dir / "test_summary.json", summary)
    if visualization["enabled"]:
        missing = chosen_ids - set(visuals)
        if missing:
            raise RuntimeError(
                f"Selected visualization samples were not evaluated: {sorted(missing)}"
            )
        selected = [visuals[row["sample_id"]] for row in chosen]
        if shared:
            save_shared_test_visualization(selected, visualization, result_dir)
        else:
            save_test_visualization(selected, visualization, result_dir)
    if logger is not None:
        if shared:
            logger.info(
                "Shared test completed: samples=%d psnr_cs=%.8f psnr_sc=%.8f",
                summary["num_samples"],
                summary["mean_psnr_color_then_scatter"],
                summary["mean_psnr_scatter_then_color"],
            )
        else:
            logger.info(
                "Test completed: samples=%d mean_psnr_rgb=%.8f mean_ssim_rgb=%.8f",
                summary["num_samples"],
                summary["mean_psnr_rgb"],
                summary["mean_ssim_rgb"],
            )
    return summary


def run_standalone(
    config_path: Path = V2_CONFIG_PATH,
    *,
    run_dir_override: Optional[Path] = None,
    allow_overwrite_override: bool = False,
) -> Dict[str, Any]:
    current = load_v2_config(config_path, entry_point="test_v2")
    run_value: Any = run_dir_override or current["test"]["run_dir"]
    if run_value is None:
        raise ValueError("Provide --run-dir or set test.run_dir in configV2.yaml.")
    run_dir = _resolve_run_dir(run_value)
    snapshot = run_dir / "config.json"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Experiment config snapshot is missing: {snapshot}")
    with snapshot.open("r", encoding="utf-8") as handle:
        saved = validate_v2_config(json.load(handle), entry_point="test_v2")
    logger = build_logger(
        f"uie3_v2_test_{run_dir.name}",
        run_dir / "log/test.log",
        console=bool(saved["logging"]["console"]),
        file_enabled=bool(saved["logging"]["save_test_log"]),
    )
    try:
        result = execute_test(
            saved,
            run_dir=run_dir,
            checkpoint_key=current["test"]["checkpoint"],
            allow_overwrite=(
                allow_overwrite_override
                or bool(current["test"]["allow_overwrite"])
            ),
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
        logger.exception("Standalone v2 test failed")
        raise
    atomic_write_json(
        run_dir / "status.json",
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
    parser.add_argument("--config", type=Path, default=V2_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_standalone(
            args.config,
            run_dir_override=args.run_dir,
            allow_overwrite_override=args.allow_overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
