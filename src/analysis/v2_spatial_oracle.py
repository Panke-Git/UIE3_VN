"""Multi-scale spatial Oracle analysis for V2 shared-order validation runs.

The command performs inference with each run's best validation-PSNR checkpoint.
It never reads the test split and never modifies an experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from src.analysis.v2_order_validation import (
    DEFAULT_SEEDS,
    SeedValidationResult,
    compute_image_level_oracle,
    discover_seed_run,
    load_seed_validation_result,
    validate_configs_across_seeds,
)
from src.common.experiment.logging_utils import atomic_write_csv, atomic_write_json


DEFAULT_REGION_SIZES = (128, 64, 32)
DEFAULT_OUTPUT_DIR = Path("analysis_results/v2_validation_spatial_oracle")
SSE_TIE_REL_TOLERANCE = 1.0e-12
SSE_TIE_ABS_TOLERANCE = 1.0e-12
INFERENCE_REL_TOLERANCE = 1.0e-7
INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE = 1.0e-3
INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE = 1.0e-4
INFERENCE_PER_IMAGE_PSNR_ABS_TOLERANCE = 5.0e-3
INFERENCE_PER_IMAGE_SSIM_ABS_TOLERANCE = 1.0e-4
MONOTONIC_REL_TOLERANCE = 1.0e-7
MONOTONIC_ABS_TOLERANCE = 1.0e-8
GAIN_THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
TOP_VISUALIZATIONS_PER_SCALE = 5


@dataclass(frozen=True)
class RegionBounds:
    """One non-overlapping top-left-aligned image region."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def pixels(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)


@dataclass(frozen=True)
class TileChoice:
    """SSE-based choice for one spatial region."""

    bounds: RegionBounds
    choice: str
    sse_cs: float
    sse_sc: float


@dataclass(frozen=True)
class ReconstructedOracle:
    """A complete Oracle tensor and the mask/choices used to build it."""

    prediction: Any
    selected_cs_mask: Any
    tiles: tuple[TileChoice, ...]
    total_sse: float
    selected_cs_pixels: int
    selected_sc_pixels: int


@dataclass(frozen=True)
class MeasuredOracle:
    """A reconstructed Oracle plus its whole-image project metrics."""

    reconstructed: ReconstructedOracle
    psnr: float
    ssim: float


@dataclass(frozen=True)
class VisualizationCandidate:
    """Small top-K CPU cache retained only until panel files are written."""

    seed: int
    sample_id: str
    region_size: int
    gain_over_whole: float
    psnr_cs: float
    psnr_sc: float
    whole_psnr: float
    spatial_psnr: float
    input_tensor: Any
    target_tensor: Any
    prediction_cs: Any
    prediction_sc: Any
    oracle_prediction: Any
    selected_cs_mask: Any
    data_range: float


def validate_region_sizes(values: Sequence[int]) -> tuple[int, ...]:
    """Require positive, strictly descending, exactly refining grid sizes."""

    sizes = tuple(values)
    if not sizes:
        raise ValueError("region_sizes must contain at least one size.")
    for value in sizes:
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Every region size must be a positive integer, got {value!r}."
            )
    for larger, smaller in zip(sizes, sizes[1:]):
        if larger <= smaller:
            raise ValueError(
                "region_sizes must be strictly ordered from large to small; "
                f"got adjacent values {larger}, {smaller}."
            )
        if larger % smaller != 0:
            raise ValueError(
                "Adjacent region sizes must have an exact refinement relation: "
                f"{larger} % {smaller} != 0."
            )
    return sizes


def region_grid(height: int, width: int, region_size: int) -> tuple[RegionBounds, ...]:
    """Return a complete natural-truncation grid with no padding or overlap."""

    if type(height) is not int or height <= 0 or type(width) is not int or width <= 0:
        raise ValueError(f"Image height/width must be positive, got {height}x{width}.")
    if type(region_size) is not int or region_size <= 0:
        raise ValueError(
            f"region_size must be a positive integer, got {region_size!r}."
        )
    return tuple(
        RegionBounds(
            y0=y0,
            y1=min(y0 + region_size, height),
            x0=x0,
            x1=min(x0 + region_size, width),
        )
        for y0 in range(0, height, region_size)
        for x0 in range(0, width, region_size)
    )


def _validate_oracle_tensors(
    prediction_cs: Any, prediction_sc: Any, target: Any, *, data_range: float
) -> None:
    import torch

    if prediction_cs.shape != prediction_sc.shape or prediction_cs.shape != target.shape:
        raise ValueError(
            "CS, SC, and target tensor shapes must match; got "
            f"{prediction_cs.shape}, {prediction_sc.shape}, and {target.shape}."
        )
    if prediction_cs.ndim != 4 or prediction_cs.shape[0] != 1 or prediction_cs.shape[1] != 3:
        raise ValueError(
            "Spatial Oracle requires one RGB image [1,3,H,W], got "
            f"{tuple(prediction_cs.shape)}."
        )
    if not all(tensor.is_floating_point() for tensor in (prediction_cs, prediction_sc, target)):
        raise TypeError("Spatial Oracle tensors must use floating-point dtypes.")
    if not all(torch.isfinite(tensor).all() for tensor in (prediction_cs, prediction_sc, target)):
        raise ValueError("Spatial Oracle tensors must contain only finite values.")
    if not math.isfinite(float(data_range)) or float(data_range) <= 0.0:
        raise ValueError(f"data_range must be finite and > 0, got {data_range!r}.")
    if torch.any(target < 0.0) or torch.any(target > float(data_range)):
        raise ValueError(f"Spatial Oracle target must lie in [0,{float(data_range)}].")


def reconstruct_spatial_oracle(
    prediction_cs: Any,
    prediction_sc: Any,
    target: Any,
    *,
    region_size: Optional[int],
    data_range: float = 1.0,
    tie_rel_tolerance: float = SSE_TIE_REL_TOLERANCE,
    tie_abs_tolerance: float = SSE_TIE_ABS_TOLERANCE,
) -> ReconstructedOracle:
    """Choose CS/SC by regional RGB SSE and reconstruct one complete image.

    ``region_size=None`` treats the complete image as one region. Predictions
    are clamped exactly as the project's RGB metric implementation does.
    """

    import torch

    _validate_oracle_tensors(
        prediction_cs, prediction_sc, target, data_range=data_range
    )
    clamped_cs = prediction_cs.clamp(0.0, float(data_range))
    clamped_sc = prediction_sc.clamp(0.0, float(data_range))
    height, width = target.shape[-2:]
    regions = (
        (RegionBounds(0, height, 0, width),)
        if region_size is None
        else region_grid(height, width, region_size)
    )
    oracle = torch.empty_like(clamped_cs)
    selected_cs_mask = torch.empty(
        (height, width), dtype=torch.bool, device=target.device
    )
    tiles: list[TileChoice] = []
    selected_cs_pixels = 0
    selected_sc_pixels = 0
    selected_sse = 0.0
    for bounds in regions:
        slices = (..., slice(bounds.y0, bounds.y1), slice(bounds.x0, bounds.x1))
        cs_region = clamped_cs[slices]
        sc_region = clamped_sc[slices]
        target_region = target[slices]
        # Float64 reduction keeps the selection deterministic for large regions.
        sse_cs = float(torch.square(cs_region.double() - target_region.double()).sum())
        sse_sc = float(torch.square(sc_region.double() - target_region.double()).sum())
        tied = math.isclose(
            sse_cs,
            sse_sc,
            rel_tol=tie_rel_tolerance,
            abs_tol=tie_abs_tolerance,
        )
        if tied:
            choice = "tie"
            use_cs = True
        elif sse_cs < sse_sc:
            choice = "CS"
            use_cs = True
        else:
            choice = "SC"
            use_cs = False
        oracle[slices] = cs_region if use_cs else sc_region
        selected_cs_mask[
            bounds.y0 : bounds.y1, bounds.x0 : bounds.x1
        ] = use_cs
        if use_cs:
            selected_cs_pixels += bounds.pixels
            selected_sse += sse_cs
        else:
            selected_sc_pixels += bounds.pixels
            selected_sse += sse_sc
        tiles.append(
            TileChoice(
                bounds=bounds,
                choice=choice,
                sse_cs=sse_cs,
                sse_sc=sse_sc,
            )
        )
    if selected_cs_pixels + selected_sc_pixels != height * width:
        raise RuntimeError("Spatial Oracle grid did not select every image pixel.")
    return ReconstructedOracle(
        prediction=oracle,
        selected_cs_mask=selected_cs_mask,
        tiles=tuple(tiles),
        total_sse=selected_sse,
        selected_cs_pixels=selected_cs_pixels,
        selected_sc_pixels=selected_sc_pixels,
    )


def measure_reconstructed_oracle(
    reconstructed: ReconstructedOracle,
    target: Any,
    metrics_config: Mapping[str, Any],
) -> MeasuredOracle:
    """Measure the complete reconstructed image, never an average of tile PSNR."""

    from src.common.metrics.image_metrics import (
        rgb_psnr_per_image,
        rgb_ssim_per_image,
    )

    psnr = rgb_psnr_per_image(
        reconstructed.prediction,
        target,
        data_range=float(metrics_config["data_range"]),
        crop_border=int(metrics_config["crop_border"]),
    )
    ssim = rgb_ssim_per_image(
        reconstructed.prediction,
        target,
        data_range=float(metrics_config["data_range"]),
        crop_border=int(metrics_config["crop_border"]),
        window_size=int(metrics_config["ssim_window_size"]),
        sigma=float(metrics_config["ssim_sigma"]),
    )
    return MeasuredOracle(
        reconstructed=reconstructed,
        psnr=float(psnr[0].detach().cpu()),
        ssim=float(ssim[0].detach().cpu()),
    )


def compute_spatial_oracle(
    prediction_cs: Any,
    prediction_sc: Any,
    target: Any,
    *,
    region_size: Optional[int],
    metrics_config: Mapping[str, Any],
) -> MeasuredOracle:
    """Reconstruct by regional SSE, then apply original whole-image metrics."""

    reconstructed = reconstruct_spatial_oracle(
        prediction_cs,
        prediction_sc,
        target,
        region_size=region_size,
        data_range=float(metrics_config["data_range"]),
    )
    return measure_reconstructed_oracle(reconstructed, target, metrics_config)


def require_oracle_monotonicity(
    levels: Sequence[tuple[str, MeasuredOracle]],
    *,
    rel_tolerance: float = MONOTONIC_REL_TOLERANCE,
    abs_tolerance: float = MONOTONIC_ABS_TOLERANCE,
) -> None:
    """Reject a finer refinement whose SSE rises or whose PSNR falls."""

    for (coarse_name, coarse), (fine_name, fine) in zip(levels, levels[1:]):
        sse_tolerance = max(
            abs_tolerance, rel_tolerance * max(abs(coarse.reconstructed.total_sse), 1.0)
        )
        if fine.reconstructed.total_sse > coarse.reconstructed.total_sse + sse_tolerance:
            raise RuntimeError(
                f"Spatial Oracle SSE monotonicity violated: {fine_name} SSE="
                f"{fine.reconstructed.total_sse!r} exceeds {coarse_name} SSE="
                f"{coarse.reconstructed.total_sse!r}."
            )
        if fine.psnr + abs_tolerance < coarse.psnr:
            raise RuntimeError(
                f"Spatial Oracle PSNR monotonicity violated: {fine_name} PSNR="
                f"{fine.psnr!r} is below {coarse_name} PSNR={coarse.psnr!r}."
            )


def require_validation_split(split: str) -> None:
    """Hard guard preventing reuse of this analysis for test data."""

    if split != "validation":
        raise ValueError(
            f"V2 spatial Oracle analysis only permits split='validation'; got {split!r}."
        )


def validate_spatial_analysis_metadata(
    *,
    expected_seed: int,
    run_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    split: str = "validation",
) -> Dict[str, Any]:
    """Apply the existing strict V2 checkpoint/run provenance validation."""

    require_validation_split(split)
    from src.v2.config import validate_v2_config
    from src.v2.test_v2 import require_v2_test_checkpoint_match

    config = validate_v2_config(
        dict(run_config), entry_point="v2 spatial Oracle run config"
    )
    variant = config["experiment"]["variant"]
    if variant != "shared_order_diagnostic":
        raise ValueError(
            "V2 spatial Oracle requires variant='shared_order_diagnostic'; "
            f"got {variant!r}."
        )
    config_seed = config["experiment"]["seed"]
    if config_seed != expected_seed:
        raise ValueError(
            f"Spatial Oracle seed mismatch: expected {expected_seed}, "
            f"config has {config_seed!r}."
        )
    require_v2_test_checkpoint_match(
        run_config=config,
        checkpoint=checkpoint,
        sidecar=sidecar,
        shared=True,
    )
    return config


def resolve_device(device_name: str, torch_module: Any) -> Any:
    """Resolve auto/cpu/cuda without silently downgrading an explicit CUDA request."""

    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"device must be one of 'auto', 'cpu', or 'cuda'; got {device_name!r}."
        )
    if device_name == "auto":
        device_name = "cuda" if torch_module.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch_module.device(device_name)


def configure_inference_backend(config: Mapping[str, Any]) -> None:
    """Mirror the training run's seed and deterministic/cuDNN settings."""

    from src.common.experiment.seed import set_global_seed

    set_global_seed(
        int(config["experiment"]["seed"]),
        deterministic=bool(config["training"]["deterministic"]),
    )


def _single_string(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    values = list(value)
    if len(values) != 1:
        raise ValueError(
            f"Validation batch field {field!r} must contain one value, got {values!r}."
        )
    return str(values[0])


def _path_metrics(
    prediction: Any, target: Any, metrics: Mapping[str, Any]
) -> tuple[float, float]:
    from src.common.metrics.image_metrics import (
        rgb_psnr_per_image,
        rgb_ssim_per_image,
    )

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
    return float(psnr[0].detach().cpu()), float(ssim[0].detach().cpu())


def _require_close(
    *,
    seed: int,
    name: str,
    actual: float,
    expected: float,
    abs_tolerance: float,
    rel_tolerance: float = INFERENCE_REL_TOLERANCE,
) -> None:
    if not math.isclose(
        actual, expected, rel_tol=rel_tolerance, abs_tol=abs_tolerance
    ):
        difference = abs(actual - expected)
        allowed = max(
            abs_tolerance,
            rel_tolerance * max(abs(actual), abs(expected)),
        )
        raise ValueError(
            f"Seed {seed} validation inference regression mismatch for {name}: "
            f"saved={expected!r}, recomputed={actual!r}, "
            f"abs_diff={difference!r}, allowed={allowed!r}."
        )


def validate_inference_regression(
    rows: Sequence[Mapping[str, Any]],
    saved_summary: Mapping[str, Any],
    *,
    seed: int,
    rel_tolerance: float = INFERENCE_REL_TOLERANCE,
    psnr_abs_tolerance: float = INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE,
    ssim_abs_tolerance: float = INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE,
) -> Dict[str, float]:
    """Require re-inferred fixed-path validation metrics to match the snapshot."""

    if not rows:
        raise ValueError(f"Seed {seed} validation inference produced no samples.")
    saved_count = saved_summary.get("num_samples")
    if type(saved_count) is not int or saved_count != len(rows):
        raise ValueError(
            f"Seed {seed} validation inference sample count mismatch: "
            f"saved={saved_count!r}, recomputed={len(rows)}."
        )
    field_pairs = {
        "mean_psnr_color_then_scatter": "psnr_cs",
        "mean_psnr_scatter_then_color": "psnr_sc",
        "mean_ssim_color_then_scatter": "ssim_cs",
        "mean_ssim_scatter_then_color": "ssim_sc",
    }
    recomputed: Dict[str, float] = {}
    for summary_field, row_field in field_pairs.items():
        actual = statistics.fmean(float(row[row_field]) for row in rows)
        expected_raw = saved_summary.get(summary_field)
        if type(expected_raw) not in (int, float) or not math.isfinite(
            float(expected_raw)
        ):
            raise ValueError(
                f"Seed {seed} saved summary {summary_field} must be finite, "
                f"got {expected_raw!r}."
            )
        expected = float(expected_raw)
        _require_close(
            seed=seed,
            name=summary_field,
            actual=actual,
            expected=expected,
            rel_tolerance=rel_tolerance,
            abs_tolerance=(
                psnr_abs_tolerance
                if row_field.startswith("psnr_")
                else ssim_abs_tolerance
            ),
        )
        recomputed[summary_field] = actual
    return recomputed


def validate_sample_inference_regression(
    *,
    seed_result: SeedValidationResult,
    sample_id: str,
    input_relative_path: str,
    gt_relative_path: str,
    psnr_cs: float,
    psnr_sc: float,
    ssim_cs: float,
    ssim_sc: float,
) -> None:
    """Fail before spatial work if one re-inferred image misses its saved metrics."""

    key = (input_relative_path, gt_relative_path)
    saved = seed_result.samples.get(key)
    if saved is None:
        raise ValueError(
            f"Seed {seed_result.seed} re-inferred validation sample {key!r} is "
            "absent from best_psnr_validation_order_comparison.csv."
        )
    if saved.sample_id != sample_id:
        raise ValueError(
            f"Seed {seed_result.seed} validation sample_id mismatch for {key!r}: "
            f"saved={saved.sample_id!r}, re-inferred={sample_id!r}."
        )
    comparisons = (
        (
            "per-image PSNR CS",
            psnr_cs,
            saved.psnr_color_then_scatter,
            INFERENCE_PER_IMAGE_PSNR_ABS_TOLERANCE,
        ),
        (
            "per-image PSNR SC",
            psnr_sc,
            saved.psnr_scatter_then_color,
            INFERENCE_PER_IMAGE_PSNR_ABS_TOLERANCE,
        ),
        (
            "per-image SSIM CS",
            ssim_cs,
            saved.ssim_color_then_scatter,
            INFERENCE_PER_IMAGE_SSIM_ABS_TOLERANCE,
        ),
        (
            "per-image SSIM SC",
            ssim_sc,
            saved.ssim_scatter_then_color,
            INFERENCE_PER_IMAGE_SSIM_ABS_TOLERANCE,
        ),
    )
    for name, actual, expected, abs_tolerance in comparisons:
        _require_close(
            seed=seed_result.seed,
            name=f"{name} for sample {sample_id}",
            actual=actual,
            expected=expected,
            abs_tolerance=abs_tolerance,
        )


def _read_prior_oracle_rows(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "seed",
            "oracle_psnr",
            "psnr_oracle_selected_mean_ssim",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Existing whole-image Oracle CSV {path} is missing fields: {missing}."
            )
        rows: Dict[int, Dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                seed = int(row["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Existing whole-image Oracle CSV {path} row {row_number} "
                    "has an invalid seed."
                ) from exc
            if seed in rows:
                raise ValueError(
                    f"Existing whole-image Oracle CSV {path} duplicates seed {seed}."
                )
            rows[seed] = dict(row)
    return rows


def validate_whole_oracle_regression(
    rows: Sequence[Mapping[str, Any]],
    seed_result: SeedValidationResult,
    *,
    prior_oracle_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Match tensor-derived whole-image Oracle to the saved per-image CSV result."""

    actual_psnr = statistics.fmean(float(row["whole_oracle_psnr"]) for row in rows)
    actual_ssim = statistics.fmean(float(row["whole_oracle_ssim"]) for row in rows)
    expected = compute_image_level_oracle(seed_result)
    _require_close(
        seed=seed_result.seed,
        name="whole-image Oracle PSNR from best validation comparison CSV",
        actual=actual_psnr,
        expected=float(expected["oracle_psnr"]),
        abs_tolerance=INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE,
    )
    _require_close(
        seed=seed_result.seed,
        name="whole-image Oracle selected-path SSIM from best validation comparison CSV",
        actual=actual_ssim,
        expected=float(expected["psnr_oracle_selected_mean_ssim"]),
        abs_tolerance=INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE,
    )
    if prior_oracle_row is not None:
        try:
            prior_psnr = float(prior_oracle_row["oracle_psnr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Seed {seed_result.seed} existing oracle_per_seed.csv has an "
                "invalid oracle_psnr."
            ) from exc
        _require_close(
            seed=seed_result.seed,
            name="existing oracle_per_seed.csv oracle_psnr",
            actual=actual_psnr,
            expected=prior_psnr,
            abs_tolerance=INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE,
        )
        try:
            prior_ssim = float(prior_oracle_row["psnr_oracle_selected_mean_ssim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Seed {seed_result.seed} existing oracle_per_seed.csv has an "
                "invalid psnr_oracle_selected_mean_ssim."
            ) from exc
        _require_close(
            seed=seed_result.seed,
            name="existing oracle_per_seed.csv selected-path SSIM",
            actual=actual_ssim,
            expected=prior_ssim,
            abs_tolerance=INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE,
        )
    return {"whole_oracle_psnr": actual_psnr, "whole_oracle_ssim": actual_ssim}


def _tile_rows(
    *,
    seed: int,
    sample_id: str,
    region_size: int,
    measured: MeasuredOracle,
) -> list[Dict[str, Any]]:
    return [
        {
            "seed": seed,
            "sample_id": sample_id,
            "region_size": region_size,
            "y0": tile.bounds.y0,
            "y1": tile.bounds.y1,
            "x0": tile.bounds.x0,
            "x1": tile.bounds.x1,
            "choice": tile.choice,
        }
        for tile in measured.reconstructed.tiles
    ]


def _consider_visualization_candidate(
    bucket: list[VisualizationCandidate],
    candidate: VisualizationCandidate,
    *,
    limit: int,
) -> None:
    bucket.append(candidate)
    bucket.sort(
        key=lambda item: (
            -item.gain_over_whole,
            item.seed,
            item.sample_id,
        )
    )
    del bucket[limit:]


def _visualization_candidate_needed(
    bucket: Sequence[VisualizationCandidate],
    *,
    gain_over_whole: float,
    seed: int,
    sample_id: str,
    limit: int,
) -> bool:
    if len(bucket) < limit:
        return True
    candidate_key = (-gain_over_whole, seed, sample_id)
    worst = bucket[-1]
    worst_key = (-worst.gain_over_whole, worst.seed, worst.sample_id)
    return candidate_key < worst_key


def aggregate_seed_rows(
    rows: list[Dict[str, Any]], seed: int, region_sizes: Sequence[int]
) -> Dict[str, Any]:
    """Aggregate by arithmetic means of per-image project metrics."""

    if not rows:
        raise ValueError(f"Seed {seed} has no spatial Oracle image rows.")
    mean_psnr_cs = statistics.fmean(float(row["psnr_cs"]) for row in rows)
    mean_psnr_sc = statistics.fmean(float(row["psnr_sc"]) for row in rows)
    mean_ssim_cs = statistics.fmean(float(row["ssim_cs"]) for row in rows)
    mean_ssim_sc = statistics.fmean(float(row["ssim_sc"]) for row in rows)
    best_fixed_order = "CS" if mean_psnr_cs >= mean_psnr_sc else "SC"
    best_fixed_psnr = max(mean_psnr_cs, mean_psnr_sc)
    best_fixed_ssim = mean_ssim_cs if best_fixed_order == "CS" else mean_ssim_sc
    fixed_field = "psnr_cs" if best_fixed_order == "CS" else "psnr_sc"
    for row in rows:
        fixed_image_psnr = float(row[fixed_field])
        row["whole_oracle_gain_over_best_fixed"] = (
            float(row["whole_oracle_psnr"]) - fixed_image_psnr
        )
        for size in region_sizes:
            row[f"oracle_{size}_gain_over_best_fixed"] = (
                float(row[f"oracle_{size}_psnr"]) - fixed_image_psnr
            )
            row[f"oracle_{size}_gain_over_whole"] = (
                float(row[f"oracle_{size}_psnr"])
                - float(row["whole_oracle_psnr"])
            )

    whole_psnr = statistics.fmean(
        float(row["whole_oracle_psnr"]) for row in rows
    )
    whole_ssim = statistics.fmean(
        float(row["whole_oracle_ssim"]) for row in rows
    )
    output: Dict[str, Any] = {
        "seed": seed,
        "num_samples": len(rows),
        "mean_psnr_cs": mean_psnr_cs,
        "mean_psnr_sc": mean_psnr_sc,
        "mean_ssim_cs": mean_ssim_cs,
        "mean_ssim_sc": mean_ssim_sc,
        "best_fixed_order": best_fixed_order,
        "best_fixed_psnr": best_fixed_psnr,
        "best_fixed_ssim": best_fixed_ssim,
        "whole_oracle_psnr": whole_psnr,
        "whole_oracle_ssim": whole_ssim,
        "whole_gain_over_fixed": whole_psnr - best_fixed_psnr,
    }
    previous_label = "whole"
    previous_psnr = whole_psnr
    for size in region_sizes:
        oracle_psnr = statistics.fmean(
            float(row[f"oracle_{size}_psnr"]) for row in rows
        )
        oracle_ssim = statistics.fmean(
            float(row[f"oracle_{size}_ssim"]) for row in rows
        )
        output[f"oracle_{size}_psnr"] = oracle_psnr
        output[f"oracle_{size}_ssim"] = oracle_ssim
        output[f"gain_{size}_over_fixed"] = oracle_psnr - best_fixed_psnr
        output[f"gain_{size}_over_whole"] = oracle_psnr - whole_psnr
        output[f"gain_{size}_over_{previous_label}"] = oracle_psnr - previous_psnr
        total_pixels = sum(
            int(row["height"]) * int(row["width"]) for row in rows
        )
        cs_pixels = sum(
            float(row[f"pixels_{size}_cs_rate"])
            * int(row["height"])
            * int(row["width"])
            for row in rows
        )
        output[f"cs_pixel_rate_{size}"] = cs_pixels / total_pixels
        output[f"sc_pixel_rate_{size}"] = 1.0 - output[f"cs_pixel_rate_{size}"]
        previous_label = str(size)
        previous_psnr = oracle_psnr
    return output


def analyze_seed(
    seed_result: SeedValidationResult,
    *,
    region_sizes: Sequence[int],
    device: Any,
    prior_oracle_row: Optional[Mapping[str, Any]] = None,
    visualization_limit: int = TOP_VISUALIZATIONS_PER_SCALE,
) -> tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    Dict[str, Any],
    Dict[int, list[VisualizationCandidate]],
    Dict[str, Any],
]:
    """Load one seed only, infer validation, validate regressions, and release it."""

    import torch

    from src.common.data.dataloader import build_validation_dataloader
    from src.common.experiment.checkpoint import load_checkpoint
    from src.v1.trainer import require_finite_tensor
    from src.v2.model import build_v2_model

    sizes = validate_region_sizes(region_sizes)
    checkpoint_path = seed_result.run_dir / "best/best_psnr.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Seed {seed_result.seed} best validation checkpoint is missing: "
            f"{checkpoint_path}"
        )
    model = build_v2_model(
        variant="shared_order_diagnostic",
        model_config=seed_result.config["model"],
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        torch_module=torch,
        map_location="cpu",
        restore_training_state=False,
        restore_rng=False,
    )
    config = validate_spatial_analysis_metadata(
        expected_seed=seed_result.seed,
        run_config=seed_result.config,
        checkpoint=checkpoint,
        sidecar=seed_result.checkpoint_sidecar,
        split="validation",
    )
    # A fresh analysis process otherwise keeps PyTorch's default cuDNN flags,
    # which differ from training when training.deterministic is false.
    configure_inference_backend(config)
    model.to(device).eval()
    # This helper accesses data.validation_manifest only; test loaders are never built.
    loader = build_validation_dataloader(config)
    image_rows: list[Dict[str, Any]] = []
    tile_rows: list[Dict[str, Any]] = []
    candidates: Dict[int, list[VisualizationCandidate]] = {
        size: [] for size in sizes
    }
    metrics = config["metrics"]
    data_range = float(metrics["data_range"])
    total_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            require_finite_tensor("spatial Oracle validation input", inputs)
            require_finite_tensor("spatial Oracle validation target", targets)
            if inputs.shape[0] != 1:
                raise ValueError(
                    "Spatial Oracle validation DataLoader must use batch_size=1."
                )
            sample_id = _single_string(batch["sample_id"], "sample_id")
            input_path = _single_string(
                batch["input_relative_path"], "input_relative_path"
            )
            gt_path = _single_string(batch["gt_relative_path"], "gt_relative_path")
            prediction_cs = model.forward_color_then_scatter(inputs)
            prediction_sc = model.forward_scatter_then_color(inputs)
            if prediction_cs.shape != targets.shape or prediction_sc.shape != targets.shape:
                raise ValueError(
                    f"Seed {seed_result.seed} sample {sample_id!r} prediction/target "
                    "shape mismatch."
                )
            require_finite_tensor(
                "spatial Oracle CS validation prediction", prediction_cs
            )
            require_finite_tensor(
                "spatial Oracle SC validation prediction", prediction_sc
            )
            psnr_cs, ssim_cs = _path_metrics(prediction_cs, targets, metrics)
            psnr_sc, ssim_sc = _path_metrics(prediction_sc, targets, metrics)
            # The saved per-image validation comparison is checked before any
            # spatial Oracle computation for this sample.
            validate_sample_inference_regression(
                seed_result=seed_result,
                sample_id=sample_id,
                input_relative_path=input_path,
                gt_relative_path=gt_path,
                psnr_cs=psnr_cs,
                psnr_sc=psnr_sc,
                ssim_cs=ssim_cs,
                ssim_sc=ssim_sc,
            )
            whole = compute_spatial_oracle(
                prediction_cs,
                prediction_sc,
                targets,
                region_size=None,
                metrics_config=metrics,
            )
            measured_by_size: Dict[int, MeasuredOracle] = {}
            for size in sizes:
                measured_by_size[size] = compute_spatial_oracle(
                    prediction_cs,
                    prediction_sc,
                    targets,
                    region_size=size,
                    metrics_config=metrics,
                )
            require_oracle_monotonicity(
                [("whole", whole)]
                + [(str(size), measured_by_size[size]) for size in sizes]
            )
            height, width = targets.shape[-2:]
            row: Dict[str, Any] = {
                "seed": seed_result.seed,
                "sample_id": sample_id,
                "input_relative_path": input_path,
                "gt_relative_path": gt_path,
                "height": int(height),
                "width": int(width),
                "psnr_cs": psnr_cs,
                "psnr_sc": psnr_sc,
                "ssim_cs": ssim_cs,
                "ssim_sc": ssim_sc,
                "whole_oracle_psnr": whole.psnr,
                "whole_oracle_ssim": whole.ssim,
                "whole_oracle_selected_order": whole.reconstructed.tiles[0].choice,
            }
            total_pixels = int(height) * int(width)
            for size in sizes:
                measured = measured_by_size[size]
                choices = [tile.choice for tile in measured.reconstructed.tiles]
                row.update(
                    {
                        f"oracle_{size}_psnr": measured.psnr,
                        f"oracle_{size}_ssim": measured.ssim,
                        f"tiles_{size}_total": len(choices),
                        f"tiles_{size}_cs": choices.count("CS"),
                        f"tiles_{size}_sc": choices.count("SC"),
                        f"tiles_{size}_tie": choices.count("tie"),
                        f"pixels_{size}_cs_rate": (
                            measured.reconstructed.selected_cs_pixels / total_pixels
                        ),
                        f"pixels_{size}_sc_rate": (
                            measured.reconstructed.selected_sc_pixels / total_pixels
                        ),
                    }
                )
                tile_rows.extend(
                    _tile_rows(
                        seed=seed_result.seed,
                        sample_id=sample_id,
                        region_size=size,
                        measured=measured,
                    )
                )
                gain_over_whole = measured.psnr - whole.psnr
                if _visualization_candidate_needed(
                    candidates[size],
                    gain_over_whole=gain_over_whole,
                    seed=seed_result.seed,
                    sample_id=sample_id,
                    limit=visualization_limit,
                ):
                    candidate = VisualizationCandidate(
                        seed=seed_result.seed,
                        sample_id=sample_id,
                        region_size=size,
                        gain_over_whole=gain_over_whole,
                        psnr_cs=psnr_cs,
                        psnr_sc=psnr_sc,
                        whole_psnr=whole.psnr,
                        spatial_psnr=measured.psnr,
                        input_tensor=inputs[0].detach().cpu().clone(),
                        target_tensor=targets[0].detach().cpu().clone(),
                        prediction_cs=prediction_cs[0]
                        .detach()
                        .clamp(0.0, data_range)
                        .cpu()
                        .clone(),
                        prediction_sc=prediction_sc[0]
                        .detach()
                        .clamp(0.0, data_range)
                        .cpu()
                        .clone(),
                        oracle_prediction=measured.reconstructed.prediction[0]
                        .detach()
                        .cpu()
                        .clone(),
                        selected_cs_mask=measured.reconstructed.selected_cs_mask
                        .detach()
                        .cpu()
                        .clone(),
                        data_range=data_range,
                    )
                    _consider_visualization_candidate(
                        candidates[size], candidate, limit=visualization_limit
                    )
            image_rows.append(row)
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                print(
                    f"seed={seed_result.seed} validation="
                    f"{batch_index}/{total_batches} sample={sample_id}",
                    flush=True,
                )
            del inputs, targets, prediction_cs, prediction_sc, whole, measured_by_size

    fixed_regression = validate_inference_regression(
        image_rows, seed_result.validation_summary, seed=seed_result.seed
    )
    whole_regression = validate_whole_oracle_regression(
        image_rows, seed_result, prior_oracle_row=prior_oracle_row
    )
    per_seed_row = aggregate_seed_rows(image_rows, seed_result.seed, sizes)
    regression = {
        "seed": seed_result.seed,
        "status": "passed",
        "tolerances": {
            "relative": INFERENCE_REL_TOLERANCE,
            "aggregate_psnr_absolute_db": (
                INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE
            ),
            "aggregate_ssim_absolute": INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE,
            "per_image_psnr_absolute_db": (
                INFERENCE_PER_IMAGE_PSNR_ABS_TOLERANCE
            ),
            "per_image_ssim_absolute": INFERENCE_PER_IMAGE_SSIM_ABS_TOLERANCE,
        },
        "fixed_path_metrics": fixed_regression,
        "whole_oracle_metrics": whole_regression,
        "saved_best_validation_epoch": seed_result.validation_summary["epoch"],
        "saved_best_validation_global_step": seed_result.validation_summary[
            "global_step"
        ],
    }
    del loader, model, checkpoint
    return image_rows, tile_rows, per_seed_row, candidates, regression


def classify_spatial_mix(cs_pixel_rate: float, sc_pixel_rate: float) -> str:
    """Classify a selection mask using the specified strict 90% boundary."""

    if not math.isclose(
        cs_pixel_rate + sc_pixel_rate, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9
    ):
        raise ValueError(
            "CS and SC pixel rates must sum to one, got "
            f"{cs_pixel_rate!r} and {sc_pixel_rate!r}."
        )
    if cs_pixel_rate > 0.90:
        return "mostly_cs"
    if sc_pixel_rate > 0.90:
        return "mostly_sc"
    return "mixed"


def compute_spatial_choice_stability(
    tile_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    region_sizes: Sequence[int],
) -> Dict[str, Any]:
    """Align exact grid coordinates and describe cross-seed local choices."""

    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(
            f"Spatial choice stability requires three unique seeds, got {list(seeds)}."
        )
    by_size: Dict[str, Any] = {}
    for size in region_sizes:
        by_seed: Dict[int, Dict[tuple[Any, ...], str]] = {
            seed: {} for seed in seeds
        }
        for row in tile_rows:
            if int(row["region_size"]) != size:
                continue
            seed = int(row["seed"])
            if seed not in by_seed:
                raise ValueError(f"Unexpected tile seed {seed} for region size {size}.")
            key = (
                str(row["sample_id"]),
                int(row["y0"]),
                int(row["y1"]),
                int(row["x0"]),
                int(row["x1"]),
            )
            if key in by_seed[seed]:
                raise ValueError(
                    f"Duplicate spatial tile for seed {seed}, size {size}, key {key}."
                )
            choice = str(row["choice"])
            if choice not in {"CS", "SC", "tie"}:
                raise ValueError(f"Invalid tile choice {choice!r} for key {key}.")
            by_seed[seed][key] = choice
        reference_seed = seeds[0]
        reference_keys = set(by_seed[reference_seed])
        if not reference_keys:
            raise ValueError(f"No tile choices found for region size {size}.")
        for seed in seeds[1:]:
            keys = set(by_seed[seed])
            if keys != reference_keys:
                raise ValueError(
                    f"Spatial tile set mismatch for region size {size}, seed {seed}: "
                    f"missing={sorted(reference_keys - keys)}, "
                    f"extra={sorted(keys - reference_keys)}."
                )
        pairwise: Dict[str, Any] = {}
        for index, left_seed in enumerate(seeds):
            for right_seed in seeds[index + 1 :]:
                same = different = tie_involved = 0
                for key in reference_keys:
                    left = by_seed[left_seed][key]
                    right = by_seed[right_seed][key]
                    if "tie" in (left, right):
                        tie_involved += 1
                    elif left == right:
                        same += 1
                    else:
                        different += 1
                non_tie = same + different
                pairwise[f"{left_seed}_vs_{right_seed}"] = {
                    "same_choice_count": same,
                    "different_choice_count": different,
                    "tie_involved_count": tie_involved,
                    "non_tie_comparison_count": non_tie,
                    "same_choice_rate": same / non_tie if non_tie else None,
                }
        all_cs = all_sc = all_tie = 0
        any_tie = 0
        for key in reference_keys:
            choices = [by_seed[seed][key] for seed in seeds]
            if "tie" in choices:
                any_tie += 1
            if all(choice == "CS" for choice in choices):
                all_cs += 1
            elif all(choice == "SC" for choice in choices):
                all_sc += 1
            elif all(choice == "tie" for choice in choices):
                all_tie += 1
        all_same = all_cs + all_sc
        by_size[str(size)] = {
            "num_aligned_tiles": len(reference_keys),
            "pairwise": pairwise,
            "all_three_same_choice_count": all_same,
            "all_three_same_choice_rate": all_same / len(reference_keys),
            "all_three_CS_count": all_cs,
            "all_three_SC_count": all_sc,
            "all_three_tie_count": all_tie,
            "any_tie_count": any_tie,
        }
    return {
        "split": "validation",
        "selection_metric": "regional RGB SSE",
        "tie_handling": (
            "ties are reported separately and deterministically reconstructed as CS"
        ),
        "pairwise_rate_denominator": "non_tie_aligned_tiles",
        "seeds": list(seeds),
        "region_sizes": list(region_sizes),
        "by_region_size": by_size,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of empty values.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Gain distribution requires non-empty finite values.")
    output: Dict[str, Any] = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "p05": _percentile(values, 0.05),
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "thresholds": {},
    }
    for threshold in GAIN_THRESHOLDS:
        count = sum(value > threshold for value in values)
        output["thresholds"][f"{threshold:.2f}"] = {
            "count": count,
            "rate": count / len(values),
        }
    return output


def _mean_and_sample_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("Cross-seed aggregation requires at least one value.")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def build_spatial_oracle_summary(
    image_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    region_sizes: Sequence[int],
    regression_checks: Sequence[Mapping[str, Any]],
    run_dirs: Sequence[Path],
) -> Dict[str, Any]:
    """Build required descriptive aggregation with no significance claims."""

    aggregate_fields = [
        "best_fixed_psnr",
        "whole_oracle_psnr",
        "whole_gain_over_fixed",
    ]
    previous_label = "whole"
    for size in region_sizes:
        aggregate_fields.extend(
            [
                f"oracle_{size}_psnr",
                f"gain_{size}_over_fixed",
                f"gain_{size}_over_whole",
            ]
        )
        if previous_label != "whole":
            aggregate_fields.append(f"gain_{size}_over_{previous_label}")
        previous_label = str(size)
    aggregates = {
        field: _mean_and_sample_std(
            [float(row[field]) for row in per_seed_rows]
        )
        for field in aggregate_fields
    }
    gain_distributions: Dict[str, Any] = {}
    mixing: Dict[str, Any] = {}
    for size in region_sizes:
        gain_distributions[str(size)] = {
            "gain_over_best_fixed": _distribution(
                [
                    float(row[f"oracle_{size}_gain_over_best_fixed"])
                    for row in image_rows
                ]
            ),
            "gain_over_whole": _distribution(
                [
                    float(row[f"oracle_{size}_gain_over_whole"])
                    for row in image_rows
                ]
            ),
        }
        categories = [
            classify_spatial_mix(
                float(row[f"pixels_{size}_cs_rate"]),
                float(row[f"pixels_{size}_sc_rate"]),
            )
            for row in image_rows
        ]
        total = len(categories)
        mixing[str(size)] = {
            "mostly_cs_count": categories.count("mostly_cs"),
            "mostly_sc_count": categories.count("mostly_sc"),
            "mixed_image_count": categories.count("mixed"),
            "mostly_cs_rate": categories.count("mostly_cs") / total,
            "mostly_sc_rate": categories.count("mostly_sc") / total,
            "mixed_image_rate": categories.count("mixed") / total,
            "definition": (
                "mostly_cs if CS pixel rate > 0.90; mostly_sc if SC pixel "
                "rate > 0.90; otherwise mixed"
            ),
        }
    return {
        "split": "validation",
        "selection_metric": "regional RGB SSE",
        "region_sizes": list(region_sizes),
        "grid": "non_overlapping_top_left_aligned",
        "border_handling": "natural_truncation_no_resize_no_padding",
        "prediction_clamp": [0.0, "data_range"],
        "tie_reconstruction": "choose_CS_and_report_tie_separately",
        "pixel_rate_semantics": "actual_pixels; tie regions count as selected CS",
        "metric_semantics": "mean_of_per_image_rgb_metrics",
        "per_image_fixed_gain_reference": (
            "the per-image metric of that seed's dataset-selected best fixed order"
        ),
        "seeds": list(seeds),
        "num_seed_image_rows": len(image_rows),
        "run_directories": [str(path) for path in run_dirs],
        "per_seed": [dict(row) for row in per_seed_rows],
        "cross_seed_mean_and_sample_std": aggregates,
        "per_image_gain_distributions": gain_distributions,
        "spatial_mixing": mixing,
        "inference_regression_checks": [dict(row) for row in regression_checks],
    }


def _per_image_fieldnames(region_sizes: Sequence[int]) -> list[str]:
    fields = [
        "seed",
        "sample_id",
        "input_relative_path",
        "gt_relative_path",
        "height",
        "width",
        "psnr_cs",
        "psnr_sc",
        "ssim_cs",
        "ssim_sc",
        "whole_oracle_psnr",
        "whole_oracle_ssim",
        "whole_oracle_selected_order",
        "whole_oracle_gain_over_best_fixed",
    ]
    for size in region_sizes:
        fields.extend(
            [
                f"oracle_{size}_psnr",
                f"oracle_{size}_ssim",
                f"oracle_{size}_gain_over_best_fixed",
                f"oracle_{size}_gain_over_whole",
                f"tiles_{size}_total",
                f"tiles_{size}_cs",
                f"tiles_{size}_sc",
                f"tiles_{size}_tie",
                f"pixels_{size}_cs_rate",
                f"pixels_{size}_sc_rate",
            ]
        )
    return fields


def _per_seed_fieldnames(region_sizes: Sequence[int]) -> list[str]:
    fields = [
        "seed",
        "num_samples",
        "mean_psnr_cs",
        "mean_psnr_sc",
        "mean_ssim_cs",
        "mean_ssim_sc",
        "best_fixed_order",
        "best_fixed_psnr",
        "best_fixed_ssim",
        "whole_oracle_psnr",
        "whole_oracle_ssim",
        "whole_gain_over_fixed",
    ]
    previous = "whole"
    for size in region_sizes:
        fields.extend(
            [
                f"oracle_{size}_psnr",
                f"oracle_{size}_ssim",
                f"gain_{size}_over_fixed",
                f"gain_{size}_over_whole",
            ]
        )
        if previous != "whole":
            fields.append(f"gain_{size}_over_{previous}")
        fields.extend(
            [f"cs_pixel_rate_{size}", f"sc_pixel_rate_{size}"]
        )
        previous = str(size)
    return fields


def build_analysis_markdown(
    summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    *,
    region_sizes: Sequence[int],
) -> str:
    """Render the required objective, human-readable validation report."""

    lines = [
        "# V2 Multi-scale Validation Spatial Oracle Analysis",
        "",
        "This report uses only the V2 shared-order best-checkpoint validation split. "
        "Regional choices minimize RGB SSE after prediction clamping; every PSNR and "
        "SSIM value is measured on a complete reconstructed image.",
        "",
        "## Table 1: Per-seed PSNR",
        "",
    ]
    headers = ["Seed", "Best fixed", "Whole Oracle", *(str(size) for size in region_sizes)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    for row in summary["per_seed"]:
        values = [
            str(row["seed"]),
            f"{float(row['best_fixed_psnr']):.6f}",
            (
                f"{float(row['whole_oracle_psnr']):.6f} "
                f"(+{float(row['whole_gain_over_fixed']):.6f})"
            ),
        ]
        values.extend(
            f"{float(row[f'oracle_{size}_psnr']):.6f} "
            f"(+{float(row[f'gain_{size}_over_fixed']):.6f})"
            for size in region_sizes
        )
        lines.append("| " + " | ".join(values) + " |")

    aggregates = summary["cross_seed_mean_and_sample_std"]
    lines.extend(
        [
            "",
            "## Table 2: Three-seed mean ± sample std",
            "",
            "| Selection granularity | PSNR | Gain over fixed | Gain over whole |",
            "|---|---:|---:|---:|",
        ]
    )

    def rendered(field: str) -> str:
        values = aggregates[field]
        return f"{float(values['mean']):.6f} ± {float(values['sample_std']):.6f}"

    lines.append(f"| Best fixed | {rendered('best_fixed_psnr')} | 0 | n/a |")
    lines.append(
        f"| Whole image | {rendered('whole_oracle_psnr')} | "
        f"{rendered('whole_gain_over_fixed')} | 0 |"
    )
    for size in region_sizes:
        lines.append(
            f"| {size}×{size} | {rendered(f'oracle_{size}_psnr')} | "
            f"{rendered(f'gain_{size}_over_fixed')} | "
            f"{rendered(f'gain_{size}_over_whole')} |"
        )

    lines.extend(
        [
            "",
            "## Table 3: Spatial mixing",
            "",
            "| Region size | Mostly CS | Mostly SC | Mixed |",
            "|---:|---:|---:|---:|",
        ]
    )
    for size in region_sizes:
        values = summary["spatial_mixing"][str(size)]
        lines.append(
            f"| {size} | {values['mostly_cs_count']} | "
            f"{values['mostly_sc_count']} | {values['mixed_image_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table 4: Cross-seed tile preference agreement",
            "",
            "Rates use non-tie aligned tiles; tie-involved comparisons are reported "
            "separately in `spatial_choice_stability.json`.",
            "",
            "| Region size | 1234–3407 | 1234–3520 | 3407–3520 | All-three |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    seed_values = list(summary["seeds"])
    pairs = [
        f"{seed_values[0]}_vs_{seed_values[1]}",
        f"{seed_values[0]}_vs_{seed_values[2]}",
        f"{seed_values[1]}_vs_{seed_values[2]}",
    ]
    for size in region_sizes:
        values = stability["by_region_size"][str(size)]
        rates = []
        for pair in pairs:
            rate = values["pairwise"][pair]["same_choice_rate"]
            rates.append("n/a" if rate is None else f"{100.0 * float(rate):.2f}%")
        lines.append(
            f"| {size} | {rates[0]} | {rates[1]} | {rates[2]} | "
            f"{100.0 * float(values['all_three_same_choice_rate']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "If spatial Oracle headroom is substantially larger than the whole-image "
            "Oracle and local choices are reasonably reproducible across seeds, a "
            "spatial order-aware model may merit further study. If the measured spatial "
            "upper bound remains limited, a spatial gate has limited attainable PSNR "
            "headroom under the current C/S operators. No materiality threshold or "
            "statistical-significance claim is imposed here.",
            "",
        ]
    )
    return "\n".join(lines)


def _tensor_to_pil(tensor: Any, *, data_range: float) -> Any:
    import numpy as np
    from PIL import Image

    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, data_range)
        .div(data_range)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def _mask_to_pil(mask: Any) -> Any:
    import numpy as np
    from PIL import Image

    boolean = mask.detach().cpu().numpy().astype(bool)
    array = np.empty((*boolean.shape, 3), dtype=np.uint8)
    array[boolean] = np.array([0, 170, 255], dtype=np.uint8)  # CS: blue
    array[~boolean] = np.array([255, 140, 0], dtype=np.uint8)  # SC: orange
    return Image.fromarray(array, mode="RGB")


def save_visualization_panels(
    candidates: Mapping[int, Sequence[VisualizationCandidate]],
    output_dir: Path,
) -> list[Path]:
    """Save deterministic top-gain panels only, never the full prediction set."""

    from PIL import Image, ImageDraw

    destination = output_dir / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    labels = ("Input", "GT", "C→S", "S→C", "Spatial Oracle", "Selection Mask")
    for size in sorted(candidates, reverse=True):
        ordered = sorted(
            candidates[size],
            key=lambda item: (-item.gain_over_whole, item.seed, item.sample_id),
        )
        for rank, candidate in enumerate(ordered, start=1):
            images = (
                _tensor_to_pil(candidate.input_tensor, data_range=candidate.data_range),
                _tensor_to_pil(candidate.target_tensor, data_range=candidate.data_range),
                _tensor_to_pil(candidate.prediction_cs, data_range=candidate.data_range),
                _tensor_to_pil(candidate.prediction_sc, data_range=candidate.data_range),
                _tensor_to_pil(candidate.oracle_prediction, data_range=candidate.data_range),
                _mask_to_pil(candidate.selected_cs_mask),
            )
            width, height = images[0].size
            header_height = 58
            label_height = 18
            panel = Image.new(
                "RGB", (width * len(images), header_height + label_height + height), "white"
            )
            draw = ImageDraw.Draw(panel)
            title = (
                f"seed={candidate.seed}  sample={candidate.sample_id}  "
                f"region={size}  gain_over_whole={candidate.gain_over_whole:.6f} dB\n"
                f"PSNR CS={candidate.psnr_cs:.6f}  SC={candidate.psnr_sc:.6f}  "
                f"Whole={candidate.whole_psnr:.6f}  Spatial={candidate.spatial_psnr:.6f}"
            )
            draw.text((4, 4), title, fill="black")
            for column, (label, image) in enumerate(zip(labels, images)):
                x = column * width
                draw.text((x + 4, header_height), label, fill="black")
                panel.paste(image, (x, header_height + label_height))
            safe_sample = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in candidate.sample_id
            )
            path = destination / (
                f"region_{size}_rank_{rank:02d}_seed{candidate.seed}_{safe_sample}.png"
            )
            panel.save(path, format="PNG")
            written.append(path)
    return written


def write_analysis_outputs(
    *,
    output_dir: Path | str,
    image_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    candidates: Mapping[int, Sequence[VisualizationCandidate]],
    region_sizes: Sequence[int],
) -> Dict[str, Path]:
    """Write all required CSV/JSON/Markdown outputs after every check passes."""

    destination = Path(output_dir).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "spatial_oracle_per_image.csv": destination / "spatial_oracle_per_image.csv",
        "spatial_oracle_per_seed.csv": destination / "spatial_oracle_per_seed.csv",
        "spatial_oracle_summary.json": destination / "spatial_oracle_summary.json",
        "spatial_choice_stability.json": destination / "spatial_choice_stability.json",
        "analysis_summary.md": destination / "analysis_summary.md",
    }
    atomic_write_csv(
        outputs["spatial_oracle_per_image.csv"],
        image_rows,
        _per_image_fieldnames(region_sizes),
    )
    atomic_write_csv(
        outputs["spatial_oracle_per_seed.csv"],
        per_seed_rows,
        _per_seed_fieldnames(region_sizes),
    )
    atomic_write_json(outputs["spatial_oracle_summary.json"], summary)
    atomic_write_json(outputs["spatial_choice_stability.json"], stability)
    outputs["analysis_summary.md"].write_text(
        build_analysis_markdown(summary, stability, region_sizes=region_sizes),
        encoding="utf-8",
    )
    save_visualization_panels(candidates, destination)
    return outputs


def _prior_oracle_path(output_dir: Path) -> Optional[Path]:
    candidates = (
        output_dir.parent / "v2_validation_order/oracle_per_seed.csv",
        Path("analysis_results/v2_validation_order/oracle_per_seed.csv")
        .expanduser()
        .resolve(strict=False),
    )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    return None


def run_analysis(
    *,
    experiments_root: Path | str = Path("experiments"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    region_sizes: Sequence[int] = DEFAULT_REGION_SIZES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    device_name: str = "auto",
    split: str = "validation",
) -> Dict[str, Path]:
    """Run one-model-at-a-time inference and write validation-only analysis."""

    require_validation_split(split)
    sizes = validate_region_sizes(region_sizes)
    seed_values = tuple(seeds)
    if len(seed_values) != 3 or len(set(seed_values)) != 3:
        raise ValueError(
            f"Spatial Oracle analysis requires exactly three unique seeds, got {seed_values}."
        )
    import torch

    device = resolve_device(device_name, torch)
    root = Path(experiments_root).expanduser().resolve(strict=False)
    seed_results = [
        load_seed_validation_result(discover_seed_run(root, seed=seed), seed=seed)
        for seed in seed_values
    ]
    validate_configs_across_seeds(seed_results)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    prior_path = _prior_oracle_path(destination)
    prior_rows = _read_prior_oracle_rows(prior_path) if prior_path else {}
    if prior_path:
        missing_prior = [seed for seed in seed_values if seed not in prior_rows]
        if missing_prior:
            raise ValueError(
                f"Existing whole-image Oracle CSV {prior_path} is missing seeds "
                f"{missing_prior}."
            )

    all_image_rows: list[Dict[str, Any]] = []
    all_tile_rows: list[Dict[str, Any]] = []
    per_seed_rows: list[Dict[str, Any]] = []
    regressions: list[Dict[str, Any]] = []
    top_candidates: Dict[int, list[VisualizationCandidate]] = {
        size: [] for size in sizes
    }
    for seed_result in seed_results:
        print(
            f"Loading seed {seed_result.seed} best-validation checkpoint from "
            f"{seed_result.run_dir}",
            flush=True,
        )
        image_rows, tile_rows, per_seed, candidates, regression = analyze_seed(
            seed_result,
            region_sizes=sizes,
            device=device,
            prior_oracle_row=prior_rows.get(seed_result.seed),
        )
        all_image_rows.extend(image_rows)
        all_tile_rows.extend(tile_rows)
        per_seed_rows.append(per_seed)
        regressions.append(regression)
        for size in sizes:
            for candidate in candidates[size]:
                _consider_visualization_candidate(
                    top_candidates[size],
                    candidate,
                    limit=TOP_VISUALIZATIONS_PER_SCALE,
                )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stability = compute_spatial_choice_stability(
        all_tile_rows, seeds=seed_values, region_sizes=sizes
    )
    summary = build_spatial_oracle_summary(
        all_image_rows,
        per_seed_rows,
        seeds=seed_values,
        region_sizes=sizes,
        regression_checks=regressions,
        run_dirs=[result.run_dir for result in seed_results],
    )
    return write_analysis_outputs(
        output_dir=destination,
        image_rows=all_image_rows,
        per_seed_rows=per_seed_rows,
        summary=summary,
        stability=stability,
        candidates=top_candidates,
        region_sizes=sizes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--region-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_REGION_SIZES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run_analysis(
            experiments_root=args.experiments_root,
            seeds=args.seeds,
            region_sizes=args.region_sizes,
            output_dir=args.output_dir,
            device_name=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote V2 validation spatial Oracle analysis to {args.output_dir}")
    for name in outputs:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
