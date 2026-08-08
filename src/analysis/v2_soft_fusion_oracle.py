"""Convex regional soft-fusion Oracle for V2 shared-order validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.analysis import v2_spatial_oracle as spatial
from src.analysis.v2_order_validation import (
    DEFAULT_SEEDS,
    SeedValidationResult,
    discover_seed_run,
    load_seed_validation_result,
    validate_configs_across_seeds,
)
from src.common.experiment.logging_utils import atomic_write_csv, atomic_write_json


DEFAULT_REGION_SIZES = spatial.DEFAULT_REGION_SIZES
DEFAULT_OUTPUT_DIR = Path("analysis_results/v2_validation_soft_fusion_oracle")
ALPHA_DENOMINATOR_EPS = 1.0e-12
ALPHA_NEAR_BOUNDARY = 0.05
GAIN_THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.10)
TOP_VISUALIZATIONS_PER_GRANULARITY = 5


@dataclass(frozen=True)
class SoftRegion:
    """One region's analytic convex coefficient and diagnostics."""

    bounds: spatial.RegionBounds
    alpha_raw: float
    alpha_star: float
    denominator: float
    degenerate: bool

    @property
    def pixels(self) -> int:
        return self.bounds.pixels


@dataclass(frozen=True)
class ReconstructedSoftOracle:
    """A complete convex-fusion prediction and its per-region coefficients."""

    prediction: Any
    alpha_map: Any
    regions: tuple[SoftRegion, ...]
    total_sse: float


@dataclass(frozen=True)
class MeasuredSoftOracle:
    """A soft reconstruction measured with the project's whole-image metrics."""

    reconstructed: ReconstructedSoftOracle
    psnr: float
    ssim: float


@dataclass(frozen=True)
class SoftVisualizationCandidate:
    seed: int
    sample_id: str
    granularity: str
    gain_over_hard: float
    psnr_cs: float
    psnr_sc: float
    hard_psnr: float
    soft_psnr: float
    mean_alpha: float
    interior_pixel_rate: float
    input_tensor: Any
    target_tensor: Any
    prediction_cs: Any
    prediction_sc: Any
    hard_prediction: Any
    soft_prediction: Any
    alpha_map: Any
    data_range: float


def alpha_category(alpha: float) -> str:
    if alpha <= ALPHA_NEAR_BOUNDARY:
        return "near_SC"
    if alpha >= 1.0 - ALPHA_NEAR_BOUNDARY:
        return "near_CS"
    return "soft"


def reconstruct_soft_fusion_oracle(
    prediction_cs: Any,
    prediction_sc: Any,
    target: Any,
    *,
    region_size: Optional[int],
    data_range: float = 1.0,
    denominator_eps: float = ALPHA_DENOMINATOR_EPS,
) -> ReconstructedSoftOracle:
    """Analytically minimize regional RGB SSE over one scalar alpha in [0,1]."""

    import torch

    spatial._validate_oracle_tensors(
        prediction_cs, prediction_sc, target, data_range=data_range
    )
    if not math.isfinite(float(denominator_eps)) or denominator_eps < 0.0:
        raise ValueError(
            f"denominator_eps must be finite and >= 0, got {denominator_eps!r}."
        )
    clamped_cs = prediction_cs.clamp(0.0, float(data_range))
    clamped_sc = prediction_sc.clamp(0.0, float(data_range))
    height, width = target.shape[-2:]
    bounds_values = (
        (spatial.RegionBounds(0, height, 0, width),)
        if region_size is None
        else spatial.region_grid(height, width, region_size)
    )
    prediction = torch.empty_like(clamped_cs)
    alpha_map = torch.empty(
        (height, width), dtype=torch.float32, device=target.device
    )
    regions: list[SoftRegion] = []
    total_sse = 0.0
    for bounds in bounds_values:
        slices = (..., slice(bounds.y0, bounds.y1), slice(bounds.x0, bounds.x1))
        a = clamped_cs[slices]
        b = clamped_sc[slices]
        g = target[slices]
        # Only scalar reductions use float64. Predictions remain in their
        # inference dtype and no dataset-sized float64 cache is created.
        difference = a.double() - b.double()
        residual = g.double() - b.double()
        numerator = float((residual * difference).sum())
        denominator = float((difference * difference).sum())
        if denominator <= denominator_eps:
            alpha_raw = 0.5
            alpha_star = 0.5
            degenerate = True
        else:
            alpha_raw = numerator / denominator
            alpha_star = min(1.0, max(0.0, alpha_raw))
            degenerate = False
        fused = alpha_star * a + (1.0 - alpha_star) * b
        prediction[slices] = fused
        alpha_map[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = alpha_star
        total_sse += float(((fused.double() - g.double()) ** 2).sum())
        regions.append(
            SoftRegion(
                bounds=bounds,
                alpha_raw=alpha_raw,
                alpha_star=alpha_star,
                denominator=denominator,
                degenerate=degenerate,
            )
        )
    return ReconstructedSoftOracle(
        prediction=prediction,
        alpha_map=alpha_map,
        regions=tuple(regions),
        total_sse=total_sse,
    )


def compute_soft_fusion_oracle(
    prediction_cs: Any,
    prediction_sc: Any,
    target: Any,
    *,
    region_size: Optional[int],
    metrics_config: Mapping[str, Any],
) -> MeasuredSoftOracle:
    """Reconstruct the complete convex Oracle, then compute whole-image metrics."""

    reconstructed = reconstruct_soft_fusion_oracle(
        prediction_cs,
        prediction_sc,
        target,
        region_size=region_size,
        data_range=float(metrics_config["data_range"]),
    )
    psnr, ssim = spatial._path_metrics(
        reconstructed.prediction, target, metrics_config
    )
    return MeasuredSoftOracle(
        reconstructed=reconstructed,
        psnr=psnr,
        ssim=ssim,
    )


def require_soft_not_worse_than_hard(
    soft_oracle: MeasuredSoftOracle,
    hard_oracle: spatial.MeasuredOracle,
    *,
    label: str,
) -> None:
    """Enforce that convex alpha endpoints contain every hard choice."""

    tolerance = max(
        spatial.MONOTONIC_SSE_ABS_TOLERANCE,
        spatial.MONOTONIC_REL_TOLERANCE
        * max(abs(hard_oracle.reconstructed.total_sse), 1.0),
    )
    if (
        soft_oracle.reconstructed.total_sse
        > hard_oracle.reconstructed.total_sse + tolerance
    ):
        raise ValueError(
            f"Soft Oracle is worse than Hard Oracle at {label}: soft SSE="
            f"{soft_oracle.reconstructed.total_sse!r}, hard SSE="
            f"{hard_oracle.reconstructed.total_sse!r}."
        )
    if (
        soft_oracle.psnr + spatial.MONOTONIC_PSNR_ABS_TOLERANCE
        < hard_oracle.psnr
    ):
        raise ValueError(
            f"Soft Oracle PSNR is below Hard Oracle at {label}: soft="
            f"{soft_oracle.psnr!r}, hard={hard_oracle.psnr!r}, "
            f"abs_diff={hard_oracle.psnr - soft_oracle.psnr!r}, "
            f"allowed={spatial.MONOTONIC_PSNR_ABS_TOLERANCE!r}."
        )


def summarize_alpha_regions(regions: Sequence[SoftRegion]) -> Dict[str, Any]:
    """Compute unweighted region statistics and actual-pixel-weighted rates."""

    if not regions:
        raise ValueError("Alpha aggregation requires at least one region.")
    alphas = [region.alpha_star for region in regions]
    total_regions = len(regions)
    total_pixels = sum(region.pixels for region in regions)

    def count_region(predicate: Any) -> int:
        return sum(bool(predicate(region)) for region in regions)

    def count_pixels(predicate: Any) -> int:
        return sum(region.pixels for region in regions if predicate(region))

    predicates = {
        "near_sc": lambda region: region.alpha_star <= 0.05,
        "near_cs": lambda region: region.alpha_star >= 0.95,
        "interior_soft": lambda region: 0.05 < region.alpha_star < 0.95,
        "interior_0p10_0p90": lambda region: 0.10 < region.alpha_star < 0.90,
        "interior_0p25_0p75": lambda region: 0.25 < region.alpha_star < 0.75,
        "raw_below_zero": lambda region: region.alpha_raw < 0.0,
        "raw_above_one": lambda region: region.alpha_raw > 1.0,
        "raw_inside_unit_interval": lambda region: 0.0 <= region.alpha_raw <= 1.0,
        "degenerate": lambda region: region.degenerate,
    }

    def percentile(quantile: float) -> float:
        ordered = sorted(alphas)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    output: Dict[str, Any] = {
        "num_regions": total_regions,
        "num_pixels": total_pixels,
        "mean_alpha": statistics.fmean(alphas),
        "median_alpha": statistics.median(alphas),
        "std_alpha": statistics.stdev(alphas) if len(alphas) >= 2 else 0.0,
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
    }
    for name, predicate in predicates.items():
        region_count = count_region(predicate)
        pixel_count = count_pixels(predicate)
        output[f"{name}_region_count"] = region_count
        output[f"{name}_region_rate"] = region_count / total_regions
        output[f"{name}_pixel_count"] = pixel_count
        output[f"{name}_pixel_rate"] = pixel_count / total_pixels
    return output


def _alpha_records(
    *,
    seed: int,
    sample_id: str,
    granularity: str,
    regions: Sequence[SoftRegion],
) -> list[Dict[str, Any]]:
    return [
        {
            "seed": seed,
            "sample_id": sample_id,
            "granularity": granularity,
            "y0": region.bounds.y0,
            "y1": region.bounds.y1,
            "x0": region.bounds.x0,
            "x1": region.bounds.x1,
            "pixels": region.pixels,
            "alpha_raw": region.alpha_raw,
            "alpha_star": region.alpha_star,
            "category": alpha_category(region.alpha_star),
            "degenerate": region.degenerate,
        }
        for region in regions
    ]


def _candidate_key(candidate: SoftVisualizationCandidate) -> tuple[Any, ...]:
    return (-candidate.gain_over_hard, candidate.seed, candidate.sample_id)


def _candidate_needed(
    bucket: Sequence[SoftVisualizationCandidate],
    *,
    gain: float,
    seed: int,
    sample_id: str,
    limit: int,
) -> bool:
    if limit <= 0:
        return False
    if len(bucket) < limit:
        return True
    return (-gain, seed, sample_id) < _candidate_key(bucket[-1])


def _consider_candidate(
    bucket: list[SoftVisualizationCandidate],
    candidate: SoftVisualizationCandidate,
    *,
    limit: int,
) -> None:
    bucket.append(candidate)
    bucket.sort(key=_candidate_key)
    del bucket[limit:]


class SoftFusionCollector:
    """Sample callback plugged into the proven hard-spatial inference loop."""

    def __init__(
        self,
        region_sizes: Sequence[int],
        *,
        visualization_limit: int = TOP_VISUALIZATIONS_PER_GRANULARITY,
    ) -> None:
        self.region_sizes = spatial.validate_region_sizes(region_sizes)
        self.visualization_limit = visualization_limit
        self.image_rows: list[Dict[str, Any]] = []
        self.alpha_records: list[Dict[str, Any]] = []
        self.candidates: Dict[str, list[SoftVisualizationCandidate]] = {
            "whole": [],
            **{str(size): [] for size in self.region_sizes},
        }

    def __call__(self, **context: Any) -> None:
        seed_result: SeedValidationResult = context["seed_result"]
        sample_id = str(context["sample_id"])
        prediction_cs = context["prediction_cs"]
        prediction_sc = context["prediction_sc"]
        targets = context["targets"]
        metrics = context["metrics_config"]
        hard_whole: spatial.MeasuredOracle = context["hard_whole"]
        hard_by_size: Mapping[int, spatial.MeasuredOracle] = context["hard_by_size"]

        soft_whole = compute_soft_fusion_oracle(
            prediction_cs,
            prediction_sc,
            targets,
            region_size=None,
            metrics_config=metrics,
        )
        soft_by_size = {
            size: compute_soft_fusion_oracle(
                prediction_cs,
                prediction_sc,
                targets,
                region_size=size,
                metrics_config=metrics,
            )
            for size in self.region_sizes
        }
        require_soft_not_worse_than_hard(
            soft_whole, hard_whole, label="whole"
        )
        for size in self.region_sizes:
            require_soft_not_worse_than_hard(
                soft_by_size[size], hard_by_size[size], label=str(size)
            )
        spatial.require_oracle_monotonicity(
            [("whole", soft_whole)]
            + [(str(size), soft_by_size[size]) for size in self.region_sizes]
        )

        whole_alpha = summarize_alpha_regions(soft_whole.reconstructed.regions)
        height, width = targets.shape[-2:]
        row: Dict[str, Any] = {
            "seed": seed_result.seed,
            "sample_id": sample_id,
            "input_relative_path": str(context["input_relative_path"]),
            "gt_relative_path": str(context["gt_relative_path"]),
            "height": int(height),
            "width": int(width),
            "psnr_cs": float(context["psnr_cs"]),
            "psnr_sc": float(context["psnr_sc"]),
            "ssim_cs": float(context["ssim_cs"]),
            "ssim_sc": float(context["ssim_sc"]),
            "hard_whole_psnr": hard_whole.psnr,
            "soft_whole_psnr": soft_whole.psnr,
            "soft_whole_ssim": soft_whole.ssim,
            "alpha_whole": soft_whole.reconstructed.regions[0].alpha_star,
            "alpha_whole_raw": soft_whole.reconstructed.regions[0].alpha_raw,
            "alpha_whole_degenerate": soft_whole.reconstructed.regions[0].degenerate,
            "soft_whole_gain_over_hard": soft_whole.psnr - hard_whole.psnr,
        }
        measured_values: list[
            tuple[str, MeasuredSoftOracle, spatial.MeasuredOracle, Dict[str, Any]]
        ] = [("whole", soft_whole, hard_whole, whole_alpha)]
        self.alpha_records.extend(
            _alpha_records(
                seed=seed_result.seed,
                sample_id=sample_id,
                granularity="whole",
                regions=soft_whole.reconstructed.regions,
            )
        )
        for size in self.region_sizes:
            soft = soft_by_size[size]
            hard = hard_by_size[size]
            alpha = summarize_alpha_regions(soft.reconstructed.regions)
            row.update(
                {
                    f"hard_{size}_psnr": hard.psnr,
                    f"soft_{size}_psnr": soft.psnr,
                    f"soft_{size}_ssim": soft.ssim,
                    f"soft_{size}_gain_over_hard": soft.psnr - hard.psnr,
                    f"soft_{size}_gain_over_soft_whole": (
                        soft.psnr - soft_whole.psnr
                    ),
                    f"alpha_{size}_mean": alpha["mean_alpha"],
                    f"alpha_{size}_std": alpha["std_alpha"],
                    f"alpha_{size}_interior_pixel_rate": alpha[
                        "interior_soft_pixel_rate"
                    ],
                    f"alpha_{size}_near_cs_pixel_rate": alpha[
                        "near_cs_pixel_rate"
                    ],
                    f"alpha_{size}_near_sc_pixel_rate": alpha[
                        "near_sc_pixel_rate"
                    ],
                    f"alpha_{size}_degenerate_region_count": alpha[
                        "degenerate_region_count"
                    ],
                    f"alpha_{size}_degenerate_pixel_rate": alpha[
                        "degenerate_pixel_rate"
                    ],
                }
            )
            self.alpha_records.extend(
                _alpha_records(
                    seed=seed_result.seed,
                    sample_id=sample_id,
                    granularity=str(size),
                    regions=soft.reconstructed.regions,
                )
            )
            measured_values.append((str(size), soft, hard, alpha))
        self.image_rows.append(row)

        data_range = float(metrics["data_range"])
        inputs = context["inputs"]
        for granularity, soft, hard, alpha in measured_values:
            gain = soft.psnr - hard.psnr
            bucket = self.candidates[granularity]
            if not _candidate_needed(
                bucket,
                gain=gain,
                seed=seed_result.seed,
                sample_id=sample_id,
                limit=self.visualization_limit,
            ):
                continue
            candidate = SoftVisualizationCandidate(
                seed=seed_result.seed,
                sample_id=sample_id,
                granularity=granularity,
                gain_over_hard=gain,
                psnr_cs=float(context["psnr_cs"]),
                psnr_sc=float(context["psnr_sc"]),
                hard_psnr=hard.psnr,
                soft_psnr=soft.psnr,
                mean_alpha=float(alpha["mean_alpha"]),
                interior_pixel_rate=float(alpha["interior_soft_pixel_rate"]),
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
                hard_prediction=hard.reconstructed.prediction[0]
                .detach()
                .cpu()
                .clone(),
                soft_prediction=soft.reconstructed.prediction[0]
                .detach()
                .cpu()
                .clone(),
                alpha_map=soft.reconstructed.alpha_map.detach().cpu().clone(),
                data_range=data_range,
            )
            _consider_candidate(
                bucket, candidate, limit=self.visualization_limit
            )


def _regions_from_records(records: Sequence[Mapping[str, Any]]) -> list[SoftRegion]:
    return [
        SoftRegion(
            bounds=spatial.RegionBounds(
                int(row["y0"]),
                int(row["y1"]),
                int(row["x0"]),
                int(row["x1"]),
            ),
            alpha_raw=float(row["alpha_raw"]),
            alpha_star=float(row["alpha_star"]),
            denominator=0.0,
            degenerate=bool(row["degenerate"]),
        )
        for row in records
    ]


def aggregate_soft_seed_rows(
    image_rows: list[Dict[str, Any]],
    hard_per_seed: Mapping[str, Any],
    alpha_records: Sequence[Mapping[str, Any]],
    *,
    region_sizes: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate soft metrics as means of per-image RGB metrics."""

    if not image_rows:
        raise ValueError("Soft fusion seed aggregation requires image rows.")
    seed = int(hard_per_seed["seed"])
    best_order = str(hard_per_seed["best_fixed_order"])
    fixed_field = "psnr_cs" if best_order == "CS" else "psnr_sc"
    for row in image_rows:
        fixed_psnr = float(row[fixed_field])
        row["soft_whole_gain_over_best_fixed"] = (
            float(row["soft_whole_psnr"]) - fixed_psnr
        )
        for size in region_sizes:
            row[f"soft_{size}_gain_over_best_fixed"] = (
                float(row[f"soft_{size}_psnr"]) - fixed_psnr
            )

    output: Dict[str, Any] = {
        "seed": seed,
        "num_samples": len(image_rows),
        "mean_psnr_cs": statistics.fmean(float(row["psnr_cs"]) for row in image_rows),
        "mean_psnr_sc": statistics.fmean(float(row["psnr_sc"]) for row in image_rows),
        "best_fixed_order": best_order,
        "best_fixed_psnr": float(hard_per_seed["best_fixed_psnr"]),
        "best_fixed_ssim": float(hard_per_seed["best_fixed_ssim"]),
        "hard_whole_psnr": float(hard_per_seed["whole_oracle_psnr"]),
        "hard_whole_ssim": float(hard_per_seed["whole_oracle_ssim"]),
        "soft_whole_psnr": statistics.fmean(
            float(row["soft_whole_psnr"]) for row in image_rows
        ),
        "soft_whole_ssim": statistics.fmean(
            float(row["soft_whole_ssim"]) for row in image_rows
        ),
    }
    output["soft_whole_gain_over_hard"] = (
        output["soft_whole_psnr"] - output["hard_whole_psnr"]
    )
    output["soft_whole_gain_over_fixed"] = (
        output["soft_whole_psnr"] - output["best_fixed_psnr"]
    )
    for size in region_sizes:
        output[f"hard_{size}_psnr"] = float(
            hard_per_seed[f"oracle_{size}_psnr"]
        )
        output[f"hard_{size}_ssim"] = float(
            hard_per_seed[f"oracle_{size}_ssim"]
        )
        output[f"soft_{size}_psnr"] = statistics.fmean(
            float(row[f"soft_{size}_psnr"]) for row in image_rows
        )
        output[f"soft_{size}_ssim"] = statistics.fmean(
            float(row[f"soft_{size}_ssim"]) for row in image_rows
        )
        output[f"soft_{size}_gain_over_hard"] = (
            output[f"soft_{size}_psnr"] - output[f"hard_{size}_psnr"]
        )
        output[f"soft_{size}_gain_over_fixed"] = (
            output[f"soft_{size}_psnr"] - output["best_fixed_psnr"]
        )
        output[f"soft_{size}_gain_over_soft_whole"] = (
            output[f"soft_{size}_psnr"] - output["soft_whole_psnr"]
        )

    for granularity in ("whole", *(str(size) for size in region_sizes)):
        selected = [
            row for row in alpha_records if str(row["granularity"]) == granularity
        ]
        alpha = summarize_alpha_regions(_regions_from_records(selected))
        prefix = f"alpha_{granularity}"
        for field in (
            "mean_alpha",
            "median_alpha",
            "std_alpha",
            "near_cs_region_rate",
            "near_sc_region_rate",
            "interior_soft_region_rate",
            "near_cs_pixel_rate",
            "near_sc_pixel_rate",
            "interior_soft_pixel_rate",
            "interior_0p10_0p90_region_rate",
            "interior_0p10_0p90_pixel_rate",
            "interior_0p25_0p75_region_rate",
            "interior_0p25_0p75_pixel_rate",
            "raw_below_zero_region_rate",
            "raw_above_one_region_rate",
            "raw_inside_unit_interval_region_rate",
            "degenerate_region_rate",
            "degenerate_pixel_rate",
        ):
            output[f"{prefix}_{field}"] = alpha[field]
    return output


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Return Pearson r without SciPy; constant inputs have undefined r."""

    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must have the same non-zero length.")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(centered_left, centered_right)
    ) / denominator


def compute_alpha_stability(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    granularities: Sequence[str],
) -> Dict[str, Any]:
    """Align alpha by exact sample/grid coordinates across three seeds."""

    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Alpha stability requires exactly three unique seeds.")
    output: Dict[str, Any] = {}
    for granularity in granularities:
        by_seed: Dict[int, Dict[tuple[Any, ...], Mapping[str, Any]]] = {
            seed: {} for seed in seeds
        }
        for row in records:
            if str(row["granularity"]) != granularity:
                continue
            seed = int(row["seed"])
            if seed not in by_seed:
                raise ValueError(f"Unexpected alpha seed {seed}.")
            key = (
                str(row["sample_id"]),
                int(row["y0"]),
                int(row["y1"]),
                int(row["x0"]),
                int(row["x1"]),
            )
            if key in by_seed[seed]:
                raise ValueError(
                    f"Duplicate alpha region for seed {seed}, "
                    f"granularity {granularity}, key {key}."
                )
            by_seed[seed][key] = row
        reference_keys = set(by_seed[seeds[0]])
        if not reference_keys:
            raise ValueError(f"No alpha regions for granularity {granularity}.")
        for seed in seeds[1:]:
            if set(by_seed[seed]) != reference_keys:
                raise ValueError(
                    f"Alpha region set mismatch for seed {seed}, "
                    f"granularity {granularity}."
                )
        ordered_keys = sorted(reference_keys)
        pairwise: Dict[str, Any] = {}
        for index, left_seed in enumerate(seeds):
            for right_seed in seeds[index + 1 :]:
                left = [
                    float(by_seed[left_seed][key]["alpha_star"])
                    for key in ordered_keys
                ]
                right = [
                    float(by_seed[right_seed][key]["alpha_star"])
                    for key in ordered_keys
                ]
                same_category = sum(
                    str(by_seed[left_seed][key]["category"])
                    == str(by_seed[right_seed][key]["category"])
                    for key in ordered_keys
                )
                pairwise[f"{left_seed}_vs_{right_seed}"] = {
                    "pearson_correlation": pearson_correlation(left, right),
                    "same_category_count": same_category,
                    "same_category_rate": same_category / len(ordered_keys),
                }
        all_same = sum(
            len(
                {
                    str(by_seed[seed][key]["category"])
                    for seed in seeds
                }
            )
            == 1
            for key in ordered_keys
        )
        output[granularity] = {
            "num_aligned_regions": len(ordered_keys),
            "pairwise": pairwise,
            "all_three_same_category_count": all_same,
            "all_three_same_category_rate": all_same / len(ordered_keys),
        }
    return {
        "split": "validation",
        "diagnostic": "cross_seed_alpha_stability_not_model_accuracy",
        "categories": {
            "near_SC": "alpha <= 0.05",
            "soft": "0.05 < alpha < 0.95",
            "near_CS": "alpha >= 0.95",
        },
        "seeds": list(seeds),
        "by_granularity": output,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gain_distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Soft gain distribution requires finite non-empty values.")
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
        output["thresholds"][f"{threshold:.3f}"] = {
            "count": count,
            "rate": count / len(values),
        }
    return output


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def build_soft_summary(
    image_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    alpha_records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    region_sizes: Sequence[int],
    regression_checks: Sequence[Mapping[str, Any]],
    hard_regression: Mapping[str, Any],
    run_dirs: Sequence[Path],
) -> Dict[str, Any]:
    granularities = ("whole", *(str(size) for size in region_sizes))
    fields = [
        "best_fixed_psnr",
        "hard_whole_psnr",
        "soft_whole_psnr",
        "soft_whole_gain_over_hard",
        "soft_whole_gain_over_fixed",
    ]
    for size in region_sizes:
        fields.extend(
            [
                f"hard_{size}_psnr",
                f"soft_{size}_psnr",
                f"soft_{size}_gain_over_hard",
                f"soft_{size}_gain_over_fixed",
                f"soft_{size}_gain_over_soft_whole",
            ]
        )
    aggregate = {
        field: _mean_std([float(row[field]) for row in per_seed_rows])
        for field in fields
    }
    alpha_aggregate = {}
    gains = {}
    for granularity in granularities:
        selected_alpha = [
            row
            for row in alpha_records
            if str(row["granularity"]) == granularity
        ]
        alpha_aggregate[granularity] = summarize_alpha_regions(
            _regions_from_records(selected_alpha)
        )
        gain_field = (
            "soft_whole_gain_over_hard"
            if granularity == "whole"
            else f"soft_{granularity}_gain_over_hard"
        )
        gains[granularity] = gain_distribution(
            [float(row[gain_field]) for row in image_rows]
        )
    return {
        "split": "validation",
        "oracle_type": "convex_soft_fusion",
        "alpha_constraint": [0.0, 1.0],
        "alpha_granularity": "one_scalar_per_region_shared_by_RGB_and_pixels",
        "selection_objective": "RGB squared error",
        "region_sizes": list(region_sizes),
        "grid": "non_overlapping_top_left_aligned",
        "border_handling": "natural_truncation_no_resize_no_padding",
        "metric_semantics": "mean_of_per_image_rgb_metrics",
        "prediction_clamp": [0.0, "data_range"],
        "denominator_epsilon": ALPHA_DENOMINATOR_EPS,
        "seeds": list(seeds),
        "run_directories": [str(path) for path in run_dirs],
        "per_seed": [dict(row) for row in per_seed_rows],
        "cross_seed_mean_and_sample_std": aggregate,
        "alpha_distribution": alpha_aggregate,
        "per_image_soft_gain_over_hard_distribution": gains,
        "validation_replay_regression": [dict(row) for row in regression_checks],
        "previous_hard_spatial_regression": dict(hard_regression),
    }


def load_previous_hard_analysis(
    directory: Path,
) -> Optional[tuple[Dict[int, Dict[str, str]], Dict[str, Any]]]:
    """Load both prior hard outputs if the analysis directory exists."""

    if not directory.exists():
        return None
    per_seed_path = directory / "spatial_oracle_per_seed.csv"
    summary_path = directory / "spatial_oracle_summary.json"
    missing = [path for path in (per_seed_path, summary_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Previous hard spatial analysis is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    with per_seed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[int, Dict[str, str]] = {}
        for row in reader:
            seed = int(row["seed"])
            if seed in rows:
                raise ValueError(f"Previous hard analysis duplicates seed {seed}.")
            rows[seed] = dict(row)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError("Previous hard spatial summary must be a JSON object.")
    return rows, summary


def find_previous_hard_analysis(output_dir: Path) -> Optional[Path]:
    """Find the sibling/default hard analysis without requiring it to exist."""

    candidates = (
        output_dir.parent / "v2_validation_spatial_oracle",
        Path("analysis_results/v2_validation_spatial_oracle")
        .expanduser()
        .resolve(strict=False),
    )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved
    return None


def validate_previous_hard_analysis(
    current_rows: Sequence[Mapping[str, Any]],
    previous: Optional[tuple[Mapping[int, Mapping[str, Any]], Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    region_sizes: Sequence[int],
) -> Dict[str, Any]:
    if previous is None:
        return {"status": "skipped", "reason": "previous hard analysis not found"}
    previous_rows, previous_summary = previous
    if previous_summary.get("split") != "validation":
        raise ValueError("Previous hard spatial summary is not validation-only.")
    if list(previous_summary.get("region_sizes", [])) != list(region_sizes):
        raise ValueError("Previous hard spatial region sizes do not match this run.")
    fields = [
        ("best_fixed_psnr", "best_fixed_psnr"),
        ("hard_whole_psnr", "whole_oracle_psnr"),
        *((f"hard_{size}_psnr", f"oracle_{size}_psnr") for size in region_sizes),
    ]
    ssim_fields = [
        ("best_fixed_ssim", "best_fixed_ssim"),
        ("hard_whole_ssim", "whole_oracle_ssim"),
        *((f"hard_{size}_ssim", f"oracle_{size}_ssim") for size in region_sizes),
    ]
    by_seed = {int(row["seed"]): row for row in current_rows}
    differences: Dict[str, Any] = {}
    for seed in seeds:
        if seed not in previous_rows:
            raise ValueError(f"Previous hard spatial analysis is missing seed {seed}.")
        differences[str(seed)] = {}
        for current_field, previous_field in fields:
            actual = float(by_seed[seed][current_field])
            expected = float(previous_rows[seed][previous_field])
            spatial._require_close(
                seed=seed,
                name=f"previous hard spatial {previous_field}",
                actual=actual,
                expected=expected,
                abs_tolerance=spatial.INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE,
            )
            differences[str(seed)][previous_field] = actual - expected
        for current_field, previous_field in ssim_fields:
            actual = float(by_seed[seed][current_field])
            expected = float(previous_rows[seed][previous_field])
            spatial._require_close(
                seed=seed,
                name=f"previous hard spatial {previous_field}",
                actual=actual,
                expected=expected,
                abs_tolerance=spatial.INFERENCE_AGGREGATE_SSIM_ABS_TOLERANCE,
            )
            differences[str(seed)][previous_field] = actual - expected
    previous_aggregate = previous_summary.get("cross_seed_mean_and_sample_std")
    if not isinstance(previous_aggregate, Mapping):
        raise ValueError("Previous hard summary lacks cross-seed aggregation.")
    for current_field, previous_field in fields:
        current_mean = statistics.fmean(
            float(row[current_field]) for row in current_rows
        )
        expected_mean = float(previous_aggregate[previous_field]["mean"])
        spatial._require_close(
            seed=-1,
            name=f"previous hard spatial aggregate {previous_field}",
            actual=current_mean,
            expected=expected_mean,
            abs_tolerance=spatial.INFERENCE_AGGREGATE_PSNR_ABS_TOLERANCE,
        )
    return {"status": "passed", "differences_by_seed": differences}


def _per_image_fields(region_sizes: Sequence[int]) -> list[str]:
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
        "hard_whole_psnr",
        "soft_whole_psnr",
        "soft_whole_ssim",
        "alpha_whole",
        "alpha_whole_raw",
        "alpha_whole_degenerate",
        "soft_whole_gain_over_hard",
        "soft_whole_gain_over_best_fixed",
    ]
    for size in region_sizes:
        fields.extend(
            [
                f"hard_{size}_psnr",
                f"soft_{size}_psnr",
                f"soft_{size}_ssim",
                f"soft_{size}_gain_over_hard",
                f"soft_{size}_gain_over_best_fixed",
                f"soft_{size}_gain_over_soft_whole",
                f"alpha_{size}_mean",
                f"alpha_{size}_std",
                f"alpha_{size}_interior_pixel_rate",
                f"alpha_{size}_near_cs_pixel_rate",
                f"alpha_{size}_near_sc_pixel_rate",
                f"alpha_{size}_degenerate_region_count",
                f"alpha_{size}_degenerate_pixel_rate",
            ]
        )
    return fields


def _per_seed_fields(region_sizes: Sequence[int]) -> list[str]:
    fields = [
        "seed",
        "num_samples",
        "mean_psnr_cs",
        "mean_psnr_sc",
        "best_fixed_order",
        "best_fixed_psnr",
        "best_fixed_ssim",
        "hard_whole_psnr",
        "hard_whole_ssim",
        "soft_whole_psnr",
        "soft_whole_ssim",
        "soft_whole_gain_over_hard",
        "soft_whole_gain_over_fixed",
    ]
    for size in region_sizes:
        fields.extend(
            [
                f"hard_{size}_psnr",
                f"hard_{size}_ssim",
                f"soft_{size}_psnr",
                f"soft_{size}_ssim",
                f"soft_{size}_gain_over_hard",
                f"soft_{size}_gain_over_fixed",
                f"soft_{size}_gain_over_soft_whole",
            ]
        )
    alpha_fields = (
        "mean_alpha",
        "median_alpha",
        "std_alpha",
        "near_cs_region_rate",
        "near_sc_region_rate",
        "interior_soft_region_rate",
        "near_cs_pixel_rate",
        "near_sc_pixel_rate",
        "interior_soft_pixel_rate",
        "interior_0p10_0p90_region_rate",
        "interior_0p10_0p90_pixel_rate",
        "interior_0p25_0p75_region_rate",
        "interior_0p25_0p75_pixel_rate",
        "raw_below_zero_region_rate",
        "raw_above_one_region_rate",
        "raw_inside_unit_interval_region_rate",
        "degenerate_region_rate",
        "degenerate_pixel_rate",
    )
    for granularity in ("whole", *(str(size) for size in region_sizes)):
        fields.extend(f"alpha_{granularity}_{field}" for field in alpha_fields)
    return fields


def build_analysis_markdown(
    summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    *,
    region_sizes: Sequence[int],
) -> str:
    aggregate = summary["cross_seed_mean_and_sample_std"]
    granularities = ("whole", *(str(size) for size in region_sizes))

    def mean_std(field: str) -> str:
        values = aggregate[field]
        return f"{float(values['mean']):.6f} ± {float(values['sample_std']):.6f}"

    lines = [
        "# V2 Multi-scale Regional Soft Fusion Oracle Analysis",
        "",
        "This validation-only Oracle analytically chooses one convex CS/SC alpha per "
        "region. All metrics are computed on complete reconstructed images.",
        "",
        "## Table 1: Hard versus soft PSNR (three-seed mean ± sample std)",
        "",
        "| Granularity | Best fixed | Hard Oracle | Soft Oracle | Soft gain over Hard | Soft gain over Fixed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| Whole | {mean_std('best_fixed_psnr')} | {mean_std('hard_whole_psnr')} | "
        f"{mean_std('soft_whole_psnr')} | {mean_std('soft_whole_gain_over_hard')} | "
        f"{mean_std('soft_whole_gain_over_fixed')} |"
    )
    for size in region_sizes:
        lines.append(
            f"| {size}×{size} | {mean_std('best_fixed_psnr')} | "
            f"{mean_std(f'hard_{size}_psnr')} | {mean_std(f'soft_{size}_psnr')} | "
            f"{mean_std(f'soft_{size}_gain_over_hard')} | "
            f"{mean_std(f'soft_{size}_gain_over_fixed')} |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Alpha behavior",
            "",
            "| Granularity | Near-SC | Interior Soft | Near-CS | Raw alpha inside [0,1] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for granularity in granularities:
        alpha = summary["alpha_distribution"][granularity]
        label = "Whole" if granularity == "whole" else f"{granularity}×{granularity}"
        lines.append(
            f"| {label} | {100 * float(alpha['near_sc_region_rate']):.2f}% | "
            f"{100 * float(alpha['interior_soft_region_rate']):.2f}% | "
            f"{100 * float(alpha['near_cs_region_rate']):.2f}% | "
            f"{100 * float(alpha['raw_inside_unit_interval_region_rate']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Table 3: Per-image soft gain over hard coverage",
            "",
            "| Granularity | >0.001 dB | >0.005 dB | >0.01 dB | >0.05 dB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for granularity in granularities:
        thresholds = summary["per_image_soft_gain_over_hard_distribution"][granularity][
            "thresholds"
        ]
        label = "Whole" if granularity == "whole" else f"{granularity}×{granularity}"
        lines.append(
            f"| {label} | {100 * float(thresholds['0.001']['rate']):.2f}% | "
            f"{100 * float(thresholds['0.005']['rate']):.2f}% | "
            f"{100 * float(thresholds['0.010']['rate']):.2f}% | "
            f"{100 * float(thresholds['0.050']['rate']):.2f}% |"
        )

    seed_values = list(summary["seeds"])
    pairs = (
        f"{seed_values[0]}_vs_{seed_values[1]}",
        f"{seed_values[0]}_vs_{seed_values[2]}",
        f"{seed_values[1]}_vs_{seed_values[2]}",
    )
    lines.extend(
        [
            "",
            "## Table 4: Cross-seed alpha stability",
            "",
            f"| Granularity | Pearson {pairs[0]} | Pearson {pairs[1]} | Pearson {pairs[2]} | All-three category agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for granularity in granularities:
        values = stability["by_granularity"][granularity]
        rendered = []
        for pair in pairs:
            correlation = values["pairwise"][pair]["pearson_correlation"]
            rendered.append("n/a" if correlation is None else f"{float(correlation):.4f}")
        label = "Whole" if granularity == "whole" else f"{granularity}×{granularity}"
        lines.append(
            f"| {label} | {rendered[0]} | {rendered[1]} | {rendered[2]} | "
            f"{100 * float(values['all_three_same_category_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "If Soft Oracle materially exceeds Hard Oracle while many coefficients are "
            "interior, the paths contain complementary information not captured by hard "
            "selection. If Soft and Hard are nearly equal and coefficients concentrate "
            "near zero or one, convex blending has limited additional headroom. This "
            "report defines no materiality threshold and makes no significance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _alpha_map_to_pil(alpha_map: Any) -> Any:
    import numpy as np
    from PIL import Image

    alpha = alpha_map.detach().float().cpu().clamp(0.0, 1.0).numpy()
    output = np.empty((*alpha.shape, 3), dtype=np.float32)
    lower = alpha <= 0.5
    fraction_lower = (alpha[lower] * 2.0)[:, None]
    output[lower] = (
        np.array([0.0, 70.0, 255.0]) * (1.0 - fraction_lower)
        + np.array([255.0, 255.0, 255.0]) * fraction_lower
    )
    fraction_upper = ((alpha[~lower] - 0.5) * 2.0)[:, None]
    output[~lower] = (
        np.array([255.0, 255.0, 255.0]) * (1.0 - fraction_upper)
        + np.array([255.0, 70.0, 0.0]) * fraction_upper
    )
    return Image.fromarray(np.rint(output).astype(np.uint8), mode="RGB")


def save_visualizations(
    candidates: Mapping[str, Sequence[SoftVisualizationCandidate]],
    output_dir: Path,
) -> list[Path]:
    from PIL import Image, ImageDraw

    destination = output_dir / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)
    granularities = list(candidates)
    labels = (
        "Input",
        "GT",
        "C→S",
        "S→C",
        "Hard Oracle",
        "Soft Oracle",
        "Alpha Map",
    )
    written: list[Path] = []
    for granularity in granularities:
        ordered = sorted(candidates[granularity], key=_candidate_key)
        for rank, candidate in enumerate(ordered, start=1):
            images = (
                spatial._tensor_to_pil(
                    candidate.input_tensor, data_range=candidate.data_range
                ),
                spatial._tensor_to_pil(
                    candidate.target_tensor, data_range=candidate.data_range
                ),
                spatial._tensor_to_pil(
                    candidate.prediction_cs, data_range=candidate.data_range
                ),
                spatial._tensor_to_pil(
                    candidate.prediction_sc, data_range=candidate.data_range
                ),
                spatial._tensor_to_pil(
                    candidate.hard_prediction, data_range=candidate.data_range
                ),
                spatial._tensor_to_pil(
                    candidate.soft_prediction, data_range=candidate.data_range
                ),
                _alpha_map_to_pil(candidate.alpha_map),
            )
            width, height = images[0].size
            header_height = 58
            label_height = 18
            panel = Image.new(
                "RGB", (width * len(images), header_height + label_height + height), "white"
            )
            draw = ImageDraw.Draw(panel)
            draw.text(
                (4, 4),
                (
                    f"seed={candidate.seed}  sample={candidate.sample_id}  "
                    f"granularity={granularity}  soft_gain_over_hard="
                    f"{candidate.gain_over_hard:.6f} dB\n"
                    f"PSNR CS={candidate.psnr_cs:.6f}  SC={candidate.psnr_sc:.6f}  "
                    f"Hard={candidate.hard_psnr:.6f}  Soft={candidate.soft_psnr:.6f}  "
                    f"mean alpha={candidate.mean_alpha:.4f}  interior pixels="
                    f"{100 * candidate.interior_pixel_rate:.2f}%"
                ),
                fill="black",
            )
            for column, (label, image) in enumerate(zip(labels, images)):
                x = column * width
                draw.text((x + 4, header_height), label, fill="black")
                panel.paste(image, (x, header_height + label_height))
            safe_sample = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in candidate.sample_id
            )
            path = destination / (
                f"soft_{granularity}_rank_{rank:02d}_seed{candidate.seed}_"
                f"{safe_sample}.png"
            )
            panel.save(path, format="PNG")
            written.append(path)
    return written


def write_outputs(
    *,
    output_dir: Path,
    image_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    candidates: Mapping[str, Sequence[SoftVisualizationCandidate]],
    region_sizes: Sequence[int],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "soft_fusion_per_image.csv": output_dir / "soft_fusion_per_image.csv",
        "soft_fusion_per_seed.csv": output_dir / "soft_fusion_per_seed.csv",
        "soft_fusion_summary.json": output_dir / "soft_fusion_summary.json",
        "soft_fusion_alpha_stability.json": output_dir
        / "soft_fusion_alpha_stability.json",
        "analysis_summary.md": output_dir / "analysis_summary.md",
    }
    atomic_write_csv(
        outputs["soft_fusion_per_image.csv"], image_rows, _per_image_fields(region_sizes)
    )
    atomic_write_csv(
        outputs["soft_fusion_per_seed.csv"], per_seed_rows, _per_seed_fields(region_sizes)
    )
    atomic_write_json(outputs["soft_fusion_summary.json"], summary)
    atomic_write_json(outputs["soft_fusion_alpha_stability.json"], stability)
    outputs["analysis_summary.md"].write_text(
        build_analysis_markdown(summary, stability, region_sizes=region_sizes),
        encoding="utf-8",
    )
    save_visualizations(candidates, output_dir)
    return outputs


def run_analysis(
    *,
    experiments_root: Path | str = Path("experiments"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    region_sizes: Sequence[int] = DEFAULT_REGION_SIZES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    device_name: str = "auto",
    split: str = "validation",
) -> Dict[str, Path]:
    spatial.require_validation_split(split)
    sizes = spatial.validate_region_sizes(region_sizes)
    seed_values = tuple(seeds)
    if len(seed_values) != 3 or len(set(seed_values)) != 3:
        raise ValueError(
            f"Soft fusion analysis requires exactly three unique seeds, got {seed_values}."
        )
    import torch

    device = spatial.resolve_device(device_name, torch)
    root = Path(experiments_root).expanduser().resolve(strict=False)
    seed_results = [
        load_seed_validation_result(discover_seed_run(root, seed=seed), seed=seed)
        for seed in seed_values
    ]
    validate_configs_across_seeds(seed_results)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    previous_hard_directory = find_previous_hard_analysis(destination)
    previous_hard = (
        load_previous_hard_analysis(previous_hard_directory)
        if previous_hard_directory is not None
        else None
    )

    all_image_rows: list[Dict[str, Any]] = []
    all_alpha_records: list[Dict[str, Any]] = []
    per_seed_rows: list[Dict[str, Any]] = []
    regressions: list[Dict[str, Any]] = []
    top_candidates: Dict[str, list[SoftVisualizationCandidate]] = {
        "whole": [],
        **{str(size): [] for size in sizes},
    }
    for seed_result in seed_results:
        print(
            f"Loading seed {seed_result.seed} through the shared hard-spatial "
            f"validation inference path: {seed_result.run_dir}",
            flush=True,
        )
        collector = SoftFusionCollector(sizes)
        _, _, hard_per_seed, _, regression = spatial.analyze_seed(
            seed_result,
            region_sizes=sizes,
            device=device,
            visualization_limit=0,
            sample_callback=collector,
        )
        soft_per_seed = aggregate_soft_seed_rows(
            collector.image_rows,
            hard_per_seed,
            collector.alpha_records,
            region_sizes=sizes,
        )
        all_image_rows.extend(collector.image_rows)
        all_alpha_records.extend(collector.alpha_records)
        per_seed_rows.append(soft_per_seed)
        regressions.append(regression)
        for granularity, values in collector.candidates.items():
            for candidate in values:
                _consider_candidate(
                    top_candidates[granularity],
                    candidate,
                    limit=TOP_VISUALIZATIONS_PER_GRANULARITY,
                )
        del collector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hard_regression = validate_previous_hard_analysis(
        per_seed_rows,
        previous_hard,
        seeds=seed_values,
        region_sizes=sizes,
    )
    granularities = ("whole", *(str(size) for size in sizes))
    stability = compute_alpha_stability(
        all_alpha_records,
        seeds=seed_values,
        granularities=granularities,
    )
    summary = build_soft_summary(
        all_image_rows,
        per_seed_rows,
        all_alpha_records,
        seeds=seed_values,
        region_sizes=sizes,
        regression_checks=regressions,
        hard_regression=hard_regression,
        run_dirs=[result.run_dir for result in seed_results],
    )
    return write_outputs(
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
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--region-sizes", type=int, nargs="+", default=list(DEFAULT_REGION_SIZES)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    print(f"Wrote V2 validation soft-fusion Oracle analysis to {args.output_dir}")
    for name in outputs:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
