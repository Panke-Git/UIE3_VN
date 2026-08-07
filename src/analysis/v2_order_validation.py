"""Analyze V2 shared-order results from best-checkpoint validation snapshots.

This module is intentionally offline: it reads JSON and CSV artifacts only. It
does not import a model, load a checkpoint, inspect images, or read test data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from src.common.experiment.logging_utils import atomic_write_csv, atomic_write_json


DEFAULT_SEEDS = (1234, 3407, 3520)
SIGN_TOLERANCE = 1.0e-8
VALUE_REL_TOLERANCE = 1.0e-10
VALUE_ABS_TOLERANCE = 1.0e-10
PSNR_MARGINS = (0.01, 0.05, 0.10)
DELTA_DEFINITION = "CS-SC (color_then_scatter minus scatter_then_color)"

CONFIG_FILENAME = "config.json"
SIDECAR_FILENAME = "best/best_psnr.json"
COMPARISON_FILENAME = "result/best_psnr_validation_order_comparison.csv"
SUMMARY_FILENAME = "result/best_psnr_validation_summary.json"

CSV_REQUIRED_FIELDS = (
    "sample_id",
    "input_relative_path",
    "gt_relative_path",
    "psnr_color_then_scatter",
    "psnr_scatter_then_color",
    "delta_psnr_cs_minus_sc",
    "ssim_color_then_scatter",
    "ssim_scatter_then_color",
    "delta_ssim_cs_minus_sc",
)
NUMERIC_CSV_FIELDS = CSV_REQUIRED_FIELDS[3:]
STRICT_CONFIG_SECTIONS = (
    "data",
    "model",
    "order_study",
    "loss",
    "optimizer",
    "scheduler",
    "training",
    "metrics",
    "checkpoint",
)

SampleKey = tuple[str, str]


@dataclass(frozen=True)
class ValidationSample:
    """One image's two-path metrics from one seed."""

    sample_id: str
    input_relative_path: str
    gt_relative_path: str
    psnr_color_then_scatter: float
    psnr_scatter_then_color: float
    delta_psnr_cs_minus_sc: float
    ssim_color_then_scatter: float
    ssim_scatter_then_color: float
    delta_ssim_cs_minus_sc: float

    @property
    def key(self) -> SampleKey:
        return (self.input_relative_path, self.gt_relative_path)


@dataclass(frozen=True)
class SeedValidationResult:
    """Validated best-validation inputs for one completed shared V2 run."""

    seed: int
    run_dir: Path
    config: Mapping[str, Any]
    checkpoint_sidecar: Mapping[str, Any]
    validation_summary: Mapping[str, Any]
    samples: Mapping[SampleKey, ValidationSample]


@dataclass(frozen=True)
class AlignedValidationSample:
    """One validation identity aligned across all requested seeds."""

    sample_id: str
    input_relative_path: str
    gt_relative_path: str
    by_seed: Mapping[int, ValidationSample]


def _read_json_mapping(path: Path, *, seed: int, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Seed {seed} run has unreadable {label} JSON at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Seed {seed} run {label} must be a JSON object: {path}"
        )
    return value


def _finite_number(value: Any, *, seed: int, location: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(
            f"Seed {seed} {location} must be numeric, got {value!r}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"Seed {seed} {location} must be finite, got {value!r}."
        )
    return result


def _required_integer(value: Any, *, seed: int, location: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Seed {seed} {location} must be an integer >= {minimum}, "
            f"got {value!r}."
        )
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=VALUE_REL_TOLERANCE,
        abs_tol=VALUE_ABS_TOLERANCE,
    )


def _parse_csv_number(raw: Optional[str], *, seed: int, row: int, field: str) -> float:
    try:
        value = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Seed {seed} CSV row {row} field {field!r} is not numeric: {raw!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Seed {seed} CSV row {row} field {field!r} must be finite, "
            f"got {raw!r}."
        )
    return value


def _read_comparison_csv(path: Path, *, seed: int) -> Dict[SampleKey, ValidationSample]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(
            f"Seed {seed} comparison CSV cannot be opened at {path}: {exc}"
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [field for field in CSV_REQUIRED_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"Seed {seed} comparison CSV {path} is missing required fields: "
                f"{missing}."
            )
        samples: Dict[SampleKey, ValidationSample] = {}
        sample_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            identifiers: Dict[str, str] = {}
            for field in CSV_REQUIRED_FIELDS[:3]:
                raw = row.get(field)
                if raw is None or not raw.strip():
                    raise ValueError(
                        f"Seed {seed} CSV row {row_number} field {field!r} "
                        "must not be empty."
                    )
                identifiers[field] = raw
            sample_id = identifiers["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(
                    f"Seed {seed} comparison CSV contains duplicate sample_id "
                    f"{sample_id!r} at row {row_number}."
                )
            values = {
                field: _parse_csv_number(
                    row.get(field), seed=seed, row=row_number, field=field
                )
                for field in NUMERIC_CSV_FIELDS
            }
            expected_psnr_delta = (
                values["psnr_color_then_scatter"]
                - values["psnr_scatter_then_color"]
            )
            expected_ssim_delta = (
                values["ssim_color_then_scatter"]
                - values["ssim_scatter_then_color"]
            )
            if not _close(values["delta_psnr_cs_minus_sc"], expected_psnr_delta):
                raise ValueError(
                    f"Seed {seed} CSV row {row_number} PSNR delta mismatch: "
                    "delta_psnr_cs_minus_sc must equal "
                    "psnr_color_then_scatter - psnr_scatter_then_color."
                )
            if not _close(values["delta_ssim_cs_minus_sc"], expected_ssim_delta):
                raise ValueError(
                    f"Seed {seed} CSV row {row_number} SSIM delta mismatch: "
                    "delta_ssim_cs_minus_sc must equal "
                    "ssim_color_then_scatter - ssim_scatter_then_color."
                )
            sample = ValidationSample(**identifiers, **values)
            if sample.key in samples:
                raise ValueError(
                    f"Seed {seed} comparison CSV contains duplicate image identity "
                    f"{sample.key!r} at row {row_number}."
                )
            sample_ids.add(sample_id)
            samples[sample.key] = sample
    return samples


def _validate_summary_against_samples(
    *, seed: int, summary: Mapping[str, Any], samples: Mapping[SampleKey, ValidationSample]
) -> None:
    num_samples = _required_integer(
        summary.get("num_samples"),
        seed=seed,
        location="best validation summary num_samples",
        minimum=1,
    )
    if len(samples) != num_samples:
        raise ValueError(
            f"Seed {seed} best validation row count mismatch: CSV has "
            f"{len(samples)} rows but summary num_samples={num_samples}."
        )

    expected_means = {
        "mean_psnr_color_then_scatter": statistics.fmean(
            sample.psnr_color_then_scatter for sample in samples.values()
        ),
        "mean_psnr_scatter_then_color": statistics.fmean(
            sample.psnr_scatter_then_color for sample in samples.values()
        ),
        "mean_ssim_color_then_scatter": statistics.fmean(
            sample.ssim_color_then_scatter for sample in samples.values()
        ),
        "mean_ssim_scatter_then_color": statistics.fmean(
            sample.ssim_scatter_then_color for sample in samples.values()
        ),
        "mean_delta_psnr_cs_minus_sc": statistics.fmean(
            sample.delta_psnr_cs_minus_sc for sample in samples.values()
        ),
        "mean_delta_ssim_cs_minus_sc": statistics.fmean(
            sample.delta_ssim_cs_minus_sc for sample in samples.values()
        ),
    }
    expected_means["mean_path_psnr"] = 0.5 * (
        expected_means["mean_psnr_color_then_scatter"]
        + expected_means["mean_psnr_scatter_then_color"]
    )
    for field, expected in expected_means.items():
        actual = _finite_number(
            summary.get(field),
            seed=seed,
            location=f"best validation summary {field}",
        )
        if not _close(actual, expected):
            raise ValueError(
                f"Seed {seed} best validation summary {field}={actual!r} "
                f"does not match the per-image CSV mean {expected!r}."
            )


def validate_seed_result(
    *,
    seed: int,
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_sidecar: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    samples: Mapping[SampleKey, ValidationSample],
) -> SeedValidationResult:
    """Validate one seed's identity, provenance, and best-validation rows."""

    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError(f"Seed {seed} config experiment section must be an object.")
    variant = experiment.get("variant")
    if variant != "shared_order_diagnostic":
        raise ValueError(
            f"Seed {seed} run {run_dir} must use variant="
            f"'shared_order_diagnostic'; got {variant!r}."
        )
    config_seed = experiment.get("seed")
    if config_seed != seed or type(config_seed) is not int:
        raise ValueError(
            f"Seed {seed} run {run_dir} config seed mismatch: "
            f"experiment.seed={config_seed!r}."
        )
    missing_sections = [section for section in STRICT_CONFIG_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(
            f"Seed {seed} config is missing sections required for cross-seed "
            f"comparison: {missing_sections}."
        )

    selection_metric = checkpoint_sidecar.get("selection_metric")
    if selection_metric != "validation_mean_path_psnr":
        raise ValueError(
            f"Seed {seed} best/best_psnr.json selection_metric must equal "
            f"'validation_mean_path_psnr'; got {selection_metric!r}."
        )
    sidecar_epoch = _required_integer(
        checkpoint_sidecar.get("epoch"),
        seed=seed,
        location="best/best_psnr.json epoch",
        minimum=0,
    )
    summary_epoch = _required_integer(
        validation_summary.get("epoch"),
        seed=seed,
        location="best validation summary epoch",
        minimum=0,
    )
    if sidecar_epoch != summary_epoch:
        raise ValueError(
            f"Seed {seed} checkpoint epoch mismatch: best/best_psnr.json has "
            f"epoch={sidecar_epoch}, but best validation summary has "
            f"epoch={summary_epoch}."
        )
    sidecar_step = _required_integer(
        checkpoint_sidecar.get("global_step"),
        seed=seed,
        location="best/best_psnr.json global_step",
        minimum=0,
    )
    summary_step = _required_integer(
        validation_summary.get("global_step"),
        seed=seed,
        location="best validation summary global_step",
        minimum=0,
    )
    if sidecar_step != summary_step:
        raise ValueError(
            f"Seed {seed} checkpoint global_step mismatch: best/best_psnr.json "
            f"has global_step={sidecar_step}, but best validation summary has "
            f"global_step={summary_step}."
        )
    sidecar_psnr = _finite_number(
        checkpoint_sidecar.get("psnr"),
        seed=seed,
        location="best/best_psnr.json psnr",
    )
    summary_psnr = _finite_number(
        validation_summary.get("mean_path_psnr"),
        seed=seed,
        location="best validation summary mean_path_psnr",
    )
    if not _close(sidecar_psnr, summary_psnr):
        raise ValueError(
            f"Seed {seed} best checkpoint PSNR mismatch: sidecar psnr="
            f"{sidecar_psnr!r}, summary mean_path_psnr={summary_psnr!r}."
        )
    _validate_summary_against_samples(
        seed=seed, summary=validation_summary, samples=samples
    )
    return SeedValidationResult(
        seed=seed,
        run_dir=run_dir,
        config=copy.deepcopy(dict(config)),
        checkpoint_sidecar=copy.deepcopy(dict(checkpoint_sidecar)),
        validation_summary=copy.deepcopy(dict(validation_summary)),
        samples=dict(samples),
    )


def load_seed_validation_result(
    run_dir: Path | str, *, seed: int
) -> SeedValidationResult:
    """Load and validate one seed without reading latest-validation or test files."""

    resolved_run = Path(run_dir).expanduser().resolve(strict=False)
    paths = {
        "config": resolved_run / CONFIG_FILENAME,
        "checkpoint sidecar": resolved_run / SIDECAR_FILENAME,
        "best validation comparison": resolved_run / COMPARISON_FILENAME,
        "best validation summary": resolved_run / SUMMARY_FILENAME,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Seed {seed} run {resolved_run} is missing required {label} "
                f"file: {path}"
            )
    config = _read_json_mapping(paths["config"], seed=seed, label="config")
    sidecar = _read_json_mapping(
        paths["checkpoint sidecar"], seed=seed, label="checkpoint sidecar"
    )
    summary = _read_json_mapping(
        paths["best validation summary"], seed=seed, label="best validation summary"
    )
    samples = _read_comparison_csv(paths["best validation comparison"], seed=seed)
    return validate_seed_result(
        seed=seed,
        run_dir=resolved_run,
        config=config,
        checkpoint_sidecar=sidecar,
        validation_summary=summary,
        samples=samples,
    )


def _normalized_cross_seed_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    experiment = normalized.get("experiment")
    if isinstance(experiment, MutableMapping):
        experiment.pop("name", None)
        experiment.pop("seed", None)
    training = normalized.get("training")
    if isinstance(training, MutableMapping):
        training.pop("resume", None)
    test = normalized.get("test")
    if isinstance(test, MutableMapping):
        test.pop("run_dir", None)
        test.pop("allow_overwrite", None)
    return normalized


def _first_difference(left: Any, right: Any, prefix: str = "config") -> str:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}"
            if key not in left:
                return f"{path} is missing from the reference seed"
            if key not in right:
                return f"{path} is missing from the compared seed"
            difference = _first_difference(left[key], right[key], path)
            if difference:
                return difference
        return ""
    if left != right:
        return f"{prefix} differs ({left!r} != {right!r})"
    return ""


def validate_configs_across_seeds(results: Sequence[SeedValidationResult]) -> None:
    """Require identical experiment semantics except explicitly run-local fields."""

    if not results:
        raise ValueError("At least one seed result is required.")
    reference = results[0]
    reference_config = _normalized_cross_seed_config(reference.config)
    for result in results[1:]:
        compared = _normalized_cross_seed_config(result.config)
        if compared != reference_config:
            difference = _first_difference(reference_config, compared)
            raise ValueError(
                f"Cross-seed config mismatch between seed {reference.seed} and "
                f"seed {result.seed}: {difference}. Only experiment.name, "
                "experiment.seed, training.resume, test.run_dir, and "
                "test.allow_overwrite may differ."
            )


def align_samples_across_seeds(
    results: Sequence[SeedValidationResult],
) -> list[AlignedValidationSample]:
    """Align by input/GT path identity, never by source CSV row order."""

    if not results:
        raise ValueError("At least one seed result is required for alignment.")
    reference = results[0]
    reference_keys = set(reference.samples)
    for result in results[1:]:
        keys = set(result.samples)
        missing = sorted(reference_keys - keys)
        extra = sorted(keys - reference_keys)
        if missing or extra:
            raise ValueError(
                f"Seed {result.seed} validation sample set does not match seed "
                f"{reference.seed}: missing={missing}, extra={extra}."
            )
    aligned: list[AlignedValidationSample] = []
    for key in sorted(reference_keys):
        by_seed = {result.seed: result.samples[key] for result in results}
        sample_ids = {sample.sample_id for sample in by_seed.values()}
        if len(sample_ids) != 1:
            details = {seed: sample.sample_id for seed, sample in by_seed.items()}
            raise ValueError(
                f"Cross-seed sample_id mismatch for image identity {key!r}: "
                f"{details}."
            )
        aligned.append(
            AlignedValidationSample(
                sample_id=next(iter(sample_ids)),
                input_relative_path=key[0],
                gt_relative_path=key[1],
                by_seed=by_seed,
            )
        )
    return aligned


def _preference(delta: float, tolerance: float = SIGN_TOLERANCE) -> str:
    if delta > tolerance:
        return "CS"
    if delta < -tolerance:
        return "SC"
    return "tie"


def _majority_preference(preferences: Iterable[str]) -> str:
    values = list(preferences)
    required = len(values) // 2 + 1
    if values.count("CS") >= required:
        return "CS"
    if values.count("SC") >= required:
        return "SC"
    return "none"


def _threshold_label(values: Sequence[float], threshold: float) -> str:
    if all(value >= threshold for value in values):
        return "CS"
    if all(value <= -threshold for value in values):
        return "SC"
    return "unstable"


def _margin_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def compute_cross_seed_per_image(
    aligned: Sequence[AlignedValidationSample], seeds: Sequence[int]
) -> list[Dict[str, Any]]:
    """Compute deterministic per-image deltas and preference stability."""

    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for sample standard deviation.")
    rows: list[Dict[str, Any]] = []
    for item in aligned:
        psnr_values = [
            item.by_seed[seed].delta_psnr_cs_minus_sc for seed in seeds
        ]
        ssim_values = [
            item.by_seed[seed].delta_ssim_cs_minus_sc for seed in seeds
        ]
        psnr_preferences = [_preference(value) for value in psnr_values]
        ssim_preferences = [_preference(value) for value in ssim_values]
        row: Dict[str, Any] = {
            "sample_id": item.sample_id,
            "input_relative_path": item.input_relative_path,
            "gt_relative_path": item.gt_relative_path,
        }
        for seed, value in zip(seeds, psnr_values):
            row[f"delta_psnr_seed{seed}"] = value
        for seed, value in zip(seeds, ssim_values):
            row[f"delta_ssim_seed{seed}"] = value
        row.update(
            {
                "mean_delta_psnr": statistics.fmean(psnr_values),
                "median_delta_psnr": statistics.median(psnr_values),
                "std_delta_psnr": statistics.stdev(psnr_values),
                "mean_delta_ssim": statistics.fmean(ssim_values),
                "median_delta_ssim": statistics.median(ssim_values),
                "std_delta_ssim": statistics.stdev(ssim_values),
                "cs_psnr_win_count": psnr_preferences.count("CS"),
                "sc_psnr_win_count": psnr_preferences.count("SC"),
                "cs_ssim_win_count": ssim_preferences.count("CS"),
                "sc_ssim_win_count": ssim_preferences.count("SC"),
                "all_three_psnr_same_preference": len(set(psnr_preferences)) == 1,
                "majority_psnr_preference": _majority_preference(psnr_preferences),
                "all_three_ssim_same_preference": len(set(ssim_preferences)) == 1,
                "majority_ssim_preference": _majority_preference(ssim_preferences),
            }
        )
        for threshold in PSNR_MARGINS:
            row[f"stable_preference_psnr_{_margin_label(threshold)}"] = (
                _threshold_label(psnr_values, threshold)
            )
        rows.append(row)
    return rows


def _pairwise_sign_agreement(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], metric: str
) -> Dict[str, Dict[str, Any]]:
    agreement: Dict[str, Dict[str, Any]] = {}
    for index, left_seed in enumerate(seeds):
        for right_seed in seeds[index + 1 :]:
            same = different = tie_involved = 0
            for row in rows:
                left = _preference(float(row[f"delta_{metric}_seed{left_seed}"]))
                right = _preference(float(row[f"delta_{metric}_seed{right_seed}"]))
                if "tie" in (left, right):
                    tie_involved += 1
                elif left == right:
                    same += 1
                else:
                    different += 1
            non_tie = same + different
            agreement[f"{left_seed}_vs_{right_seed}"] = {
                "same_sign_count": same,
                "different_sign_count": different,
                "tie_involved_count": tie_involved,
                "non_tie_comparison_count": non_tie,
                "same_sign_rate": same / non_tie if non_tie else None,
            }
    return agreement


def compute_sign_agreement(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], *, metric: str = "psnr"
) -> Dict[str, Any]:
    """Compute pairwise and all-seed directional agreement for PSNR or SSIM."""

    if metric not in {"psnr", "ssim"}:
        raise ValueError(f"metric must be 'psnr' or 'ssim', got {metric!r}.")
    all_cs = all_sc = all_tie = 0
    for row in rows:
        preferences = [
            _preference(float(row[f"delta_{metric}_seed{seed}"])) for seed in seeds
        ]
        if all(value == "CS" for value in preferences):
            all_cs += 1
        elif all(value == "SC" for value in preferences):
            all_sc += 1
        elif all(value == "tie" for value in preferences):
            all_tie += 1
    same = all_cs + all_sc
    total = len(rows)
    return {
        "pairwise": _pairwise_sign_agreement(rows, seeds, metric),
        "all_three_same_sign_count": same,
        "all_three_same_sign_rate": same / total if total else None,
        "all_three_CS_count": all_cs,
        "all_three_SC_count": all_sc,
        "all_three_tie_count": all_tie,
    }


def compute_threshold_stability(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    thresholds: Sequence[float] = PSNR_MARGINS,
) -> Dict[str, Dict[str, Any]]:
    """Count strict all-seed PSNR stability at each requested margin."""

    total = len(rows)
    output: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        counts = {"CS": 0, "SC": 0, "unstable": 0}
        for row in rows:
            values = [float(row[f"delta_psnr_seed{seed}"]) for seed in seeds]
            counts[_threshold_label(values, threshold)] += 1
        output[f"{threshold:.2f}"] = {
            "stable_CS_count": counts["CS"],
            "stable_SC_count": counts["SC"],
            "unstable_count": counts["unstable"],
            "stable_CS_rate": counts["CS"] / total if total else None,
            "stable_SC_rate": counts["SC"] / total if total else None,
            "unstable_rate": counts["unstable"] / total if total else None,
        }
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _delta_distribution(values: Sequence[float], *, metric: str) -> Dict[str, Any]:
    if not values:
        raise ValueError("Delta distribution requires at least one value.")
    prefix = f"delta_{metric}"
    result: Dict[str, Any] = {
        f"mean_{prefix}": statistics.fmean(values),
        f"median_{prefix}": statistics.median(values),
        f"std_{prefix}": statistics.stdev(values) if len(values) >= 2 else 0.0,
        f"mean_abs_{prefix}": statistics.fmean(abs(value) for value in values),
        f"median_abs_{prefix}": statistics.median(abs(value) for value in values),
        "percentiles": {
            "p05": _percentile(values, 0.05),
            "p25": _percentile(values, 0.25),
            "p50": _percentile(values, 0.50),
            "p75": _percentile(values, 0.75),
            "p95": _percentile(values, 0.95),
        },
    }
    if metric == "psnr":
        result["absolute_margin"] = {}
        for threshold in PSNR_MARGINS:
            count = sum(abs(value) > threshold for value in values)
            result["absolute_margin"][f"{threshold:.2f}"] = {
                "count": count,
                "rate": count / len(values),
            }
    return result


def compute_delta_statistics(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Return per-seed PSNR/SSIM distributions and cross-seed mean PSNR distribution."""

    psnr: Dict[str, Dict[str, Any]] = {}
    ssim: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        psnr[str(seed)] = _delta_distribution(
            [float(row[f"delta_psnr_seed{seed}"]) for row in rows], metric="psnr"
        )
        ssim[str(seed)] = _delta_distribution(
            [float(row[f"delta_ssim_seed{seed}"]) for row in rows], metric="ssim"
        )
    mean_psnr = _delta_distribution(
        [float(row["mean_delta_psnr"]) for row in rows], metric="psnr"
    )
    return psnr, ssim, mean_psnr


def compute_image_level_oracle(
    result: SeedValidationResult, *, tolerance: float = SIGN_TOLERANCE
) -> Dict[str, Any]:
    """Compute the whole-image PSNR Oracle and its path-corresponding SSIM."""

    samples = [result.samples[key] for key in sorted(result.samples)]
    if not samples:
        raise ValueError(f"Seed {result.seed} has no validation samples.")
    mean_psnr_cs = statistics.fmean(
        sample.psnr_color_then_scatter for sample in samples
    )
    mean_psnr_sc = statistics.fmean(
        sample.psnr_scatter_then_color for sample in samples
    )
    best_fixed_order = "CS" if mean_psnr_cs >= mean_psnr_sc else "SC"
    best_fixed_psnr = max(mean_psnr_cs, mean_psnr_sc)
    oracle_psnr_values: list[float] = []
    selected_ssim_values: list[float] = []
    metricwise_ssim_values: list[float] = []
    cs_count = sc_count = tie_count = 0
    for sample in samples:
        psnr_cs = sample.psnr_color_then_scatter
        psnr_sc = sample.psnr_scatter_then_color
        oracle_psnr_values.append(max(psnr_cs, psnr_sc))
        metricwise_ssim_values.append(
            max(sample.ssim_color_then_scatter, sample.ssim_scatter_then_color)
        )
        delta = psnr_cs - psnr_sc
        if delta > tolerance:
            cs_count += 1
            selected_ssim_values.append(sample.ssim_color_then_scatter)
        elif delta < -tolerance:
            sc_count += 1
            selected_ssim_values.append(sample.ssim_scatter_then_color)
        else:
            tie_count += 1
            # A near-tie is still resolved to the numerically larger PSNR path;
            # exact ties use CS. This keeps the associated SSIM on a path that
            # actually realizes the reported max PSNR.
            selected_ssim_values.append(
                sample.ssim_color_then_scatter
                if psnr_cs >= psnr_sc
                else sample.ssim_scatter_then_color
            )
    oracle_psnr = statistics.fmean(oracle_psnr_values)
    count = len(samples)
    return {
        "seed": result.seed,
        "mean_psnr_cs": mean_psnr_cs,
        "mean_psnr_sc": mean_psnr_sc,
        "best_fixed_order": best_fixed_order,
        "best_fixed_psnr": best_fixed_psnr,
        "oracle_psnr": oracle_psnr,
        "oracle_gain_psnr": oracle_psnr - best_fixed_psnr,
        "oracle_select_cs_count": cs_count,
        "oracle_select_sc_count": sc_count,
        "oracle_tie_count": tie_count,
        "oracle_select_cs_rate": cs_count / count,
        "oracle_select_sc_rate": sc_count / count,
        "psnr_oracle_selected_mean_ssim": statistics.fmean(selected_ssim_values),
        "metricwise_ssim_oracle": statistics.fmean(metricwise_ssim_values),
        "metricwise_ssim_oracle_status": "diagnostic_only",
    }


def _aggregate_oracle_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fields = (
        "best_fixed_psnr",
        "oracle_psnr",
        "oracle_gain_psnr",
        "oracle_select_cs_rate",
        "oracle_select_sc_rate",
    )
    output: Dict[str, Any] = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        mean = statistics.fmean(values)
        sample_std = statistics.stdev(values) if len(values) >= 2 else 0.0
        output[field] = {
            "mean": mean,
            "sample_std": sample_std,
            "mean_plus_minus_sample_std": f"{mean:.10g} ± {sample_std:.10g}",
        }
    return output


def build_cross_seed_summary(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> Dict[str, Any]:
    """Build the machine-readable cross-seed stability summary."""

    psnr_agreement = compute_sign_agreement(rows, seeds, metric="psnr")
    ssim_agreement = compute_sign_agreement(rows, seeds, metric="ssim")
    psnr_statistics, ssim_statistics, mean_statistics = compute_delta_statistics(
        rows, seeds
    )
    return {
        "seeds": list(seeds),
        "num_samples": len(rows),
        "delta_definition": DELTA_DEFINITION,
        "sign_equality_tolerance": SIGN_TOLERANCE,
        "sample_standard_deviation_ddof": 1,
        "per_seed_delta_statistics": psnr_statistics,
        "per_seed_delta_ssim_statistics": ssim_statistics,
        "cross_seed_mean_delta_statistics": mean_statistics,
        "pairwise_seed_sign_agreement": psnr_agreement["pairwise"],
        "pairwise_seed_sign_agreement_ssim": ssim_agreement["pairwise"],
        "all_three_same_sign_count": psnr_agreement["all_three_same_sign_count"],
        "all_three_same_sign_rate": psnr_agreement["all_three_same_sign_rate"],
        "all_three_CS_count": psnr_agreement["all_three_CS_count"],
        "all_three_SC_count": psnr_agreement["all_three_SC_count"],
        "all_three_tie_count": psnr_agreement["all_three_tie_count"],
        "ssim_all_seed_sign_agreement": {
            key: value for key, value in ssim_agreement.items() if key != "pairwise"
        },
        "threshold_stability": compute_threshold_stability(rows, seeds),
    }


def build_oracle_summary(
    oracle_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> Dict[str, Any]:
    """Build descriptive, non-inferential aggregation of per-seed Oracles."""

    return {
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "oracle_scope": "whole_image_validation_only",
        "oracle_definition": "mean_i(max(PSNR_CS_i, PSNR_SC_i))",
        "fixed_order_definition": "max(mean_i(PSNR_CS_i), mean_i(PSNR_SC_i))",
        "oracle_gain_definition": "oracle_psnr - best_fixed_psnr",
        "psnr_tie_tolerance": SIGN_TOLERANCE,
        "psnr_oracle_selected_ssim_definition": (
            "mean SSIM from the same path selected by per-image PSNR; "
            "a numerical PSNR tie uses its higher exact-PSNR path, then CS "
            "for an exact tie"
        ),
        "per_seed": [dict(row) for row in oracle_rows],
        "cross_seed_mean_and_sample_std": _aggregate_oracle_rows(oracle_rows),
        "metricwise_ssim_oracle": {
            "status": "diagnostic_only",
            "description": "Per-image maximum SSIM, independent of the PSNR Oracle.",
            "per_seed": {
                str(row["seed"]): row["metricwise_ssim_oracle"] for row in oracle_rows
            },
        },
    }


def _format_float(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_analysis_summary(
    results: Sequence[SeedValidationResult],
    cross_summary: Mapping[str, Any],
    oracle_rows: Sequence[Mapping[str, Any]],
    oracle_summary: Mapping[str, Any],
) -> str:
    """Render an objective Markdown summary from computed validation results."""

    lines = [
        "# V2 Validation Order Analysis",
        "",
        "This is an offline analysis of shared-order **best validation** artifacts. "
        "It does not use test results and does not run model inference.",
        "",
        "## Inputs",
        "",
        "| Seed | Run | Best validation epoch | Global step | Images |",
        "|---:|---|---:|---:|---:|",
    ]
    for result in results:
        summary = result.validation_summary
        lines.append(
            f"| {result.seed} | `{result.run_dir}` | {summary['epoch']} | "
            f"{summary['global_step']} | {summary['num_samples']} |"
        )
    lines.extend(
        [
            "",
            f"Aligned validation images: **{cross_summary['num_samples']}**.",
            "",
            f"Delta direction everywhere: `{DELTA_DEFINITION}`.",
            "",
            "## Per-seed fixed-order metrics",
            "",
            "| Seed | Mean PSNR CS | Mean PSNR SC | Mean delta PSNR (CS-SC) |",
            "|---:|---:|---:|---:|",
        ]
    )
    oracle_by_seed = {int(row["seed"]): row for row in oracle_rows}
    delta_statistics = cross_summary["per_seed_delta_statistics"]
    for result in results:
        oracle = oracle_by_seed[result.seed]
        stats = delta_statistics[str(result.seed)]
        lines.append(
            f"| {result.seed} | {_format_float(oracle['mean_psnr_cs'])} | "
            f"{_format_float(oracle['mean_psnr_sc'])} | "
            f"{_format_float(stats['mean_delta_psnr'])} |"
        )
    lines.extend(["", "## Seed preference agreement", ""])
    lines.append("| Seed pair | Same sign | Different sign | Tie involved | Same-sign rate (non-ties) |")
    lines.append("|---|---:|---:|---:|---:|")
    for pair, values in cross_summary["pairwise_seed_sign_agreement"].items():
        rate = values["same_sign_rate"]
        rendered_rate = "n/a" if rate is None else f"{100.0 * rate:.2f}%"
        lines.append(
            f"| {pair.replace('_vs_', ' vs ')} | {values['same_sign_count']} | "
            f"{values['different_sign_count']} | {values['tie_involved_count']} | "
            f"{rendered_rate} |"
        )
    all_rate = cross_summary["all_three_same_sign_rate"]
    lines.extend(
        [
            "",
            f"All three seeds have the same non-tie PSNR direction for "
            f"**{cross_summary['all_three_same_sign_count']} / "
            f"{cross_summary['num_samples']}** images "
            f"({100.0 * all_rate:.2f}%).",
            "",
            "## PSNR margin stability",
            "",
            "Stable means every seed reaches the same direction and the stated margin; "
            "a 2/3 majority is not called stable.",
            "",
            "| Margin (dB) | Stable CS | Stable SC | Unstable |",
            "|---:|---:|---:|---:|",
        ]
    )
    for threshold, values in cross_summary["threshold_stability"].items():
        lines.append(
            f"| {threshold} | {values['stable_CS_count']} | "
            f"{values['stable_SC_count']} | {values['unstable_count']} |"
        )
    lines.extend(
        [
            "",
            "## Whole-image PSNR Oracle",
            "",
            "For each validation image, the Oracle takes the higher PSNR of CS and SC. "
            "Oracle gain is relative to the better single fixed order. The associated "
            "SSIM comes from the PSNR-selected path.",
            "",
            "| Seed | Best fixed order | Best fixed PSNR | Oracle PSNR | Oracle gain |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in oracle_rows:
        lines.append(
            f"| {row['seed']} | {row['best_fixed_order']} | "
            f"{_format_float(row['best_fixed_psnr'])} | "
            f"{_format_float(row['oracle_psnr'])} | "
            f"{_format_float(row['oracle_gain_psnr'])} |"
        )
    gain = oracle_summary["cross_seed_mean_and_sample_std"]["oracle_gain_psnr"]
    lines.extend(
        [
            "",
            "Oracle gain across seeds (mean ± sample std): "
            f"**{_format_float(gain['mean'])} ± "
            f"{_format_float(gain['sample_std'])} dB**.",
            "",
            "## Interpretation boundary",
            "",
            "If Oracle gain is very small, image-level hard order selection has limited "
            "headroom. If it is materially larger than the best fixed path, image-level "
            "order selection may warrant further study. This report does not define a "
            "materiality threshold or claim statistical significance.",
            "",
        ]
    )
    return "\n".join(lines)


def _cross_seed_fieldnames(seeds: Sequence[int]) -> list[str]:
    return [
        "sample_id",
        "input_relative_path",
        "gt_relative_path",
        *(f"delta_psnr_seed{seed}" for seed in seeds),
        *(f"delta_ssim_seed{seed}" for seed in seeds),
        "mean_delta_psnr",
        "median_delta_psnr",
        "std_delta_psnr",
        "mean_delta_ssim",
        "median_delta_ssim",
        "std_delta_ssim",
        "cs_psnr_win_count",
        "sc_psnr_win_count",
        "cs_ssim_win_count",
        "sc_ssim_win_count",
        "all_three_psnr_same_preference",
        "majority_psnr_preference",
        "all_three_ssim_same_preference",
        "majority_ssim_preference",
        "stable_preference_psnr_0p01",
        "stable_preference_psnr_0p05",
        "stable_preference_psnr_0p10",
    ]


ORACLE_FIELDNAMES = (
    "seed",
    "mean_psnr_cs",
    "mean_psnr_sc",
    "best_fixed_order",
    "best_fixed_psnr",
    "oracle_psnr",
    "oracle_gain_psnr",
    "oracle_select_cs_count",
    "oracle_select_sc_count",
    "oracle_tie_count",
    "oracle_select_cs_rate",
    "oracle_select_sc_rate",
    "psnr_oracle_selected_mean_ssim",
    "metricwise_ssim_oracle",
    "metricwise_ssim_oracle_status",
)


def write_analysis_outputs(
    results: Sequence[SeedValidationResult], output_dir: Path | str
) -> Dict[str, Path]:
    """Validate, calculate, and atomically write all required analysis artifacts."""

    seeds = [result.seed for result in results]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(
            "V2 validation order analysis requires exactly three unique seeds; "
            f"got {seeds}."
        )
    validate_configs_across_seeds(results)
    aligned = align_samples_across_seeds(results)
    cross_rows = compute_cross_seed_per_image(aligned, seeds)
    cross_summary = build_cross_seed_summary(cross_rows, seeds)
    oracle_rows = [compute_image_level_oracle(result) for result in results]
    oracle_summary = build_oracle_summary(oracle_rows, seeds)
    markdown = build_analysis_summary(
        results, cross_summary, oracle_rows, oracle_summary
    )

    destination = Path(output_dir).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    fieldnames = _cross_seed_fieldnames(seeds)
    outputs = {
        "cross_seed_per_image.csv": destination / "cross_seed_per_image.csv",
        "cross_seed_summary.json": destination / "cross_seed_summary.json",
        "oracle_per_seed.csv": destination / "oracle_per_seed.csv",
        "oracle_summary.json": destination / "oracle_summary.json",
        "stable_cs_samples_0p05.csv": destination / "stable_cs_samples_0p05.csv",
        "stable_sc_samples_0p05.csv": destination / "stable_sc_samples_0p05.csv",
        "unstable_samples_0p05.csv": destination / "unstable_samples_0p05.csv",
        "analysis_summary.md": destination / "analysis_summary.md",
    }
    atomic_write_csv(outputs["cross_seed_per_image.csv"], cross_rows, fieldnames)
    atomic_write_json(outputs["cross_seed_summary.json"], cross_summary)
    atomic_write_csv(outputs["oracle_per_seed.csv"], oracle_rows, ORACLE_FIELDNAMES)
    atomic_write_json(outputs["oracle_summary.json"], oracle_summary)

    sorted_rows = sorted(
        cross_rows,
        key=lambda row: (
            -abs(float(row["mean_delta_psnr"])),
            str(row["input_relative_path"]),
            str(row["gt_relative_path"]),
        ),
    )
    label_to_filename = {
        "CS": "stable_cs_samples_0p05.csv",
        "SC": "stable_sc_samples_0p05.csv",
        "unstable": "unstable_samples_0p05.csv",
    }
    for label, filename in label_to_filename.items():
        subset = [
            row
            for row in sorted_rows
            if row["stable_preference_psnr_0p05"] == label
        ]
        atomic_write_csv(outputs[filename], subset, fieldnames)
    outputs["analysis_summary.md"].write_text(markdown, encoding="utf-8")
    return outputs


def run_analysis(
    *,
    experiments_root: Path | str = Path("experiments"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: Path | str = Path("analysis_results/v2_validation_order"),
) -> Dict[str, Path]:
    """Discover the requested completed runs and create the offline reports."""

    seed_list = list(seeds)
    if len(seed_list) != 3 or len(set(seed_list)) != 3:
        raise ValueError(
            "--seeds must contain exactly three unique integer seeds; "
            f"got {seed_list}."
        )
    root = Path(experiments_root).expanduser().resolve(strict=False)
    results = [
        load_seed_validation_result(discover_seed_run(root, seed=seed), seed=seed)
        for seed in seed_list
    ]
    return write_analysis_outputs(results, output_dir)


def discover_seed_run(experiments_root: Path | str, *, seed: int) -> Path:
    """Find one shared-order run, including the standard V2 timestamp suffix."""

    root = Path(experiments_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Seed {seed} experiments root does not exist or is not a directory: {root}"
        )
    legacy_names = {
        f"shared_order_diagnostic_seed{seed}",
        f"v2_shared_order_diagnostic_seed{seed}",
    }
    timestamped_name = re.compile(
        rf"^v2_shared_order_diagnostic_seed{seed}_"
        rf"\d{{8}}_\d{{6}}(?:_[0-9a-f]{{6}})?$"
    )
    matches = sorted(
        (
            path.resolve(strict=False)
            for path in root.iterdir()
            if path.is_dir()
            and (path.name in legacy_names or timestamped_name.fullmatch(path.name))
        ),
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(
            f"Seed {seed} shared-order run was not found under {root}. Expected "
            f"v2_shared_order_diagnostic_seed{seed}_YYYYMMDD_HHMMSS."
        )
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Seed {seed} has multiple shared-order runs under {root}; refusing "
            f"to choose one silently: {rendered}"
        )
    return matches[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline cross-seed analysis of V2 shared-order best-validation "
            "artifacts. No checkpoint or test data is read."
        )
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help=(
            "Directory containing timestamped "
            "v2_shared_order_diagnostic_seed<seed> runs."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Exactly three completed shared-order seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/v2_validation_order"),
        help="Destination for CSV, JSON, and Markdown analysis outputs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = run_analysis(
        experiments_root=args.experiments_root,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(f"Wrote V2 best-validation order analysis to {args.output_dir}")
    for name in outputs:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
