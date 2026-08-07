from __future__ import annotations

import csv
import json
import math
import statistics

import pytest

from src.analysis.v2_order_validation import (
    COMPARISON_FILENAME,
    DEFAULT_SEEDS,
    SUMMARY_FILENAME,
    align_samples_across_seeds,
    compute_cross_seed_per_image,
    compute_image_level_oracle,
    compute_threshold_stability,
    discover_seed_run,
    load_seed_validation_result,
    run_analysis,
)


def _sample(
    sample_id: str,
    psnr_cs: float,
    psnr_sc: float,
    *,
    ssim_cs: float = 0.8,
    ssim_sc: float = 0.7,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "input_relative_path": f"input/{sample_id}.png",
        "gt_relative_path": f"gt/{sample_id}.png",
        "psnr_color_then_scatter": psnr_cs,
        "psnr_scatter_then_color": psnr_sc,
        "delta_psnr_cs_minus_sc": psnr_cs - psnr_sc,
        "ssim_color_then_scatter": ssim_cs,
        "ssim_scatter_then_color": ssim_sc,
        "delta_ssim_cs_minus_sc": ssim_cs - ssim_sc,
    }


def _config(seed: int, *, variant: str = "shared_order_diagnostic") -> dict:
    return {
        "experiment": {
            "version": "v2",
            "name": f"{variant}_seed{seed}",
            "variant": variant,
            "seed": seed,
            "output_root": "experiments",
        },
        "data": {"validation_manifest": "splits/validation.tsv"},
        "model": {"backbone": {"width": 4}},
        "order_study": {"shared_checkpoint_metric": "mean_path_psnr"},
        "loss": {"name": "charbonnier"},
        "optimizer": {"name": "AdamW"},
        "scheduler": {"name": "none"},
        "training": {"epochs": 2, "resume": None},
        "metrics": {"data_range": 1.0},
        "checkpoint": {"primary": "psnr"},
        "test": {"run_dir": None, "allow_overwrite": False},
        "logging": {"console": True},
    }


def _summary(rows: list[dict[str, object]], *, epoch: int = 4) -> dict:
    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    mean_cs = mean("psnr_color_then_scatter")
    mean_sc = mean("psnr_scatter_then_color")
    return {
        "epoch": epoch,
        "global_step": 50,
        "num_samples": len(rows),
        "mean_psnr_color_then_scatter": mean_cs,
        "mean_psnr_scatter_then_color": mean_sc,
        "mean_ssim_color_then_scatter": mean("ssim_color_then_scatter"),
        "mean_ssim_scatter_then_color": mean("ssim_scatter_then_color"),
        "mean_delta_psnr_cs_minus_sc": mean("delta_psnr_cs_minus_sc"),
        "mean_delta_ssim_cs_minus_sc": mean("delta_ssim_cs_minus_sc"),
        "mean_path_psnr": 0.5 * (mean_cs + mean_sc),
    }


def _write_run(
    experiments_root,
    seed: int,
    rows: list[dict[str, object]],
    *,
    variant: str = "shared_order_diagnostic",
    config_seed: int | None = None,
    sidecar_epoch: int = 4,
    summary_epoch: int = 4,
):
    run = experiments_root / f"shared_order_diagnostic_seed{seed}"
    (run / "best").mkdir(parents=True)
    (run / "result").mkdir()
    config = _config(seed if config_seed is None else config_seed, variant=variant)
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    summary = _summary(rows, epoch=summary_epoch)
    (run / SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")
    sidecar = {
        "selection_metric": "validation_mean_path_psnr",
        "epoch": sidecar_epoch,
        "global_step": 50,
        "psnr": summary["mean_path_psnr"],
    }
    (run / "best/best_psnr.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with (run / COMPARISON_FILENAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run


def _write_three_runs(experiments_root, rows=None):
    rows = rows or [
        _sample("a", 30.0, 29.0, ssim_cs=0.70, ssim_sc=0.90),
        _sample("b", 28.0, 29.0, ssim_cs=0.95, ssim_sc=0.80),
        _sample("c", 30.0, 30.0, ssim_cs=0.60, ssim_sc=0.99),
    ]
    return [
        _write_run(experiments_root, seed, list(rows)) for seed in DEFAULT_SEEDS
    ]


def test_three_valid_seeds_can_be_analyzed_and_all_outputs_exist(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    _write_three_runs(experiments)
    output = tmp_path / "analysis"
    outputs = run_analysis(experiments_root=experiments, output_dir=output)
    expected = {
        "cross_seed_per_image.csv",
        "cross_seed_summary.json",
        "oracle_per_seed.csv",
        "oracle_summary.json",
        "stable_cs_samples_0p05.csv",
        "stable_sc_samples_0p05.csv",
        "unstable_samples_0p05.csv",
        "analysis_summary.md",
    }
    assert set(outputs) == expected
    assert all(path.is_file() for path in outputs.values())
    summary = json.loads(
        outputs["cross_seed_summary.json"].read_text(encoding="utf-8")
    )
    assert summary["seeds"] == list(DEFAULT_SEEDS)
    assert summary["num_samples"] == 3
    assert summary["delta_definition"].startswith("CS-SC")
    markdown = outputs["analysis_summary.md"].read_text(encoding="utf-8")
    assert "best validation" in markdown
    assert "does not use test" in markdown


def test_timestamped_v2_run_directories_are_discovered(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    timestamps = ("20260805_232118", "20260806_081226", "20260806_081254")
    for seed, timestamp in zip(DEFAULT_SEEDS, timestamps):
        original = _write_run(experiments, seed, [_sample("a", 30.0, 29.0)])
        timestamped = experiments / (
            f"v2_shared_order_diagnostic_seed{seed}_{timestamp}"
        )
        original.rename(timestamped)
        assert discover_seed_run(experiments, seed=seed) == timestamped.resolve()
    outputs = run_analysis(
        experiments_root=experiments, output_dir=tmp_path / "analysis"
    )
    assert outputs["analysis_summary.md"].is_file()


def test_multiple_timestamped_runs_for_one_seed_are_rejected(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    original = _write_run(experiments, 1234, [_sample("a", 30.0, 29.0)])
    first = experiments / "v2_shared_order_diagnostic_seed1234_20260805_232118"
    original.rename(first)
    second = experiments / "v2_shared_order_diagnostic_seed1234_20260806_010101"
    second.mkdir()
    with pytest.raises(ValueError, match="multiple shared-order runs"):
        discover_seed_run(experiments, seed=1234)


@pytest.mark.parametrize(
    "relative_path",
    ["config.json", "best/best_psnr.json", COMPARISON_FILENAME, SUMMARY_FILENAME],
)
def test_missing_required_seed_file_fails_with_seed_and_path(
    tmp_path, relative_path
) -> None:
    experiments = tmp_path / "experiments"
    runs = _write_three_runs(experiments)
    target = runs[1] / relative_path
    target.unlink()
    with pytest.raises(FileNotFoundError, match=rf"Seed 3407.*{target.name}"):
        run_analysis(experiments_root=experiments, output_dir=tmp_path / "out")


def test_non_shared_variant_is_rejected(tmp_path) -> None:
    run = _write_run(
        tmp_path,
        1234,
        [_sample("a", 30.0, 29.0)],
        variant="color_then_scatter",
    )
    with pytest.raises(ValueError, match="must use variant=.*shared_order_diagnostic"):
        load_seed_validation_result(run, seed=1234)


def test_config_seed_mismatch_is_rejected(tmp_path) -> None:
    run = _write_run(
        tmp_path,
        1234,
        [_sample("a", 30.0, 29.0)],
        config_seed=99,
    )
    with pytest.raises(ValueError, match="config seed mismatch"):
        load_seed_validation_result(run, seed=1234)


def test_checkpoint_and_best_summary_epoch_mismatch_is_rejected(tmp_path) -> None:
    run = _write_run(
        tmp_path,
        1234,
        [_sample("a", 30.0, 29.0)],
        sidecar_epoch=3,
        summary_epoch=4,
    )
    with pytest.raises(ValueError, match="checkpoint epoch mismatch"):
        load_seed_validation_result(run, seed=1234)


@pytest.mark.parametrize("extra", [False, True])
def test_cross_seed_missing_or_extra_image_is_rejected(tmp_path, extra) -> None:
    experiments = tmp_path / "experiments"
    base = [_sample("a", 30.0, 29.0), _sample("b", 28.0, 29.0)]
    _write_run(experiments, 1234, base)
    _write_run(experiments, 3407, base)
    changed = list(base)
    if extra:
        changed.append(_sample("c", 31.0, 30.0))
    else:
        changed.pop()
    _write_run(experiments, 3520, changed)
    with pytest.raises(ValueError, match="sample set does not match"):
        run_analysis(experiments_root=experiments, output_dir=tmp_path / "out")


def test_duplicate_sample_id_is_rejected(tmp_path) -> None:
    first = _sample("a", 30.0, 29.0)
    second = _sample("b", 28.0, 29.0)
    second["sample_id"] = "a"
    run = _write_run(tmp_path, 1234, [first, second])
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_seed_validation_result(run, seed=1234)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_csv_metric_is_rejected(tmp_path, bad_value) -> None:
    row = _sample("a", 30.0, 29.0)
    row["psnr_color_then_scatter"] = bad_value
    run = _write_run(tmp_path, 1234, [row])
    with pytest.raises(ValueError, match="must be finite"):
        load_seed_validation_result(run, seed=1234)


def test_csv_recomputed_delta_mismatch_is_rejected(tmp_path) -> None:
    row = _sample("a", 30.0, 29.0)
    row["delta_psnr_cs_minus_sc"] = 123.0
    run = _write_run(tmp_path, 1234, [row])
    with pytest.raises(ValueError, match="PSNR delta mismatch"):
        load_seed_validation_result(run, seed=1234)


def test_different_csv_row_orders_align_by_image_paths(tmp_path) -> None:
    rows = [
        _sample("b", 28.0, 29.0),
        _sample("a", 30.0, 29.0),
        _sample("c", 30.0, 30.0),
    ]
    experiments = tmp_path / "experiments"
    _write_run(experiments, 1234, rows)
    _write_run(experiments, 3407, list(reversed(rows)))
    _write_run(experiments, 3520, [rows[1], rows[2], rows[0]])
    results = [
        load_seed_validation_result(
            experiments / f"shared_order_diagnostic_seed{seed}", seed=seed
        )
        for seed in DEFAULT_SEEDS
    ]
    aligned = align_samples_across_seeds(results)
    assert [item.sample_id for item in aligned] == ["a", "b", "c"]


def test_cross_seed_hand_calculated_statistics_and_preferences(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    deltas_by_seed = {
        1234: [0.06, -0.06, 0.06],
        3407: [0.07, -0.07, 0.07],
        3520: [0.08, -0.08, 0.04],
    }
    for seed, deltas in deltas_by_seed.items():
        rows = [
            _sample(
                sample_id,
                30.0 + delta,
                30.0,
                ssim_cs=0.80 + delta / 10.0,
                ssim_sc=0.80,
            )
            for sample_id, delta in zip(("stable_cs", "stable_sc", "majority"), deltas)
        ]
        _write_run(experiments, seed, rows)
    results = [
        load_seed_validation_result(
            experiments / f"shared_order_diagnostic_seed{seed}", seed=seed
        )
        for seed in DEFAULT_SEEDS
    ]
    rows = compute_cross_seed_per_image(
        align_samples_across_seeds(results), DEFAULT_SEEDS
    )
    by_id = {row["sample_id"]: row for row in rows}
    stable_cs = by_id["stable_cs"]
    assert stable_cs["mean_delta_psnr"] == pytest.approx(0.07)
    assert stable_cs["median_delta_psnr"] == pytest.approx(0.07)
    assert stable_cs["std_delta_psnr"] == pytest.approx(0.01)
    assert stable_cs["cs_psnr_win_count"] == 3
    assert stable_cs["sc_psnr_win_count"] == 0
    assert stable_cs["all_three_psnr_same_preference"] is True
    assert stable_cs["majority_psnr_preference"] == "CS"
    assert stable_cs["stable_preference_psnr_0p05"] == "CS"
    assert by_id["stable_sc"]["stable_preference_psnr_0p05"] == "SC"
    majority = by_id["majority"]
    assert majority["mean_delta_psnr"] == pytest.approx(0.17 / 3.0)
    assert majority["majority_psnr_preference"] == "CS"
    assert majority["stable_preference_psnr_0p05"] == "unstable"
    stability = compute_threshold_stability(rows, DEFAULT_SEEDS)
    assert stability["0.05"]["stable_CS_count"] == 1
    assert stability["0.05"]["stable_SC_count"] == 1
    assert stability["0.05"]["unstable_count"] == 1


def test_oracle_matches_manual_values_and_uses_psnr_selected_ssim(tmp_path) -> None:
    rows = [
        _sample("image1", 30.0, 29.0, ssim_cs=0.70, ssim_sc=0.90),
        _sample("image2", 28.0, 29.0, ssim_cs=0.95, ssim_sc=0.80),
        _sample("image3", 30.0, 30.0, ssim_cs=0.60, ssim_sc=0.99),
    ]
    run = _write_run(tmp_path, 1234, rows)
    result = load_seed_validation_result(run, seed=1234)
    oracle = compute_image_level_oracle(result)
    assert oracle["mean_psnr_cs"] == pytest.approx(88.0 / 3.0)
    assert oracle["mean_psnr_sc"] == pytest.approx(88.0 / 3.0)
    assert oracle["best_fixed_order"] == "CS"
    assert oracle["best_fixed_psnr"] == pytest.approx(88.0 / 3.0)
    assert oracle["oracle_psnr"] == pytest.approx(89.0 / 3.0)
    assert oracle["oracle_gain_psnr"] == pytest.approx(1.0 / 3.0)
    assert oracle["oracle_select_cs_count"] == 1
    assert oracle["oracle_select_sc_count"] == 1
    assert oracle["oracle_tie_count"] == 1
    assert oracle["psnr_oracle_selected_mean_ssim"] == pytest.approx(0.70)
    assert oracle["metricwise_ssim_oracle"] == pytest.approx((0.90 + 0.95 + 0.99) / 3)
    assert oracle["metricwise_ssim_oracle_status"] == "diagnostic_only"


def test_cross_seed_config_semantic_mismatch_is_rejected(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    runs = _write_three_runs(experiments)
    config_path = runs[2] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["optimizer"]["name"] = "SGD"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"config.optimizer.name differs"):
        run_analysis(experiments_root=experiments, output_dir=tmp_path / "out")


def test_output_0p05_subsets_are_strict_and_sorted_by_absolute_mean(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    values = {
        1234: [("cs", 0.06), ("sc", -0.20), ("mixed", 0.06)],
        3407: [("cs", 0.08), ("sc", -0.18), ("mixed", 0.07)],
        3520: [("cs", 0.07), ("sc", -0.19), ("mixed", 0.04)],
    }
    for seed, seed_values in values.items():
        rows = [
            _sample(name, 30.0 + delta, 30.0)
            for name, delta in seed_values
        ]
        _write_run(experiments, seed, rows)
    outputs = run_analysis(
        experiments_root=experiments, output_dir=tmp_path / "out"
    )

    def ids(filename):
        with outputs[filename].open("r", encoding="utf-8", newline="") as handle:
            return [row["sample_id"] for row in csv.DictReader(handle)]

    assert ids("stable_cs_samples_0p05.csv") == ["cs"]
    assert ids("stable_sc_samples_0p05.csv") == ["sc"]
    assert ids("unstable_samples_0p05.csv") == ["mixed"]
    oracle_summary = json.loads(
        outputs["oracle_summary.json"].read_text(encoding="utf-8")
    )
    gain = oracle_summary["cross_seed_mean_and_sample_std"]["oracle_gain_psnr"]
    assert math.isfinite(gain["mean"])
    assert math.isfinite(gain["sample_std"])
