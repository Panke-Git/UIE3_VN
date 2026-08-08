from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.analysis import v2_spatial_oracle as spatial  # noqa: E402
from src.analysis.v2_soft_fusion_oracle import (  # noqa: E402
    SoftRegion,
    aggregate_soft_seed_rows,
    build_soft_summary,
    compute_alpha_stability,
    compute_soft_fusion_oracle,
    pearson_correlation,
    reconstruct_soft_fusion_oracle,
    require_soft_not_worse_than_hard,
    summarize_alpha_regions,
    write_outputs,
)
from src.common.metrics.image_metrics import (  # noqa: E402
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)


METRICS = {
    "data_range": 1.0,
    "crop_border": 0,
    "ssim_window_size": 1,
    "ssim_sigma": 1.0,
}


def test_analytic_alpha_reconstructs_exact_midpoint_target() -> None:
    target = torch.full((1, 3, 4, 4), 0.5)
    cs = torch.full_like(target, 0.4)
    sc = torch.full_like(target, 0.6)
    result = reconstruct_soft_fusion_oracle(cs, sc, target, region_size=None)
    region = result.regions[0]
    assert region.alpha_raw == pytest.approx(0.5)
    assert region.alpha_star == pytest.approx(0.5)
    assert torch.allclose(result.prediction, target)
    assert result.total_sse == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    ("target_value", "expected_raw", "expected_alpha"),
    [(0.3, -0.5, 0.0), (0.7, 1.5, 1.0)],
)
def test_alpha_raw_is_clipped_to_convex_interval(
    target_value, expected_raw, expected_alpha
) -> None:
    target = torch.full((1, 3, 2, 2), target_value)
    cs = torch.full_like(target, 0.6)
    sc = torch.full_like(target, 0.4)
    result = reconstruct_soft_fusion_oracle(cs, sc, target, region_size=None)
    region = result.regions[0]
    assert region.alpha_raw == pytest.approx(expected_raw)
    assert region.alpha_star == expected_alpha
    expected = cs if expected_alpha == 1.0 else sc
    assert torch.equal(result.prediction, expected)


def test_degenerate_equal_paths_use_half_alpha_without_changing_prediction() -> None:
    target = torch.zeros((1, 3, 3, 3))
    cs = torch.full_like(target, 0.25)
    result = reconstruct_soft_fusion_oracle(cs, cs.clone(), target, region_size=2)
    assert all(region.degenerate for region in result.regions)
    assert all(region.alpha_star == 0.5 for region in result.regions)
    assert torch.equal(result.prediction, cs)


def test_soft_is_not_worse_than_hard_and_refines_monotonically() -> None:
    generator = torch.Generator().manual_seed(23)
    target = torch.rand((1, 3, 256, 256), generator=generator)
    cs = (target + 0.09 * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    sc = (target + 0.09 * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    soft_levels = []
    for label, size in (("whole", None), ("128", 128), ("64", 64), ("32", 32)):
        hard = spatial.compute_spatial_oracle(
            cs, sc, target, region_size=size, metrics_config=METRICS
        )
        soft = compute_soft_fusion_oracle(
            cs, sc, target, region_size=size, metrics_config=METRICS
        )
        require_soft_not_worse_than_hard(soft, hard, label=label)
        assert soft.reconstructed.total_sse <= hard.reconstructed.total_sse + 1.0e-8
        soft_levels.append((label, soft))
    spatial.require_oracle_monotonicity(soft_levels)
    sses = [result.reconstructed.total_sse for _, result in soft_levels]
    psnrs = [result.psnr for _, result in soft_levels]
    assert sses == sorted(sses, reverse=True)
    assert psnrs == sorted(psnrs)


def test_soft_hard_check_allows_float32_psnr_reduction_roundoff() -> None:
    hard = SimpleNamespace(
        reconstructed=SimpleNamespace(total_sse=2.0),
        psnr=32.660850524902344,
    )
    soft = SimpleNamespace(
        reconstructed=SimpleNamespace(total_sse=1.99),
        psnr=32.66084671020508,
    )
    require_soft_not_worse_than_hard(soft, hard, label="32")

    materially_lower = SimpleNamespace(
        reconstructed=SimpleNamespace(total_sse=1.99),
        psnr=32.66082,
    )
    with pytest.raises(ValueError, match=r"abs_diff=.*allowed=1e-05"):
        require_soft_not_worse_than_hard(materially_lower, hard, label="32")


def test_soft_metric_uses_complete_reconstruction_not_patch_psnr_mean() -> None:
    target = torch.zeros((1, 3, 3, 5))
    cs = torch.full_like(target, 0.1)
    sc = torch.full_like(target, 0.3)
    cs[:, :, :, 4:] = 0.4
    sc[:, :, :, 4:] = 0.2
    measured = compute_soft_fusion_oracle(
        cs, sc, target, region_size=4, metrics_config=METRICS
    )
    direct = float(rgb_psnr_per_image(measured.reconstructed.prediction, target)[0])
    patch_values = []
    for region in measured.reconstructed.regions:
        bounds = region.bounds
        prediction = measured.reconstructed.prediction[
            :, :, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1
        ]
        patch_target = target[:, :, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        patch_values.append(float(rgb_psnr_per_image(prediction, patch_target)[0]))
    assert measured.psnr == pytest.approx(direct)
    assert measured.psnr != pytest.approx(sum(patch_values) / len(patch_values))


def test_alpha_is_solved_from_clamped_predictions() -> None:
    target = torch.ones((1, 3, 4, 4))
    cs = torch.full_like(target, 2.0)
    sc = torch.full_like(target, 0.9)
    result = reconstruct_soft_fusion_oracle(cs, sc, target, region_size=2)
    assert all(region.alpha_star == pytest.approx(1.0) for region in result.regions)
    assert torch.equal(result.prediction, target)


def test_ssim_is_reported_on_sse_optimal_soft_prediction() -> None:
    target = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
    cs = target * 0.8
    sc = target * 1.1
    result = compute_soft_fusion_oracle(
        cs, sc, target, region_size=4, metrics_config=METRICS
    )
    expected = rgb_ssim_per_image(
        result.reconstructed.prediction,
        target,
        data_range=1.0,
        crop_border=0,
        window_size=1,
        sigma=1.0,
    )
    assert result.ssim == pytest.approx(float(expected[0]))
    assert all(
        region.alpha_star
        == pytest.approx(
            min(1.0, max(0.0, region.alpha_raw))
        )
        for region in result.reconstructed.regions
    )


def test_validated_regression_tolerance_accepts_real_cuda_drift() -> None:
    saved = 33.117183685302734
    recomputed = 33.11731719970703
    rows = [
        {
            "psnr_cs": recomputed,
            "psnr_sc": 30.0,
            "ssim_cs": 0.8,
            "ssim_sc": 0.7,
        }
    ]
    summary = {
        "num_samples": 1,
        "mean_psnr_color_then_scatter": saved,
        "mean_psnr_scatter_then_color": 30.0,
        "mean_ssim_color_then_scatter": 0.8,
        "mean_ssim_scatter_then_color": 0.7,
    }
    diagnostics = spatial.validate_inference_regression(rows, summary, seed=7)
    assert diagnostics["max_abs_psnr_difference"] == pytest.approx(
        abs(recomputed - saved)
    )
    bad = dict(summary, mean_psnr_color_then_scatter=recomputed - 0.05)
    with pytest.raises(ValueError, match="regression mismatch"):
        spatial.validate_inference_regression(rows, bad, seed=7)


def test_replay_regression_collects_all_sample_differences_before_deciding() -> None:
    rows = [
        {"psnr_cs": 30.0, "psnr_sc": 29.0, "ssim_cs": 0.8, "ssim_sc": 0.7},
        {"psnr_cs": 30.0, "psnr_sc": 29.0, "ssim_cs": 0.8, "ssim_sc": 0.7},
    ]
    summary = {
        "num_samples": 2,
        "mean_psnr_color_then_scatter": 30.0,
        "mean_psnr_scatter_then_color": 29.0,
        "mean_ssim_color_then_scatter": 0.8,
        "mean_ssim_scatter_then_color": 0.7,
    }
    differences = [
        {
            "psnr_cs_difference": 0.00539398193359375,
            "psnr_sc_difference": -1.0e-4,
            "ssim_cs_difference": 2.0e-6,
            "ssim_sc_difference": -1.0e-6,
        },
        {
            "psnr_cs_difference": -1.2e-4,
            "psnr_sc_difference": 1.1e-4,
            "ssim_cs_difference": -1.5e-6,
            "ssim_sc_difference": 1.0e-6,
        },
    ]
    diagnostics = spatial.validate_inference_regression(
        rows, summary, seed=7, sample_differences=differences
    )
    assert diagnostics["max_abs_psnr_difference"] == pytest.approx(
        0.00539398193359375
    )
    assert diagnostics["p95_abs_psnr_difference"] > 0.0
    bad = [dict(differences[0], psnr_cs_difference=0.05), differences[1]]
    with pytest.raises(ValueError, match=r"checking all 2 samples.*0.05"):
        spatial.validate_inference_regression(
            rows, summary, seed=7, sample_differences=bad
        )


def _region(y0, y1, alpha, raw, *, degenerate=False):
    return SoftRegion(
        bounds=spatial.RegionBounds(y0, y1, 0, 2),
        alpha_raw=raw,
        alpha_star=alpha,
        denominator=0.0 if degenerate else 1.0,
        degenerate=degenerate,
    )


def test_alpha_aggregation_has_region_and_pixel_weighted_rates() -> None:
    regions = [
        _region(0, 1, 0.0, -0.2),
        _region(1, 4, 0.5, 0.5, degenerate=True),
        _region(4, 5, 1.0, 1.2),
    ]
    result = summarize_alpha_regions(regions)
    assert result["near_sc_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["interior_soft_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["near_cs_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["near_sc_pixel_rate"] == pytest.approx(0.2)
    assert result["interior_soft_pixel_rate"] == pytest.approx(0.6)
    assert result["near_cs_pixel_rate"] == pytest.approx(0.2)
    assert result["raw_below_zero_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["raw_inside_unit_interval_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["raw_above_one_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["degenerate_region_rate"] == pytest.approx(1.0 / 3.0)
    assert result["degenerate_pixel_rate"] == pytest.approx(0.6)


def _alpha_record(seed, x0, alpha):
    category = "near_SC" if alpha <= 0.05 else "near_CS" if alpha >= 0.95 else "soft"
    return {
        "seed": seed,
        "sample_id": "sample",
        "granularity": "32",
        "y0": 0,
        "y1": 2,
        "x0": x0,
        "x1": x0 + 2,
        "alpha_star": alpha,
        "category": category,
    }


def test_cross_seed_alpha_pearson_and_category_agreement() -> None:
    values = {
        1234: (0.0, 0.5, 1.0),
        3407: (0.1, 0.5, 0.9),
        3520: (1.0, 0.5, 0.0),
    }
    records = [
        _alpha_record(seed, index * 2, alpha)
        for seed, alphas in values.items()
        for index, alpha in enumerate(alphas)
    ]
    result = compute_alpha_stability(
        records, seeds=[1234, 3407, 3520], granularities=["32"]
    )["by_granularity"]["32"]
    assert result["pairwise"]["1234_vs_3407"]["pearson_correlation"] == pytest.approx(1.0)
    assert result["pairwise"]["1234_vs_3520"]["pearson_correlation"] == pytest.approx(-1.0)
    assert result["pairwise"]["1234_vs_3407"]["same_category_rate"] == pytest.approx(1.0 / 3.0)
    assert result["all_three_same_category_rate"] == pytest.approx(1.0 / 3.0)
    assert pearson_correlation([1.0, 1.0], [0.0, 1.0]) is None


def test_soft_seed_aggregation_reports_key_gains() -> None:
    rows = [
        {
            "seed": 7,
            "psnr_cs": 30.0,
            "psnr_sc": 29.0,
            "soft_whole_psnr": 30.5,
            "soft_whole_ssim": 0.8,
            "soft_32_psnr": 31.0,
            "soft_32_ssim": 0.81,
        },
        {
            "seed": 7,
            "psnr_cs": 28.0,
            "psnr_sc": 29.0,
            "soft_whole_psnr": 29.5,
            "soft_whole_ssim": 0.7,
            "soft_32_psnr": 30.0,
            "soft_32_ssim": 0.71,
        },
    ]
    hard = {
        "seed": 7,
        "best_fixed_order": "CS",
        "best_fixed_psnr": 29.0,
        "best_fixed_ssim": 0.75,
        "whole_oracle_psnr": 29.5,
        "whole_oracle_ssim": 0.76,
        "oracle_32_psnr": 30.0,
        "oracle_32_ssim": 0.77,
    }
    alpha_records = []
    for sample in ("a", "b"):
        for granularity in ("whole", "32"):
            alpha_records.append(
                {
                    "seed": 7,
                    "sample_id": sample,
                    "granularity": granularity,
                    "y0": 0,
                    "y1": 2,
                    "x0": 0,
                    "x1": 2,
                    "alpha_raw": 0.5,
                    "alpha_star": 0.5,
                    "degenerate": False,
                }
            )
    result = aggregate_soft_seed_rows(rows, hard, alpha_records, region_sizes=[32])
    assert result["soft_whole_psnr"] == 30.0
    assert result["soft_whole_gain_over_hard"] == 0.5
    assert result["soft_32_psnr"] == 30.5
    assert result["soft_32_gain_over_hard"] == 0.5
    assert result["soft_32_gain_over_fixed"] == 1.5
    assert result["alpha_32_interior_soft_region_rate"] == 1.0


def test_required_soft_outputs_are_written_from_toy_results(tmp_path) -> None:
    all_rows = []
    all_alpha = []
    per_seed = []
    for seed in (1234, 3407, 3520):
        rows = []
        alpha_records = []
        for index, sample in enumerate(("a", "b")):
            psnr_cs = 30.0 - index
            psnr_sc = 29.0 + index
            row = {
                "seed": seed,
                "sample_id": sample,
                "input_relative_path": f"input/{sample}.png",
                "gt_relative_path": f"gt/{sample}.png",
                "height": 2,
                "width": 2,
                "psnr_cs": psnr_cs,
                "psnr_sc": psnr_sc,
                "ssim_cs": 0.8,
                "ssim_sc": 0.7,
                "hard_whole_psnr": 30.0,
                "soft_whole_psnr": 30.25,
                "soft_whole_ssim": 0.81,
                "alpha_whole": 0.5,
                "alpha_whole_raw": 0.5,
                "alpha_whole_degenerate": False,
                "soft_whole_gain_over_hard": 0.25,
                "hard_32_psnr": 30.5,
                "soft_32_psnr": 31.0,
                "soft_32_ssim": 0.82,
                "soft_32_gain_over_hard": 0.5,
                "soft_32_gain_over_soft_whole": 0.75,
                "alpha_32_mean": 0.5,
                "alpha_32_std": 0.0,
                "alpha_32_interior_pixel_rate": 1.0,
                "alpha_32_near_cs_pixel_rate": 0.0,
                "alpha_32_near_sc_pixel_rate": 0.0,
                "alpha_32_degenerate_region_count": 0,
                "alpha_32_degenerate_pixel_rate": 0.0,
            }
            rows.append(row)
            for granularity in ("whole", "32"):
                alpha_records.append(
                    {
                        "seed": seed,
                        "sample_id": sample,
                        "granularity": granularity,
                        "y0": 0,
                        "y1": 2,
                        "x0": 0,
                        "x1": 2,
                        "pixels": 4,
                        "alpha_raw": 0.5,
                        "alpha_star": 0.5,
                        "category": "soft",
                        "degenerate": False,
                    }
                )
        hard = {
            "seed": seed,
            "best_fixed_order": "CS",
            "best_fixed_psnr": 29.5,
            "best_fixed_ssim": 0.75,
            "whole_oracle_psnr": 30.0,
            "whole_oracle_ssim": 0.76,
            "oracle_32_psnr": 30.5,
            "oracle_32_ssim": 0.77,
        }
        per_seed.append(
            aggregate_soft_seed_rows(rows, hard, alpha_records, region_sizes=[32])
        )
        all_rows.extend(rows)
        all_alpha.extend(alpha_records)
    stability = compute_alpha_stability(
        all_alpha,
        seeds=[1234, 3407, 3520],
        granularities=["whole", "32"],
    )
    summary = build_soft_summary(
        all_rows,
        per_seed,
        all_alpha,
        seeds=[1234, 3407, 3520],
        region_sizes=[32],
        regression_checks=[{"seed": seed, "status": "passed"} for seed in (1234, 3407, 3520)],
        hard_regression={"status": "skipped"},
        run_dirs=[tmp_path / str(seed) for seed in (1234, 3407, 3520)],
    )
    outputs = write_outputs(
        output_dir=tmp_path / "soft",
        image_rows=all_rows,
        per_seed_rows=per_seed,
        summary=summary,
        stability=stability,
        candidates={"whole": [], "32": []},
        region_sizes=[32],
    )
    assert set(outputs) == {
        "soft_fusion_per_image.csv",
        "soft_fusion_per_seed.csv",
        "soft_fusion_summary.json",
        "soft_fusion_alpha_stability.json",
        "analysis_summary.md",
    }
    assert all(path.is_file() for path in outputs.values())
