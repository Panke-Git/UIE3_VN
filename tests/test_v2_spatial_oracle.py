from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from src.analysis.v2_spatial_oracle import (  # noqa: E402
    VisualizationCandidate,
    aggregate_seed_rows,
    build_spatial_oracle_summary,
    classify_spatial_mix,
    configure_inference_backend,
    compute_spatial_choice_stability,
    compute_spatial_oracle,
    reconstruct_spatial_oracle,
    region_grid,
    require_oracle_monotonicity,
    require_validation_split,
    validate_inference_regression,
    validate_region_sizes,
    validate_spatial_analysis_metadata,
    write_analysis_outputs,
)
from src.common.metrics.image_metrics import (  # noqa: E402
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)
from src.v2.config import load_v2_config  # noqa: E402


METRICS = {
    "data_range": 1.0,
    "crop_border": 0,
    "ssim_window_size": 1,
    "ssim_sigma": 1.0,
}


@pytest.mark.parametrize(
    ("height", "width", "size"),
    [(256, 256, 64), (250, 190, 64), (31, 45, 64), (130, 65, 64)],
)
def test_region_grid_covers_every_pixel_exactly_once(height, width, size) -> None:
    coverage = torch.zeros((height, width), dtype=torch.int32)
    regions = region_grid(height, width, size)
    for region in regions:
        coverage[region.y0 : region.y1, region.x0 : region.x1] += 1
        assert 0 <= region.y0 < region.y1 <= height
        assert 0 <= region.x0 < region.x1 <= width
        assert region.y1 - region.y0 <= size
        assert region.x1 - region.x0 <= size
    assert torch.all(coverage == 1)
    assert regions[-1].y1 == height
    assert regions[-1].x1 == width


def test_region_size_validation() -> None:
    assert validate_region_sizes([128, 64, 32]) == (128, 64, 32)
    for invalid in ([], [0], [64, 128], [128, 48, 32], [128, 64, 64]):
        with pytest.raises(ValueError):
            validate_region_sizes(invalid)


def test_spatial_oracle_reconstructs_hand_calculated_left_and_right() -> None:
    target = torch.zeros((1, 3, 4, 4))
    cs = torch.empty_like(target)
    sc = torch.empty_like(target)
    cs[:, :, :, :2] = 0.1
    cs[:, :, :, 2:] = 0.9
    sc[:, :, :, :2] = 0.8
    sc[:, :, :, 2:] = 0.2
    whole = reconstruct_spatial_oracle(cs, sc, target, region_size=None)
    spatial = reconstruct_spatial_oracle(cs, sc, target, region_size=2)
    assert whole.tiles[0].choice == "SC"
    assert torch.equal(whole.prediction, sc)
    expected = torch.cat((cs[:, :, :, :2], sc[:, :, :, 2:]), dim=3)
    assert torch.equal(spatial.prediction, expected)
    assert spatial.total_sse < whole.total_sse
    assert {tile.choice for tile in spatial.tiles} == {"CS", "SC"}


def test_final_psnr_uses_complete_oracle_not_mean_tile_psnr() -> None:
    target = torch.zeros((1, 3, 3, 5))
    cs = torch.full_like(target, 0.1)
    sc = torch.full_like(target, 0.3)
    cs[:, :, :, 4:] = 0.4
    sc[:, :, :, 4:] = 0.2
    reconstructed = reconstruct_spatial_oracle(cs, sc, target, region_size=4)
    measured = compute_spatial_oracle(
        cs, sc, target, region_size=4, metrics_config=METRICS
    )
    expected = float(
        rgb_psnr_per_image(reconstructed.prediction, target)[0]
    )
    tile_psnr = []
    for tile in reconstructed.tiles:
        bounds = tile.bounds
        prediction = reconstructed.prediction[
            :, :, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1
        ]
        tile_target = target[:, :, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        tile_psnr.append(float(rgb_psnr_per_image(prediction, tile_target)[0]))
    assert measured.psnr == pytest.approx(expected)
    assert measured.psnr != pytest.approx(sum(tile_psnr) / len(tile_psnr))


def test_oracle_selection_uses_clamped_predictions() -> None:
    target = torch.ones((1, 3, 4, 4))
    cs = torch.full_like(target, 2.0)
    sc = torch.full_like(target, 0.9)
    reconstructed = reconstruct_spatial_oracle(cs, sc, target, region_size=2)
    assert all(tile.choice == "CS" for tile in reconstructed.tiles)
    assert torch.equal(reconstructed.prediction, target)


def test_refined_oracle_sse_and_psnr_are_monotonic() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.rand((1, 3, 256, 256), generator=generator)
    cs = (target + 0.08 * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    sc = (target + 0.08 * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    levels = [("whole", compute_spatial_oracle(cs, sc, target, region_size=None, metrics_config=METRICS))]
    levels.extend(
        (
            str(size),
            compute_spatial_oracle(cs, sc, target, region_size=size, metrics_config=METRICS),
        )
        for size in (128, 64, 32)
    )
    require_oracle_monotonicity(levels)
    sses = [level.reconstructed.total_sse for _, level in levels]
    psnrs = [level.psnr for _, level in levels]
    assert sses == sorted(sses, reverse=True)
    assert psnrs == sorted(psnrs)


def test_ssim_is_measured_from_psnr_sse_selected_reconstruction() -> None:
    target = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
    cs = target.clone()
    sc = 1.0 - target
    cs[:, :, :, 4:] += 0.3
    sc[:, :, :, :4] += 0.3
    measured = compute_spatial_oracle(
        cs, sc, target, region_size=4, metrics_config=METRICS
    )
    direct = rgb_ssim_per_image(
        measured.reconstructed.prediction,
        target,
        data_range=1.0,
        crop_border=0,
        window_size=1,
        sigma=1.0,
    )
    assert measured.ssim == pytest.approx(float(direct[0]))
    for tile in measured.reconstructed.tiles:
        expected = "CS" if tile.sse_cs < tile.sse_sc else "SC"
        assert tile.choice == expected


def _checkpoint_metadata(config):
    checkpoint = {
        "config": config,
        "seed": config["experiment"]["seed"],
        "epoch": 4,
        "global_step": 50,
        "train_loss": 0.2,
        "val_loss": 0.1,
        "psnr": 20.0,
        "ssim": 0.8,
        "learning_rate": 2.0e-4,
        "validation_psnr_cs": 21.0,
        "validation_psnr_sc": 19.0,
        "validation_ssim_cs": 0.81,
        "validation_ssim_sc": 0.79,
        "validation_loss_cs": 0.09,
        "validation_loss_sc": 0.11,
        "validation_mean_path_psnr": 20.0,
        "validation_mean_path_ssim": 0.8,
    }
    sidecar = {
        "selection_metric": "validation_mean_path_psnr",
        "checkpoint": "best_psnr.pt",
        "epoch": 4,
        "global_step": 50,
        "train_loss": 0.2,
        "val_loss": 0.1,
        "psnr": 20.0,
        "ssim": 0.8,
        "learning_rate": 2.0e-4,
    }
    return checkpoint, sidecar


def test_spatial_metadata_reuses_strict_shared_checkpoint_validation() -> None:
    config = load_v2_config(variant="shared_order_diagnostic", seed=7)
    checkpoint, sidecar = _checkpoint_metadata(config)
    validated = validate_spatial_analysis_metadata(
        expected_seed=7,
        run_config=config,
        checkpoint=checkpoint,
        sidecar=sidecar,
    )
    assert validated["experiment"]["variant"] == "shared_order_diagnostic"
    bad_sidecar = dict(sidecar, selection_metric="validation_ssim")
    with pytest.raises(ValueError, match="selection_metric mismatch"):
        validate_spatial_analysis_metadata(
            expected_seed=7,
            run_config=config,
            checkpoint=checkpoint,
            sidecar=bad_sidecar,
        )


def test_spatial_split_variant_and_seed_guards() -> None:
    shared = load_v2_config(variant="shared_order_diagnostic", seed=7)
    checkpoint, sidecar = _checkpoint_metadata(shared)
    with pytest.raises(ValueError, match="only permits split='validation'"):
        validate_spatial_analysis_metadata(
            expected_seed=7,
            run_config=shared,
            checkpoint=checkpoint,
            sidecar=sidecar,
            split="test",
        )
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_spatial_analysis_metadata(
            expected_seed=8,
            run_config=shared,
            checkpoint=checkpoint,
            sidecar=sidecar,
        )
    fixed = load_v2_config(variant="color_then_scatter", seed=7)
    fixed_checkpoint, fixed_sidecar = _checkpoint_metadata(fixed)
    fixed_sidecar["selection_metric"] = "validation_psnr"
    with pytest.raises(ValueError, match="requires variant='shared_order_diagnostic'"):
        validate_spatial_analysis_metadata(
            expected_seed=7,
            run_config=fixed,
            checkpoint=fixed_checkpoint,
            sidecar=fixed_sidecar,
        )


@pytest.mark.parametrize("deterministic", [False, True])
def test_inference_backend_matches_training_config(
    monkeypatch, deterministic: bool
) -> None:
    from src.common.experiment import seed as seed_module

    calls = []

    def fake_set_global_seed(seed: int, *, deterministic: bool = False) -> None:
        calls.append((seed, deterministic))

    monkeypatch.setattr(seed_module, "set_global_seed", fake_set_global_seed)
    configure_inference_backend(
        {
            "experiment": {"seed": 3407},
            "training": {"deterministic": deterministic},
        }
    )
    assert calls == [(3407, deterministic)]


def test_validation_only_dataloader_never_requests_test_split(monkeypatch) -> None:
    from torch.utils.data import Dataset, SequentialSampler

    from src.common.data import dataloader as dataloader_module

    requested = []

    class ToyDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"value": index}

    def fake_dataset(config, split):
        requested.append(split)
        return ToyDataset()

    monkeypatch.setattr(dataloader_module, "_dataset", fake_dataset)
    config = {
        "experiment": {"seed": 7},
        "data": {"num_workers": 0, "pin_memory": False},
    }
    loader = dataloader_module.build_validation_dataloader(config)
    assert requested == ["validation"]
    assert loader.batch_size == 1
    assert isinstance(loader.sampler, SequentialSampler)
    assert list(loader)[0]["value"].tolist() == [0]


def test_inference_regression_match_and_mismatch() -> None:
    rows = [
        {"psnr_cs": 30.0, "psnr_sc": 29.0, "ssim_cs": 0.8, "ssim_sc": 0.7},
        {"psnr_cs": 28.0, "psnr_sc": 29.0, "ssim_cs": 0.6, "ssim_sc": 0.9},
    ]
    summary = {
        "num_samples": 2,
        "mean_psnr_color_then_scatter": 29.0,
        "mean_psnr_scatter_then_color": 29.0,
        "mean_ssim_color_then_scatter": 0.7,
        "mean_ssim_scatter_then_color": 0.8,
    }
    recomputed = validate_inference_regression(rows, summary, seed=7)
    assert recomputed["mean_psnr_color_then_scatter"] == 29.0
    bad = dict(summary, mean_psnr_color_then_scatter=29.1)
    with pytest.raises(ValueError, match="inference regression mismatch"):
        validate_inference_regression(rows, bad, seed=7)


def test_inference_regression_uses_metric_specific_cuda_replay_tolerances() -> None:
    rows = [
        {
            "psnr_cs": 30.0005,
            "psnr_sc": 29.0,
            "ssim_cs": 0.80005,
            "ssim_sc": 0.7,
        }
    ]
    summary = {
        "num_samples": 1,
        "mean_psnr_color_then_scatter": 30.0,
        "mean_psnr_scatter_then_color": 29.0,
        "mean_ssim_color_then_scatter": 0.8,
        "mean_ssim_scatter_then_color": 0.7,
    }
    validate_inference_regression(rows, summary, seed=7)

    bad_psnr = dict(summary, mean_psnr_color_then_scatter=29.999)
    with pytest.raises(ValueError, match=r"abs_diff=.*allowed="):
        validate_inference_regression(rows, bad_psnr, seed=7)

    bad_ssim = dict(summary, mean_ssim_color_then_scatter=0.7998)
    with pytest.raises(ValueError, match=r"abs_diff=.*allowed="):
        validate_inference_regression(rows, bad_ssim, seed=7)


def _aggregate_rows(seed: int):
    return [
        {
            "seed": seed,
            "sample_id": "small",
            "height": 2,
            "width": 2,
            "psnr_cs": 30.0,
            "psnr_sc": 29.0,
            "ssim_cs": 0.8,
            "ssim_sc": 0.7,
            "whole_oracle_psnr": 30.0,
            "whole_oracle_ssim": 0.8,
            "oracle_2_psnr": 31.0,
            "oracle_2_ssim": 0.81,
            "pixels_2_cs_rate": 0.5,
            "pixels_2_sc_rate": 0.5,
        },
        {
            "seed": seed,
            "sample_id": "large",
            "height": 2,
            "width": 6,
            "psnr_cs": 20.0,
            "psnr_sc": 22.0,
            "ssim_cs": 0.5,
            "ssim_sc": 0.6,
            "whole_oracle_psnr": 23.0,
            "whole_oracle_ssim": 0.65,
            "oracle_2_psnr": 24.0,
            "oracle_2_ssim": 0.66,
            "pixels_2_cs_rate": 0.25,
            "pixels_2_sc_rate": 0.75,
        },
    ]


def test_aggregation_uses_mean_per_image_and_pixel_weighted_rates() -> None:
    rows = _aggregate_rows(7)
    aggregate = aggregate_seed_rows(rows, 7, [2])
    assert aggregate["mean_psnr_cs"] == 25.0
    assert aggregate["mean_psnr_sc"] == 25.5
    assert aggregate["best_fixed_order"] == "SC"
    assert aggregate["best_fixed_psnr"] == 25.5
    assert aggregate["whole_oracle_psnr"] == 26.5
    assert aggregate["oracle_2_psnr"] == 27.5
    assert aggregate["gain_2_over_fixed"] == 2.0
    assert aggregate["gain_2_over_whole"] == 1.0
    assert aggregate["cs_pixel_rate_2"] == pytest.approx(5.0 / 16.0)
    assert classify_spatial_mix(0.95, 0.05) == "mostly_cs"
    assert classify_spatial_mix(0.05, 0.95) == "mostly_sc"
    assert classify_spatial_mix(0.90, 0.10) == "mixed"


def _tile(seed, sample, x0, choice):
    return {
        "seed": seed,
        "sample_id": sample,
        "region_size": 2,
        "y0": 0,
        "y1": 2,
        "x0": x0,
        "x1": x0 + 2,
        "choice": choice,
    }


def test_cross_seed_tile_choice_agreement_aligns_coordinates_and_tracks_ties() -> None:
    rows = []
    choices = {
        1234: ("CS", "CS", "tie"),
        3407: ("CS", "SC", "tie"),
        3520: ("CS", "SC", "tie"),
    }
    for seed, values in choices.items():
        rows.extend(
            _tile(seed, "sample", index * 2, choice)
            for index, choice in enumerate(values)
        )
    stability = compute_spatial_choice_stability(
        rows, seeds=[1234, 3407, 3520], region_sizes=[2]
    )["by_region_size"]["2"]
    assert stability["pairwise"]["1234_vs_3407"]["same_choice_rate"] == 0.5
    assert stability["pairwise"]["3407_vs_3520"]["same_choice_rate"] == 1.0
    assert stability["all_three_same_choice_count"] == 1
    assert stability["all_three_same_choice_rate"] == pytest.approx(1.0 / 3.0)
    assert stability["all_three_CS_count"] == 1
    assert stability["all_three_SC_count"] == 0
    assert stability["all_three_tie_count"] == 1


def test_required_outputs_are_written_from_toy_aggregates(tmp_path) -> None:
    image_rows = []
    per_seed = []
    tiles = []
    for seed in (1234, 3407, 3520):
        seed_rows = _aggregate_rows(seed)
        per_seed.append(aggregate_seed_rows(seed_rows, seed, [2]))
        for row in seed_rows:
            row.update(
                {
                    "input_relative_path": f"input/{row['sample_id']}.png",
                    "gt_relative_path": f"gt/{row['sample_id']}.png",
                    "whole_oracle_selected_order": "CS",
                    "tiles_2_total": 1,
                    "tiles_2_cs": 1,
                    "tiles_2_sc": 0,
                    "tiles_2_tie": 0,
                }
            )
        image_rows.extend(seed_rows)
        tiles.extend(_tile(seed, sample, 0, "CS") for sample in ("small", "large"))
    stability = compute_spatial_choice_stability(
        tiles, seeds=[1234, 3407, 3520], region_sizes=[2]
    )
    summary = build_spatial_oracle_summary(
        image_rows,
        per_seed,
        seeds=[1234, 3407, 3520],
        region_sizes=[2],
        regression_checks=[{"seed": seed, "status": "passed"} for seed in (1234, 3407, 3520)],
        run_dirs=[tmp_path / str(seed) for seed in (1234, 3407, 3520)],
    )
    visual = VisualizationCandidate(
        seed=1234,
        sample_id="small",
        region_size=2,
        gain_over_whole=1.0,
        psnr_cs=30.0,
        psnr_sc=29.0,
        whole_psnr=30.0,
        spatial_psnr=31.0,
        input_tensor=torch.zeros((3, 2, 2)),
        target_tensor=torch.zeros((3, 2, 2)),
        prediction_cs=torch.full((3, 2, 2), 0.1),
        prediction_sc=torch.full((3, 2, 2), 0.2),
        oracle_prediction=torch.full((3, 2, 2), 0.1),
        selected_cs_mask=torch.tensor([[True, False], [True, False]]),
        data_range=1.0,
    )
    outputs = write_analysis_outputs(
        output_dir=tmp_path / "analysis",
        image_rows=image_rows,
        per_seed_rows=per_seed,
        summary=summary,
        stability=stability,
        candidates={2: [visual]},
        region_sizes=[2],
    )
    assert set(outputs) == {
        "spatial_oracle_per_image.csv",
        "spatial_oracle_per_seed.csv",
        "spatial_oracle_summary.json",
        "spatial_choice_stability.json",
        "analysis_summary.md",
    }
    assert all(path.is_file() for path in outputs.values())
    panels = list((tmp_path / "analysis/visualizations").glob("*.png"))
    assert len(panels) == 1
    saved = json.loads(
        outputs["spatial_oracle_summary.json"].read_text(encoding="utf-8")
    )
    assert saved["split"] == "validation"
    assert saved["selection_metric"] == "regional RGB SSE"
    assert saved["metric_semantics"] == "mean_of_per_image_rgb_metrics"
    assert math.isfinite(
        saved["per_image_gain_distributions"]["2"]["gain_over_whole"]["mean"]
    )
    require_validation_split("validation")
