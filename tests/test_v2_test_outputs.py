from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.common.experiment.checkpoint import BestTracker, build_checkpoint_payload, save_checkpoint  # noqa: E402
from src.v1.trainer import _make_grad_scaler  # noqa: E402
from src.v2.config import load_v2_config, validate_v2_config  # noqa: E402
from src.v2.model import build_v2_model  # noqa: E402
from src.v2.test_v2 import execute_test  # noqa: E402
from src.v2.train_v2 import _extend_shared_checkpoint  # noqa: E402


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir(parents=True)
    lines = []
    for index in range(2):
        grid = np.arange(16 * 17 * 3, dtype=np.uint16).reshape(16, 17, 3)
        input_array = ((grid + index * 13) % 180).astype(np.uint8)
        gt_array = np.clip(input_array.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        Image.fromarray(input_array, mode="RGB").save(root / f"input/{index}.png")
        Image.fromarray(gt_array, mode="RGB").save(root / f"gt/{index}.png")
        lines.append(f"sample{index}\tinput/{index}.png\tgt/{index}.png\n")
    manifest = tmp_path / "pairs.tsv"
    manifest.write_text("".join(lines), encoding="utf-8")
    return root, manifest


def _config(tmp_path: Path, variant: str):
    root, manifest = _write_dataset(tmp_path)
    config = load_v2_config(variant=variant, seed=5)
    config["data"].update(
        root=str(root),
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        test_manifest=str(manifest),
        num_workers=0,
        pin_memory=False,
        patch_size=8,
        batch_size=1,
    )
    config["model"]["backbone"].update(
        width=4, enc_blk_nums=[0], middle_blk_num=0, dec_blk_nums=[0]
    )
    config["model"]["color_operator"]["hidden_channels"] = 4
    config["model"]["scattering_operator"]["hidden_channels"] = 4
    config["test"].update(output_size=16, save_all_enhanced_images=True)
    config["test"]["visualization"].update(
        num_samples=2,
        grid_rows=2,
        cell_width=16,
        cell_height=16,
        add_labels=False,
        preserve_aspect_ratio=False,
    )
    config["logging"].update(console=False, save_test_log=False)
    return validate_v2_config(config)


def _run_dir(tmp_path: Path, config):
    run_dir = tmp_path / "run"
    for path in (run_dir / "best", run_dir / "result", run_dir / "log"):
        path.mkdir(parents=True, exist_ok=True)
    model = build_v2_model(
        variant=config["experiment"]["variant"], model_config=config["model"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    tracker = BestTracker(psnr=10.0, ssim=0.5, val_loss=0.2)
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=_make_grad_scaler(False),
        epoch=1,
        global_step=4,
        config=config,
        seed=config["experiment"]["seed"],
        train_loss=0.3,
        val_loss=0.2,
        psnr=10.0,
        ssim=0.5,
        learning_rate=2.0e-4,
        best_tracker=tracker,
        torch_module=torch,
        git_commit=None,
    )
    if config["experiment"]["variant"] == "shared_order_diagnostic":
        _extend_shared_checkpoint(
            payload,
            train_result={
                "train_loss_joint": 0.3,
                "train_loss_cs": 0.2,
                "train_loss_sc": 0.4,
            },
            validation={
                "psnr_cs": 11.0,
                "psnr_sc": 9.0,
                "ssim_cs": 0.6,
                "ssim_sc": 0.4,
                "val_loss_cs": 0.1,
                "val_loss_sc": 0.3,
                "mean_path_psnr": 10.0,
                "mean_path_ssim": 0.5,
            },
        )
    save_checkpoint(
        run_dir / "best/best_psnr.pt",
        payload=payload,
        torch_module=torch,
        selection_metric=(
            "validation_mean_path_psnr"
            if config["experiment"]["variant"] == "shared_order_diagnostic"
            else "validation_psnr"
        ),
    )
    return run_dir


def test_single_output_test_summary_and_overwrite_guard(tmp_path) -> None:
    config = _config(tmp_path, "color_only")
    run_dir = _run_dir(tmp_path, config)
    summary = execute_test(
        config,
        run_dir=run_dir,
        checkpoint_key="best_psnr",
        allow_overwrite=False,
    )
    assert summary["num_samples"] == 2
    assert summary["all_metrics_finite"] is True
    assert (run_dir / "result/test_metrics.csv").is_file()
    with Image.open(run_dir / "result/test_grid_2x3.png") as grid:
        assert grid.size == (48, 32)
    with Image.open(
        run_dir / "result/test_all_enhanced/sample0_enhanced.png"
    ) as image:
        assert image.size == (16, 16)
    with pytest.raises(FileExistsError, match="allow_overwrite=false"):
        execute_test(
            config,
            run_dir=run_dir,
            checkpoint_key="best_psnr",
            allow_overwrite=False,
        )


def test_shared_test_writes_two_metrics_comparison_and_four_column_grid(tmp_path) -> None:
    config = _config(tmp_path, "shared_order_diagnostic")
    run_dir = _run_dir(tmp_path, config)
    summary = execute_test(
        config,
        run_dir=run_dir,
        checkpoint_key="best_psnr",
        allow_overwrite=False,
    )
    result = run_dir / "result"
    assert summary["checkpoint_selection_source"] == "validation_mean_path_psnr"
    assert summary["num_samples"] == 2
    assert summary["all_metrics_finite"] is True
    for filename in (
        "test_metrics_color_then_scatter.csv",
        "test_metrics_scatter_then_color.csv",
        "test_order_comparison.csv",
        "test_summary.json",
    ):
        assert (result / filename).is_file()
    with (result / "test_order_comparison.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "delta_psnr_cs_minus_sc" in rows[0]
    with Image.open(result / "test_grid_2x4.png") as grid:
        assert grid.size == (64, 32)
    for order in ("color_then_scatter", "scatter_then_color"):
        with Image.open(
            result / f"test_all_enhanced/{order}/sample0_enhanced.png"
        ) as image:
            assert image.size == (16, 16)


def test_test_rejects_missing_best_psnr_sidecar_before_inference(tmp_path) -> None:
    config = _config(tmp_path, "color_only")
    run_dir = _run_dir(tmp_path, config)
    (run_dir / "best/best_psnr.json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        execute_test(
            config,
            run_dir=run_dir,
            checkpoint_key="best_psnr",
            allow_overwrite=False,
        )
    assert not (run_dir / "result/test_metrics.csv").exists()
