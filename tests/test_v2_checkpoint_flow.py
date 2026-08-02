from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from src.common.experiment.checkpoint import (  # noqa: E402
    BestTracker,
    build_checkpoint_payload,
    load_checkpoint,
)
from src.v1.model import build_nafnet_small  # noqa: E402
from src.v1.trainer import _make_grad_scaler  # noqa: E402
from src.v2.config import load_v2_config  # noqa: E402
from src.v2.model import build_v2_model  # noqa: E402
from src.v2.train_v2 import (  # noqa: E402
    _copy_resume_artifacts,
    _extend_shared_checkpoint,
    _save_epoch_checkpoints,
    _write_best_psnr_validation_snapshot,
    _write_validation_outputs,
    require_v2_resume_config_match,
)
from src.v2.test_v2 import require_v2_test_checkpoint_match  # noqa: E402
from src.common.experiment.experiment import RunPaths  # noqa: E402


def _small_config(variant="color_only"):
    config = load_v2_config(variant=variant, seed=7)
    config["model"]["backbone"].update(
        width=4, enc_blk_nums=[0], middle_blk_num=0, dec_blk_nums=[0]
    )
    config["model"]["color_operator"]["hidden_channels"] = 4
    config["model"]["scattering_operator"]["hidden_channels"] = 4
    return config


def _payload(config, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    scaler = _make_grad_scaler(False)
    tracker = BestTracker(psnr=20.0, ssim=0.8, val_loss=0.1)
    return build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        epoch=0,
        global_step=2,
        config=config,
        seed=config["experiment"]["seed"],
        train_loss=0.2,
        val_loss=0.1,
        psnr=20.0,
        ssim=0.8,
        learning_rate=2.0e-4,
        best_tracker=tracker,
        torch_module=torch,
        git_commit=None,
    )


def _sidecar(payload, selection_metric):
    return {
        "selection_metric": selection_metric,
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "train_loss": payload["train_loss"],
        "val_loss": payload["val_loss"],
        "psnr": payload["psnr"],
        "ssim": payload["ssim"],
        "learning_rate": payload["learning_rate"],
        "checkpoint": "best_psnr.pt",
    }


def _shared_payload(config):
    model = build_v2_model(
        variant="shared_order_diagnostic", model_config=config["model"]
    )
    payload = _payload(config, model)
    _extend_shared_checkpoint(
        payload,
        train_result={
            "train_loss_joint": 0.2,
            "train_loss_cs": 0.1,
            "train_loss_sc": 0.3,
        },
        validation={
            "psnr_cs": 21.0,
            "psnr_sc": 19.0,
            "ssim_cs": 0.81,
            "ssim_sc": 0.79,
            "val_loss_cs": 0.09,
            "val_loss_sc": 0.11,
            "mean_path_psnr": 20.0,
            "mean_path_ssim": 0.8,
        },
    )
    return payload


def _normal_validation(psnr):
    return {
        "num_samples": 1,
        "val_loss": 0.1,
        "psnr_rgb": float(psnr),
        "ssim_rgb": 0.8,
        "per_image": [
            {
                "sample_id": "sample",
                "input_relative_path": "input.png",
                "gt_relative_path": "gt.png",
                "psnr_rgb": float(psnr),
                "ssim_rgb": 0.8,
            }
        ],
    }


def _shared_validation(mean_psnr):
    psnr_cs = float(mean_psnr) + 1.0
    psnr_sc = float(mean_psnr) - 1.0
    common = {
        "sample_id": "sample",
        "input_relative_path": "input.png",
        "gt_relative_path": "gt.png",
    }
    return {
        "num_samples": 1,
        "val_loss": 0.1,
        "joint_val_loss": 0.1,
        "val_loss_cs": 0.09,
        "val_loss_sc": 0.11,
        "psnr_rgb": float(mean_psnr),
        "ssim_rgb": 0.8,
        "psnr_cs": psnr_cs,
        "psnr_sc": psnr_sc,
        "ssim_cs": 0.81,
        "ssim_sc": 0.79,
        "mean_path_psnr": float(mean_psnr),
        "mean_path_ssim": 0.8,
        "mean_delta_psnr_cs_minus_sc": 2.0,
        "mean_delta_ssim_cs_minus_sc": 0.02,
        "color_then_scatter_psnr_win_count": 1,
        "scatter_then_color_psnr_win_count": 0,
        "psnr_tie_count": 0,
        "winner_tolerance": 1.0e-8,
        "per_image_cs": [{**common, "psnr_rgb": psnr_cs, "ssim_rgb": 0.81}],
        "per_image_sc": [{**common, "psnr_rgb": psnr_sc, "ssim_rgb": 0.79}],
        "comparison": [
            {
                **common,
                "psnr_color_then_scatter": psnr_cs,
                "psnr_scatter_then_color": psnr_sc,
                "delta_psnr_cs_minus_sc": 2.0,
                "ssim_color_then_scatter": 0.81,
                "ssim_scatter_then_color": 0.79,
                "delta_ssim_cs_minus_sc": 0.02,
                "winner_psnr": "color_then_scatter",
                "winner_ssim": "color_then_scatter",
            }
        ],
    }


def _paths(root):
    paths = RunPaths(
        root=root,
        log=root / "log",
        best=root / "best",
        checkpoint=root / "checkpoint",
        result=root / "result",
        test_samples=root / "result/test_samples",
    )
    for path in (paths.log, paths.best, paths.checkpoint, paths.result, paths.test_samples):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def test_normal_v2_checkpoint_round_trip(tmp_path) -> None:
    config = _small_config()
    model = build_v2_model(
        variant=config["experiment"]["variant"], model_config=config["model"]
    )
    payload = _payload(config, model)
    path = tmp_path / "normal.pt"
    torch.save(payload, path)
    target = build_v2_model(
        variant=config["experiment"]["variant"], model_config=config["model"]
    )
    loaded = load_checkpoint(path, model=target, torch_module=torch)
    assert loaded["global_step"] == 2
    assert list(model.state_dict()) == list(target.state_dict())


def test_shared_checkpoint_extensions_and_mean_path_selection(tmp_path) -> None:
    config = _small_config("shared_order_diagnostic")
    model = build_v2_model(
        variant="shared_order_diagnostic", model_config=config["model"]
    )
    payload = _payload(config, model)
    train_result = {
        "train_loss_joint": 0.2,
        "train_loss_cs": 0.1,
        "train_loss_sc": 0.3,
    }
    validation = {
        "psnr_cs": 21.0,
        "psnr_sc": 19.0,
        "ssim_cs": 0.81,
        "ssim_sc": 0.79,
        "val_loss_cs": 0.09,
        "val_loss_sc": 0.11,
        "mean_path_psnr": 20.0,
        "mean_path_ssim": 0.8,
    }
    _extend_shared_checkpoint(
        payload, train_result=train_result, validation=validation
    )
    assert payload["validation_mean_path_psnr"] == 20.0
    assert payload["train_loss"] == 0.2
    paths = _paths(tmp_path / "run")
    _save_epoch_checkpoints(
        paths=paths,
        config=config,
        payload=payload,
        updates={"psnr": True, "ssim": False, "val_loss": False},
        epoch=0,
        torch_module=torch,
        shared=True,
    )
    sidecar = json.loads(
        (paths.best / "best_psnr.json").read_text(encoding="utf-8")
    )
    assert sidecar["selection_metric"] == "validation_mean_path_psnr"
    target = build_v2_model(
        variant="shared_order_diagnostic", model_config=config["model"]
    )
    loaded = load_checkpoint(
        paths.best / "best_psnr.pt", model=target, torch_module=torch
    )
    for key in (
        "train_loss_joint",
        "train_loss_cs",
        "train_loss_sc",
        "validation_psnr_cs",
        "validation_psnr_sc",
        "validation_mean_path_psnr",
    ):
        assert key in loaded


def test_resume_rejects_any_non_ignored_semantic_change() -> None:
    saved = _small_config()
    current = copy.deepcopy(saved)
    current["training"]["resume"] = "some.pt"
    current["test"]["run_dir"] = "run"
    current["test"]["allow_overwrite"] = True
    require_v2_resume_config_match(current, saved)
    current["model"]["color_operator"]["matrix_scale"] = 0.2
    with pytest.raises(ValueError, match="does not match v2 semantics"):
        require_v2_resume_config_match(current, saved)


def test_v1_state_dict_strict_loads_into_v2_baseline(tmp_path) -> None:
    config = _small_config("baseline")
    v1 = build_nafnet_small(**config["model"]["backbone"])
    v2 = build_v2_model(variant="baseline", model_config=config["model"])
    payload = _payload(config, v1)
    path = tmp_path / "v1-compatible.pt"
    torch.save(payload, path)
    load_checkpoint(path, model=v2, torch_module=torch)
    assert list(v1.state_dict()) == list(v2.state_dict())


def _write_validation_stage(
    paths, tracker, validation, *, epoch, shared, save_best_psnr=True
):
    updates = tracker.consider(
        psnr=validation["psnr_rgb"],
        ssim=validation["ssim_rgb"],
        val_loss=validation["val_loss"],
    )
    _write_validation_outputs(
        paths=paths,
        validation=validation,
        tracker=tracker,
        epoch=epoch,
        global_step=epoch + 10,
        shared=shared,
    )
    _write_best_psnr_validation_snapshot(
        paths=paths,
        validation=validation,
        tracker=tracker,
        epoch=epoch,
        global_step=epoch + 10,
        shared=shared,
        updates=updates,
        save_best_psnr=save_best_psnr,
    )


def test_normal_best_validation_snapshot_tracks_best_not_latest(tmp_path) -> None:
    paths = _paths(tmp_path / "normal")
    tracker = BestTracker()
    _write_validation_stage(
        paths, tracker, _normal_validation(20.0), epoch=0, shared=False
    )
    _write_validation_stage(
        paths, tracker, _normal_validation(19.0), epoch=1, shared=False
    )
    latest = json.loads(
        (paths.result / "validation_summary.json").read_text(encoding="utf-8")
    )
    best = json.loads(
        (paths.result / "best_psnr_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["epoch"] == 1
    assert best["epoch"] == 0
    assert best["checkpoint"] == "best_psnr.pt"
    assert best["checkpoint_selection_source"] == "validation_psnr"
    assert best["mean_psnr_rgb"] == best["best_validation_psnr"] == 20.0
    _write_validation_stage(
        paths, tracker, _normal_validation(21.0), epoch=2, shared=False
    )
    best = json.loads(
        (paths.result / "best_psnr_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert best["epoch"] == 2
    assert best["global_step"] == 12
    assert best["mean_psnr_rgb"] == best["best_validation_psnr"] == 21.0


def test_best_snapshot_checkpoint_and_sidecar_share_selection_state(tmp_path) -> None:
    config = _small_config("color_only")
    model = build_v2_model(variant="color_only", model_config=config["model"])
    payload = _payload(config, model)
    validation = _normal_validation(20.0)
    tracker = BestTracker()
    updates = tracker.consider(psnr=20.0, ssim=0.8, val_loss=0.1)
    paths = _paths(tmp_path / "paired")
    _write_best_psnr_validation_snapshot(
        paths=paths,
        validation=validation,
        tracker=tracker,
        epoch=payload["epoch"],
        global_step=payload["global_step"],
        shared=False,
        updates=updates,
        save_best_psnr=True,
    )
    _save_epoch_checkpoints(
        paths=paths,
        config=config,
        payload=payload,
        updates=updates,
        epoch=payload["epoch"],
        torch_module=torch,
        shared=False,
    )
    summary = json.loads(
        (paths.result / "best_psnr_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    sidecar = json.loads(
        (paths.best / "best_psnr.json").read_text(encoding="utf-8")
    )
    target = build_v2_model(variant="color_only", model_config=config["model"])
    loaded = load_checkpoint(
        paths.best / "best_psnr.pt", model=target, torch_module=torch
    )
    assert summary["epoch"] == sidecar["epoch"] == loaded["epoch"]
    assert (
        summary["global_step"]
        == sidecar["global_step"]
        == loaded["global_step"]
    )
    assert summary["mean_psnr_rgb"] == sidecar["psnr"] == loaded["psnr"]


def test_shared_best_validation_snapshot_tracks_best_not_latest(tmp_path) -> None:
    paths = _paths(tmp_path / "shared")
    tracker = BestTracker()
    _write_validation_stage(
        paths, tracker, _shared_validation(20.0), epoch=0, shared=True
    )
    _write_validation_stage(
        paths, tracker, _shared_validation(19.0), epoch=1, shared=True
    )
    latest = json.loads(
        (paths.result / "validation_summary.json").read_text(encoding="utf-8")
    )
    best_path = paths.result / "best_psnr_validation_summary.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert latest["epoch"] == 1
    assert best["epoch"] == 0
    assert best["mean_path_psnr"] == best["best_validation_mean_path_psnr"]
    for filename in (
        "best_psnr_validation_metrics_color_then_scatter.csv",
        "best_psnr_validation_metrics_scatter_then_color.csv",
        "best_psnr_validation_order_comparison.csv",
    ):
        assert (paths.result / filename).is_file()
    _write_validation_stage(
        paths, tracker, _shared_validation(21.0), epoch=2, shared=True
    )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert best["epoch"] == 2
    assert best["mean_path_psnr"] == best["best_validation_mean_path_psnr"] == 21.0


def test_best_snapshot_is_not_written_when_checkpoint_switch_is_off(tmp_path) -> None:
    paths = _paths(tmp_path / "disabled")
    tracker = BestTracker()
    _write_validation_stage(
        paths,
        tracker,
        _normal_validation(20.0),
        epoch=0,
        shared=False,
        save_best_psnr=False,
    )
    assert not (paths.result / "best_psnr_validation_summary.json").exists()


def test_resume_copies_existing_best_validation_snapshots(tmp_path) -> None:
    source = tmp_path / "source"
    resume_path = source / "checkpoint/last.pt"
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"checkpoint-placeholder")
    source_result = source / "result"
    source_result.mkdir()
    filenames = (
        "best_psnr_validation_summary.json",
        "best_psnr_validation_metrics_color_then_scatter.csv",
        "best_psnr_validation_order_comparison.csv",
    )
    for filename in filenames:
        (source_result / filename).write_text(filename, encoding="utf-8")
    paths = _paths(tmp_path / "destination")
    _copy_resume_artifacts(
        resume_path,
        paths,
        through_epoch=0,
        save_metrics_history=False,
    )
    for filename in filenames:
        assert (paths.result / filename).read_text(encoding="utf-8") == filename


def test_valid_normal_and_shared_test_provenance_is_accepted() -> None:
    normal_config = _small_config("color_then_scatter")
    normal_model = build_v2_model(
        variant="color_then_scatter", model_config=normal_config["model"]
    )
    normal_payload = _payload(normal_config, normal_model)
    require_v2_test_checkpoint_match(
        run_config=normal_config,
        checkpoint=normal_payload,
        sidecar=_sidecar(normal_payload, "validation_psnr"),
        shared=False,
    )

    shared_config = _small_config("shared_order_diagnostic")
    shared_payload = _shared_payload(shared_config)
    require_v2_test_checkpoint_match(
        run_config=shared_config,
        checkpoint=shared_payload,
        sidecar=_sidecar(shared_payload, "validation_mean_path_psnr"),
        shared=True,
    )


def test_test_provenance_rejects_compatible_wrong_order_checkpoint() -> None:
    checkpoint_config = _small_config("color_then_scatter")
    run_config = _small_config("scatter_then_color")
    model = build_v2_model(
        variant="color_then_scatter", model_config=checkpoint_config["model"]
    )
    payload = _payload(checkpoint_config, model)
    with pytest.raises(ValueError, match="does not match v2 semantics"):
        require_v2_test_checkpoint_match(
            run_config=run_config,
            checkpoint=payload,
            sidecar=_sidecar(payload, "validation_psnr"),
            shared=False,
        )


@pytest.mark.parametrize("mismatch", ["seed", "color", "scatter"])
def test_test_provenance_rejects_seed_or_operator_config_mismatch(mismatch) -> None:
    run_config = _small_config("color_then_scatter")
    model = build_v2_model(
        variant="color_then_scatter", model_config=run_config["model"]
    )
    payload = _payload(run_config, model)
    if mismatch == "seed":
        payload["seed"] += 1
        message = "seed does not match"
    else:
        payload["config"] = copy.deepcopy(payload["config"])
        section = "color_operator" if mismatch == "color" else "scattering_operator"
        field = "matrix_scale" if mismatch == "color" else "residual_max"
        payload["config"]["model"][section][field] *= 2.0
        message = "does not match v2 semantics"
    with pytest.raises(ValueError, match=message):
        require_v2_test_checkpoint_match(
            run_config=run_config,
            checkpoint=payload,
            sidecar=_sidecar(payload, "validation_psnr"),
            shared=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_metric", "validation_ssim", "selection_metric"),
        ("epoch", 99, "epoch"),
        ("global_step", 99, "global_step"),
        ("checkpoint", "wrong.pt", "checkpoint must equal"),
        ("psnr", 99.0, "best_psnr.json psnr mismatch"),
    ],
)
def test_test_provenance_rejects_sidecar_mismatch(field, value, message) -> None:
    config = _small_config()
    model = build_v2_model(variant="color_only", model_config=config["model"])
    payload = _payload(config, model)
    sidecar = _sidecar(payload, "validation_psnr")
    sidecar[field] = value
    with pytest.raises(ValueError, match=message):
        require_v2_test_checkpoint_match(
            run_config=config,
            checkpoint=payload,
            sidecar=sidecar,
            shared=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(psnr=19.0), "psnr/validation_mean_path_psnr"),
        (
            lambda p: p.update(psnr=19.5, validation_mean_path_psnr=19.5),
            "validation mean-path PSNR",
        ),
        (lambda p: p.update(val_loss=0.2), "joint validation loss"),
    ],
)
def test_shared_test_provenance_rejects_invalid_derived_values(mutation, message) -> None:
    config = _small_config("shared_order_diagnostic")
    payload = _shared_payload(config)
    mutation(payload)
    sidecar = _sidecar(payload, "validation_mean_path_psnr")
    with pytest.raises(ValueError, match=message):
        require_v2_test_checkpoint_match(
            run_config=config,
            checkpoint=payload,
            sidecar=sidecar,
            shared=True,
        )
