from __future__ import annotations

import copy
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest
import yaml

from src.common.experiment.config import (
    PROJECT_ROOT,
    V1_CONFIG_PATH,
    load_v1_config,
    validate_v1_config,
)
from src.v1 import test_v1, train_v1


@contextmanager
def _temporary_v1_config(
    directory: Path, *, filename: str | None = None, version: str = "v1"
) -> Iterator[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    if filename is None:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="configV1_custom_", suffix=".yaml", dir=directory
        )
        os.close(descriptor)
        path = Path(raw_path)
    else:
        path = directory / filename
    contents = V1_CONFIG_PATH.read_text(encoding="utf-8")
    if version != "v1":
        contents = contents.replace("version: v1", f"version: {version}", 1)
    path.write_text(contents, encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_yaml_parses_and_has_matching_v1_seed() -> None:
    config = load_v1_config()
    assert config["experiment"]["version"] == "v1"
    seed = config["experiment"]["seed"]
    assert config["experiment"]["name"] == f"NAFNet_small_seed{seed}"


def test_checked_in_smoke_config_loads() -> None:
    config = load_v1_config(PROJECT_ROOT / "configs/configV1_smoke.yaml")
    assert config["experiment"]["version"] == "v1"
    assert config["test"]["output_size"] == 256
    assert config["test"]["save_all_enhanced_images"] is True
    assert config["test"]["visualization"]["cell_width"] == 256
    assert config["test"]["visualization"]["cell_height"] == 256


def test_default_config_is_the_train_and_test_cli_default() -> None:
    assert train_v1.build_parser().parse_args([]).config == V1_CONFIG_PATH
    assert test_v1.build_parser().parse_args([]).config == V1_CONFIG_PATH


def test_repository_custom_v1_config_can_load() -> None:
    with _temporary_v1_config(PROJECT_ROOT / "configs") as config_path:
        config_path.resolve().relative_to(PROJECT_ROOT.resolve())
        config = load_v1_config(config_path, entry_point="train_v1")
    assert config["experiment"]["version"] == "v1"


def test_tmp_smoke_v1_config_can_load() -> None:
    smoke_dir = Path(tempfile.mkdtemp(prefix="UIE3_VN_v1_smoke_", dir="/tmp"))
    try:
        with _temporary_v1_config(
            smoke_dir, filename="configV1_smoke.yaml"
        ) as config_path:
            config = load_v1_config(config_path, entry_point="train_v1")
    finally:
        shutil.rmtree(smoke_dir)
    assert config["experiment"]["version"] == "v1"


def test_custom_manifests_and_two_epochs_can_load(tmp_path: Path) -> None:
    config = load_v1_config()
    manifest_paths = {}
    for split in ("train", "validation", "test"):
        manifest_path = tmp_path / f"{split}.tsv"
        manifest_path.write_text(
            f"{split}_sample\tinput.png\tgt.png\n", encoding="utf-8"
        )
        key = f"{split}_manifest"
        config["data"][key] = str(manifest_path)
        manifest_paths[key] = str(manifest_path)
    config["training"]["epochs"] = 2
    config_path = tmp_path / "configV1_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    loaded = load_v1_config(config_path, entry_point="train_v1")

    assert loaded["training"]["epochs"] == 2
    assert {
        key: loaded["data"][key] for key in manifest_paths
    } == manifest_paths


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    config = load_v1_config()
    config["data"]["train_manifest"] = str(tmp_path / "missing.tsv")
    with pytest.raises(FileNotFoundError, match="data.train_manifest"):
        validate_v1_config(config)


def test_non_utf8_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.tsv"
    manifest_path.write_bytes(b"\xff\xfe\x00")
    config = load_v1_config()
    config["data"]["test_manifest"] = str(manifest_path)
    with pytest.raises(ValueError, match="data.test_manifest cannot be read"):
        validate_v1_config(config)


@pytest.mark.parametrize(
    ("entry_point", "runner"),
    [("train_v1", train_v1.run), ("test_v1", test_v1.run_standalone)],
)
def test_v2_config_is_rejected_by_v1_entry_points(
    entry_point: str, runner: Callable[[Path], object]
) -> None:
    with _temporary_v1_config(Path("/tmp"), version="v2") as config_path:
        with pytest.raises(
            ValueError,
            match=rf"{entry_point} requires a v1 configuration; got version=v2",
        ):
            runner(config_path)


@pytest.mark.parametrize("runner", [train_v1.run, test_v1.run_standalone])
def test_missing_config_is_rejected_by_v1_entry_points(
    tmp_path: Path, runner: Callable[[Path], object]
) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        runner(missing_path)


def test_other_version_like_fields_are_checked() -> None:
    config = load_v1_config()
    config["model"]["model_version"] = "v2"
    with pytest.raises(
        ValueError,
        match="train_v1 requires a v1 configuration; got model_version=v2",
    ):
        validate_v1_config(config, entry_point="train_v1")


def test_experiment_name_does_not_lock_the_seed() -> None:
    config = load_v1_config(V1_CONFIG_PATH)
    config["experiment"]["seed"] = 2027
    assert validate_v1_config(config)["experiment"]["seed"] == 2027


def test_supported_tunable_values_are_yaml_controlled(tmp_path: Path) -> None:
    config = load_v1_config(V1_CONFIG_PATH)
    config["experiment"]["output_root"] = str(tmp_path / "runs")
    config["data"].update(
        {
            "patch_size": 128,
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": False,
            "pad_if_smaller": False,
        }
    )
    config["data"]["augmentation"] = {
        "hflip": False,
        "vflip": False,
        "rot90": False,
    }
    config["model"].update(
        {
            "width": 16,
            "enc_blk_nums": [1, 1],
            "middle_blk_num": 2,
            "dec_blk_nums": [1, 1],
        }
    )
    config["loss"]["epsilon"] = 0.01
    config["optimizer"].update(
        {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "betas": [0.8, 0.95],
        }
    )
    config["training"].update(
        {
            "epochs": 3,
            "amp": False,
            "deterministic": True,
            "validate_every": 2,
            "save_every": 5,
            "gradient_clip_norm": 1.0,
            "fail_on_nonfinite": False,
        }
    )
    config["checkpoint"].update(
        {
            "save_best_ssim": False,
            "save_best_val_loss": False,
            "save_last": False,
            "save_periodic": False,
        }
    )
    config["test"]["auto_run_after_training"] = False
    config["test"]["visualization"].update(
        {
            "num_samples": 4,
            "grid_rows": 4,
            "cell_width": 320,
            "cell_height": 240,
            "preserve_aspect_ratio": False,
            "add_labels": False,
        }
    )
    config["metrics"].update(
        {
            "data_range": 2.0,
            "crop_border": 1,
            "ssim_window_size": 7,
            "ssim_sigma": 1.0,
        }
    )
    config["logging"].update(
        {
            "console": False,
            "save_train_log": False,
            "save_validation_log": False,
            "save_test_log": False,
            "save_metrics_history_json": False,
            "log_every_steps": 3,
        }
    )

    validated = validate_v1_config(config)

    assert validated == config


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "batch_size", 0),
        ("data", "num_workers", -1),
        ("loss", "epsilon", 0.0),
        ("optimizer", "learning_rate", -0.1),
        ("training", "epochs", 0),
        ("training", "validate_every", 0),
        ("metrics", "data_range", 0.0),
    ],
)
def test_invalid_ranges_or_fixed_semantics_are_rejected(
    section: str, key: str, value: object
) -> None:
    config = copy.deepcopy(load_v1_config(V1_CONFIG_PATH))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_v1_config(config)
