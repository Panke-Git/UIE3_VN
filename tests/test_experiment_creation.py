from __future__ import annotations

import json
import re
from datetime import datetime

from src.common.experiment import experiment as experiment_module
from src.common.experiment.config import V1_CONFIG_PATH, load_v1_config


def test_unique_second_precision_run_and_config_snapshot(
    tmp_path, monkeypatch
) -> None:
    config = load_v1_config(V1_CONFIG_PATH)
    monkeypatch.setattr(experiment_module, "PROJECT_ROOT", tmp_path)
    moment = datetime(2026, 7, 30, 22, 58, 0)
    first = experiment_module.create_experiment(config, now=moment)
    second = experiment_module.create_experiment(config, now=moment)

    assert re.fullmatch(
        r"v1_NAFNet_small_seed1234_20260730_225800", first.root.name
    )
    assert re.fullmatch(
        r"v1_NAFNet_small_seed1234_20260730_225800_[0-9a-f]{6}",
        second.root.name,
    )
    assert first.root != second.root
    with (first.root / "config.json").open("r", encoding="utf-8") as handle:
        assert json.load(handle) == config
    assert (first.root / "log").is_dir()
    assert (first.root / "best").is_dir()
    assert (first.root / "checkpoint").is_dir()
    assert (first.root / "result/test_samples").is_dir()
