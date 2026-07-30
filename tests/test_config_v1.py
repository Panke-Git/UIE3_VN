from __future__ import annotations

import copy

import pytest

from src.common.experiment.config import (
    V1_CONFIG_PATH,
    load_v1_config,
    validate_v1_config,
)


def test_yaml_parses_and_has_matching_v1_seed() -> None:
    config = load_v1_config(V1_CONFIG_PATH)
    assert config["experiment"]["version"] == "v1"
    assert config["experiment"]["name"] == "NAFNet_small_seed1234"
    assert config["experiment"]["seed"] == 1234


def test_name_seed_mismatch_is_rejected() -> None:
    config = load_v1_config(V1_CONFIG_PATH)
    config["experiment"]["seed"] = 2027
    with pytest.raises(ValueError, match="matching experiment.seed"):
        validate_v1_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "batch_size", 0),
        ("data", "num_workers", -1),
        ("loss", "epsilon", 0.0),
        ("optimizer", "learning_rate", -0.1),
        ("training", "epochs", 0),
        ("training", "validate_every", 2),
        ("metrics", "data_range", 255.0),
    ],
)
def test_invalid_ranges_or_fixed_semantics_are_rejected(
    section: str, key: str, value: object
) -> None:
    config = copy.deepcopy(load_v1_config(V1_CONFIG_PATH))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_v1_config(config)
