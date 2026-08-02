from __future__ import annotations

import copy
import json

import pytest

from src.common.experiment import experiment as experiment_module
from src.common.experiment.config import V1_CONFIG_PATH, load_v1_config, validate_v1_config
from src.v2.config import (
    VALID_VARIANTS,
    V2_CONFIG_PATH,
    load_v2_config,
    validate_v2_config,
)
from src.v2.train_v2 import build_parser, resolve_training_config
from src.v2.test_v2 import run_standalone


def test_checked_in_v2_config_loads() -> None:
    config = load_v2_config()
    assert config["experiment"] == {
        "version": "v2",
        "name": "color_only_seed1234",
        "variant": "color_only",
        "seed": 1234,
        "output_root": "experiments",
    }


@pytest.mark.parametrize("variant", sorted(VALID_VARIANTS))
def test_all_six_variants_resolve_with_consistent_name(variant: str) -> None:
    config = load_v2_config(variant=variant, seed=77)
    assert config["experiment"]["name"] == f"{variant}_seed77"
    expected_columns = 4 if variant == "shared_order_diagnostic" else 3
    assert config["test"]["visualization"]["grid_columns"] == expected_columns


def test_cli_overrides_are_applied_before_snapshot(tmp_path, monkeypatch) -> None:
    args = build_parser().parse_args(
        ["--config", str(V2_CONFIG_PATH), "--variant", "scatter_only", "--seed", "9"]
    )
    config = resolve_training_config(
        args.config, variant=args.variant, seed=args.seed
    )
    config["experiment"]["output_root"] = "runs"
    monkeypatch.setattr(experiment_module, "PROJECT_ROOT", tmp_path)
    paths = experiment_module.create_experiment(config)
    saved = json.loads((paths.root / "config.json").read_text(encoding="utf-8"))
    assert saved["experiment"]["variant"] == "scatter_only"
    assert saved["experiment"]["seed"] == 9
    assert saved["experiment"]["name"] == "scatter_only_seed9"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda c: c["experiment"].update(variant="unknown"), "variant"),
        (lambda c: c["experiment"].update(version="v1"), "requires a v2"),
        (lambda c: c["experiment"].update(name="wrong"), "must match"),
        (lambda c: c["model"]["color_operator"].update(matrix_scale=0), "matrix_scale"),
        (lambda c: c["model"]["scattering_operator"].update(t_min=0), "t_min"),
        (
            lambda c: c["model"]["scattering_operator"].update(initial_t=0.1),
            "t_min < initial_t",
        ),
        (
            lambda c: c["order_study"].update(color_then_scatter_loss_weight=0.4),
            "both equal 0.5",
        ),
    ],
)
def test_invalid_v2_semantics_are_rejected(mutator, message: str) -> None:
    config = copy.deepcopy(load_v2_config())
    mutator(config)
    with pytest.raises(ValueError, match=message):
        validate_v2_config(config)


def test_unknown_key_is_not_silently_ignored() -> None:
    config = load_v2_config()
    config["model"]["color_operator"]["unused"] = 1
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_v2_config(config)


def test_loader_does_not_repair_inconsistent_name_without_cli_override(
    tmp_path,
) -> None:
    text = V2_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "name: color_only_seed1234", "name: inconsistent", 1
    )
    path = tmp_path / "bad-name.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must match variant and seed"):
        load_v2_config(path)


def test_v1_and_v2_validators_reject_the_other_version() -> None:
    with pytest.raises(ValueError, match="requires a v2 configuration"):
        validate_v2_config(load_v1_config(), entry_point="train_v2")
    with pytest.raises(ValueError, match="requires a v1 configuration"):
        validate_v1_config(load_v2_config(), entry_point="train_v1")
    with pytest.raises(ValueError, match="train_v2 requires a v2 configuration"):
        resolve_training_config(V1_CONFIG_PATH)
    with pytest.raises(ValueError, match="test_v2 requires a v2 configuration"):
        run_standalone(V1_CONFIG_PATH)
