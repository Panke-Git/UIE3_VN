"""Strict, isolated configuration loading for v2 order-study experiments."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.common.experiment.config import PROJECT_ROOT, validate_v1_config


V2_CONFIG_PATH = PROJECT_ROOT / "configs/configV2.yaml"
VALID_VARIANTS = frozenset(
    {
        "baseline",
        "color_only",
        "scatter_only",
        "color_then_scatter",
        "scatter_then_color",
        "shared_order_diagnostic",
    }
)
SINGLE_OUTPUT_VARIANTS = VALID_VARIANTS - {"shared_order_diagnostic"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMON_SECTIONS = {
    "experiment",
    "data",
    "model",
    "order_study",
    "loss",
    "optimizer",
    "scheduler",
    "training",
    "checkpoint",
    "test",
    "metrics",
    "logging",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _strict_keys(
    value: Mapping[str, Any], name: str, expected: Sequence[str]
) -> None:
    missing = sorted(set(expected) - set(value))
    if missing:
        raise ValueError(f"Config section {name!r} is missing keys: {missing}.")
    unexpected = sorted(set(value) - set(expected))
    if unexpected:
        raise ValueError(
            f"Config section {name!r} contains unsupported keys: {unexpected}."
        )


def _integer(
    value: Mapping[str, Any], section: str, key: str, *, minimum: int
) -> int:
    result = value.get(key)
    if type(result) is not int or result < minimum:
        raise ValueError(
            f"{section}.{key} must be an integer >= {minimum}, got {result!r}."
        )
    return result


def _number(
    value: Mapping[str, Any],
    section: str,
    key: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    raw = value.get(key)
    if type(raw) not in (int, float):
        raise ValueError(f"{section}.{key} must be numeric, got {raw!r}.")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{section}.{key} must be finite, got {raw!r}.")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(
                f"{section}.{key} must be {relation} {minimum}, got {raw!r}."
            )
    if maximum is not None:
        invalid = result >= maximum if strict_maximum else result > maximum
        if invalid:
            relation = "<" if strict_maximum else "<="
            raise ValueError(
                f"{section}.{key} must be {relation} {maximum}, got {raw!r}."
            )
    return result


def _expected_visualization(variant: str) -> tuple[list[str], int]:
    if variant == "shared_order_diagnostic":
        return [
            "input",
            "color_then_scatter",
            "scatter_then_color",
            "gt",
        ], 4
    return ["input", "enhanced", "gt"], 3


def _common_v1_view(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a temporary v1-shaped view to reuse unchanged common validation."""

    common = copy.deepcopy(dict(config))
    common.pop("order_study")
    experiment = common["experiment"]
    experiment.pop("variant")
    experiment["version"] = "v1"
    common["model"] = copy.deepcopy(common["model"]["backbone"])
    visualization = common["test"]["visualization"]
    visualization["columns"] = ["input", "enhanced", "gt"]
    visualization["grid_columns"] = 3
    return common


def validate_v2_config(
    config: Any, *, entry_point: str = "v2"
) -> Dict[str, Any]:
    """Validate the complete v2 schema without modifying v1 validation."""

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a mapping.")
    experiment_value = config.get("experiment")
    version = (
        experiment_value.get("version")
        if isinstance(experiment_value, Mapping)
        else None
    )
    if version != "v2":
        raise ValueError(
            f"{entry_point} requires a v2 configuration; got version={version}."
        )
    unexpected_sections = sorted(set(config) - _COMMON_SECTIONS)
    missing_sections = sorted(_COMMON_SECTIONS - set(config))
    if missing_sections:
        raise ValueError(f"YAML config is missing sections: {missing_sections}.")
    if unexpected_sections:
        raise ValueError(
            f"YAML config contains unsupported sections: {unexpected_sections}."
        )

    experiment = _mapping(config["experiment"], "experiment")
    _strict_keys(
        experiment,
        "experiment",
        ("version", "name", "variant", "seed", "output_root"),
    )
    variant = experiment.get("variant")
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"experiment.variant must be one of {sorted(VALID_VARIANTS)}, "
            f"got {variant!r}."
        )
    seed = _integer(experiment, "experiment", "seed", minimum=0)
    name = experiment.get("name")
    expected_name = f"{variant}_seed{seed}"
    if name != expected_name:
        raise ValueError(
            "experiment.name must match variant and seed: "
            f"expected {expected_name!r}, got {name!r}."
        )
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError("experiment.name contains unsafe characters.")

    model = _mapping(config["model"], "model")
    _strict_keys(
        model,
        "model",
        ("backbone", "color_operator", "scattering_operator"),
    )
    _mapping(model["backbone"], "model.backbone")

    color = _mapping(model["color_operator"], "model.color_operator")
    _strict_keys(
        color,
        "model.color_operator",
        ("hidden_channels", "matrix_scale", "bias_scale"),
    )
    _integer(color, "model.color_operator", "hidden_channels", minimum=1)
    _number(
        color,
        "model.color_operator",
        "matrix_scale",
        minimum=0.0,
        strict_minimum=True,
    )
    _number(
        color,
        "model.color_operator",
        "bias_scale",
        minimum=0.0,
        strict_minimum=True,
    )

    scatter = _mapping(
        model["scattering_operator"], "model.scattering_operator"
    )
    _strict_keys(
        scatter,
        "model.scattering_operator",
        (
            "hidden_channels",
            "t_min",
            "eps",
            "residual_max",
            "initial_t",
            "initial_A",
        ),
    )
    _integer(scatter, "model.scattering_operator", "hidden_channels", minimum=1)
    t_min = _number(
        scatter,
        "model.scattering_operator",
        "t_min",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    initial_t = _number(
        scatter,
        "model.scattering_operator",
        "initial_t",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    if not t_min < initial_t:
        raise ValueError(
            "model.scattering_operator must satisfy t_min < initial_t."
        )
    _number(
        scatter,
        "model.scattering_operator",
        "initial_A",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    for key in ("eps", "residual_max"):
        _number(
            scatter,
            "model.scattering_operator",
            key,
            minimum=0.0,
            strict_minimum=True,
        )

    order_study = _mapping(config["order_study"], "order_study")
    _strict_keys(
        order_study,
        "order_study",
        (
            "color_then_scatter_loss_weight",
            "scatter_then_color_loss_weight",
            "shared_checkpoint_metric",
        ),
    )
    color_weight = _number(
        order_study,
        "order_study",
        "color_then_scatter_loss_weight",
    )
    scatter_weight = _number(
        order_study,
        "order_study",
        "scatter_then_color_loss_weight",
    )
    if color_weight != 0.5 or scatter_weight != 0.5:
        raise ValueError("Shared order loss weights must both equal 0.5.")
    if not math.isclose(color_weight + scatter_weight, 1.0):
        raise ValueError("Shared order loss weights must sum to 1.0.")
    if order_study.get("shared_checkpoint_metric") != "mean_path_psnr":
        raise ValueError(
            "order_study.shared_checkpoint_metric must equal "
            "'mean_path_psnr'."
        )

    visualization = _mapping(config["test"]["visualization"], "test.visualization")
    expected_columns, expected_count = _expected_visualization(str(variant))
    if visualization.get("columns") != expected_columns:
        raise ValueError(
            f"test.visualization.columns for {variant} must equal "
            f"{expected_columns}."
        )
    if visualization.get("grid_columns") != expected_count:
        raise ValueError(
            f"test.visualization.grid_columns for {variant} must equal "
            f"{expected_count}."
        )

    # The temporary view reuses every unchanged V1 rule: manifests, optimizer,
    # training, checkpoint, metrics, logging, and visualization dimensions.
    validate_v1_config(_common_v1_view(config), entry_point="v2_common")
    return copy.deepcopy(config)


def resolve_v2_config_path(path: Path | str = V2_CONFIG_PATH) -> Path:
    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    return config_path


def apply_v2_overrides(
    config: Mapping[str, Any],
    *,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply CLI overrides and update all structurally derived fields."""

    resolved = copy.deepcopy(dict(config))
    if variant is None and seed is None:
        return resolved
    experiment = _mapping(resolved.get("experiment"), "experiment")
    final_variant = experiment.get("variant") if variant is None else variant
    if final_variant not in VALID_VARIANTS:
        raise ValueError(
            f"variant override must be one of {sorted(VALID_VARIANTS)}, "
            f"got {final_variant!r}."
        )
    final_seed = experiment.get("seed") if seed is None else seed
    if type(final_seed) is not int or final_seed < 0:
        raise ValueError(
            f"seed override must be a non-negative integer, got {final_seed!r}."
        )
    resolved["experiment"]["variant"] = final_variant
    resolved["experiment"]["seed"] = final_seed
    resolved["experiment"]["name"] = f"{final_variant}_seed{final_seed}"
    if variant is not None:
        columns, count = _expected_visualization(final_variant)
        resolved["test"]["visualization"]["columns"] = columns
        resolved["test"]["visualization"]["grid_columns"] = count
    return resolved


def load_v2_config(
    path: Path | str = V2_CONFIG_PATH,
    *,
    entry_point: str = "v2",
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Load YAML, apply optional CLI overrides, then validate the final config."""

    config_path = resolve_v2_config_path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the v2 configuration.") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        return validate_v2_config(raw, entry_point=entry_point)
    resolved = apply_v2_overrides(raw, variant=variant, seed=seed)
    return validate_v2_config(resolved, entry_point=entry_point)
