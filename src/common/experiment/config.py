"""Load and validate v1 YAML configurations without a schema framework."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V1_CONFIG_PATH = PROJECT_ROOT / "configs/configV1.yaml"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION_KEYS = {"version", "variant", "model_version"}


def _is_version_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _VERSION_KEYS or normalized.endswith("_version")


def _declared_versions(config: Mapping[str, Any]) -> list[tuple[str, Any]]:
    declared: list[tuple[str, Any]] = []

    def visit(value: Mapping[str, Any]) -> None:
        for key, child in value.items():
            if _is_version_key(key):
                declared.append((str(key), child))
            if isinstance(child, Mapping):
                visit(child)

    visit(config)
    return declared


def _require_v1_version(config: Mapping[str, Any], entry_point: str) -> None:
    declared = _declared_versions(config)
    if not declared:
        raise ValueError(
            f"{entry_point} requires a v1 configuration; got no version field."
        )
    for key, value in declared:
        if not isinstance(value, str) or value.strip().casefold() != "v1":
            raise ValueError(
                f"{entry_point} requires a v1 configuration; got {key}={value}."
            )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _require_keys(section: Mapping[str, Any], name: str, keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"Config section {name!r} is missing keys: {missing}.")
    unexpected = sorted(set(section) - set(keys))
    if unexpected:
        raise ValueError(
            f"Config section {name!r} contains unsupported keys: {unexpected}."
        )


def _bool(section: Mapping[str, Any], section_name: str, key: str) -> bool:
    value = section.get(key)
    if type(value) is not bool:
        raise ValueError(f"{section_name}.{key} must be boolean, got {value!r}.")
    return value


def _integer(
    section: Mapping[str, Any],
    section_name: str,
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = section.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{section_name}.{key} must be an integer >= {minimum}, got {value!r}."
        )
    return value


def _number(
    section: Mapping[str, Any],
    section_name: str,
    key: str,
    *,
    minimum: Optional[float] = None,
    strictly_greater: bool = False,
) -> float:
    value = section.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{section_name}.{key} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{section_name}.{key} must be finite, got {value!r}.")
    if minimum is not None:
        invalid = result <= minimum if strictly_greater else result < minimum
        if invalid:
            relation = ">" if strictly_greater else ">="
            raise ValueError(
                f"{section_name}.{key} must be {relation} {minimum}, got {value!r}."
            )
    return result


def _string(section: Mapping[str, Any], section_name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty string.")
    return value


def _nullable_string(
    section: Mapping[str, Any], section_name: str, key: str
) -> Optional[str]:
    value = section.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{section_name}.{key} must be null or a non-empty string.")
    return value


def _integer_list(
    section: Mapping[str, Any], section_name: str, key: str
) -> list[int]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{section_name}.{key} must be a non-empty integer list.")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(
            f"{section_name}.{key} values must be integers >= 0, got {value!r}."
        )
    return list(value)


def _validate_relative_path(value: str, field: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a project-relative path without '..'.")


def validate_v1_config(
    config: Any, *, entry_point: str = "v1"
) -> Dict[str, Any]:
    """Validate the v1 schema, supported components, types, and safe ranges."""

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a mapping.")
    _require_v1_version(config, entry_point)
    required_sections = (
        "experiment",
        "data",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "checkpoint",
        "test",
        "metrics",
        "logging",
    )
    unexpected_sections = sorted(set(config) - set(required_sections))
    if unexpected_sections:
        raise ValueError(
            f"YAML config contains unsupported sections: {unexpected_sections}."
        )
    for name in required_sections:
        _section(config, name)

    experiment = _section(config, "experiment")
    _require_keys(experiment, "experiment", ("version", "name", "seed", "output_root"))
    if _string(experiment, "experiment", "version") != "v1":
        raise ValueError("experiment.version must equal 'v1'.")
    name = _string(experiment, "experiment", "name")
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            "experiment.name must use only letters, digits, '.', '_' and '-'."
        )
    _integer(experiment, "experiment", "seed")
    _string(experiment, "experiment", "output_root")

    data = _section(config, "data")
    _require_keys(
        data,
        "data",
        (
            "root",
            "train_manifest",
            "validation_manifest",
            "test_manifest",
            "patch_size",
            "batch_size",
            "num_workers",
            "pin_memory",
            "pad_if_smaller",
            "augmentation",
        ),
    )
    _string(data, "data", "root")
    for key in ("train_manifest", "validation_manifest", "test_manifest"):
        value = _string(data, "data", key)
        manifest_path = resolve_manifest_path(value)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"data.{key} does not exist or is not a file: {manifest_path}"
            )
        try:
            manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"data.{key} cannot be read as UTF-8: {manifest_path}"
            ) from exc
    _integer(data, "data", "patch_size", minimum=1)
    _integer(data, "data", "batch_size", minimum=1)
    _integer(data, "data", "num_workers")
    _bool(data, "data", "pin_memory")
    _bool(data, "data", "pad_if_smaller")
    augmentation = _section(data, "augmentation")
    _require_keys(augmentation, "data.augmentation", ("hflip", "vflip", "rot90"))
    for key in ("hflip", "vflip", "rot90"):
        _bool(augmentation, "data.augmentation", key)

    model = _section(config, "model")
    _require_keys(
        model,
        "model",
        (
            "type",
            "img_channel",
            "width",
            "enc_blk_nums",
            "middle_blk_num",
            "dec_blk_nums",
        ),
    )
    if _string(model, "model", "type") != "nafnet_small":
        raise ValueError("v1 supports only model.type='nafnet_small'.")
    if _integer(model, "model", "img_channel", minimum=1) != 3:
        raise ValueError("v1 RGB data and metrics require model.img_channel=3.")
    _integer(model, "model", "width", minimum=1)
    encoder_blocks = _integer_list(model, "model", "enc_blk_nums")
    _integer(model, "model", "middle_blk_num")
    decoder_blocks = _integer_list(model, "model", "dec_blk_nums")
    if len(encoder_blocks) != len(decoder_blocks):
        raise ValueError(
            "model.enc_blk_nums and model.dec_blk_nums must have equal lengths."
        )

    loss = _section(config, "loss")
    _require_keys(loss, "loss", ("name", "epsilon"))
    if _string(loss, "loss", "name").casefold() != "charbonnier":
        raise ValueError("loss.name must equal 'charbonnier'.")
    _number(
        loss, "loss", "epsilon", minimum=0.0, strictly_greater=True
    )

    optimizer = _section(config, "optimizer")
    _require_keys(
        optimizer,
        "optimizer",
        ("name", "learning_rate", "weight_decay", "betas"),
    )
    if _string(optimizer, "optimizer", "name").casefold() != "adamw":
        raise ValueError("optimizer.name must equal 'AdamW'.")
    _number(
        optimizer,
        "optimizer",
        "learning_rate",
        minimum=0.0,
        strictly_greater=True,
    )
    _number(optimizer, "optimizer", "weight_decay", minimum=0.0)
    betas = optimizer.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(type(value) not in (int, float) for value in betas)
        or not all(
            math.isfinite(float(value)) and 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        raise ValueError("optimizer.betas must be two numeric values in [0,1).")

    scheduler = _section(config, "scheduler")
    _require_keys(scheduler, "scheduler", ("name",))
    if str(scheduler.get("name", "")).casefold() != "none":
        raise ValueError("scheduler.name must equal 'none' for v1.")

    training = _section(config, "training")
    _require_keys(
        training,
        "training",
        (
            "epochs",
            "amp",
            "deterministic",
            "validate_every",
            "save_every",
            "gradient_clip_norm",
            "resume",
            "fail_on_nonfinite",
        ),
    )
    _integer(training, "training", "epochs", minimum=1)
    _bool(training, "training", "amp")
    _bool(training, "training", "deterministic")
    _integer(training, "training", "validate_every", minimum=1)
    _integer(training, "training", "save_every", minimum=1)
    gradient_clip = training.get("gradient_clip_norm")
    if gradient_clip is not None:
        _number(
            training,
            "training",
            "gradient_clip_norm",
            minimum=0.0,
            strictly_greater=True,
        )
    _nullable_string(training, "training", "resume")
    _bool(training, "training", "fail_on_nonfinite")

    checkpoint = _section(config, "checkpoint")
    _require_keys(
        checkpoint,
        "checkpoint",
        (
            "primary",
            "save_best_psnr",
            "save_best_ssim",
            "save_best_val_loss",
            "save_last",
            "save_periodic",
        ),
    )
    if _string(checkpoint, "checkpoint", "primary").casefold() != "psnr":
        raise ValueError("checkpoint.primary must equal 'psnr'.")
    for key in (
        "save_best_psnr",
        "save_best_ssim",
        "save_best_val_loss",
        "save_last",
        "save_periodic",
    ):
        _bool(checkpoint, "checkpoint", key)

    test = _section(config, "test")
    _require_keys(
        test,
        "test",
        (
            "auto_run_after_training",
            "checkpoint",
            "run_dir",
            "allow_overwrite",
            "save_all_enhanced_images",
            "visualization",
        ),
    )
    auto_test = _bool(test, "test", "auto_run_after_training")
    if _string(test, "test", "checkpoint") != "best_psnr":
        raise ValueError("test.checkpoint must equal 'best_psnr'.")
    _nullable_string(test, "test", "run_dir")
    _bool(test, "test", "allow_overwrite")
    _bool(test, "test", "save_all_enhanced_images")
    visualization = _section(test, "visualization")
    _require_keys(
        visualization,
        "test.visualization",
        (
            "enabled",
            "num_samples",
            "random_seed",
            "columns",
            "grid_rows",
            "grid_columns",
            "cell_width",
            "cell_height",
            "preserve_aspect_ratio",
            "add_labels",
        ),
    )
    visualization_enabled = _bool(
        visualization, "test.visualization", "enabled"
    )
    num_samples = _integer(
        visualization, "test.visualization", "num_samples", minimum=1
    )
    _integer(visualization, "test.visualization", "random_seed")
    if visualization.get("columns") != ["input", "enhanced", "gt"]:
        raise ValueError(
            "test.visualization.columns must equal [input, enhanced, gt]."
        )
    grid_rows = _integer(
        visualization, "test.visualization", "grid_rows", minimum=1
    )
    grid_columns = _integer(
        visualization, "test.visualization", "grid_columns", minimum=1
    )
    if grid_columns != len(visualization["columns"]):
        raise ValueError(
            "test.visualization.grid_columns must match the number of columns."
        )
    if visualization_enabled and grid_rows < num_samples:
        raise ValueError(
            "test.visualization.grid_rows must be >= num_samples when enabled."
        )
    _integer(visualization, "test.visualization", "cell_width", minimum=1)
    cell_height = _integer(
        visualization, "test.visualization", "cell_height", minimum=1
    )
    _bool(visualization, "test.visualization", "preserve_aspect_ratio")
    add_labels = _bool(visualization, "test.visualization", "add_labels")
    if visualization_enabled and add_labels and cell_height <= 24:
        raise ValueError(
            "test.visualization.cell_height must be > 24 when labels are enabled."
        )

    metrics = _section(config, "metrics")
    _require_keys(
        metrics,
        "metrics",
        ("data_range", "crop_border", "ssim_window_size", "ssim_sigma"),
    )
    _number(metrics, "metrics", "data_range", minimum=0.0, strictly_greater=True)
    _integer(metrics, "metrics", "crop_border")
    _integer(metrics, "metrics", "ssim_window_size", minimum=1)
    _number(metrics, "metrics", "ssim_sigma", minimum=0.0, strictly_greater=True)

    logging = _section(config, "logging")
    _require_keys(
        logging,
        "logging",
        (
            "console",
            "save_train_log",
            "save_validation_log",
            "save_test_log",
            "save_metrics_history_json",
        ),
    )
    for key in (
        "console",
        "save_train_log",
        "save_validation_log",
        "save_test_log",
        "save_metrics_history_json",
    ):
        _bool(logging, "logging", key)
    if auto_test and not checkpoint["save_best_psnr"]:
        raise ValueError(
            "test.auto_run_after_training=true requires "
            "checkpoint.save_best_psnr=true."
        )
    return copy.deepcopy(config)


def resolve_v1_config_path(path: Path | str = V1_CONFIG_PATH) -> Path:
    """Resolve an arbitrary v1 config path and require it to be a file."""

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    return config_path


def load_v1_config(
    path: Path | str = V1_CONFIG_PATH, *, entry_point: str = "v1"
) -> Dict[str, Any]:
    """Read one YAML file and return an independent validated dictionary."""

    config_path = resolve_v1_config_path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the v1 configuration.") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_v1_config(config, entry_point=entry_point)


def require_canonical_v1_config(path: Path | str) -> Path:
    """Backward-compatible alias for unrestricted v1 path resolution."""

    return resolve_v1_config_path(path)


def resolve_manifest_path(path_value: str) -> Path:
    """Resolve a YAML-provided manifest path without constraining its location."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (PROJECT_ROOT / path).resolve(strict=False)


def resolve_output_root(
    path_value: str, *, project_root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a YAML-provided experiment output root."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (project_root / path).resolve(strict=False)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a safe path relative to this independent project."""

    _validate_relative_path(relative_path, "project path")
    return (PROJECT_ROOT / relative_path).resolve(strict=False)
