"""Reproducible random test sampling and YAML-sized comparison grids."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from PIL import Image, ImageDraw, ImageFont

from .logging_utils import atomic_write_json


BACKGROUND = (24, 24, 24)
TEXT_COLOR = (255, 255, 255)


def select_visualization_records(
    records: Iterable[Mapping[str, Any]], *, num_samples: int, random_seed: int
) -> List[Dict[str, Any]]:
    """Sort by sample_id, then sample distinct records with an isolated RNG."""

    ordered = sorted((dict(record) for record in records), key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("Visualization candidates contain duplicate sample_id values.")
    count = min(num_samples, len(ordered))
    return random.Random(random_seed).sample(ordered, count)


def _rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy().convert("RGB")
    with Image.open(Path(value)) as image:
        image.load()
        return image.convert("RGB")


def _cell(
    image: Image.Image,
    *,
    width: int,
    height: int,
    label: str,
    preserve_aspect_ratio: bool,
    add_labels: bool,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    label_height = 24 if add_labels else 0
    available_height = height - label_height
    source = image.copy().convert("RGB")
    if preserve_aspect_ratio:
        source.thumbnail((width, available_height), Image.Resampling.LANCZOS)
    else:
        source = source.resize((width, available_height), Image.Resampling.LANCZOS)
    left = (width - source.width) // 2
    top = label_height + (available_height - source.height) // 2
    canvas.paste(source, (left, top))
    if add_labels:
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), label, fill=TEXT_COLOR, font=ImageFont.load_default())
    return canvas


def build_visualization_grid(
    samples: List[Mapping[str, Any]],
    visualization_config: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Create a configured RGB canvas without modifying source images."""

    rows = int(visualization_config["grid_rows"])
    columns = list(visualization_config["columns"])
    grid_columns = int(visualization_config["grid_columns"])
    if columns != ["input", "enhanced", "gt"] or grid_columns != 3:
        raise ValueError("Visualization columns must be Input / Enhanced / GT.")
    if len(samples) > rows:
        raise ValueError(f"Received {len(samples)} samples for a {rows}-row grid.")
    cell_width = int(visualization_config["cell_width"])
    cell_height = int(visualization_config["cell_height"])
    grid = Image.new(
        "RGB", (cell_width * grid_columns, cell_height * rows), BACKGROUND
    )
    titles = {"input": "Input", "enhanced": "Enhanced", "gt": "GT"}
    for row_index, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        for column_index, key in enumerate(columns):
            cell = _cell(
                _rgb_image(sample[key]),
                width=cell_width,
                height=cell_height,
                label=f"{sample_id} | {titles[key]}",
                preserve_aspect_ratio=bool(
                    visualization_config["preserve_aspect_ratio"]
                ),
                add_labels=bool(visualization_config["add_labels"]),
            )
            grid.paste(cell, (column_index * cell_width, row_index * cell_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="PNG")
    return destination


def save_test_visualization(
    selected_samples: List[Mapping[str, Any]],
    visualization_config: Mapping[str, Any],
    result_dir: Path,
) -> Dict[str, Any]:
    """Save selected enhanced images, selection provenance, and the grid."""

    samples_dir = result_dir / "test_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    json_samples: List[Dict[str, str]] = []
    grid_samples: List[Dict[str, Any]] = []
    for sample in selected_samples:
        sample_id = str(sample["sample_id"])
        enhanced_path = samples_dir / f"{sample_id}_enhanced.png"
        _rgb_image(sample["enhanced"]).save(enhanced_path, format="PNG")
        json_samples.append(
            {
                "sample_id": sample_id,
                "input_relative_path": str(sample["input_relative_path"]),
                "gt_relative_path": str(sample["gt_relative_path"]),
                "enhanced_path": str(enhanced_path.relative_to(result_dir.parent)),
            }
        )
        grid_samples.append(
            {
                "sample_id": sample_id,
                "input": sample["input"],
                "enhanced": enhanced_path,
                "gt": sample["gt"],
            }
        )
    payload = {
        "selection_method": "sorted_sample_id_then_random_sample",
        "random_seed": int(visualization_config["random_seed"]),
        "requested_num_samples": int(visualization_config["num_samples"]),
        "actual_num_samples": len(selected_samples),
        "samples": json_samples,
    }
    atomic_write_json(result_dir / "test_visualization_samples.json", payload)
    grid_filename = (
        f"test_grid_{int(visualization_config['grid_rows'])}x"
        f"{int(visualization_config['grid_columns'])}.png"
    )
    build_visualization_grid(
        grid_samples,
        visualization_config,
        result_dir / grid_filename,
    )
    return payload
