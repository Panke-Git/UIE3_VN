"""Four-column visualization for the shared v2 order diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from PIL import Image

from src.common.experiment.logging_utils import atomic_write_json
from src.common.experiment.visualization import BACKGROUND, _cell, _rgb_image


SHARED_COLUMNS = [
    "input",
    "color_then_scatter",
    "scatter_then_color",
    "gt",
]


def build_shared_visualization_grid(
    samples: List[Mapping[str, Any]],
    visualization_config: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Build an exact Input / C→S / S→C / GT comparison grid."""

    rows = int(visualization_config["grid_rows"])
    columns = list(visualization_config["columns"])
    if columns != SHARED_COLUMNS or int(visualization_config["grid_columns"]) != 4:
        raise ValueError("Shared visualization requires four fixed order columns.")
    if len(samples) > rows:
        raise ValueError(f"Received {len(samples)} samples for a {rows}-row grid.")
    width = int(visualization_config["cell_width"])
    height = int(visualization_config["cell_height"])
    grid = Image.new("RGB", (width * 4, height * rows), BACKGROUND)
    titles = {
        "input": "Input",
        "color_then_scatter": "C→S",
        "scatter_then_color": "S→C",
        "gt": "GT",
    }
    for row_index, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        for column_index, key in enumerate(columns):
            cell = _cell(
                _rgb_image(sample[key]),
                width=width,
                height=height,
                label=f"{sample_id} | {titles[key]}",
                preserve_aspect_ratio=bool(
                    visualization_config["preserve_aspect_ratio"]
                ),
                add_labels=bool(visualization_config["add_labels"]),
            )
            grid.paste(cell, (column_index * width, row_index * height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="PNG")
    return destination


def save_shared_test_visualization(
    selected_samples: List[Mapping[str, Any]],
    visualization_config: Mapping[str, Any],
    result_dir: Path,
) -> Dict[str, Any]:
    """Save both selected order outputs, provenance JSON, and a four-column grid."""

    samples_root = result_dir / "test_samples"
    cs_dir = samples_root / "color_then_scatter"
    sc_dir = samples_root / "scatter_then_color"
    cs_dir.mkdir(parents=True, exist_ok=True)
    sc_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    grid_samples: List[Dict[str, Any]] = []
    for sample in selected_samples:
        sample_id = str(sample["sample_id"])
        cs_path = cs_dir / f"{sample_id}_enhanced.png"
        sc_path = sc_dir / f"{sample_id}_enhanced.png"
        _rgb_image(sample["color_then_scatter"]).save(cs_path, format="PNG")
        _rgb_image(sample["scatter_then_color"]).save(sc_path, format="PNG")
        rows.append(
            {
                "sample_id": sample_id,
                "input_relative_path": str(sample["input_relative_path"]),
                "gt_relative_path": str(sample["gt_relative_path"]),
                "color_then_scatter_enhanced_path": str(
                    cs_path.relative_to(result_dir.parent)
                ),
                "scatter_then_color_enhanced_path": str(
                    sc_path.relative_to(result_dir.parent)
                ),
            }
        )
        grid_samples.append(
            {
                "sample_id": sample_id,
                "input": sample["input"],
                "color_then_scatter": cs_path,
                "scatter_then_color": sc_path,
                "gt": sample["gt"],
            }
        )
    payload = {
        "selection_method": "sorted_sample_id_then_random_sample",
        "random_seed": int(visualization_config["random_seed"]),
        "requested_num_samples": int(visualization_config["num_samples"]),
        "actual_num_samples": len(selected_samples),
        "samples": rows,
    }
    atomic_write_json(result_dir / "test_visualization_samples.json", payload)
    filename = f"test_grid_{int(visualization_config['grid_rows'])}x4.png"
    build_shared_visualization_grid(
        grid_samples, visualization_config, result_dir / filename
    )
    return payload
