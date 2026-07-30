from __future__ import annotations

import json

from PIL import Image

from src.common.experiment.visualization import (
    save_test_visualization,
    select_visualization_records,
)


def _records(count: int):
    return [
        {
            "sample_id": f"{index:02d}",
            "input_relative_path": f"input/{index}.png",
            "gt_relative_path": f"gt/{index}.png",
        }
        for index in reversed(range(count))
    ]


def _visual(record):
    return {
        **record,
        "input": Image.new("RGB", (7, 5), (255, 0, 0)),
        "enhanced": Image.new("RGB", (7, 5), (0, 255, 0)),
        "gt": Image.new("RGB", (7, 5), (0, 0, 255)),
    }


def _config():
    return {
        "num_samples": 10,
        "random_seed": 3407,
        "columns": ["input", "enhanced", "gt"],
        "grid_rows": 10,
        "grid_columns": 3,
        "cell_width": 32,
        "cell_height": 24,
        "preserve_aspect_ratio": False,
        "add_labels": False,
    }


def test_random_selection_is_reproducible_and_unique() -> None:
    first = select_visualization_records(
        _records(15), num_samples=10, random_seed=3407
    )
    second = select_visualization_records(
        _records(15), num_samples=10, random_seed=3407
    )
    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in second
    ]
    assert len({row["sample_id"] for row in first}) == 10


def test_grid_json_column_order_and_source_immutability(tmp_path) -> None:
    records = select_visualization_records(
        _records(12), num_samples=10, random_seed=3407
    )
    samples = [_visual(record) for record in records]
    before = [sample["input"].tobytes() for sample in samples]
    payload = save_test_visualization(samples, _config(), tmp_path)
    assert payload["actual_num_samples"] == 10
    assert before == [sample["input"].tobytes() for sample in samples]
    with Image.open(tmp_path / "test_grid_10x3.png") as grid:
        assert grid.mode == "RGB"
        assert grid.size == (32 * 3, 24 * 10)
        assert grid.getpixel((16, 12)) == (255, 0, 0)
        assert grid.getpixel((32 + 16, 12)) == (0, 255, 0)
        assert grid.getpixel((64 + 16, 12)) == (0, 0, 255)
    with (tmp_path / "test_visualization_samples.json").open(
        "r", encoding="utf-8"
    ) as handle:
        saved = json.load(handle)
    assert saved["random_seed"] == 3407
    assert len(saved["samples"]) == 10


def test_fewer_than_ten_uses_every_sample_without_repetition(tmp_path) -> None:
    selected = select_visualization_records(
        _records(4), num_samples=10, random_seed=3407
    )
    assert len(selected) == 4
    assert len({row["sample_id"] for row in selected}) == 4
    payload = save_test_visualization(
        [_visual(record) for record in selected], _config(), tmp_path
    )
    assert payload["actual_num_samples"] == 4
