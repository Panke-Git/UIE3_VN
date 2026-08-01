from __future__ import annotations

import json

import pytest
from PIL import Image

from src.common.experiment.experiment import run_training_and_auto_test
from src.v1.test_v1 import _resize_output_image, ensure_test_outputs_available
from src.v1.train_v1 import _progress_log_due, _validation_due


def _run(tmp_path, training, testing, progress=None):
    return run_training_and_auto_test(
        training_callable=training,
        test_callable=testing,
        status_path=tmp_path / "status.json",
        error_path=tmp_path / "error.json",
        progress=progress or {"epoch": 1, "global_step": 10},
        auto_test=True,
    )


def test_all_epochs_success_calls_only_best_psnr(tmp_path) -> None:
    calls = []
    _run(
        tmp_path,
        lambda: calls.append("training") or "trained",
        lambda checkpoint: calls.append(checkpoint) or "tested",
    )
    assert calls == ["training", "best_psnr"]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall"] == "COMPLETED"
    assert status["test"] == "COMPLETED"


def test_training_failure_does_not_call_test(tmp_path) -> None:
    calls = []

    def fail():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        _run(tmp_path, fail, lambda checkpoint: calls.append(checkpoint))
    assert calls == []
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall"] == "FAILED"
    assert status["test"] == "NOT_RUN"
    assert (tmp_path / "error.json").is_file()


def test_test_failure_preserves_completed_training(tmp_path) -> None:
    def fail_test(checkpoint):
        assert checkpoint == "best_psnr"
        raise RuntimeError("test failed")

    with pytest.raises(RuntimeError, match="test failed"):
        _run(tmp_path, lambda: "trained", fail_test)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["overall"] == "PARTIAL_FAILURE"
    assert status["training"] == "COMPLETED"
    assert status["test"] == "FAILED"


def test_existing_test_output_is_rejected_without_overwrite(tmp_path) -> None:
    result = tmp_path / "result"
    result.mkdir()
    (result / "test_metrics.csv").write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="allow_overwrite=false"):
        ensure_test_outputs_available(result, allow_overwrite=False)
    ensure_test_outputs_available(result, allow_overwrite=True)


def test_validation_cadence_always_includes_final_epoch() -> None:
    assert [
        epoch
        for epoch in range(5)
        if _validation_due(epoch, total_epochs=5, validate_every=2)
    ] == [1, 3, 4]


def test_progress_logging_includes_first_interval_and_last_batch() -> None:
    assert [
        batch
        for batch in range(1, 24)
        if _progress_log_due(batch, total_batches=23, log_every_steps=10)
    ] == [1, 10, 20, 23]


def test_saved_test_image_is_resized_to_configured_square() -> None:
    source = Image.new("RGB", (19, 31), (10, 20, 30))
    resized = _resize_output_image(source, 256)
    assert resized.size == (256, 256)
    assert source.size == (19, 31)
