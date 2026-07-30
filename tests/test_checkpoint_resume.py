from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.common.experiment.checkpoint import (  # noqa: E402
    BestTracker,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from src.common.experiment.config import V1_CONFIG_PATH, load_v1_config  # noqa: E402


class FakeScaler:
    def __init__(self):
        self.value = {"scale": 128.0}

    def state_dict(self):
        return dict(self.value)

    def load_state_dict(self, value):
        self.value = dict(value)


def test_checkpoint_save_load_and_resume_state(tmp_path) -> None:
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    scaler = FakeScaler()
    config = load_v1_config(V1_CONFIG_PATH)
    tracker = BestTracker(psnr=25.0, ssim=0.9, val_loss=0.04)
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        epoch=9,
        global_step=123,
        config=config,
        seed=1234,
        train_loss=0.05,
        val_loss=0.04,
        psnr=25.0,
        ssim=0.9,
        learning_rate=2.0e-4,
        best_tracker=tracker,
        torch_module=torch,
        git_commit=None,
    )
    path = save_checkpoint(
        tmp_path / "best_psnr.pt",
        payload=payload,
        torch_module=torch,
        selection_metric="validation_psnr",
    )
    target = torch.nn.Linear(3, 3)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1.0)
    target_scaler = FakeScaler()
    loaded = load_checkpoint(
        path,
        model=target,
        torch_module=torch,
        optimizer=target_optimizer,
        scaler=target_scaler,
        restore_training_state=True,
        restore_rng=True,
    )
    assert loaded["epoch"] == 9
    assert loaded["global_step"] == 123
    assert target_scaler.state_dict() == scaler.state_dict()
    for source_parameter, target_parameter in zip(
        model.parameters(), target.parameters()
    ):
        torch.testing.assert_close(source_parameter, target_parameter)
    assert path.with_suffix(".json").is_file()
