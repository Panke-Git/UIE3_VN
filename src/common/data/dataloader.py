"""Construct deterministic-worker train, validation, and test DataLoaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.common.experiment.config import resolve_manifest_path
from src.common.experiment.seed import seed_worker

from .paired_dataset import PairedImageDataset


def _dataset(config: Dict[str, Any], split: str) -> PairedImageDataset:
    data = config["data"]
    training = split == "train"
    augmentation = data["augmentation"]
    return PairedImageDataset(
        str(Path(data["root"]).expanduser()),
        str(resolve_manifest_path(data[f"{split}_manifest"])),
        training=training,
        patch_size=int(data["patch_size"]) if training else None,
        enable_hflip=bool(augmentation["hflip"]) if training else False,
        enable_vflip=bool(augmentation["vflip"]) if training else False,
        enable_rot90=bool(augmentation["rot90"]) if training else False,
        pad_if_smaller=bool(data["pad_if_smaller"]) if training else False,
    )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_dataloaders(
    config: Dict[str, Any], *, include_test: bool = False
) -> Dict[str, DataLoader]:
    """Build train/validation loaders, optionally including the full test split."""

    data = config["data"]
    seed = int(config["experiment"]["seed"])
    common = {
        "num_workers": int(data["num_workers"]),
        "pin_memory": bool(data["pin_memory"]),
        "worker_init_fn": seed_worker,
    }
    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            _dataset(config, "train"),
            batch_size=int(data["batch_size"]),
            shuffle=True,
            generator=_generator(seed),
            **common,
        ),
        "validation": build_validation_dataloader(config),
    }
    if include_test:
        loaders["test"] = DataLoader(
            _dataset(config, "test"),
            batch_size=1,
            shuffle=False,
            generator=_generator(seed + 2),
            **common,
        )
    return loaders


def build_test_dataloader(config: Dict[str, Any]) -> DataLoader:
    """Build only the complete, ordered, non-augmented test loader."""

    data = config["data"]
    seed = int(config["experiment"]["seed"])
    return DataLoader(
        _dataset(config, "test"),
        batch_size=1,
        shuffle=False,
        num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=_generator(seed + 2),
    )


def build_validation_dataloader(config: Dict[str, Any]) -> DataLoader:
    """Build only the complete, ordered, non-augmented validation loader."""

    data = config["data"]
    seed = int(config["experiment"]["seed"])
    return DataLoader(
        _dataset(config, "validation"),
        batch_size=1,
        shuffle=False,
        num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=_generator(seed + 1),
    )
