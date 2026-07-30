"""Paired crop, padding, and augmentation preserved from the UIE3 baseline."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def paired_reflect_pad(
    input_array: np.ndarray,
    target_array: np.ndarray,
    pad_height: int,
    pad_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    try:
        return (
            np.pad(input_array, padding, mode="reflect"),
            np.pad(target_array, padding, mode="reflect"),
        )
    except ValueError as exc:
        raise ValueError(
            "Reflect padding failed; the source image may be too small for the "
            f"requested padding (bottom={pad_height}, right={pad_width})."
        ) from exc


def paired_training_transform(
    input_array: np.ndarray,
    target_array: np.ndarray,
    *,
    sample_id: str,
    patch_size: int | None,
    enable_hflip: bool,
    enable_vflip: bool,
    enable_rot90: bool,
    pad_if_smaller: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical random spatial operations to one input/GT pair."""

    if patch_size is not None:
        height, width = input_array.shape[:2]
        pad_height = max(0, patch_size - height)
        pad_width = max(0, patch_size - width)
        if pad_height or pad_width:
            if not pad_if_smaller:
                raise ValueError(
                    f"Sample {sample_id!r} has size {height}x{width}, smaller than "
                    f"patch_size={patch_size}, and pad_if_smaller is false."
                )
            input_array, target_array = paired_reflect_pad(
                input_array, target_array, pad_height, pad_width
            )
            height, width = input_array.shape[:2]
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)
        input_array = input_array[top : top + patch_size, left : left + patch_size]
        target_array = target_array[top : top + patch_size, left : left + patch_size]

    if enable_hflip and random.random() < 0.5:
        input_array = np.flip(input_array, axis=1)
        target_array = np.flip(target_array, axis=1)
    if enable_vflip and random.random() < 0.5:
        input_array = np.flip(input_array, axis=0)
        target_array = np.flip(target_array, axis=0)
    if enable_rot90:
        rotations = random.randint(0, 3)
        if rotations:
            input_array = np.rot90(input_array, rotations, axes=(0, 1))
            target_array = np.rot90(target_array, rotations, axes=(0, 1))
    return input_array, target_array
