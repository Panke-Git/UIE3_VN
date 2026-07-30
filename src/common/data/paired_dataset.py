"""Strict TSV-backed paired RGB dataset migrated from the verified baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import paired_training_transform


@dataclass(frozen=True)
class ManifestEntry:
    """One exact-ID input/GT pair from a three-column manifest."""

    sample_id: str
    input_relative_path: str
    gt_relative_path: str
    input_path: Path
    gt_path: Path


def _resolve_dataset_file(
    dataset_root: Path, relative_text: str, line_number: int
) -> Path:
    if not relative_text:
        raise ValueError(f"Manifest line {line_number} contains an empty image path.")
    relative_path = Path(relative_text)
    if relative_path.is_absolute():
        raise ValueError(
            f"Manifest line {line_number} must use a relative path, got "
            f"{relative_text!r}."
        )
    resolved = (dataset_root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest line {line_number} escapes dataset_root: {relative_text!r}."
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(
            f"Manifest line {line_number} references a missing file: {resolved}"
        )
    if not resolved.is_file():
        raise ValueError(
            f"Manifest line {line_number} does not reference a regular file: {resolved}"
        )
    return resolved


def read_manifest(dataset_root: Path, manifest_path: Path) -> List[ManifestEntry]:
    """Read an existing, non-empty, unique-ID, three-column TSV."""

    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist or is not a file: {manifest_path}"
        )
    entries: List[ManifestEntry] = []
    sample_ids = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            columns = line.split("\t")
            if len(columns) != 3:
                raise ValueError(
                    f"Manifest line {line_number} must contain exactly 3 "
                    f"tab-separated columns; found {len(columns)}."
                )
            sample_id, input_relative_path, gt_relative_path = columns
            if not sample_id or not sample_id.strip():
                raise ValueError(
                    f"Manifest line {line_number} contains an empty sample_id."
                )
            if (
                Path(sample_id).name != sample_id
                or sample_id in {".", ".."}
                or "/" in sample_id
                or "\\" in sample_id
            ):
                raise ValueError(
                    f"Manifest sample_id {sample_id!r} is unsafe for result filenames."
                )
            if sample_id in sample_ids:
                raise ValueError(
                    f"Manifest contains duplicate sample_id {sample_id!r} at "
                    f"line {line_number}."
                )
            sample_ids.add(sample_id)
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    input_relative_path=input_relative_path,
                    gt_relative_path=gt_relative_path,
                    input_path=_resolve_dataset_file(
                        dataset_root, input_relative_path, line_number
                    ),
                    gt_path=_resolve_dataset_file(
                        dataset_root, gt_relative_path, line_number
                    ),
                )
            )
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return entries


def load_rgb_array(path: Path) -> np.ndarray:
    """Decode with Pillow, convert to RGB float32, and scale by exactly 255."""

    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image as RGB: {path}: {exc}") from exc
    return array / np.float32(255.0)


class PairedImageDataset(Dataset):
    """Load paired `[3,H,W]` RGB float32 tensors in `[0,1]`.

    Random crop and augmentation execute only when ``training=True``. Validation
    and test therefore always return the complete original image.
    """

    def __init__(
        self,
        dataset_root: str,
        manifest_path: str,
        *,
        training: bool,
        patch_size: Optional[int] = None,
        enable_hflip: bool = False,
        enable_vflip: bool = False,
        enable_rot90: bool = False,
        pad_if_smaller: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            raise FileNotFoundError(
                "dataset_root does not exist or is not a directory: "
                f"{self.dataset_root}"
            )
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        if patch_size is not None and (
            type(patch_size) is not int or patch_size <= 0
        ):
            raise ValueError(
                f"patch_size must be a positive integer or None, got {patch_size!r}."
            )
        self.training = bool(training)
        self.patch_size = patch_size
        self.enable_hflip = bool(enable_hflip)
        self.enable_vflip = bool(enable_vflip)
        self.enable_rot90 = bool(enable_rot90)
        self.pad_if_smaller = bool(pad_if_smaller)
        self.entries = read_manifest(self.dataset_root, self.manifest_path)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        return torch.from_numpy(contiguous).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        input_array = load_rgb_array(entry.input_path)
        target_array = load_rgb_array(entry.gt_path)
        if input_array.shape != target_array.shape:
            raise ValueError(
                f"Input/GT size mismatch for sample {entry.sample_id!r}: "
                f"input={input_array.shape[:2]}, GT={target_array.shape[:2]}."
            )
        if self.training:
            input_array, target_array = paired_training_transform(
                input_array,
                target_array,
                sample_id=entry.sample_id,
                patch_size=self.patch_size,
                enable_hflip=self.enable_hflip,
                enable_vflip=self.enable_vflip,
                enable_rot90=self.enable_rot90,
                pad_if_smaller=self.pad_if_smaller,
            )
        input_tensor = self._to_tensor(input_array)
        target_tensor = self._to_tensor(target_array)
        if input_tensor.shape[0] != 3 or target_tensor.shape[0] != 3:
            raise RuntimeError(f"RGB conversion failed for sample {entry.sample_id!r}.")
        return {
            "sample_id": entry.sample_id,
            "input": input_tensor,
            "target": target_tensor,
            "input_relative_path": entry.input_relative_path,
            "gt_relative_path": entry.gt_relative_path,
        }
