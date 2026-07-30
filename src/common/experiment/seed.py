"""Global and DataLoader-worker seed handling from the verified baseline."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker-specific PyTorch seed."""

    if type(worker_id) is not int or worker_id < 0:
        raise ValueError(
            f"worker_id must be a non-negative integer, got {worker_id!r}."
        )
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
