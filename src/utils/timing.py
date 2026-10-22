"""Small timing helpers for training/profile scripts."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import torch


def synchronized_time(device: str | torch.device = "cpu") -> float:
    """Return a wall-clock timestamp after synchronizing CUDA work if needed."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return time.perf_counter()


def percent(part: float, total: float) -> float:
    """Return ``part / total`` as a percentage, guarded for zero totals."""

    return 0.0 if total <= 0.0 else 100.0 * float(part) / float(total)


def format_profile_block(
    title: str,
    items: Iterable[tuple[str, float]],
    *,
    total_seconds: float,
) -> str:
    """Format timing rows as a readable newline block."""

    lines = [title]
    for label, seconds in items:
        lines.append(f"  {label:<22} {seconds:>9.4f}s  {percent(seconds, total_seconds):>6.2f}%")
    lines.append(f"  {'total':<22} {total_seconds:>9.4f}s  {100.0:>6.2f}%")
    return "\n".join(lines)


def mean_seconds(records: list[dict[str, float]], key: str) -> float:
    """Mean timing value for a key across profile records."""

    values = [float(record.get(key, 0.0)) for record in records]
    return float(np.mean(values)) if values else 0.0


def sum_seconds(records: list[dict[str, float]], key: str) -> float:
    """Total timing value for a key across profile records."""

    values = [float(record.get(key, 0.0)) for record in records]
    return float(np.sum(values)) if values else 0.0
