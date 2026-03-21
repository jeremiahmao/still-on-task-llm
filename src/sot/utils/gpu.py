"""GPU memory and compute cost tracking."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class ComputeStats:
    """Tracks GPU-hours and peak memory for a single run."""
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    device: str = "cuda:0"

    @property
    def gpu_hours(self) -> float:
        return self.elapsed_seconds / 3600


@contextmanager
def track_compute(device: str = "cuda:0"):
    """Context manager that tracks wall-clock time and peak GPU memory.

    Usage:
        with track_compute() as stats:
            # ... training code ...
        print(f"GPU-hours: {stats.gpu_hours:.2f}, Peak mem: {stats.peak_memory_gb:.1f} GB")
    """
    stats = ComputeStats(device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    stats.start_time = time.monotonic()
    try:
        yield stats
    finally:
        stats.elapsed_seconds = time.monotonic() - stats.start_time
        if torch.cuda.is_available():
            stats.peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
