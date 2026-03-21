"""Compute cost tracking: GPU-hours and peak memory per method."""

from sot.utils.gpu import ComputeStats


def format_compute_report(stats: ComputeStats, method_name: str, scale: int) -> dict:
    """Format compute stats into a standardized report dict."""
    return {
        "method": method_name,
        "scale": scale,
        "gpu_hours": round(stats.gpu_hours, 3),
        "elapsed_seconds": round(stats.elapsed_seconds, 1),
        "peak_memory_gb": round(stats.peak_memory_gb, 2),
        "device": stats.device,
    }
