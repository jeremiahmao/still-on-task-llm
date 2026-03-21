"""Custom training callbacks for wandb logging and GPU tracking."""

import torch
from transformers import TrainerCallback

import wandb


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and logs is not None:
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
