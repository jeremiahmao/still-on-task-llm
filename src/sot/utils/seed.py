"""Reproducibility: seed all RNGs."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds for random, numpy, torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
