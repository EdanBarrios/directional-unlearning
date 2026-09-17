"""Single entry point for randomness. Nothing else calls random.seed or torch.manual_seed."""

import random

import numpy as np


def rng(seed: int) -> random.Random:
    """Independent RNG stream for one purpose (dataset build, shuffle order)."""
    return random.Random(seed)


def set_seed(seed: int) -> None:
    """Seed every global RNG. Call once at the top of a training or eval script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
