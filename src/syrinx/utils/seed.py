"""Global seeding for reproducible runs.

Seeds Python's ``random``, NumPy's legacy global RNG, and torch (CPU and every
accelerator). Capturing and restoring RNG state for checkpoint resume is part of
the training pipeline, not this module.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MAX_SEED = 2**32 - 1
"""NumPy's legacy seeder accepts seeds in [0, 2**32 - 1]."""


def seed_everything(seed: int) -> None:
    """Seed every global RNG Syrinx uses.

    ``torch.manual_seed`` seeds the CPU generator and all accelerator generators
    (MPS, CUDA). Determinism of GPU *kernels* is a separate matter and is not
    guaranteed on MPS; exact-resume tests run on CPU (PLAN.md Phase 4c).

    Args:
        seed: Seed in ``[0, 2**32 - 1]``.

    Raises:
        TypeError: If ``seed`` is not an int (bools are rejected).
        ValueError: If ``seed`` is out of range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.debug("Seeded python, numpy and torch with %d", seed)
