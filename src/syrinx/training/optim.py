"""AdamW with decoupled weight decay on matrices only, and a warmup + cosine LR schedule.

The schedule is a pure function of the optimizer step, applied with :func:`set_lr` before
every step. There is no ``LRScheduler`` object, so a checkpoint only needs the step number
to restore the learning rate exactly.
"""

import math
from typing import Any

import torch
from torch import nn

from syrinx.config import TrainConfig

ADAM_EPS = 1e-8
"""AdamW epsilon (PyTorch default)."""

NO_DECAY_NAMES = ("embed_tokens.weight",)
"""2-D parameters exempt from weight decay: the (tied) embedding."""


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split trainable parameters into a decayed group and a non-decayed group.

    Decayed: every parameter with ``ndim >= 2`` whose name is not in ``NO_DECAY_NAMES``
    (q/k/v/o, gate/up/down, and an untied ``lm_head``). Not decayed: the embedding and
    every 1-D parameter (RMSNorm weights). Parameters with ``requires_grad=False`` are
    left out. ``named_parameters()`` yields a shared (tied) tensor once, so each tensor
    lands in exactly one group.

    Args:
        model: The model.
        weight_decay: Decay for the first group.

    Returns:
        ``[{"params": decayed, "weight_decay": weight_decay},
        {"params": not_decayed, "weight_decay": 0.0}]``, each list in
        ``named_parameters()`` order.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and name not in NO_DECAY_NAMES:
            decay.append(param)
        else:
            no_decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, train: TrainConfig) -> torch.optim.AdamW:
    """``torch.optim.AdamW(param_groups(model, train.weight_decay), lr=train.lr,
    betas=train.betas, eps=ADAM_EPS)``."""
    return torch.optim.AdamW(
        param_groups(model=model, weight_decay=train.weight_decay),
        lr=train.lr,
        betas=train.betas,
        eps=ADAM_EPS,
    )


def lr_at(step: int, peak_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate for optimizer step ``step`` (0-based).

    - Warmup, ``step < warmup_steps``: linear, ``peak_lr * (step + 1) / warmup_steps``, so
      step 0 already has a non-zero LR and step ``warmup_steps - 1`` reaches ``peak_lr``.
    - Decay, ``warmup_steps <= step < max_steps``: cosine from ``peak_lr`` to ``min_lr``,
      ``min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))`` with
      ``progress = (step - warmup_steps) / (max_steps - warmup_steps)``.
    - After, ``step >= max_steps``: ``min_lr``.

    Raises:
        ValueError: If ``step`` is negative.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set ``lr`` on every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr
