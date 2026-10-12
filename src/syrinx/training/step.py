"""One optimizer step with gradient accumulation, clipping and optional bf16 autocast, and
a forward-only evaluation pass.

The caller sets the learning rate (``optim.set_lr``) before :func:`train_step`; this
module never touches it.
"""

import contextlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from syrinx.model import lm_loss

Batch = tuple[torch.Tensor, torch.Tensor]
"""``(x, y)``, each (B, T) int64; ``y`` is ``x`` shifted by one (from the loader)."""


@dataclass(frozen=True)
class StepStats:
    """What one optimizer step reports.

    Attributes:
        loss: Mean of the micro-batch losses (unscaled; NOT divided by accum steps).
        grad_norm: Global L2 norm of all gradients BEFORE clipping.
        tokens: Target tokens processed in this step, summed over micro-batches.
    """

    loss: float
    grad_norm: float
    tokens: int


def autocast_context(device_type: str, precision: str) -> contextlib.AbstractContextManager[None]:
    """The forward-pass context for ``precision``.

    ``"fp32"`` -> ``contextlib.nullcontext()``; ``"bf16-autocast"`` ->
    ``torch.autocast(device_type=device_type, dtype=torch.bfloat16)``.

    Raises:
        ValueError: For any other precision (message names it).
    """
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16-autocast":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    raise ValueError(f"Unkown precision {precision!r}; expected 'fp32' or 'bf16-autocast'.")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    get_batch: Callable[[], Batch],
    grad_accum_steps: int,
    grad_clip: float,
    precision: str,
) -> StepStats:
    """One optimizer step over ``grad_accum_steps`` micro-batches.

    1. ``model.train()``; ``optimizer.zero_grad(set_to_none=True)``.
    2. For each micro-batch: ``x, y = get_batch()``; forward + ``lm_loss`` inside
       :func:`autocast_context` (device type from ``x``); backward on
       ``loss / grad_accum_steps``, so the summed gradient equals the gradient of the
       mean loss over all micro-batches; keep the unscaled ``loss.detach()``.
    3. ``grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)``
       (returns the norm before clipping).
    4. If ``grad_norm`` is not finite, raise BEFORE ``optimizer.step()``, so the weights
       are never updated with NaN/inf.
    5. ``optimizer.step()``.

    Call ``.item()`` (a device sync) only once per step for the loss, not per micro-batch.

    Args:
        model: The model (its parameters are what ``optimizer`` updates).
        optimizer: The optimizer, with the step's LR already set.
        get_batch: Returns the next ``(x, y)``; called exactly ``grad_accum_steps`` times.
        grad_accum_steps: Micro-batches per step; positive.
        grad_clip: Max global grad norm.
        precision: ``"fp32"`` or ``"bf16-autocast"``.

    Returns:
        :class:`StepStats`.

    Raises:
        ValueError: If ``grad_accum_steps`` is not positive (before any batch is drawn).
        FloatingPointError: If the gradient norm is NaN or inf (message includes it).
    """
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps}.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros(())
    tokens = 0
    for _ in range(grad_accum_steps):
        x, y = get_batch()
        with autocast_context(x.device.type, precision=precision):
            loss = lm_loss(model(x), y)
        (loss / grad_accum_steps).backward()
        loss_sum = loss_sum.to(loss.device) + loss.detach()
        tokens += y.numel()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    if not math.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite gradient norm {grad_norm}; skip the update.")
    optimizer.step()
    return StepStats(loss=(loss_sum / grad_accum_steps).item(), grad_norm=grad_norm, tokens=tokens)


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[Batch], precision: str) -> float:
    """Mean loss per target token over every batch (forward only).

    Batches may differ in size (the last validation batch is smaller), so the result is
    ``sum(loss_b * n_b) / sum(n_b)`` with ``n_b = y.numel()``, not a mean of batch
    means. Runs in ``eval()`` mode and restores the model's previous mode afterwards,
    even on error.

    Raises:
        ValueError: If ``batches`` is empty.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        for x, y in batches:
            with autocast_context(x.device.type, precision=precision):
                loss = lm_loss(model(x), y)
            n = y.numel()
            total += loss.item() * n
            count += n
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("evaluate() got no batches.")
    return total / count
