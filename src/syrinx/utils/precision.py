"""Empirical bf16 capability probe.

Whether bf16 autocast training works on a backend is tested by running the kernels
training uses (causal SDPA, a projection matmul, cross-entropy, backward) and
comparing against an fp32 reference. A dtype flag or a tensor allocation proves
nothing about the backward pass.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from syrinx.utils.device import synchronize

logger = logging.getLogger(__name__)

# Probe shapes: small enough to run in milliseconds, real enough to hit the kernels.
_B, _H, _T, _D, _V = 2, 4, 64, 32, 1024


@dataclass(frozen=True)
class Bf16ProbeResult:
    """Outcome of :func:`probe_bf16`.

    Attributes:
        supported: True if every probed op ran under bf16 autocast, the logits were
            actually bf16, the loss and all grads were finite, and the bf16 loss
            matched the fp32 reference within ``rtol``.
        reason: Human-readable explanation (success summary or failure cause).
        rel_err: Relative error of the bf16 loss vs the fp32 loss, or None if the
            probe failed before computing it.
    """

    supported: bool
    reason: str
    rel_err: float | None


def _make_probe_inputs(
    device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create seeded fp32 leaf tensors and targets for the probe.

    Shapes: q, k, v (B, H, T, D); weight (H*D, V), scaled by 1/sqrt(H*D) so logits
    are O(1); targets (B*T,) int64. Drawn on the CPU from a local generator, so the
    global RNG is untouched and results match across devices.

    Args:
        device: Device to place the tensors on.
        seed: Generator seed.

    Returns:
        ``(q, k, v, weight, targets)``; all but ``targets`` require grad.
    """
    generator = torch.Generator().manual_seed(seed)

    def leaf(*shape: int, scale: float = 1.0) -> torch.Tensor:
        values = torch.randn(*shape, generator=generator) * scale
        return values.to(device).requires_grad_()

    q, k, v = (leaf(_B, _H, _T, _D) for _ in range(3))
    weight = leaf(_H * _D, _V, scale=1.0 / math.sqrt(_H * _D))
    targets = torch.randint(0, _V, (_B * _T,), generator=generator).to(device)
    return q, k, v, weight, targets


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Merge attention heads: (B, H, T, D) -> (B, T, H*D).

    Transposes to (B, T, H, D) before reshaping; reshaping (B, H, T, D) directly
    would mix tokens across heads. Invariant: ``out[b, t, h*D:(h+1)*D] == x[b, h, t]``.

    Args:
        x: Tensor of shape (B, H, T, D).

    Returns:
        Tensor of shape (B, T, H*D).
    """
    batch, heads, seq, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, seq, heads * head_dim)


def _probe_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.dtype]:
    """Causal SDPA -> merge heads -> projection -> fp32 cross-entropy.

    Precision is set by the caller (inside or outside autocast).

    Returns:
        ``(loss, logits_dtype)``: the scalar fp32 loss and the dtype the logits were
        computed in, so the caller can confirm autocast actually applied.
    """
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    logits = _merge_heads(attn) @ weight
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets)
    return loss, logits.dtype


def probe_bf16(device: torch.device, rtol: float = 3e-2, seed: int = 0) -> Bf16ProbeResult:
    """Empirically test whether bf16 autocast training works on ``device``.

    Never raises for probe failures; they are reported in the result. Restores the
    caller's grad mode. Runs on any device, including cpu.

    Args:
        device: Device to probe.
        rtol: Max relative error of the bf16 loss vs the fp32 loss. bf16 has
            eps = 2**-7, so values far below ~1e-2 give false failures.
        seed: Seed for the probe inputs.

    Returns:
        The probe outcome.
    """
    grad_was_enabled = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        q, k, v, weight, targets = _make_probe_inputs(device, seed)
        ref_loss, _ = _probe_forward(q, k, v, weight, targets)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, logits_dtype = _probe_forward(q, k, v, weight, targets)
        loss.backward()
        synchronize(device)

        if logits_dtype != torch.bfloat16:
            return Bf16ProbeResult(False, f"autocast produced {logits_dtype} logits", None)
        grads = [t.grad for t in (q, k, v, weight)]
        if any(g is None for g in grads):
            return Bf16ProbeResult(False, "backward produced no gradient for a leaf", None)
        if not torch.isfinite(loss).item():
            return Bf16ProbeResult(False, f"non-finite bf16 loss {loss.item()}", None)
        if not all(torch.isfinite(g).all().item() for g in grads if g is not None):
            return Bf16ProbeResult(False, "non-finite gradient under bf16 autocast", None)

        rel_err = abs(loss.item() - ref_loss.item()) / abs(ref_loss.item())
        if rel_err > rtol:
            return Bf16ProbeResult(
                False, f"bf16 loss differs from fp32 by {rel_err:.2e} > rtol {rtol:.0e}", rel_err
            )
        return Bf16ProbeResult(
            True, f"bf16 autocast fwd+bwd ok on {device.type}, rel_err {rel_err:.2e}", rel_err
        )
    except Exception as exc:  # noqa: BLE001 -- contract: report any failure, never raise
        logger.warning("bf16 probe failed on %s: %s", device.type, exc)
        return Bf16ProbeResult(False, f"{type(exc).__name__}: {exc}", None)
    finally:
        torch.set_grad_enabled(grad_was_enabled)
