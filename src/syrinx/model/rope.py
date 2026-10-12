"""Rotary position embeddings, HF Llama "rotate_half" (split-halves) convention.

Dimension ``i < D/2`` of a head is paired with dimension ``i + D/2`` (NOT ``2i`` with
``2i + 1``, the interleaved convention); both use the frequency ``inv_freq[i]``.
Rotating q and k by an angle proportional to position makes ``q_m . k_n`` depend only on
``m - n``. Shapes: B batch, H heads, T positions, D head_dim.
"""

import torch
from torch import nn


def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """Per-pair rotation frequencies ``inv_freq[i] = theta ** (-2i / D)``, i in [0, D/2).

    Args:
        head_dim: D; even.
        theta: RoPE base (10000.0).

    Returns:
        (D/2,) float32.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
    return 1.0 / (theta**exponents)


def rope_cos_sin(
    head_dim: int, max_seq_len: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos and sin tables for positions ``0 .. max_seq_len - 1``.

    ``angles = outer(positions, inv_freq)`` is (T, D/2); it is duplicated along the last
    dimension, ``cat([angles, angles], -1)`` -> (T, D), so column ``i`` and ``i + D/2``
    share an angle. Computed in float32.

    Returns:
        ``(cos, sin)``, each (max_seq_len, D) float32.
    """
    inv_freq = rope_inv_freq(head_dim=head_dim, theta=theta)
    positions = torch.arange(max_seq_len, dtype=torch.int64).float()
    angles = torch.outer(positions, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` where x1, x2 are the first and second halves of the last
    dimension.

    Args:
        x: (..., D).

    Returns:
        (..., D).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin``, with ``cos``/``sin`` cast to ``x.dtype`` first
    (so bf16 q/k stay bf16).

    Args:
        x: (B, H, T, D).
        cos: (T, D); broadcasts over B and H.
        sin: (T, D).

    Returns:
        (B, H, T, D), same dtype as ``x``.
    """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables up to ``max_seq_len``.

    The tables are NON-persistent buffers (``register_buffer(..., persistent=False)``):
    they move with ``.to(device)`` but are not in ``state_dict()``, matching HF, whose
    checkpoints have no rotary tensors.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        """Store ``max_seq_len`` and register buffers ``cos`` and ``sin`` from
        :func:`rope_cos_sin`."""
        super().__init__()
        self.max_seq_len = max_seq_len
        cos, sin = rope_cos_sin(head_dim=head_dim, max_seq_len=max_seq_len, theta=theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The first ``seq_len`` rows of ``cos`` and ``sin``, each (seq_len, D).

        Raises:
            ValueError: If ``seq_len > max_seq_len`` (message names both).
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}.")
        return self.cos[:seq_len], self.sin[:seq_len]
