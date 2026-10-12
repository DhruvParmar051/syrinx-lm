"""Causal multi-head self-attention with RoPE on q and k (HF ``LlamaAttention`` layout,
n_kv_heads = n_heads, no biases)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from syrinx.model.rope import apply_rope


def causal_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention written out, as the reference for ``F.scaled_dot_product_attention``.

    ``scores = q @ k^T / sqrt(D)`` (B, H, T, T); positions ``j > i`` set to ``-inf``;
    softmax over the last dimension in float32, cast back to ``q.dtype``; ``@ v``.

    Args:
        q, k, v: (B, H, T, D).

    Returns:
        (B, H, T, D).
    """
    seq_len, head_dim = q.shape[-2], q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return weights @ v


class CausalSelfAttention(nn.Module):
    """``o_proj(SDPA(rope(q), rope(k), v, causal))`` with heads split from C = H * D."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Create ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``: each
        ``nn.Linear(d_model, d_model, bias=False)`` (separate, not fused: HF names).
        Store ``n_heads`` and ``head_dim = d_model // n_heads``.

        Raises:
            ValueError: If ``d_model % n_heads != 0``.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Self-attention over ``x``.

        Steps: project q, k, v (B, T, C); split heads ``view(B, T, H, D).transpose(1, 2)``
        -> (B, H, T, D); ``apply_rope`` to q and k only; ``F.scaled_dot_product_attention(
        q, k, v, is_causal=True)``; merge heads ``transpose(1, 2).reshape(B, T, C)``;
        ``o_proj``.

        Args:
            x: (B, T, C).
            cos, sin: (T, D) from ``RotaryEmbedding``.

        Returns:
            (B, T, C).
        """
        batch, seq_len, channels = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).reshape(batch, seq_len, channels)

        return self.o_proj(y)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B,T,C) -> (B, H,T,D)."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
