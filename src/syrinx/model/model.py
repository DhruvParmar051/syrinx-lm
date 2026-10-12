"""The Syrinx language model: embeddings -> L blocks -> final RMSNorm -> tied output."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from syrinx.config import ModelConfig
from syrinx.model.block import Block
from syrinx.model.norm import RMSNorm
from syrinx.model.rope import RotaryEmbedding

INIT_STD = 0.02
"""Std of the normal init for every weight matrix (GPT-2)."""


class SyrinxLM(nn.Module):
    """Decoder-only LM. ``state_dict()`` keys equal HF ``LlamaModel``'s when tied.

    Attributes (names are part of the export contract):
        config: The ``ModelConfig``.
        embed_tokens: ``nn.Embedding(V, C)``.
        layers: ``nn.ModuleList`` of ``n_layers`` ``Block``.
        norm: Final ``RMSNorm(C, norm_eps)``.
        lm_head: ``None`` when ``tie_embeddings`` (logits use ``embed_tokens.weight``),
            else ``nn.Linear(C, V, bias=False)``.
        rotary: ``RotaryEmbedding(head_dim, max_seq_len, rope_theta)``, shared by all
            layers (non-persistent buffers).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the attributes above, then call :meth:`reset_parameters`."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head: nn.Linear | None = None
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter (in place, from the current torch RNG).

        - ``o_proj.weight`` and ``down_proj.weight`` (the projections that write into the
          residual stream): normal, mean 0, std ``INIT_STD / sqrt(2 * n_layers)``, so the
          residual variance does not grow with depth (GPT-2).
        - Every other 2-D weight (q/k/v, gate/up, embedding, untied ``lm_head``):
          normal, mean 0, std ``INIT_STD``.
        - RMSNorm weights: ones.
        """
        residual_std = INIT_STD / math.sqrt(2 * self.config.n_layers)

        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)
            elif param.ndim == 2:
                nn.init.normal_(param, mean=0.0, std=INIT_STD)
            else:
                nn.init.ones_(param)

    def num_parameters(self) -> int:
        """Total parameters, each shared tensor counted once (``self.parameters()``
        already de-duplicates)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token logits.

        Steps: ``cos, sin = rotary(T)``; ``x = embed_tokens(input_ids)`` (B, T, C); each
        block; ``norm``; logits ``F.linear(x, W)`` with ``W = embed_tokens.weight`` when
        tied, else ``lm_head.weight`` -> (B, T, V).

        Args:
            input_ids: (B, T) int64 token ids, ``T <= max_seq_len``.

        Returns:
            (B, T, V) logits, dtype of the computation (float32 without autocast).

        Raises:
            ValueError: If ``input_ids`` is not 2-D, or T > ``max_seq_len`` (from
                ``rotary``).
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B,T), got shape {tuple(input_ids.shape)}.")
        seq_len = input_ids.shape[1]
        cos, sin = self.rotary(seq_len)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(x, weight=weight)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over all B*T positions, computed in float32.

    ``F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1))``. Targets are
    already shifted by the loader (``y = window[1:]``); this function does NOT shift.

    Args:
        logits: (B, T, V).
        targets: (B, T) int64.

    Returns:
        Scalar float32 loss.

    Raises:
        ValueError: If ``logits.shape[:-1] != targets.shape``.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} disagree on (B,T)."
        )
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab).float(), target=targets.reshape(-1))
