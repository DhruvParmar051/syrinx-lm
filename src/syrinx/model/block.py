"""One pre-norm transformer block (HF ``LlamaDecoderLayer`` layout)."""

import torch
from torch import nn

from syrinx.config import ModelConfig
from syrinx.model.attention import CausalSelfAttention
from syrinx.model.mlp import SwiGLU
from syrinx.model.norm import RMSNorm


class Block(nn.Module):
    """``h = x + self_attn(input_layernorm(x))``; ``out = h + mlp(post_attention_layernorm(h))``."""

    def __init__(self, config: ModelConfig) -> None:
        """Create ``input_layernorm`` and ``post_attention_layernorm`` (``RMSNorm(d_model,
        norm_eps)``), ``self_attn`` (``CausalSelfAttention(d_model, n_heads)``) and ``mlp``
        (``SwiGLU(d_model, d_ff)``)."""
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = CausalSelfAttention(config.d_model, config.n_heads)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, C); cos, sin (T, D). Returns: (B, T, C)."""
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))
