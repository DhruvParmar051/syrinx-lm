"""Syrinx decoder-only transformer: RMSNorm, RoPE (split halves), causal MHA, SwiGLU.

Module and parameter names follow HF ``LlamaModel`` so the Phase 9 export is a rename:
``SyrinxLM.state_dict()`` has exactly the keys of ``LlamaModel.state_dict()`` (tied
embeddings; prefix ``model.`` on export).
"""

from syrinx.model.attention import CausalSelfAttention, causal_attention_reference
from syrinx.model.block import Block
from syrinx.model.mlp import SwiGLU
from syrinx.model.model import INIT_STD, SyrinxLM, lm_loss
from syrinx.model.norm import RMSNorm
from syrinx.model.rope import RotaryEmbedding, apply_rope, rope_cos_sin, rope_inv_freq, rotate_half

__all__ = [
    "INIT_STD",
    "Block",
    "CausalSelfAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "SyrinxLM",
    "apply_rope",
    "causal_attention_reference",
    "lm_loss",
    "rope_cos_sin",
    "rope_inv_freq",
    "rotate_half",
]
