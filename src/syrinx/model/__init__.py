"""Syrinx decoder-only transformer: RMSNorm, RoPE (split halves), causal MHA, SwiGLU.

Module and parameter names follow HF ``LlamaModel`` so the Phase 9 export is a rename:
``SyrinxLM.state_dict()`` has exactly the keys of ``LlamaModel.state_dict()`` (tied
embeddings; prefix ``model.`` on export).
"""

from syrinx.model.norm import RMSNorm

__all__ = [
    "RMSNorm",
]
