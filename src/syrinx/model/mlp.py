"""SwiGLU feed-forward (Shazeer, 2020), HF ``LlamaMLP`` layout."""

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """``down_proj(silu(gate_proj(x)) * up_proj(x))``. Shapes: (..., C) -> (..., d_ff)
    -> (..., C)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Create ``gate_proj`` and ``up_proj`` (``nn.Linear(d_model, d_ff, bias=False)``)
        and ``down_proj`` (``nn.Linear(d_ff, d_model, bias=False)``)."""
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (..., C). Returns: (..., C)."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
