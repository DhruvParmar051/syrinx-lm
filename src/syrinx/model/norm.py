"""Root-mean-square layer norm (Zhang & Sennrich, 2019), as in HF ``LlamaRMSNorm``."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """``y = weight * (x / sqrt(mean(x**2, dim=-1) + eps))`` over the last dimension.

    No mean subtraction and no bias. Shapes: ``x`` (..., C) -> (..., C).
    """

    def __init__(self, dim: int, eps: float) -> None:
        """Store ``eps`` and create ``self.weight = nn.Parameter(torch.ones(dim))``.

        Args:
            dim: C, the size of the normalized (last) dimension.
            eps: Added to the mean square before the square root.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension and scale by ``weight``.

        The statistic is computed in float32 whatever ``x.dtype`` is (bf16 squares lose
        too much precision); the normalized tensor is cast back to ``x.dtype`` and THEN
        multiplied by ``weight`` (HF order: ``weight * normed.to(input_dtype)``).

        Args:
            x: (..., C).

        Returns:
            (..., C). Dtype follows torch promotion of ``weight`` and ``x``: bf16 input with
            a float32 weight gives float32 (as HF does); under autocast or after
            ``.to(torch.bfloat16)`` it is bf16.
        """
        input_dtype = x.dtype
        hidden = x.float()
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return self.weight * hidden.to(input_dtype)
