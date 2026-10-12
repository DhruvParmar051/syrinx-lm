"""Tests for ``syrinx.model.mlp.SwiGLU``."""

import torch
import torch.nn.functional as F

from syrinx.model import SwiGLU


def test_parameters() -> None:
    mlp = SwiGLU(16, 48)
    shapes = {n: tuple(p.shape) for n, p in mlp.named_parameters()}
    assert shapes == {
        "gate_proj.weight": (48, 16),
        "up_proj.weight": (48, 16),
        "down_proj.weight": (16, 48),
    }
    assert sum(p.numel() for p in mlp.parameters()) == 3 * 16 * 48


def test_matches_formula() -> None:
    torch.manual_seed(0)
    mlp = SwiGLU(16, 40)
    x = torch.randn(2, 5, 16)
    gate = x @ mlp.gate_proj.weight.T
    up = x @ mlp.up_proj.weight.T
    want = (gate * torch.sigmoid(gate) * up) @ mlp.down_proj.weight.T
    out = mlp(x)
    assert out.shape == (2, 5, 16)
    torch.testing.assert_close(out, want)


def test_gate_and_up_not_swapped() -> None:
    """silu is applied to gate_proj's output, not up_proj's."""
    torch.manual_seed(1)
    mlp = SwiGLU(8, 24)
    x = torch.randn(4, 8)
    swapped = mlp.down_proj(F.silu(mlp.up_proj(x)) * mlp.gate_proj(x))
    assert not torch.allclose(mlp(x), swapped)
