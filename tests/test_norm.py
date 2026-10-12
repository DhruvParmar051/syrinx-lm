"""Tests for ``syrinx.model.norm.RMSNorm``."""

import pytest
import torch

from syrinx.model import RMSNorm


def _manual(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * x / torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)


def test_weight_is_ones_and_only_parameter() -> None:
    norm = RMSNorm(16, eps=1e-5)
    params = dict(norm.named_parameters())
    assert list(params) == ["weight"]
    assert params["weight"].shape == (16,)
    assert torch.equal(params["weight"].detach(), torch.ones(16))


def test_shape_and_dtype() -> None:
    x = torch.randn(2, 3, 16)
    y = RMSNorm(16, eps=1e-5)(x)
    assert y.shape == x.shape and y.dtype == torch.float32


def test_matches_manual_formula(each_device: torch.device) -> None:
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    norm.to(each_device)
    x = (torch.randn(4, 7, 32) * 3 + 0.5).to(each_device)
    torch.testing.assert_close(norm(x), _manual(x, norm.weight.detach(), 1e-5))


def test_unit_rms_with_unit_weight(each_device: torch.device) -> None:
    """With weight = 1 every output row has RMS ~1, and no mean is subtracted."""
    x = (torch.randn(5, 64) * 4 + 10).to(each_device)
    y = RMSNorm(64, eps=1e-8).to(each_device)(x)
    torch.testing.assert_close(y.pow(2).mean(-1).sqrt().cpu(), torch.ones(5))
    assert (y.mean(-1) > 0.5).all()  # LayerNorm would centre this to ~0


def test_normalizes_last_dim_only() -> None:
    """Rows are normalized independently: scaling one row leaves the others unchanged."""
    torch.manual_seed(1)
    x = torch.randn(3, 8)
    x2 = x.clone()
    x2[0] *= 100
    norm = RMSNorm(8, eps=1e-6)
    torch.testing.assert_close(norm(x)[1:], norm(x2)[1:])
    torch.testing.assert_close(norm(x)[0], norm(x2)[0], rtol=1e-4, atol=1e-4)


def test_eps_used(each_device: torch.device) -> None:
    """eps is inside the square root: zero input gives zeros (not NaN), and a large eps
    visibly shrinks the output."""
    norm = RMSNorm(4, eps=1e-5).to(each_device)
    assert torch.equal(norm(torch.zeros(2, 4, device=each_device)).cpu(), torch.zeros(2, 4))
    x = torch.ones(1, 4, device=each_device)
    got = RMSNorm(4, eps=3.0).to(each_device)(x).cpu()
    torch.testing.assert_close(got, torch.full((1, 4), 0.5))


def test_bf16_input_statistic_in_float32(each_device: torch.device) -> None:
    """The statistic is computed in float32 and the normalized tensor is rounded to the
    input dtype BEFORE the weight multiply (HF order). With a float32 weight the product
    promotes to float32; with a bf16 weight (model cast to bf16) it stays bf16."""
    torch.manual_seed(2)
    x = (torch.randn(8, 256) * 50).to(torch.bfloat16)
    normed = _manual(x.float(), torch.ones(256), 1e-5).to(torch.bfloat16)
    norm = RMSNorm(256, eps=1e-5).to(each_device)
    y = norm(x.to(each_device))
    assert y.dtype == torch.float32
    torch.testing.assert_close(y.cpu(), normed.float(), rtol=0, atol=0)
    y16 = norm.to(torch.bfloat16)(x.to(each_device))
    assert y16.dtype == torch.bfloat16
    torch.testing.assert_close(y16.cpu(), normed, rtol=0, atol=0)


def test_matches_hf_llama_rmsnorm() -> None:
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    torch.manual_seed(3)
    ours, theirs = RMSNorm(32, eps=1e-6), llama.LlamaRMSNorm(32, eps=1e-6)
    with torch.no_grad():
        w = torch.randn(32)
        ours.weight.copy_(w)
        theirs.weight.copy_(w)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 5, 32).to(dtype)
        torch.testing.assert_close(ours(x), theirs(x), rtol=0, atol=0)
