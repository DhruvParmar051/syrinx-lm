"""Tests for ``syrinx.model.attention``."""

import math

import pytest
import torch
import torch.nn.functional as F

from syrinx.model import CausalSelfAttention, apply_rope, causal_attention_reference, rope_cos_sin


def _qkv(b: int = 2, h: int = 3, t: int = 9, d: int = 8, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(b, h, t, d, generator=gen) for _ in range(3)]


def test_reference_matches_sdpa() -> None:
    q, k, v = _qkv()
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = causal_attention_reference(q, k, v)
    assert got.shape == q.shape
    torch.testing.assert_close(got, want)


def test_reference_first_position_attends_to_itself_only() -> None:
    q, k, v = _qkv()
    torch.testing.assert_close(causal_attention_reference(q, k, v)[:, :, 0], v[:, :, 0])


def test_reference_is_causal() -> None:
    q, k, v = _qkv()
    base = causal_attention_reference(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 5:] += 10.0
    v2[:, :, 5:] -= 7.0
    out = causal_attention_reference(q, k2, v2)
    torch.testing.assert_close(out[:, :, :5], base[:, :, :5])
    assert not torch.allclose(out[:, :, 5:], base[:, :, 5:])


def test_reference_scale_is_sqrt_head_dim() -> None:
    """Two positions, hand-computed: position 1 weights are softmax([q.k0, q.k1] / sqrt(D))."""
    d = 4
    q = torch.zeros(1, 1, 2, d)
    k = torch.zeros(1, 1, 2, d)
    v = torch.tensor([[[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]]])
    q[..., 1, 0] = 2.0
    k[..., 1, 0] = 3.0  # q1.k1 = 6, q1.k0 = 0
    w1 = torch.softmax(torch.tensor([0.0, 6.0 / math.sqrt(d)]), dim=0)
    out = causal_attention_reference(q, k, v)
    torch.testing.assert_close(out[0, 0, 1, :2], w1)


def test_module_parameters() -> None:
    attn = CausalSelfAttention(32, 4)
    names = {n: p.shape for n, p in attn.named_parameters()}
    assert names == {f"{p}_proj.weight": (32, 32) for p in "qkvo"}
    assert sum(p.numel() for p in attn.parameters()) == 4 * 32 * 32
    assert attn.n_heads == 4 and attn.head_dim == 8


def test_module_bad_heads_raises() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(30, 4)


def _manual_module(attn: CausalSelfAttention, x: torch.Tensor, cos, sin) -> torch.Tensor:
    b, t, c = x.shape
    h, d = attn.n_heads, c // attn.n_heads

    def split(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(b, t, h, d).permute(0, 2, 1, 3)

    q = apply_rope(split(x @ attn.q_proj.weight.T), cos, sin)
    k = apply_rope(split(x @ attn.k_proj.weight.T), cos, sin)
    v = split(x @ attn.v_proj.weight.T)
    y = causal_attention_reference(q, k, v).permute(0, 2, 1, 3).reshape(b, t, c)
    return y @ attn.o_proj.weight.T


def test_module_matches_manual_pipeline() -> None:
    """Head split (h-major within C), RoPE on q and k only, causal attention, head merge."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(48, 6)
    x = torch.randn(2, 11, 48)
    cos, sin = rope_cos_sin(8, 11, 10000.0)
    with torch.no_grad():
        out = attn(x, cos, sin)
        want = _manual_module(attn, x, cos, sin)
    assert out.shape == (2, 11, 48)
    torch.testing.assert_close(out, want, rtol=1e-4, atol=1e-5)


def test_module_is_causal() -> None:
    """Changing token t changes no output before t."""
    torch.manual_seed(1)
    attn = CausalSelfAttention(32, 4)
    x = torch.randn(1, 10, 32)
    cos, sin = rope_cos_sin(8, 10, 10000.0)
    with torch.no_grad():
        base = attn(x, cos, sin)
        for t in (0, 4, 9):
            x2 = x.clone()
            x2[:, t] = torch.randn(32)
            out = attn(x2, cos, sin)
            torch.testing.assert_close(out[:, :t], base[:, :t])
            assert not torch.allclose(out[:, t], base[:, t])


def test_module_batch_items_independent() -> None:
    torch.manual_seed(2)
    attn = CausalSelfAttention(32, 4)
    x = torch.randn(3, 6, 32)
    cos, sin = rope_cos_sin(8, 6, 10000.0)
    with torch.no_grad():
        full = attn(x, cos, sin)
        one = attn(x[1:2], cos, sin)
    torch.testing.assert_close(full[1:2], one, rtol=1e-5, atol=1e-6)
