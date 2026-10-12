"""Tests for ``syrinx.model.rope`` (HF Llama split-halves convention)."""

import math

import pytest
import torch

from syrinx.model import RotaryEmbedding, apply_rope, rope_cos_sin, rope_inv_freq, rotate_half


def test_inv_freq_values() -> None:
    inv = rope_inv_freq(8, 10000.0)
    assert inv.shape == (4,) and inv.dtype == torch.float32
    want = torch.tensor([10000.0 ** (-i / 8) for i in (0, 2, 4, 6)])
    torch.testing.assert_close(inv, want)
    assert inv[0].item() == 1.0


def test_cos_sin_tables() -> None:
    cos, sin = rope_cos_sin(8, 16, 10000.0)
    assert cos.shape == sin.shape == (16, 8)
    assert cos.dtype == sin.dtype == torch.float32
    # split halves: column i and i + D/2 share an angle
    torch.testing.assert_close(cos[:, :4], cos[:, 4:])
    torch.testing.assert_close(sin[:, :4], sin[:, 4:])
    # position 0 is the identity rotation
    assert torch.equal(cos[0], torch.ones(8)) and torch.equal(sin[0], torch.zeros(8))
    inv = rope_inv_freq(8, 10000.0)
    torch.testing.assert_close(cos[5, :4], torch.cos(5 * inv))
    torch.testing.assert_close(sin[13, :4], torch.sin(13 * inv))


def test_rotate_half_explicit() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert torch.equal(rotate_half(x), torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]))
    batched = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(rotate_half(batched)[1, 2], torch.tensor([-22.0, -23.0, 20.0, 21.0]))


def test_apply_rope_pairs_i_with_i_plus_half() -> None:
    """One pair rotated by a known angle: dims (0, D/2) form the 2-D vector that rotates."""
    d = 4
    cos, sin = rope_cos_sin(d, 3, 10000.0)
    x = torch.zeros(1, 1, 3, d)
    x[..., 0] = 1.0  # unit vector on dim 0
    y = apply_rope(x, cos, sin)
    # position p, pair 0 has angle p * inv_freq[0] = p
    for p in range(3):
        torch.testing.assert_close(y[0, 0, p, 0], torch.tensor(math.cos(p)))
        torch.testing.assert_close(y[0, 0, p, 2], torch.tensor(math.sin(p)))
        assert y[0, 0, p, 1] == 0 and y[0, 0, p, 3] == 0


def test_position_zero_is_identity_and_shape_kept(each_device: torch.device) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 8).to(each_device)
    cos, sin = (t.to(each_device) for t in rope_cos_sin(8, 5, 10000.0))
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, :, 0], x[:, :, 0])
    assert not torch.allclose(y[:, :, 1:], x[:, :, 1:])


def test_preserves_norm(each_device: torch.device) -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 4, 64, 32).to(each_device)
    cos, sin = (t.to(each_device) for t in rope_cos_sin(32, 64, 10000.0))
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_dot_product_depends_only_on_offset(each_device: torch.device) -> None:
    """rope(q at m) . rope(k at n) == rope(q at m+s) . rope(k at n+s)."""
    torch.manual_seed(2)
    d, t = 16, 40
    q, k = torch.randn(d).to(each_device), torch.randn(d).to(each_device)
    cos, sin = (x.to(each_device) for x in rope_cos_sin(d, t, 10000.0))
    qs = apply_rope(q.expand(1, 1, t, d), cos, sin)[0, 0]
    ks = apply_rope(k.expand(1, 1, t, d), cos, sin)[0, 0]
    for m, n, s in [(3, 1, 7), (10, 10, 25), (0, 5, 30), (20, 2, 17)]:
        torch.testing.assert_close(qs[m] @ ks[n], qs[m + s] @ ks[n + s], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(qs[3] @ ks[1], qs[3] @ ks[2], atol=1e-3)


def test_bf16_stays_bf16(each_device: torch.device) -> None:
    x = torch.randn(1, 2, 4, 8).to(each_device, torch.bfloat16)
    cos, sin = (t.to(each_device) for t in rope_cos_sin(8, 4, 10000.0))
    y = apply_rope(x, cos, sin)
    assert y.dtype == torch.bfloat16
    want = (x * cos.to(torch.bfloat16)) + rotate_half(x) * sin.to(torch.bfloat16)
    assert torch.equal(y, want)


def test_rotary_embedding_module(each_device: torch.device) -> None:
    rot = RotaryEmbedding(8, 16, 10000.0).to(each_device)
    cos, sin = rot(10)
    assert cos.device.type == sin.device.type == each_device.type  # buffers move
    full_cos, full_sin = rope_cos_sin(8, 16, 10000.0)
    assert torch.equal(cos.cpu(), full_cos[:10]) and torch.equal(sin.cpu(), full_sin[:10])
    assert rot(16)[0].shape == (16, 8)
    assert list(rot.state_dict()) == []  # non-persistent: not in checkpoints
    assert {name for name, _ in rot.named_buffers()} == {"cos", "sin"}
    assert list(rot.parameters()) == []


def test_rotary_embedding_too_long_raises() -> None:
    with pytest.raises(ValueError, match="17"):
        RotaryEmbedding(8, 16, 10000.0)(17)


def test_matches_hf_llama_rotary() -> None:
    """Same tables and same rotation as HF Llama (catches the interleaved convention)."""
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    from transformers import LlamaConfig

    torch.manual_seed(3)
    b, h, t, d = 2, 3, 12, 16
    cfg = LlamaConfig(
        hidden_size=h * d, num_attention_heads=h, max_position_embeddings=t, rope_theta=500.0
    )
    hf_rot = llama.LlamaRotaryEmbedding(config=cfg)
    x = torch.randn(b, h, t, d)
    pos = torch.arange(t).unsqueeze(0)
    hf_cos, hf_sin = hf_rot(x, pos)
    cos, sin = RotaryEmbedding(d, t, 500.0)(t)
    torch.testing.assert_close(cos, hf_cos[0])
    torch.testing.assert_close(sin, hf_sin[0])
    q, k = torch.randn(b, h, t, d), torch.randn(b, h, t, d)
    hf_q, hf_k = llama.apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    torch.testing.assert_close(apply_rope(q, cos, sin), hf_q)
    torch.testing.assert_close(apply_rope(k, cos, sin), hf_k)
