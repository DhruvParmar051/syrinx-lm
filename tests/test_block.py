"""Tests for ``syrinx.model.block.Block``."""

import torch

from syrinx.config import ModelConfig
from syrinx.model import Block, RMSNorm, rope_cos_sin

CFG = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_ff=96,
    max_seq_len=16,
    rope_theta=10000.0,
    norm_eps=1e-5,
)


def _block(seed: int = 0) -> Block:
    torch.manual_seed(seed)
    block = Block(CFG)
    with torch.no_grad():  # non-trivial norm weights, so a skipped norm is visible
        for module in block.modules():
            if isinstance(module, RMSNorm):
                module.weight.uniform_(0.5, 1.5)
    return block


def test_submodules_and_parameter_count() -> None:
    block = Block(CFG)
    children = dict(block.named_children())
    assert set(children) == {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"}
    d, ff = CFG.d_model, CFG.d_ff
    assert sum(p.numel() for p in block.parameters()) == 4 * d * d + 3 * d * ff + 2 * d
    assert block.input_layernorm.eps == CFG.norm_eps


def test_pre_norm_residual_structure(each_device: torch.device) -> None:
    block = _block().to(each_device)
    x = torch.randn(2, 7, 32).to(each_device)
    cos, sin = (t.to(each_device) for t in rope_cos_sin(CFG.head_dim, 7, CFG.rope_theta))
    with torch.no_grad():
        h = x + block.self_attn(block.input_layernorm(x), cos, sin)
        want = h + block.mlp(block.post_attention_layernorm(h))
        out = block(x, cos, sin)
    assert out.shape == x.shape
    torch.testing.assert_close(out, want)


def test_is_causal(each_device: torch.device) -> None:
    block = _block(1).to(each_device)
    x = torch.randn(1, 9, 32).to(each_device)
    cos, sin = (t.to(each_device) for t in rope_cos_sin(CFG.head_dim, 9, CFG.rope_theta))
    with torch.no_grad():
        base = block(x, cos, sin)
        x2 = x.clone()
        x2[:, 6] += 1.0
        out = block(x2, cos, sin)
    torch.testing.assert_close(out[:, :6], base[:, :6])
    assert not torch.allclose(out[:, 6:], base[:, 6:])
