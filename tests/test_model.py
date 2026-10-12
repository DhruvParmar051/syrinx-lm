"""Tests for ``syrinx.model.model`` (SyrinxLM and lm_loss)."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from syrinx.config import ModelConfig, load_config, num_parameters
from syrinx.model import INIT_STD, RMSNorm, SyrinxLM, lm_loss

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["syrinx-10m", "syrinx-50m", "syrinx-125m"]

TINY = ModelConfig(
    vocab_size=128,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=192,
    max_seq_len=32,
    rope_theta=10000.0,
    norm_eps=1e-5,
)


def _model_config(name: str) -> ModelConfig:
    return load_config(ROOT / "configs" / f"{name}.yaml").model


def _tiny(seed: int = 0, **changes: object) -> SyrinxLM:
    torch.manual_seed(seed)
    return SyrinxLM(dataclasses.replace(TINY, **changes))


def _expected_keys(cfg: ModelConfig) -> set[str]:
    keys = {"embed_tokens.weight", "norm.weight"}
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        keys |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        keys |= {f"{p}self_attn.{x}_proj.weight" for x in "qkvo"}
        keys |= {f"{p}mlp.{x}_proj.weight" for x in ("gate", "up", "down")}
    if not cfg.tie_embeddings:
        keys.add("lm_head.weight")
    return keys


# --- structure and parameter count ----------------------------------------------------


@pytest.mark.parametrize("name", CONFIGS)
def test_parameter_count_matches_analytic(name: str) -> None:
    """sum(p.numel()) equals the pinned analytic count (built on the meta device)."""
    cfg = _model_config(name)
    with torch.device("meta"):
        model = SyrinxLM(cfg)
    counted = sum(p.numel() for p in model.parameters())
    assert counted == model.num_parameters() == num_parameters(cfg)
    pinned = {"syrinx-10m": 11_798_784, "syrinx-50m": 50_866_688, "syrinx-125m": 124_275_456}
    assert counted == pinned[name]


def test_tied_embeddings_share_one_tensor() -> None:
    model = _tiny()
    assert model.lm_head is None
    assert model.num_parameters() == num_parameters(TINY)
    names = [n for n, _ in model.named_parameters()]
    assert names.count("embed_tokens.weight") == 1 and not any("lm_head" in n for n in names)


def test_untied_has_separate_head() -> None:
    model = _tiny(tie_embeddings=False)
    assert model.lm_head is not None
    assert tuple(model.lm_head.weight.shape) == (TINY.vocab_size, TINY.d_model)
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    cfg = dataclasses.replace(TINY, tie_embeddings=False)
    assert model.num_parameters() == num_parameters(cfg)


@pytest.mark.parametrize("tie", [True, False])
def test_state_dict_keys(tie: bool) -> None:
    """Exactly the Llama names; no rotary tables in the checkpoint."""
    cfg = dataclasses.replace(TINY, tie_embeddings=tie)
    model = SyrinxLM(cfg)
    assert set(model.state_dict()) == _expected_keys(cfg)
    assert len(model.layers) == cfg.n_layers


def test_state_dict_matches_hf_llama_model() -> None:
    pytest.importorskip("transformers")
    from transformers import LlamaModel

    assert set(SyrinxLM(TINY).state_dict()) == set(LlamaModel(_llama_config(TINY)).state_dict())


# --- init ---------------------------------------------------------------------------


def test_init_statistics() -> None:
    """std 0.02 for weights, 0.02/sqrt(2L) for o_proj/down_proj, ones for norms."""
    cfg = _model_config("syrinx-10m")
    torch.manual_seed(0)
    model = SyrinxLM(cfg)
    resid = INIT_STD / math.sqrt(2 * cfg.n_layers)
    for name, p in model.named_parameters():
        if p.ndim == 1:
            assert torch.equal(p.detach(), torch.ones_like(p)), name
            continue
        want = resid if name.endswith(("o_proj.weight", "down_proj.weight")) else INIT_STD
        assert abs(p.std().item() - want) < 0.05 * want, name
        assert abs(p.mean().item()) < 0.05 * want, name


def test_untied_head_init() -> None:
    torch.manual_seed(0)
    model = SyrinxLM(dataclasses.replace(_model_config("syrinx-10m"), tie_embeddings=False))
    assert model.lm_head is not None
    assert abs(model.lm_head.weight.std().item() - INIT_STD) < 0.05 * INIT_STD


def test_init_is_seeded() -> None:
    a, b, c = _tiny(seed=5), _tiny(seed=5), _tiny(seed=6)
    for (name, pa), pb, pc in zip(
        a.state_dict().items(), b.state_dict().values(), c.state_dict().values(), strict=True
    ):
        assert torch.equal(pa, pb), name
        if pa.ndim == 2:
            assert not torch.equal(pa, pc), name


def test_reset_parameters_reinitializes() -> None:
    """reset_parameters overwrites every parameter, as a pure function of the RNG state."""
    a, b = _tiny(seed=0), _tiny(seed=1)
    with torch.no_grad():
        for p in a.parameters():
            p.fill_(3.0)
    for model in (a, b):
        torch.manual_seed(7)
        model.reset_parameters()
    for (name, p), q in zip(a.state_dict().items(), b.state_dict().values(), strict=True):
        assert torch.equal(p, q), name
        assert not (p == 3.0).any(), name


def test_initial_loss_near_log_vocab(each_device: torch.device) -> None:
    """Untrained 10m model on random tokens: loss ~ ln(32768) = 10.40 (near-uniform output)."""
    cfg = _model_config("syrinx-10m")
    torch.manual_seed(0)
    model = SyrinxLM(cfg).eval().to(each_device)
    tokens = torch.randint(0, cfg.vocab_size, (4, 257), generator=torch.Generator().manual_seed(1))
    tokens = tokens.to(each_device)
    with torch.no_grad():
        loss = lm_loss(model(tokens[:, :-1]), tokens[:, 1:]).item()
    assert abs(loss - math.log(cfg.vocab_size)) < 0.15, loss


# --- forward ------------------------------------------------------------------------


def test_forward_shape_and_dtype(each_device: torch.device) -> None:
    model = _tiny().to(each_device)
    out = model(torch.randint(0, TINY.vocab_size, (3, 10)).to(each_device))
    assert out.shape == (3, 10, TINY.vocab_size) and out.dtype == torch.float32
    assert out.device.type == each_device.type


def test_forward_uses_tied_embedding_as_output() -> None:
    """Logits are final hidden states times the embedding matrix (no extra head)."""
    model = _tiny()
    ids = torch.randint(0, TINY.vocab_size, (2, 6))
    captured: dict[str, torch.Tensor] = {}
    model.norm.register_forward_hook(lambda m, i, o: captured.setdefault("h", o))
    with torch.no_grad():
        logits = model(ids)
    torch.testing.assert_close(logits, captured["h"] @ model.embed_tokens.weight.T)


def test_forward_composition() -> None:
    """embed -> blocks (shared rotary tables) -> final norm -> tied output."""
    model = _tiny(seed=3)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, RMSNorm):
                m.weight.uniform_(0.5, 1.5)
    ids = torch.randint(0, TINY.vocab_size, (2, 9))
    cos, sin = model.rotary(9)
    with torch.no_grad():
        x = model.embed_tokens(ids)
        for layer in model.layers:
            x = layer(x, cos, sin)
        want = F.linear(model.norm(x), model.embed_tokens.weight)
        torch.testing.assert_close(model(ids), want)


def test_full_context_and_too_long() -> None:
    model = _tiny()
    assert model(torch.zeros(1, TINY.max_seq_len, dtype=torch.long)).shape[1] == TINY.max_seq_len
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY.max_seq_len + 1, dtype=torch.long))


def test_non_2d_input_raises() -> None:
    with pytest.raises(ValueError):
        _tiny()(torch.zeros(5, dtype=torch.long))


def test_model_is_causal(each_device: torch.device) -> None:
    """Perturbing token t leaves the logits at positions < t unchanged."""
    model = _tiny(seed=1).eval().to(each_device)
    ids = torch.randint(0, TINY.vocab_size, (2, 20), generator=torch.Generator().manual_seed(0))
    ids = ids.to(each_device)
    with torch.no_grad():
        base = model(ids)
        for t in (1, 10, 19):
            ids2 = ids.clone()
            ids2[:, t] = (ids2[:, t] + 1) % TINY.vocab_size
            out = model(ids2)
            torch.testing.assert_close(out[:, :t], base[:, :t])
            assert not torch.allclose(out[:, t], base[:, t])


def test_prefix_consistency(each_device: torch.device) -> None:
    """Logits for a prefix equal the first positions of the logits for the full sequence."""
    model = _tiny(seed=2).eval().to(each_device)
    ids = torch.randint(0, TINY.vocab_size, (1, 16), generator=torch.Generator().manual_seed(3))
    ids = ids.to(each_device)
    with torch.no_grad():
        torch.testing.assert_close(model(ids[:, :7]), model(ids)[:, :7], rtol=1e-5, atol=1e-5)


def test_every_parameter_gets_a_gradient(each_device: torch.device) -> None:
    model = _tiny().to(each_device)
    ids = torch.randint(0, TINY.vocab_size, (2, 12)).to(each_device)
    lm_loss(model(ids[:, :-1]), ids[:, 1:]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_bf16_autocast_forward(each_device: torch.device) -> None:
    """The 10m config trains under bf16 autocast: logits come out bf16 and finite, the
    loss is float32, and it stays within 0.05 of the float32 loss for the same weights."""
    cfg = _model_config("syrinx-10m")
    torch.manual_seed(0)
    model = SyrinxLM(cfg).eval().to(each_device)
    ids = torch.randint(0, cfg.vocab_size, (2, 257), generator=torch.Generator().manual_seed(1))
    ids = ids.to(each_device)
    with torch.no_grad():
        fp32_loss = lm_loss(model(ids[:, :-1]), ids[:, 1:])
        with torch.autocast(device_type=each_device.type, dtype=torch.bfloat16):
            logits = model(ids[:, :-1])
            loss = lm_loss(logits, ids[:, 1:])
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    assert loss.dtype == torch.float32
    assert abs(loss.item() - fp32_loss.item()) < 0.05, (loss.item(), fp32_loss.item())


# --- HF parity ----------------------------------------------------------------------


def _llama_config(cfg: ModelConfig):  # type: ignore[no-untyped-def]
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        max_position_embeddings=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
    )


def test_logits_match_hf_llama() -> None:
    """Our weights loaded by name into LlamaForCausalLM give the same logits."""
    pytest.importorskip("transformers")
    from transformers import LlamaForCausalLM

    model = _tiny(seed=4).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, RMSNorm):
                m.weight.uniform_(0.5, 1.5)
    hf = LlamaForCausalLM(_llama_config(TINY)).eval()
    hf.model.load_state_dict(model.state_dict(), strict=True)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(0, TINY.vocab_size, (2, TINY.max_seq_len))
    with torch.no_grad():
        torch.testing.assert_close(model(ids), hf(ids).logits, rtol=1e-4, atol=1e-5)


# --- lm_loss ------------------------------------------------------------------------


def test_lm_loss_matches_manual() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    logp = logits.log_softmax(-1)
    want = -logp.gather(-1, targets.unsqueeze(-1)).mean()
    loss = lm_loss(logits, targets)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, want)


def test_lm_loss_does_not_shift() -> None:
    """Position i is scored against targets[i]: perfect logits give ~0 loss."""
    targets = torch.randint(0, 7, (3, 4))
    logits = F.one_hot(targets, 7).float() * 50
    assert lm_loss(logits, targets).item() < 1e-6


def test_lm_loss_float32_from_bf16() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 50).to(torch.bfloat16)
    targets = torch.randint(0, 50, (2, 8))
    loss = lm_loss(logits, targets)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(
        loss, F.cross_entropy(logits.float().reshape(-1, 50), targets.reshape(-1))
    )


def test_lm_loss_shape_mismatch_raises() -> None:
    """Including (B, T) vs (T, B): same element count, so cross_entropy alone would not
    notice the misaligned targets."""
    with pytest.raises(ValueError):
        lm_loss(torch.randn(2, 5, 11), torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        lm_loss(torch.randn(2, 5, 11), torch.zeros(5, 2, dtype=torch.long))


# --- MPS ----------------------------------------------------------------------------


@pytest.mark.mps
def test_cpu_and_mps_agree(mps_device: torch.device) -> None:
    """Same weights, same tokens: MPS logits and loss match CPU within float32 tolerance."""
    cfg = _model_config("syrinx-10m")
    torch.manual_seed(0)
    model = SyrinxLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 257), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        cpu_logits = model(ids[:, :-1])
        cpu_loss = lm_loss(cpu_logits, ids[:, 1:])
        model.to(mps_device)
        assert model.rotary.cos.device.type == "mps"
        mps_logits = model(ids[:, :-1].to(mps_device))
        mps_loss = lm_loss(mps_logits, ids[:, 1:].to(mps_device))
    torch.testing.assert_close(mps_logits.cpu(), cpu_logits, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(mps_loss.cpu(), cpu_loss, rtol=1e-4, atol=1e-4)
