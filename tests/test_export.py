"""Tests for ``syrinx.export``: config, weight mapping, logit parity, tokenizer, export folder."""

import dataclasses
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from syrinx.config import ModelConfig, config_from_dict, config_to_dict, load_config
from syrinx.data.shards import EOS_ID
from syrinx.export import (
    PAD_ID,
    PARITY_TOLERANCE,
    export_checkpoint,
    generation_config,
    hf_tokenizer,
    llama_config,
    max_logit_diff,
    to_hf_model,
    to_hf_state_dict,
)
from syrinx.generate import generate
from syrinx.model import SyrinxLM
from syrinx.tokenizer import EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, encode_document, load_tokenizer
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
TINY = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_ff=96,
    max_seq_len=32,
    rope_theta=10000.0,
    norm_eps=1e-5,
)
TEXTS = [
    "Once upon a time, there was a little girl named Lily.",
    "café naïve — “quotes”, emoji 😀, tabs\tand\r\nnewlines\n\n",
    "digits 1234567890 and  double  spaces ",
    "日本語のテキスト",
]


def _model(config: ModelConfig = TINY, seed: int = 0) -> SyrinxLM:
    """A model with every weight random, so norms and both RoPE halves all matter."""
    torch.manual_seed(seed)
    model = SyrinxLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.ndim == 1 else 0.0))
    return model.eval()


def _ids(b: int, t: int, vocab: int = 64, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, vocab, (b, t), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


# llama_config -----------------------------------------------------------------------------


def test_llama_config_fields() -> None:
    c = llama_config(TINY)
    assert (c.vocab_size, c.hidden_size, c.num_hidden_layers) == (64, 32, 2)
    assert (c.num_attention_heads, c.num_key_value_heads, c.intermediate_size) == (4, 4, 96)
    assert c.max_position_embeddings == 32 and c.head_dim == 8
    assert c.rms_norm_eps == 1e-5  # HF default is 1e-6
    assert c.rope_parameters["rope_theta"] == 10000.0
    assert c.rope_parameters["rope_type"] == "default"
    assert c.hidden_act == "silu" and not c.attention_bias and not c.mlp_bias
    assert c.tie_word_embeddings is True
    assert (c.bos_token_id, c.eos_token_id, c.pad_token_id) == (EOS_ID, EOS_ID, PAD_ID) == (0, 0, 1)


def test_llama_config_follows_model_values() -> None:
    other = dataclasses.replace(TINY, rope_theta=50000.0, norm_eps=1e-6, tie_embeddings=False)
    c = llama_config(other)
    assert c.rope_parameters["rope_theta"] == 50000.0 and c.rms_norm_eps == 1e-6
    assert c.tie_word_embeddings is False


# to_hf_state_dict -----------------------------------------------------------------------------


def test_state_dict_keys_and_shapes_match_llama() -> None:
    ours = to_hf_state_dict(SyrinxLM(TINY).state_dict())
    hf = LlamaForCausalLM(llama_config(TINY)).state_dict()
    assert set(ours) == set(hf) - {"lm_head.weight"}  # tied head is not stored
    assert all(k.startswith("model.") for k in ours)
    for key, tensor in ours.items():
        assert tensor.shape == hf[key].shape, key


def test_untied_head_keeps_its_name() -> None:
    untied = dataclasses.replace(TINY, tie_embeddings=False)
    ours = to_hf_state_dict(SyrinxLM(untied).state_dict())
    hf = LlamaForCausalLM(llama_config(untied)).state_dict()
    assert "lm_head.weight" in ours and set(ours) == set(hf)


def test_to_hf_state_dict_rejects_prefixed_keys() -> None:
    with pytest.raises(ValueError, match="model."):
        to_hf_state_dict({"model.norm.weight": torch.ones(4)})


def test_to_hf_state_dict_keeps_values() -> None:
    sd = SyrinxLM(TINY).state_dict()
    ours = to_hf_state_dict(sd)
    for key, tensor in sd.items():
        assert torch.equal(ours[f"model.{key}"], tensor)


# to_hf_model and logit parity ----------------------------------------------------------------


@pytest.mark.parametrize("t", [1, 16, 32])
def test_logit_parity_fp32(t: int) -> None:
    model = _model()
    hf = to_hf_model(model, TINY)
    ids = _ids(2, t)
    with torch.no_grad():
        ours, theirs = model(ids), hf(input_ids=ids).logits
    assert theirs.shape == ours.shape == (2, t, 64)
    assert float((ours - theirs).abs().max()) < PARITY_TOLERANCE
    assert max_logit_diff(model, hf, ids) < PARITY_TOLERANCE


def test_logit_parity_untied() -> None:
    untied = dataclasses.replace(TINY, tie_embeddings=False)
    model = _model(untied)
    hf = to_hf_model(model, untied)
    assert hf.lm_head.weight.data_ptr() != hf.model.embed_tokens.weight.data_ptr()
    assert max_logit_diff(model, hf, _ids(2, 16)) < PARITY_TOLERANCE


def test_parity_check_has_teeth() -> None:
    """A wrong rope_theta leaves position 0 identical but must break later positions."""
    model = _model()
    wrong = to_hf_model(model, dataclasses.replace(TINY, rope_theta=500.0))
    ids = _ids(1, 16)
    with torch.no_grad():
        diff = (model(ids) - wrong(input_ids=ids).logits).abs()
    assert float(diff[:, 0].max()) < PARITY_TOLERANCE
    assert float(diff[:, 1:].max()) > 1e-2
    assert max_logit_diff(model, wrong, ids) > 1e-2


def test_to_hf_model_ties_head_and_is_eval_fp32() -> None:
    hf = to_hf_model(_model(), TINY)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    assert not hf.training and hf.model.embed_tokens.weight.dtype == torch.float32


def test_to_hf_model_rejects_missing_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    sd = model.state_dict()
    sd.pop("norm.weight")
    monkeypatch.setattr(model, "state_dict", lambda: sd)
    with pytest.raises(ValueError, match="key mismatch"):
        to_hf_model(model, TINY)


def test_to_hf_model_rejects_extra_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    sd = dict(model.state_dict())
    sd["extra.weight"] = torch.ones(2)
    monkeypatch.setattr(model, "state_dict", lambda: sd)
    with pytest.raises(ValueError, match="key mismatch"):
        to_hf_model(model, TINY)


def test_greedy_generate_matches_ours() -> None:
    model = _model(seed=3)
    hf = to_hf_model(model, TINY)
    prompt = _ids(1, 4, seed=5)
    ours = generate(model, prompt, 12, TINY.max_seq_len, temperature=0.0, eos_id=EOS_ID)
    theirs = hf.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=12,
        do_sample=False,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )
    assert torch.equal(ours, theirs)


# tokenizer and generation config ---------------------------------------------------------------


def test_hf_tokenizer_matches_ours(tokenizer) -> None:  # type: ignore[no-untyped-def]
    hf = hf_tokenizer(TOKENIZER, 1024)
    for text in TEXTS:
        ids = hf(text)["input_ids"]
        assert ids == encode_document(tokenizer, text, append_eos=False)  # no BOS added
        assert hf.decode(ids) == text  # lossless, no space clean-up


def test_hf_tokenizer_special_tokens() -> None:
    hf = hf_tokenizer(TOKENIZER, 1024)
    assert hf.bos_token == hf.eos_token == EOS_TOKEN and hf.pad_token == PAD_TOKEN
    assert (hf.bos_token_id, hf.eos_token_id, hf.pad_token_id) == (0, 0, 1)
    assert hf.model_max_length == 1024 and hf.clean_up_tokenization_spaces is False
    assert len(hf) == VOCAB_SIZE


def test_generation_config_fields() -> None:
    g = generation_config(1024)
    assert g.do_sample is True and g.temperature == 0.8 and g.top_k == 50 and g.max_length == 1024
    assert (g.bos_token_id, g.eos_token_id, g.pad_token_id) == (0, 0, 1)


# export_checkpoint ---------------------------------------------------------------------------------


def _checkpoint(tmp_path: Path) -> tuple[Path, ModelConfig]:
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update({"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 32})
    raw["data"]["seq_len"] = 32
    config = config_from_dict(raw)
    model = _model(config.model, seed=1)
    path = checkpoint_path(tmp_path / "ckpt", 13)
    save_checkpoint(
        path, model, build_optimizer(model, config.train), 13, {}, config_to_dict(config)
    )
    return path, config.model


def test_export_checkpoint_folder_roundtrip(tmp_path: Path, tokenizer) -> None:  # type: ignore[no-untyped-def]
    path, mcfg = _checkpoint(tmp_path)
    out = tmp_path / "export"
    model, step = export_checkpoint(path, TOKENIZER, out)
    assert step == 13 and not model.training
    for name in ("config.json", "model.safetensors", "generation_config.json", "tokenizer.json"):
        assert (out / name).exists(), name
    with safe_open(out / "model.safetensors", "pt") as fh:
        keys = list(fh.keys())
        assert not any("lm_head" in k for k in keys) and len(keys) == len(model.state_dict())
        assert all(fh.get_tensor(k).dtype == torch.float32 for k in keys)
    hf = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    assert isinstance(hf, LlamaForCausalLM)
    assert hf.config.rms_norm_eps == 1e-5 and hf.config.rope_parameters["rope_theta"] == 10000.0
    assert (hf.config.bos_token_id, hf.config.eos_token_id, hf.config.pad_token_id) == (0, 0, 1)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    assert max_logit_diff(model, hf, _ids(2, 32, vocab=mcfg.vocab_size)) < PARITY_TOLERANCE
    hf_tok = AutoTokenizer.from_pretrained(out)
    assert hf_tok(TEXTS[1])["input_ids"] == encode_document(tokenizer, TEXTS[1], append_eos=False)
    assert hf.generation_config.eos_token_id == 0 and hf.generation_config.top_k == 50


def test_export_refuses_existing_dir(tmp_path: Path) -> None:
    path, _ = _checkpoint(tmp_path)
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(FileExistsError):
        export_checkpoint(path, TOKENIZER, out)
    assert list(out.iterdir()) == []
