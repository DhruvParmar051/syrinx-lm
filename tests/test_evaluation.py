"""Tests for ``syrinx.evaluation``: byte table, full-split sums, bits per byte."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from syrinx.config import ModelConfig, config_from_dict, config_to_dict, load_config
from syrinx.evaluation import EvalResult, evaluate_split, load_model, token_byte_lengths
from syrinx.model import SyrinxLM
from syrinx.tokenizer import EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE, encode_document, load_tokenizer
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer
from syrinx.training.step import evaluate

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
V = 64
TINY = ModelConfig(
    vocab_size=V,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_ff=96,
    max_seq_len=16,
    rope_theta=10000.0,
    norm_eps=1e-5,
)
TEXTS = [
    "Once upon a time, there was a little girl named Lily.",
    "café naïve résumé — “quotes” and 😀 emoji",
    "日本語のテキストと中文",
    "digits 1234567890 and tabs\tand\r\nnewlines\n\n",
    "  leading and trailing spaces  ",
]


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


class TableModel(nn.Module):
    """``logits = table[x]``; records grad mode and training flag seen in forward."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table.clone())
        self.seen: list[tuple[bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append((torch.is_grad_enabled(), self.training))
        return self.table[x]


def _batch(b: int, t: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randint(0, V, (b, t + 1), generator=gen)
    return w[:, :-1], w[:, 1:]


def _ones(n: int = V) -> torch.Tensor:
    return torch.ones(n, dtype=torch.int64)


# token_byte_lengths -----------------------------------------------------------------------


def test_byte_lengths_shape_and_dtype(tokenizer) -> None:  # type: ignore[no-untyped-def]
    lengths = token_byte_lengths(tokenizer)
    assert lengths.shape == (VOCAB_SIZE,) and lengths.dtype == torch.int64
    assert lengths.device.type == "cpu"


@pytest.mark.parametrize("text", TEXTS)
def test_byte_lengths_match_utf8(tokenizer, text: str) -> None:  # type: ignore[no-untyped-def]
    ids = encode_document(tokenizer, text, append_eos=False)
    assert int(token_byte_lengths(tokenizer)[ids].sum()) == len(text.encode("utf-8"))


def test_special_tokens_have_zero_bytes(tokenizer) -> None:  # type: ignore[no-untyped-def]
    lengths = token_byte_lengths(tokenizer)
    for special in (EOS_TOKEN, PAD_TOKEN):
        assert int(lengths[tokenizer.token_to_id(special)]) == 0
    assert int((lengths == 0).sum()) == 2  # nothing else is empty
    assert int(lengths.min()) == 0 and int(lengths[2:].min()) >= 1


# EvalResult ---------------------------------------------------------------------------------


def test_eval_result_properties() -> None:
    r = EvalResult(loss_sum=6.0, tokens=3, bytes=4, eos_targets=0, windows=1)
    assert r.loss == pytest.approx(2.0)
    assert r.perplexity == pytest.approx(math.exp(2.0))
    assert r.bpb == pytest.approx(6.0 / (math.log(2) * 4))


# evaluate_split -------------------------------------------------------------------------------


def test_uniform_model_exact_loss_and_bpb() -> None:
    model = TableModel(torch.zeros(V, V))
    lengths = torch.full((V,), 2, dtype=torch.int64)
    r = evaluate_split(model, [_batch(3)], lengths, "fp32", eos_id=0)
    assert r.loss == pytest.approx(math.log(V), rel=1e-6)
    assert r.bpb == pytest.approx(math.log(V) / (math.log(2) * 2), rel=1e-6)  # = 3.0
    assert r.bpb == pytest.approx(3.0, rel=1e-6)


def test_sums_match_per_token_reference() -> None:
    torch.manual_seed(0)
    model = TableModel(torch.randn(V, V))
    lengths = torch.randint(1, 5, (V,))
    batches = [_batch(4, seed=1), _batch(1, seed=2)]  # uneven: last batch smaller
    r = evaluate_split(model, batches, lengths, "fp32", eos_id=0)
    nll = torch.cat(
        [
            F.cross_entropy(model.table.detach()[x].reshape(-1, V), y.reshape(-1), reduction="none")
            for x, y in batches
        ]
    )
    assert r.tokens == 5 * 8 and r.windows == 5
    assert r.loss_sum == pytest.approx(float(nll.double().sum()), rel=1e-6)
    assert r.loss == pytest.approx(float(nll.double().mean()), rel=1e-6)  # token-weighted
    assert r.bytes == int(sum(lengths[y].sum() for _, y in batches))


def test_token_weighting_not_mean_of_batch_means() -> None:
    torch.manual_seed(0)
    model = TableModel(torch.randn(V, V))
    big, small = _batch(6, seed=3), _batch(1, seed=4)
    r = evaluate_split(model, [big, small], _ones(), "fp32", eos_id=0)
    means = [
        float(F.cross_entropy(model.table.detach()[x].reshape(-1, V), y.reshape(-1)))
        for x, y in (big, small)
    ]
    assert r.loss != pytest.approx(sum(means) / 2, rel=1e-4)
    assert r.loss == pytest.approx((means[0] * 48 + means[1] * 8) / 56, rel=1e-6)


def test_loss_computed_in_float32_from_bf16_logits() -> None:
    """bf16 logits (as under autocast) must be upcast before the loss: a bf16 sum is coarse."""
    torch.manual_seed(0)
    model = TableModel(torch.randn(V, V).to(torch.bfloat16))
    batch = _batch(8, seed=8)
    r = evaluate_split(model, [batch], _ones(), "fp32", eos_id=0)
    x, y = batch
    logits = model.table.detach()[x].float().reshape(-1, V)
    expected = F.cross_entropy(logits, y.reshape(-1), reduction="sum")
    assert r.loss_sum == pytest.approx(float(expected), rel=1e-6)


def test_bytes_counted_on_targets_not_inputs() -> None:
    model = TableModel(torch.zeros(V, V))
    lengths = _ones()
    lengths[5], lengths[7] = 9, 3
    x = torch.full((2, 4), 5)
    y = torch.full((2, 4), 7)
    r = evaluate_split(model, [(x, y)], lengths, "fp32", eos_id=0)
    assert r.bytes == 8 * 3


def test_eos_in_loss_with_zero_bytes() -> None:
    torch.manual_seed(0)
    model = TableModel(torch.randn(V, V))
    lengths = _ones()
    lengths[0] = 0
    x = torch.tensor([[3, 4, 5, 6]])
    y = torch.tensor([[4, 0, 6, 0]])  # two EOS targets
    r = evaluate_split(model, [(x, y)], lengths, "fp32", eos_id=0)
    assert (r.tokens, r.bytes, r.eos_targets) == (4, 2, 2)
    full = F.cross_entropy(model.table.detach()[x].reshape(-1, V), y.reshape(-1), reduction="sum")
    assert r.loss_sum == pytest.approx(float(full), rel=1e-6)  # EOS positions included


def test_matches_training_evaluate() -> None:
    torch.manual_seed(0)
    model = SyrinxLM(TINY)
    batches = [_batch(3, seed=5), _batch(2, seed=6)]
    r = evaluate_split(model, batches, _ones(), "fp32", eos_id=0)
    assert r.loss == pytest.approx(evaluate(model, batches, "fp32"), rel=1e-5)


def test_no_grad_eval_mode_and_restores_mode() -> None:
    model = TableModel(torch.zeros(V, V))
    model.train()
    evaluate_split(model, [_batch(1)], _ones(), "fp32", eos_id=0)
    assert model.seen == [(False, False)]
    assert model.training
    model.eval()
    evaluate_split(model, [_batch(1)], _ones(), "fp32", eos_id=0)
    assert not model.training


def test_restores_mode_on_error() -> None:
    class Boom(TableModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    model = Boom(torch.zeros(V, V))
    model.train()
    with pytest.raises(RuntimeError, match="boom"):
        evaluate_split(model, [_batch(1)], _ones(), "fp32", eos_id=0)
    assert model.training


def test_empty_batches_raise() -> None:
    with pytest.raises(ValueError, match="no batches"):
        evaluate_split(TableModel(torch.zeros(V, V)), [], _ones(), "fp32", eos_id=0)


def test_zero_bytes_raise() -> None:
    zeros = torch.zeros(V, dtype=torch.int64)
    with pytest.raises(ValueError, match="0 bytes"):
        evaluate_split(TableModel(torch.zeros(V, V)), [_batch(1)], zeros, "fp32", eos_id=0)


def test_bf16_autocast_close_to_fp32(each_device: torch.device) -> None:
    torch.manual_seed(0)
    model = SyrinxLM(TINY).to(each_device)
    batches = [tuple(t.to(each_device) for t in _batch(4, seed=7))]
    lengths = _ones()  # CPU table, device batches: must still work
    fp32 = evaluate_split(model, batches, lengths, "fp32", eos_id=0)
    bf16 = evaluate_split(model, batches, lengths, "bf16-autocast", eos_id=0)
    assert bf16.tokens == fp32.tokens and bf16.bytes == fp32.bytes
    assert bf16.loss == pytest.approx(fp32.loss, rel=2e-2)
    assert math.isfinite(bf16.loss)


# load_model -----------------------------------------------------------------------------------


def test_load_model_roundtrip(tmp_path: Path) -> None:
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update({"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 32})
    raw["data"]["seq_len"] = 32
    config = config_from_dict(raw)
    torch.manual_seed(0)
    model = SyrinxLM(config.model)
    path = checkpoint_path(tmp_path, 11)
    save_checkpoint(
        path, model, build_optimizer(model, config.train), 11, {}, config_to_dict(config)
    )
    loaded, loaded_config, step = load_model(path, torch.device("cpu"))
    assert step == 11 and loaded_config == config and not loaded.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
