"""Tests for ``scripts/sample.py`` with a tiny model at the real vocabulary size."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from syrinx.config import ModelConfig, SyrinxConfig, config_from_dict, config_to_dict, load_config
from syrinx.generate import generate
from syrinx.model import SyrinxLM
from syrinx.tokenizer import EOS_TOKEN, VOCAB_SIZE, decode, encode_document, load_tokenizer
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sample_script", ROOT / "scripts" / "sample.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


def _config() -> SyrinxConfig:
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update({"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 32})
    raw["data"]["seq_len"] = 32
    return config_from_dict(raw)


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, SyrinxLM]:
    config = _config()
    torch.manual_seed(0)
    model = SyrinxLM(config.model)
    path = checkpoint_path(tmp_path / "ckpt", 7)
    save_checkpoint(
        path,
        model,
        build_optimizer(model, config.train),
        7,
        {"bit_generator": {}},
        config_to_dict(config),
    )
    return path, model


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--checkpoint", "c.pt"])
    assert args.checkpoint == Path("c.pt")
    assert args.tokenizer == Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
    assert args.prompt == list(script.DEFAULT_PROMPTS)
    assert (args.max_new_tokens, args.temperature, args.top_k, args.seed) == (200, 0.8, 50, 0)
    assert args.device is None and args.out is None


def test_parse_args_repeated_prompts(script: ModuleType) -> None:
    args = script.parse_args(["--checkpoint", "c", "--prompt", "a", "--prompt", "b c"])
    assert args.prompt == ["a", "b c"]


def test_load_model_restores_weights(script: ModuleType, checkpoint: tuple[Path, SyrinxLM]) -> None:
    path, original = checkpoint
    model, config, step = script.load_model(path, CPU)
    assert step == 7 and config == _config() and not model.training
    for (name, a), b in zip(
        original.state_dict().items(), model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b), name


def test_sample_texts_matches_generate(script: ModuleType, tokenizer, checkpoint) -> None:  # type: ignore[no-untyped-def]
    """Prompt is normalized, preceded by EOS, sampled with generator seed + i, cut at EOS."""
    path, _ = checkpoint
    model, _, _ = script.load_model(path, CPU)
    prompts = ["Once upon a time", "Tom\r\nhad a cat."]
    texts = script.sample_texts(model, tokenizer, prompts, 12, 32, 1.0, 20, 5, CPU)
    eos = tokenizer.token_to_id(EOS_TOKEN)
    for i, (prompt, text) in enumerate(zip(prompts, texts, strict=True)):
        ids = [eos] + encode_document(tokenizer, prompt.replace("\r\n", "\n"), append_eos=False)
        out = generate(
            model, torch.tensor([ids]), 12, 32, 1.0, 20, eos, torch.Generator().manual_seed(5 + i)
        )
        new = out[0, len(ids) :].tolist()
        if eos in new:
            new = new[: new.index(eos)]
        assert text == decode(tokenizer, ids[1:] + new)
        assert EOS_TOKEN not in text


def test_sample_texts_starts_with_prompt(script: ModuleType, tokenizer, checkpoint) -> None:  # type: ignore[no-untyped-def]
    model, _, _ = script.load_model(checkpoint[0], CPU)
    texts = script.sample_texts(model, tokenizer, ["Lily and Ben"], 5, 32, 0.0, None, 0, CPU)
    assert texts[0].startswith("Lily and Ben")


def test_sample_texts_seeded(script: ModuleType, tokenizer, checkpoint) -> None:  # type: ignore[no-untyped-def]
    model, _, _ = script.load_model(checkpoint[0], CPU)
    a = script.sample_texts(model, tokenizer, ["Hi"], 10, 32, 1.0, None, 3, CPU)
    b = script.sample_texts(model, tokenizer, ["Hi"], 10, 32, 1.0, None, 3, CPU)
    assert a == b


def test_main_writes_and_prints(script: ModuleType, checkpoint, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "s" / "samples.jsonl"
    argv = [
        "--checkpoint",
        str(checkpoint[0]),
        "--device",
        "cpu",
        "--prompt",
        "Once",
        "--prompt",
        "Then",
        "--max-new-tokens",
        "4",
        "--seed",
        "10",
        "--out",
        str(out),
    ]
    assert script.main(argv) == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    printed = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert written == printed and len(written) == 2
    assert [r["seed"] for r in written] == [10, 11]
    assert [r["prompt"] for r in written] == ["Once", "Then"]
    assert {r["step"] for r in written} == {7} and {r["top_k"] for r in written} == {50}
    assert all(r["text"].startswith(r["prompt"]) for r in written)
    with pytest.raises(FileExistsError):
        script.main(argv)


def test_tiny_model_is_real_vocab(tokenizer) -> None:  # type: ignore[no-untyped-def]
    assert _config().model.vocab_size == tokenizer.get_vocab_size() == VOCAB_SIZE
    assert isinstance(_config().model, ModelConfig)


def test_sample_texts_cuts_at_first_eos(
    script: ModuleType,
    tokenizer,
    checkpoint,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    """Text ends before the first generated EOS; nothing after it is kept."""
    model, _, _ = script.load_model(checkpoint[0], CPU)
    eos = tokenizer.token_to_id(EOS_TOKEN)
    a, b, c = 300, 301, 302  # ordinary (non-special) token ids

    def fake_generate(model, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
        tail = torch.tensor([[a, eos, b, c, eos]])
        return torch.cat([prompt, tail], dim=1)

    monkeypatch.setattr(script, "generate", fake_generate)
    texts = script.sample_texts(model, tokenizer, ["Once"], 5, 32, 1.0, None, 0, CPU)
    prompt_ids = encode_document(tokenizer, "Once", append_eos=False)
    assert texts == [decode(tokenizer, prompt_ids + [a])]
