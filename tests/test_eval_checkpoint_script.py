"""Tests for ``scripts/eval_checkpoint.py`` with a tiny model at the real vocabulary size."""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from syrinx.config import SyrinxConfig, config_from_dict, config_to_dict, load_config
from syrinx.data.loader import load_split, val_batches
from syrinx.data.shards import EOS_ID, shard_path, write_shard_file
from syrinx.evaluation import evaluate_split, token_byte_lengths
from syrinx.model import SyrinxLM
from syrinx.tokenizer import VOCAB_SIZE, encode_document, file_sha256, load_tokenizer
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
CPU = torch.device("cpu")
SEQ = 16
DOCS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside.",
    "The little dog was sad because he lost his ball. Then a café owner found it.",
    "Ben and Mia built a tall tower of blocks 😀 and it fell down with a crash!",
    "日本語のテキストも少し入れておきます。",
]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    path = ROOT / "scripts" / "eval_checkpoint.py"
    spec = importlib.util.spec_from_file_location("eval_checkpoint_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


def _val_dir(root: Path, tokenizer, docs: list[str]) -> Path:  # type: ignore[no-untyped-def]
    ids: list[int] = []
    for doc in docs:
        ids += encode_document(tokenizer, doc)  # EOS appended
    directory = root / "val"
    directory.mkdir(parents=True)
    write_shard_file(shard_path(directory, "val", 0), np.array(ids, dtype=np.uint16), len(docs), {})
    return directory


def _config(val_dir: Path, precision: str = "bf16-autocast") -> SyrinxConfig:
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update(
        {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": SEQ}
    )
    raw["data"].update({"seq_len": SEQ, "val_path": str(val_dir)})
    raw["train"].update({"micro_batch_size": 3, "precision": precision})
    return config_from_dict(raw)


@pytest.fixture
def setup(tmp_path: Path, tokenizer):  # type: ignore[no-untyped-def]
    val_dir = _val_dir(tmp_path, tokenizer, DOCS * 3)
    config = _config(val_dir)
    torch.manual_seed(0)
    model = SyrinxLM(config.model)
    path = checkpoint_path(tmp_path / "ckpt", 42)
    save_checkpoint(
        path, model, build_optimizer(model, config.train), 42, {}, config_to_dict(config)
    )
    return path, model, config, val_dir


def _run(script: ModuleType, capsys, argv: list[str]) -> dict:  # type: ignore[no-untyped-def]
    assert script.main(argv) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    return json.loads(lines[0])


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--checkpoint", "c.pt"])
    assert args.checkpoint == Path("c.pt") and args.tokenizer == script.DEFAULT_TOKENIZER
    assert (args.val_path, args.batch_size, args.precision, args.max_batches) == (None,) * 4
    assert args.device is None and args.out is None


def test_parse_args_rejects_unknown_precision(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.parse_args(["--checkpoint", "c", "--precision", "fp16"])


def test_main_whole_split(script: ModuleType, setup, tokenizer, capsys) -> None:  # type: ignore[no-untyped-def]
    path, model, _, val_dir = setup
    record = _run(script, capsys, ["--checkpoint", str(path), "--device", "cpu"])
    shard = load_split(val_dir, "val")[0]
    windows = (len(shard) - 1) // SEQ
    targets = np.concatenate(
        [np.asarray(shard[k * SEQ + 1 : k * SEQ + SEQ + 1]) for k in range(windows)]
    )
    lengths = token_byte_lengths(tokenizer)
    expected = evaluate_split(
        model, val_batches([shard], 3, SEQ, CPU), lengths, "bf16-autocast", EOS_ID
    )
    assert record["step"] == 42 and record["checkpoint"] == str(path)
    assert record["val_path"] == str(val_dir) and record["seq_len"] == SEQ
    assert (record["batch_size"], record["precision"], record["max_batches"]) == (
        3,
        "bf16-autocast",
        None,
    )
    assert record["windows"] == windows and record["tokens"] == windows * SEQ
    assert record["bytes"] == int(lengths[torch.from_numpy(targets.astype(np.int64))].sum())
    assert record["eos_targets"] == int((targets == EOS_ID).sum()) > 0
    assert record["loss"] == pytest.approx(expected.loss, rel=1e-6)
    assert record["perplexity"] == pytest.approx(math.exp(record["loss"]), rel=1e-9)
    assert record["bpb"] == pytest.approx(expected.bpb, rel=1e-6)
    assert record["seconds"] >= 0 and record["tokenizer_sha256"] == file_sha256(TOKENIZER)


def test_main_overrides(script: ModuleType, setup, tmp_path: Path, tokenizer, capsys) -> None:  # type: ignore[no-untyped-def]
    path, _, _, _ = setup
    other = _val_dir(tmp_path / "other", tokenizer, DOCS)
    argv = ["--checkpoint", str(path), "--device", "cpu", "--val-path", str(other)]
    record = _run(script, capsys, [*argv, "--batch-size", "2", "--precision", "fp32"])
    shard = load_split(other, "val")[0]
    assert record["val_path"] == str(other)
    assert (record["batch_size"], record["precision"]) == (2, "fp32")
    assert record["windows"] == (len(shard) - 1) // SEQ


def test_main_max_batches(script: ModuleType, setup, capsys) -> None:  # type: ignore[no-untyped-def]
    path, _, _, _ = setup
    record = _run(
        script, capsys, ["--checkpoint", str(path), "--device", "cpu", "--max-batches", "2"]
    )
    assert record["max_batches"] == 2 and record["windows"] == 6 and record["tokens"] == 6 * SEQ


def test_main_writes_out_and_never_overwrites(
    script: ModuleType, setup, tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    path, _, _, _ = setup
    out = tmp_path / "results" / "eval.json"
    argv = ["--checkpoint", str(path), "--device", "cpu", "--out", str(out)]
    printed = _run(script, capsys, argv)
    assert json.loads(out.read_text(encoding="utf-8")) == printed
    with pytest.raises(FileExistsError):
        script.main(argv)
