"""Tests for ``scripts/export_hf.py`` with a tiny model at the real vocabulary size."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from syrinx.config import config_from_dict, config_to_dict, load_config
from syrinx.model import SyrinxLM
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_hf_script", ROOT / "scripts" / "export_hf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update({"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 32})
    raw["data"]["seq_len"] = 32
    config = config_from_dict(raw)
    torch.manual_seed(0)
    model = SyrinxLM(config.model)
    path = checkpoint_path(tmp_path / "ckpt", 21)
    save_checkpoint(
        path, model, build_optimizer(model, config.train), 21, {}, config_to_dict(config)
    )
    return path


def _json_line(capsys) -> dict:  # type: ignore[no-untyped-def]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    return json.loads(lines[0])


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--checkpoint", "c.pt", "--out-dir", "o"])
    assert (args.checkpoint, args.out_dir) == (Path("c.pt"), Path("o"))
    assert args.tokenizer == script.DEFAULT_TOKENIZER
    assert (args.parity_batch, args.parity_seq_len, args.seed) == (2, None, 0)


def test_parse_args_requires_out_dir(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.parse_args(["--checkpoint", "c.pt"])


def test_main_exports_and_passes(script: ModuleType, ckpt: Path, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "export"
    assert script.main(["--checkpoint", str(ckpt), "--out-dir", str(out)]) == 0
    record = _json_line(capsys)
    assert json.loads((out / "parity.json").read_text(encoding="utf-8")) == record
    assert (
        record["step"] == 21 and record["checkpoint"] == str(ckpt) and record["out_dir"] == str(out)
    )
    assert (record["batch"], record["seq_len"]) == (2, 32)  # seq_len defaults to max_seq_len
    assert 0.0 <= record["max_abs_logit_diff"] < record["tolerance"] == 1e-4
    assert record["logits_ok"] and record["tokenizer_ok"] and record["passed"]
    assert record["transformers_version"]
    assert (out / "model.safetensors").exists() and (out / "tokenizer.json").exists()


def test_main_parity_options(script: ModuleType, ckpt: Path, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    argv = ["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "e")]
    assert script.main([*argv, "--parity-batch", "3", "--parity-seq-len", "8"]) == 0
    record = _json_line(capsys)
    assert (record["batch"], record["seq_len"]) == (3, 8)


def test_main_fails_on_logit_mismatch(
    script: ModuleType, ckpt: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(script, "max_logit_diff", lambda *a, **k: 1.0)
    assert script.main(["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "e")]) == 1
    record = _json_line(capsys)
    assert record["max_abs_logit_diff"] == 1.0
    assert not record["logits_ok"] and record["tokenizer_ok"] and not record["passed"]


def test_main_fails_on_tokenizer_mismatch(
    script: ModuleType, ckpt: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(script, "encode_document", lambda *a, **k: [5, 6, 7])
    assert script.main(["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "e")]) == 1
    record = _json_line(capsys)
    assert record["logits_ok"] and not record["tokenizer_ok"] and not record["passed"]


def test_main_refuses_existing_out_dir(script: ModuleType, ckpt: Path, tmp_path: Path) -> None:
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(FileExistsError):
        script.main(["--checkpoint", str(ckpt), "--out-dir", str(out)])
