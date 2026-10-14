"""Tests for ``scripts/check_contamination.py`` on tiny local shards and benchmark files."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from syrinx.data.shards import shard_path, write_shard_file
from syrinx.tokenizer import VOCAB_SIZE, encode_document, file_sha256, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
CTX = "A young girl stands at the edge of the diving board and looks down at the water below her"
LAMBADA = "He opened the old wooden door slowly and stepped into the dark room where his grandfather kept the maps"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    path = ROOT / "scripts" / "check_contamination.py"
    spec = importlib.util.spec_from_file_location("check_contamination_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


def _shard(directory: Path, tokenizer, docs: list[str]) -> Path:  # type: ignore[no-untyped-def]
    directory.mkdir(parents=True)
    ids: list[int] = []
    for doc in docs:
        ids += encode_document(tokenizer, doc)
    path = shard_path(directory, "train", 0)
    write_shard_file(path, np.array(ids, dtype=np.uint16), len(docs), {})
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path: Path, tokenizer):  # type: ignore[no-untyped-def]
    a = tmp_path / "a"
    b = tmp_path / "b"
    _shard(a, tokenizer, ["some clean story", f"Copied: {CTX.lower()}, then it continues.", "end"])
    _shard(b, tokenizer, ["unrelated text here", f"Quote -- {LAMBADA}!"])
    hella = _jsonl(
        tmp_path / "hellaswag.jsonl",
        [
            {"ctx": CTX, "endings": ["She jumps.", "She waves."], "label": "0"},
            {
                "ctx": "A completely different context that never appears anywhere in the shards at all",
                "endings": ["x"],
            },
            {"ctx": "Too short", "endings": ["a", "b"]},
        ],
    )
    lam = _jsonl(tmp_path / "lambada.jsonl", [{"text": "clean passage " * 10}, {"text": LAMBADA}])
    argv = ["--benchmark", f"hellaswag={hella}", "--benchmark", f"lambada={lam}"]
    argv += ["--shard-dir", str(a), "--shard-dir", str(b), "--workers", "2"]
    return argv, a, b, hella, lam


def _record(script: ModuleType, capsys, argv: list[str]) -> dict:  # type: ignore[no-untyped-def]
    assert script.main(argv) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    return json.loads(lines[0])


def test_parse_benchmark(script: ModuleType) -> None:
    assert script.parse_benchmark("piqa=data/p.jsonl") == ("piqa", Path("data/p.jsonl"))
    for bad in ("piqa", "=x.jsonl", "piqa="):
        with pytest.raises(argparse.ArgumentTypeError):
            script.parse_benchmark(bad)


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--benchmark", "lambada=l.jsonl"])
    assert args.benchmark == [("lambada", Path("l.jsonl"))]
    assert (
        args.shard_dir == list(script.DEFAULT_SHARD_DIRS)
        and args.tokenizer == script.DEFAULT_TOKENIZER
    )
    assert (args.n, args.workers, args.max_docs_per_shard, args.out) == (
        13,
        os.cpu_count(),
        None,
        None,
    )
    with pytest.raises(SystemExit):
        script.parse_args([])


def test_load_benchmark_ids_and_text(script: ModuleType, tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "arc.jsonl",
        [
            {"id": "Q1", "question": "Why?", "choices": {"text": ["A", "B"]}},
            {"question": "How?", "choices": {"text": ["C"]}},
        ],
    )
    assert script.load_benchmark("arc_easy", path) == {
        "arc_easy/Q1": "Why? A B",
        "arc_easy/1": "How? C",
    }


def test_main_finds_planted_contamination(script: ModuleType, setup, capsys) -> None:  # type: ignore[no-untyped-def]
    argv, a, b, hella, _ = setup
    record = _record(script, capsys, argv)
    assert record["n"] == 13 and record["normalization"] == "nfkc-lower-alnum-words"
    assert (
        record["tokenizer_sha256"] == file_sha256(TOKENIZER)
        and record["max_docs_per_shard"] is None
    )
    assert [s["shard"] for s in record["shards"]] == [
        str(a / "train_0000.bin"),
        str(b / "train_0000.bin"),
    ]
    assert [s["documents"] for s in record["shards"]] == [3, 2] and record["documents"] == 5
    assert (
        record["words"] == sum(s["words"] for s in record["shards"]) > 0 and record["seconds"] >= 0
    )
    h = record["benchmarks"]["hellaswag"]
    assert (h["examples"], h["unchecked"], h["checked"], h["contaminated"]) == (3, 1, 2, 1)
    assert h["contaminated_pct"] == pytest.approx(50.0) and h["contaminated_ids"] == ["hellaswag/0"]
    assert h["path"] == str(hella)
    assert h["matches"] == [{"id": "hellaswag/0", "ngram": " ".join(CTX.lower().split()[:13])}]
    lam_r = record["benchmarks"]["lambada"]
    assert (lam_r["examples"], lam_r["unchecked"], lam_r["checked"], lam_r["contaminated"]) == (
        2,
        0,
        2,
        1,
    )
    assert lam_r["contaminated_ids"] == ["lambada/1"] and lam_r[
        "contaminated_pct"
    ] == pytest.approx(50.0)


def test_main_one_worker_same_result(script: ModuleType, setup, capsys) -> None:  # type: ignore[no-untyped-def]
    argv, *_ = setup
    two = _record(script, capsys, argv)
    one = _record(script, capsys, [*argv[:-1], "1"])
    assert two["benchmarks"] == one["benchmarks"] and two["documents"] == one["documents"]


def test_main_max_docs_per_shard(script: ModuleType, setup, capsys) -> None:  # type: ignore[no-untyped-def]
    argv, *_ = setup
    record = _record(script, capsys, [*argv, "--max-docs-per-shard", "1"])
    assert record["max_docs_per_shard"] == 1 and record["documents"] == 2
    assert all(b["contaminated"] == 0 for b in record["benchmarks"].values())


def test_main_all_unchecked_pct_is_none(script: ModuleType, setup, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _, a, *_ = setup
    tiny = _jsonl(tmp_path / "tiny.jsonl", [{"text": "short"}])
    record = _record(
        script, capsys, ["--benchmark", f"lambada={tiny}", "--shard-dir", str(a), "--workers", "1"]
    )
    r = record["benchmarks"]["lambada"]
    assert (r["checked"], r["unchecked"], r["contaminated_pct"]) == (0, 1, None)


def test_main_writes_out_and_never_overwrites(
    script: ModuleType, setup, tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    argv, *_ = setup
    out = tmp_path / "res" / "contamination.json"
    printed = _record(script, capsys, [*argv, "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == printed
    with pytest.raises(FileExistsError):
        script.main([*argv, "--out", str(out)])


def test_main_missing_shards_raises(script: ModuleType, setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    argv, *_ = setup
    with pytest.raises(FileNotFoundError, match="no train"):
        script.main([argv[0], argv[1], "--shard-dir", str(tmp_path / "empty"), "--workers", "1"])


def test_main_contaminated_ids_sorted(script: ModuleType, setup, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _, a, b, *_ = setup
    rows = [{"text": LAMBADA}, {"text": "never seen " * 10}, {"text": CTX}]
    lam = _jsonl(tmp_path / "two.jsonl", rows)
    argv = [
        "--benchmark",
        f"lambada={lam}",
        "--shard-dir",
        str(a),
        "--shard-dir",
        str(b),
        "--workers",
        "1",
    ]
    record = _record(script, capsys, argv)
    r = record["benchmarks"]["lambada"]
    assert r["contaminated_ids"] == ["lambada/0", "lambada/2"] and r["contaminated"] == 2
    assert [m["id"] for m in r["matches"]] == ["lambada/0", "lambada/2"]
