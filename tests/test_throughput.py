"""Tests for ``syrinx.utils.throughput`` and ``scripts/throughput_report.py``."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from syrinx.utils.metrics import JsonlWriter
from syrinx.utils.throughput import (
    estimated_seconds,
    gate_decision,
    last_window,
    throughput_report,
    train_records,
)

ROOT = Path(__file__).resolve().parents[1]


def rec(step: int, step_time: float, tok_s: float) -> dict[str, Any]:
    return {"kind": "train", "step": step, "step_time_s": step_time, "tokens_per_sec": tok_s}


def run(n: int, every: int = 10, step_time: float = 10.0, tok_s: float = 5000.0) -> list[dict]:
    """``n`` logged records, one every ``every`` steps."""
    return [rec(every * (i + 1), step_time, tok_s) for i in range(n)]


def test_train_records_sorted_last_wins_and_skips_val() -> None:
    records = [
        rec(20, 1.0, 100.0),
        {"kind": "val", "step": 20, "val_loss": 3.0},
        rec(10, 1.0, 100.0),
        rec(20, 1.0, 200.0),  # replayed after a resume: this one counts
    ]
    out = train_records(records)
    assert [r["step"] for r in out] == [10, 20]
    assert out[1]["tokens_per_sec"] == 200.0


def test_estimated_seconds_uses_step_gaps() -> None:
    train = [rec(10, 2.0, 1.0), rec(20, 3.0, 1.0), rec(25, 4.0, 1.0)]
    assert estimated_seconds(train) == [20.0, 30.0, 20.0]


def test_last_window_is_trailing_and_long_enough() -> None:
    train = run(20)  # each record stands for 100 s
    window = last_window(train, 350.0)
    assert [r["step"] for r in window] == [170, 180, 190, 200]
    assert len(last_window(train, 400.0)) == 4
    assert len(last_window(train, 400.1)) == 5


def test_last_window_rejects_short_runs_and_bad_windows() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        last_window(run(5), 501.0)
    with pytest.raises(ValueError, match="positive"):
        last_window(run(5), 0.0)


def test_report_median_excludes_warmup() -> None:
    warmup = [rec(10 * (i + 1), 10.0, 1000.0) for i in range(10)]  # slow start, 1000 s
    steady = [rec(100 + 10 * (i + 1), 10.0, 6000.0 + i) for i in range(10)]  # 1000 s
    # One fast outlier: the median ignores it, a mean would not.
    steady[-1]["tokens_per_sec"] = 9000.0
    report = throughput_report(warmup + steady, window_seconds=900.0, proxy_tok_s=6000.0)
    assert report.first_step == 120 and report.last_step == 200 and report.samples == 9
    assert report.median_tok_s == 6005.0
    assert report.min_tok_s == 6001.0 and report.max_tok_s == 9000.0
    assert report.covered_seconds == 900.0 and report.run_seconds == 2000.0
    assert report.ratio_to_proxy == pytest.approx(6005.0 / 6000.0)
    assert not report.investigate and report.decision == "full-budget"


def test_report_flags_shortfall() -> None:
    assert throughput_report(run(20, tok_s=4600.0), 900.0, 6674.0).investigate
    assert not throughput_report(run(20, tok_s=4700.0), 900.0, 6674.0).investigate


def test_report_needs_train_records() -> None:
    with pytest.raises(ValueError, match="no train records"):
        throughput_report([{"kind": "val", "step": 10, "val_loss": 1.0}])


@pytest.mark.parametrize(
    ("tok_s", "decision"),
    [
        (4000.0, "full-budget"),
        (3999.9, "choose"),
        (2000.0, "choose"),
        (1999.9, "optimize-first"),
    ],
)
def test_gate_decision_boundaries(tok_s: float, decision: str) -> None:
    assert gate_decision(tok_s) == decision


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "throughput_report_script", ROOT / "scripts" / "throughput_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_prints_and_writes_report(script: ModuleType, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    metrics, out = tmp_path / "metrics.jsonl", tmp_path / "gate.json"
    with JsonlWriter(metrics) as writer:
        for record in run(20, tok_s=3000.0):
            writer.write(record)
    argv = [str(metrics), "--window-minutes", "5", "--proxy-tok-s", "6000", "--out", str(out)]
    assert script.main(argv) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("{"))
    result = json.loads(line)
    assert result == json.loads(out.read_text())
    assert result["metrics"] == str(metrics) and result["window_seconds"] == 300.0
    assert result["median_tok_s"] == 3000.0 and result["decision"] == "choose"
    assert result["investigate"] is True
    with pytest.raises(FileExistsError):
        script.main(argv)
