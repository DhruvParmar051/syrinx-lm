"""Tests for scripts/check_dryrun.py (CUDA dry-run pass/fail checks)."""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

from syrinx.utils.metrics import JsonlWriter

ROOT = Path(__file__).resolve().parents[1]
LN_V = math.log(32768)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_dryrun_script", ROOT / "scripts" / "check_dryrun.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(path: Path, losses: dict[int, float], grad_norm: float = 1.0) -> Path:
    with JsonlWriter(path) as writer:
        for step, loss in losses.items():
            writer.write(
                {"kind": "train", "step": step, "loss": loss, "grad_norm": grad_norm,
                 "mem_peak": 2**30 * step}
            )  # fmt: skip
        writer.write({"kind": "val", "step": max(losses), "val_loss": 1.0})
    return path


def test_start_passes_on_a_healthy_run(script: ModuleType, tmp_path: Path) -> None:
    result = script.check_start(_run(tmp_path / "m.jsonl", {1: LN_V + 0.1, 30: 8.0}), 32768)
    assert result["passed"] and result["first_step"] == 1 and result["max_mem_peak_gib"] == 30


@pytest.mark.parametrize(
    ("losses", "grad_norm", "failed"),
    [
        ({1: LN_V + 0.2, 30: 8.0}, 1.0, "initial_loss_ok"),  # bad init or wrong vocab
        ({1: LN_V, 30: LN_V + 0.01}, 1.0, "loss_falls"),  # not learning
        ({1: LN_V, 30: 8.0}, float("nan"), "all_finite"),  # NaN grad norm
    ],
)
def test_start_fails_each_check(
    script: ModuleType, tmp_path: Path, losses: dict[int, float], grad_norm: float, failed: str
) -> None:
    result = script.check_start(_run(tmp_path / "m.jsonl", losses, grad_norm), 32768)
    assert not result[failed] and not result["passed"]


def test_resume_compares_against_the_straight_run(script: ModuleType, tmp_path: Path) -> None:
    straight = _run(tmp_path / "a.jsonl", {10: 9.0, 20: 8.0, 30: 7.0})
    resumed = _run(tmp_path / "b.jsonl", {10: 9.0, 20: 8.0002, 30: 7.0004})
    repeat = _run(tmp_path / "c.jsonl", {10: 9.0001, 20: 8.0, 30: 7.0})
    result = script.check_resume(straight, resumed, repeat)
    assert result["passed"]
    assert result["max_abs_loss_diff"] == pytest.approx(4e-4)
    assert result["noise_floor"] == pytest.approx(1e-4)


def test_resume_fails_beyond_tolerance(script: ModuleType, tmp_path: Path) -> None:
    straight = _run(tmp_path / "a.jsonl", {10: 9.0, 20: 8.0})
    resumed = _run(tmp_path / "b.jsonl", {10: 9.0, 20: 8.05})  # e.g. lost AdamW state
    assert not script.check_resume(straight, resumed, None)["passed"]


def test_resumed_run_keeps_the_last_record_of_a_repeated_step(
    script: ModuleType, tmp_path: Path
) -> None:
    """The resumed run logs steps after its checkpoint twice; the survivor is compared."""
    straight = _run(tmp_path / "a.jsonl", {10: 9.0, 20: 8.0})
    path = tmp_path / "b.jsonl"
    with JsonlWriter(path) as writer:
        for step, loss in ((10, 9.0), (20, 5.0), (20, 8.0)):  # 20 run twice
            writer.write({"kind": "train", "step": step, "loss": loss, "grad_norm": 1.0})
    assert script.check_resume(straight, path, None)["max_abs_loss_diff"] == 0.0


def test_resume_refuses_runs_with_different_steps(script: ModuleType, tmp_path: Path) -> None:
    straight = _run(tmp_path / "a.jsonl", {10: 9.0, 20: 8.0})
    resumed = _run(tmp_path / "b.jsonl", {10: 9.0})
    with pytest.raises(ValueError, match="steps differ"):
        script.check_resume(straight, resumed, None)


def test_main_exit_code_and_json(script: ModuleType, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    good = _run(tmp_path / "good.jsonl", {1: LN_V, 30: 8.0})
    assert script.main(["start", str(good)]) == 0
    assert json.loads(capsys.readouterr().out.strip())["passed"] is True
    bad = _run(tmp_path / "bad.jsonl", {1: 5.0, 30: 4.0})
    assert script.main(["start", str(bad)]) == 1
