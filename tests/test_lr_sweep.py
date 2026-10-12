"""Tests for ``scripts/lr_sweep.py``: sweep configs and the LR pick rule."""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

from syrinx.config import config_to_dict, load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "syrinx-50m.yaml"


@pytest.fixture(scope="module")
def sweep() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lr_sweep", ROOT / "scripts" / "lr_sweep.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(step: int, loss: float, grad_norm: float = 1.0) -> dict:
    return {"kind": "train", "step": step, "loss": loss, "grad_norm": grad_norm}


def _val(step: int, loss: float) -> dict:
    return {"kind": "val", "step": step, "val_loss": loss}


def _records(val: float, losses: list[float] | None = None) -> list[dict]:
    losses = losses or [9.0, 7.0, 6.0, 5.5]
    steps = range(10, 10 * len(losses) + 1, 10)
    trains = [_train(s, x) for s, x in zip(steps, losses, strict=True)]
    # val at other steps must be ignored: only the compare step counts
    return [*trains, _val(100, val + 1.0), _val(300, val), _val(400, val - 1.0)]


def test_with_lr_changes_only_lr_and_min_lr(sweep: ModuleType) -> None:
    base = load_config(BASE)
    new = sweep.with_lr(base, 1.2e-3)
    a, b = config_to_dict(base), config_to_dict(new)
    assert b["train"]["lr"] == 1.2e-3
    assert b["train"]["min_lr"] == pytest.approx(1.2e-4)
    a["train"].pop("lr"), a["train"].pop("min_lr"), b["train"].pop("lr"), b["train"].pop("min_lr")
    assert a == b


def test_with_lr_rejects_non_positive(sweep: ModuleType) -> None:
    with pytest.raises(ValueError, match="positive"):
        sweep.with_lr(load_config(BASE), 0.0)


def test_lr_tag(sweep: ModuleType) -> None:
    assert sweep.lr_tag(6e-4) == "lr-6.0e-04"
    assert sweep.lr_tag(1.2e-3) == "lr-1.2e-03"


def test_summarize_stable(sweep: ModuleType) -> None:
    s = sweep.summarize_run("r", 6e-4, _records(4.0), step=300, warmup_steps=0)
    assert (s.stable, s.val_loss, s.reason) == (True, 4.0, "ok")


def test_summarize_non_finite(sweep: ModuleType) -> None:
    s = sweep.summarize_run("r", 6e-4, _records(4.0, [9.0, math.nan]), 300, 0)
    assert not s.stable and "non-finite" in s.reason
    s = sweep.summarize_run("r", 6e-4, [_train(10, 9.0, math.inf), _val(300, 4.0)], 300, 0)
    assert not s.stable


def test_summarize_spike_after_warmup_only(sweep: ModuleType) -> None:
    spiky = [9.0, 5.0, 6.5]  # 1.5 above the running min at step 30
    assert not sweep.summarize_run("r", 1e-3, _records(4.0, spiky), 300, warmup_steps=0).stable
    assert sweep.summarize_run("r", 1e-3, _records(4.0, spiky), 300, warmup_steps=30).stable
    small = [9.0, 5.0, 5.9]  # 0.9 above: within SPIKE_NATS
    assert sweep.summarize_run("r", 1e-3, _records(4.0, small), 300, warmup_steps=0).stable


def test_summarize_missing_or_bad_val(sweep: ModuleType) -> None:
    s = sweep.summarize_run("r", 1e-3, [_train(10, 9.0)], 300, 0)
    assert not s.stable and "no val loss" in s.reason
    assert not sweep.summarize_run("r", 1e-3, _records(math.nan), 300, 0).stable


def test_choose_lowest_val(sweep: ModuleType) -> None:
    S = sweep.RunSummary
    runs = [
        S("a", 3e-4, 4.10, True, "ok"),
        S("b", 6e-4, 4.00, True, "ok"),
        S("c", 1.2e-3, 4.05, True, "ok"),
    ]
    assert sweep.choose_lr(runs).lr == 6e-4


def test_choose_prefers_lower_lr_within_tolerance(sweep: ModuleType) -> None:
    S = sweep.RunSummary
    runs = [
        S("a", 3e-4, 4.20, True, "ok"),
        S("b", 6e-4, 4.005, True, "ok"),
        S("c", 1.2e-3, 4.00, True, "ok"),
    ]
    assert sweep.choose_lr(runs).lr == 6e-4
    assert sweep.choose_lr(runs, tolerance=0.001).lr == 1.2e-3


def test_choose_skips_unstable(sweep: ModuleType) -> None:
    S = sweep.RunSummary
    runs = [S("a", 3e-4, 4.2, True, "ok"), S("c", 1.2e-3, 3.0, False, "loss spike")]
    assert sweep.choose_lr(runs).lr == 3e-4
    with pytest.raises(ValueError, match="no stable"):
        sweep.choose_lr([S("c", 1.2e-3, 3.0, False, "x")])


def _fake_run(sweep: ModuleType, root: Path, lr: float, val: float, losses=None) -> Path:  # type: ignore[no-untyped-def]
    run = root / sweep.lr_tag(lr)
    from syrinx.config import save_config

    save_config(sweep.with_lr(load_config(BASE), lr), run / "config.yaml")
    lines = [json.dumps(r) for r in _records(val, losses)]
    (run / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run


def test_main_make_configs(sweep: ModuleType, tmp_path: Path) -> None:
    assert (
        sweep.main(
            [
                "make-configs",
                "--base",
                str(BASE),
                "--lrs",
                "3e-4",
                "6e-4",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert load_config(tmp_path / "lr-3.0e-04.yaml").train.lr == 3e-4
    assert load_config(tmp_path / "lr-6.0e-04.yaml").train.lr == 6e-4
    # rerunnable: existing configs are kept, not an error
    assert (
        sweep.main(
            ["make-configs", "--base", str(BASE), "--lrs", "3e-4", "--out-dir", str(tmp_path)]
        )
        == 0
    )


def test_main_pick_writes_config_and_summary(sweep: ModuleType, tmp_path: Path) -> None:
    runs = [
        _fake_run(sweep, tmp_path, lr, v) for lr, v in [(3e-4, 4.3), (6e-4, 4.2), (1.2e-3, 4.1)]
    ]
    out, summary = tmp_path / "chosen.yaml", tmp_path / "summary.json"
    argv = ["pick", "--runs", *map(str, runs), "--step", "300", "--base", str(BASE)]
    assert sweep.main([*argv, "--config-out", str(out), "--summary-out", str(summary)]) == 0
    assert load_config(out).train.lr == 1.2e-3  # differs from the base config's 6e-4
    assert load_config(out).train.max_steps == load_config(BASE).train.max_steps
    data = json.loads(summary.read_text())
    assert data["chosen"]["lr"] == 1.2e-3 and [r["lr"] for r in data["runs"]] == [
        3e-4,
        6e-4,
        1.2e-3,
    ]


def test_main_pick_run_without_metrics_is_unstable(sweep: ModuleType, tmp_path: Path) -> None:
    good = _fake_run(sweep, tmp_path, 3e-4, 4.5)
    crashed = _fake_run(sweep, tmp_path, 1.2e-3, 4.0)
    (crashed / "metrics.jsonl").unlink()
    out, summary = tmp_path / "chosen.yaml", tmp_path / "summary.json"
    argv = ["pick", "--runs", str(good), str(crashed), "--step", "300", "--base", str(BASE)]
    assert sweep.main([*argv, "--config-out", str(out), "--summary-out", str(summary)]) == 0
    runs = json.loads(summary.read_text())["runs"]
    assert runs[1]["reason"] == "no metrics" and load_config(out).train.lr == 3e-4


def test_main_pick_fails_without_stable_run(sweep: ModuleType, tmp_path: Path) -> None:
    run = _fake_run(sweep, tmp_path, 1.2e-3, 4.0, [9.0, math.nan])
    out, summary = tmp_path / "chosen.yaml", tmp_path / "summary.json"
    argv = ["pick", "--runs", str(run), "--step", "300", "--base", str(BASE)]
    assert sweep.main([*argv, "--config-out", str(out), "--summary-out", str(summary)]) == 1
    assert not out.exists()
    assert json.loads(summary.read_text())["chosen"] is None
