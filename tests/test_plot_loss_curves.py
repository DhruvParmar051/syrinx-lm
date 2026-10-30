"""Tests for scripts/plot_loss_curves.py (loading, page layout; the Chrome render is optional)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from syrinx.utils.metrics import JsonlWriter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "plot_loss_curves_script", ROOT / "scripts" / "plot_loss_curves.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, records: list[dict]) -> Path:
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write(record)
    return path


def _train(step: int, loss: float, tps: int = 1000) -> dict:
    return {"kind": "train", "step": step, "loss": loss, "tokens_seen": step * tps}


def test_resumed_steps_keep_the_last_record(script: ModuleType, tmp_path: Path) -> None:
    """A resume re-logs steps; the curve keeps the later record and stays in step order."""
    records = [
        _train(10, 5.0), _train(20, 4.0), {"kind": "val", "step": 20, "val_loss": 4.5},
        _train(30, 9.9),  # stopped here, resumed from step 20
        _train(30, 3.0), {"kind": "val", "step": 40, "val_loss": 3.5}, _train(40, 2.0),
    ]  # fmt: skip
    curves = script.load_curves("run", _write(tmp_path / "m.jsonl", records))
    assert curves.train == [(0.01, 5.0), (0.02, 4.0), (0.03, 3.0), (0.04, 2.0)]
    assert curves.val == [(0.02, 4.5), (0.04, 3.5)]


def test_missing_val_records_is_an_error(script: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs train and val"):
        script.load_curves("run", _write(tmp_path / "m.jsonl", [_train(10, 5.0)]))


@pytest.mark.parametrize(
    ("x_max", "expected"),
    [(294.9, (50, 300)), (1000.08, (200, 1000)), (2500.6, (500, 2500)), (301.0, (100, 400))],
)
def test_axis_end_covers_the_data_without_an_empty_tick(
    script: ModuleType, x_max: float, expected: tuple[float, float]
) -> None:
    assert script.axis_top(x_max) == expected


def test_page_has_one_panel_per_run(script: ModuleType) -> None:
    runs = [
        script.Curves("a <10m>", [(0.0, 9.0), (1.0, 2.0)], [(1.0, 2.5)]),
        script.Curves("b", [(0.0, 9.0), (2.0, 3.0)], [(2.0, 3.3217)]),
    ]
    page, width, height = script.render_page(runs, "Title", "Note")
    assert (width, height) == (16 + 2 * script.PANEL_SLOT, script.HEIGHT)
    assert page.count("<polyline") == 4 and "a &lt;10m&gt;" in page
    assert "val 2.50" in page and "val 3.32" in page
    assert page.count("train loss") == 1  # legend on the first panel only


def test_missing_chrome_is_reported(script: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--chrome"):
        script.render_png(tmp_path / "p.html", tmp_path / "p.png", 10, 10, tmp_path / "no-chrome")
