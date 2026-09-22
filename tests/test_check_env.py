"""Tests for ``scripts/check_env.py``."""

import datetime as dt
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check_env() -> ModuleType:
    """The script loaded as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts" / "check_env.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_on_cpu(check_env: ModuleType, clean_env: pytest.MonkeyPatch) -> None:
    """On cpu with a tiny benchmark: versions, probe, SDPA and three TFLOPS entries."""
    info = check_env.collect_env_info("cpu", matmul_n=64, iters=1)
    assert info["device"] == "cpu"
    assert info["torch"] == torch.__version__
    assert info["sdpa_causal"] == "ok"
    assert isinstance(info["bf16_autocast_supported"], bool)
    tflops = [k for k in info if k.startswith("matmul_tflops_")]
    assert len(tflops) == 3
    assert info["PYTORCH_ENABLE_MPS_FALLBACK"] == "unset"


def test_sdpa_status_reports_failure(
    check_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing SDPA call is reported as a FAIL string, not raised."""

    def broken(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("no kernel")

    monkeypatch.setattr(check_env.F, "scaled_dot_product_attention", broken)
    assert check_env._sdpa_status(torch.device("cpu")) == "FAIL: RuntimeError: no kernel"


def test_one_failing_dtype_does_not_abort(
    check_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If fp16 errors, fp32 and bf16 are still measured."""
    real = check_env.measure_matmul_tflops

    def flaky(device: torch.device, dtype: torch.dtype, **kwargs: int) -> float:
        if dtype == torch.float16:
            raise TypeError("fp16 unsupported")
        return real(device, dtype, **kwargs)

    monkeypatch.setattr(check_env, "measure_matmul_tflops", flaky)
    results = check_env._tflops_by_dtype(torch.device("cpu"), n=32, iters=1)
    assert results["fp16"] == "FAIL: TypeError: fp16 unsupported"
    assert isinstance(results["fp32"], float) and isinstance(results["bf16"], float)


def test_format_report_is_stable(check_env: ModuleType) -> None:
    """Header carries the timestamp; fields follow in insertion order."""
    stamp = dt.datetime(2026, 9, 22, 10, 0, tzinfo=dt.UTC)
    text = check_env.format_report({"torch": "2.14.0", "device": "mps"}, generated=stamp)
    assert text == (
        "# Syrinx environment report, generated 2026-09-22T10:00:00+00:00\n"
        "torch: 2.14.0\n"
        "device: mps\n"
    )


def test_main_writes_file(
    check_env: ModuleType, clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main prints the report and writes the same text to --out."""
    out = tmp_path / "sub" / "env.txt"
    code = check_env.main(
        ["--device", "cpu", "--matmul-n", "32", "--iters", "1", "--out", str(out)]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Syrinx environment report")
    assert "device: cpu" in text


def test_parse_power_source(check_env: ModuleType) -> None:
    """The source is read from ``pmset -g batt``; anything else is "unknown"."""
    batt = "Now drawing from 'Battery Power'\n -InternalBattery-0\t100%; discharging;\n"
    assert check_env._parse_power_source(batt) == "Battery Power"
    assert check_env._parse_power_source("Now drawing from 'AC Power'") == "AC Power"
    assert check_env._parse_power_source("garbage") == "unknown"


def test_parse_powermode(check_env: ModuleType) -> None:
    """The active powermode line is read from ``pmset -g``; missing gives "unknown"."""
    settings = "Currently in use:\n standby              1\n powermode            2\n"
    assert check_env._parse_powermode(settings) == "2"
    assert check_env._parse_powermode(" standby 1\n") == "unknown"


def test_report_includes_power_state(check_env: ModuleType, clean_env: pytest.MonkeyPatch) -> None:
    """The report always has power_source and pmset_powermode fields."""
    info = check_env.collect_env_info("cpu", matmul_n=32, iters=1)
    assert "power_source" in info and "pmset_powermode" in info
