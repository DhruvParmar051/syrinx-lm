"""Tests for ``syrinx.estimate`` and ``scripts/estimate_memory.py``.

Expected values are written out by hand so the tests check the formulas, not the
functions against themselves.
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest

from syrinx.config import ModelConfig, load_config
from syrinx.estimate import (
    _per_layer_activation_elements,
    activation_bytes,
    estimate_train_days,
    estimate_training_memory,
    flops_per_token,
    logits_bytes,
    parameter_state_bytes,
)
from syrinx.utils.device import bytes_to_gib

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def _model(name: str) -> ModelConfig:
    return load_config(ROOT / "configs" / f"{name}.yaml").model


def _tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=10, d_model=8, n_layers=2, n_heads=2, d_ff=16,
        max_seq_len=4, rope_theta=10000.0, norm_eps=1e-5,
    )  # fmt: skip


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- parameter state ------------------------------------------------------------------


def test_parameter_state_fp32() -> None:
    """fp32: 4N weights, 4N grads, 8N optimizer, no autocast copies."""
    assert parameter_state_bytes(100, "fp32") == {
        "params": 400, "grads": 400, "optimizer": 800, "autocast_copies": 0,
    }  # fmt: skip


def test_parameter_state_bf16() -> None:
    """bf16-autocast adds 2N bytes of weight copies."""
    assert parameter_state_bytes(100, "bf16-autocast")["autocast_copies"] == 200


@pytest.mark.parametrize("bad", ["fp16", "bf16", ""])
def test_unknown_precision_rejected(bad: str) -> None:
    """Unknown precisions raise ValueError."""
    with pytest.raises(ValueError, match="precision"):
        parameter_state_bytes(10, bad)


# --- activations and logits -----------------------------------------------------------


def test_per_layer_elements_hand_count() -> None:
    """Tiny config, B=1, T=4: 8C + 4F + 2HT per token = 64 + 64 + 16 = 144; x4 tokens."""
    assert _per_layer_activation_elements(_tiny(), micro_batch_size=1, seq_len=4) == 4 * 144


def test_activation_bytes_hand_count() -> None:
    """(L * per-layer + B*T*2C) elements times bytes per element."""
    elements = 2 * (4 * 144) + 4 * 2 * 8
    assert activation_bytes(_tiny(), 1, 4, "fp32") == elements * 4
    assert activation_bytes(_tiny(), 1, 4, "bf16-autocast") == elements * 2


def test_activations_linear_in_batch() -> None:
    """Doubling the micro-batch doubles activation bytes."""
    model = _model("syrinx-125m")
    assert activation_bytes(model, 16, 1024, "bf16-autocast") == 2 * activation_bytes(
        model, 8, 1024, "bf16-autocast"
    )


def test_activations_superlinear_in_seq_len() -> None:
    """Doubling seq_len more than doubles activations (the (B, H, T, T) term)."""
    model = _model("syrinx-125m")
    short = activation_bytes(model, 8, 512, "bf16-autocast")
    assert activation_bytes(model, 8, 1024, "bf16-autocast") > 2 * short


def test_logits_bytes_125m() -> None:
    """B=8, T=1024, V=32768: 14 B/element in bf16 (3.5 GiB), 12 B in fp32 (3.0 GiB)."""
    elements = 8 * 1024 * 32768
    assert logits_bytes(32768, 8, 1024, "bf16-autocast") == elements * 14
    assert logits_bytes(32768, 8, 1024, "fp32") == elements * 12
    assert bytes_to_gib(elements * 14) == 3.5


# --- totals ---------------------------------------------------------------------------


def test_total_applies_overhead() -> None:
    """total == round(sum(components) * overhead_factor)."""
    estimate = estimate_training_memory(_model("syrinx-10m"), 32, 1024, "bf16-autocast")
    assert estimate.total == round(sum(estimate.components.values()) * 1.25)


def test_125m_matches_plan_estimate() -> None:
    """125M, mb 8, bf16: within 10% of PLAN.md's ~16 GiB analytic estimate.

    The proxy measured 20.1 GiB driver-allocated (EXP-000); that gap is expected and
    is recorded in the experiment log, not hidden by tuning this formula.
    """
    total_gib = bytes_to_gib(
        estimate_training_memory(_model("syrinx-125m"), 8, 1024, "bf16-autocast").total
    )
    assert 16 * 0.9 <= total_gib <= 16 * 1.1


def test_estimate_rejects_bad_sizes() -> None:
    """Non-positive sizes or an overhead below 1 raise ValueError."""
    model = _model("syrinx-10m")
    with pytest.raises(ValueError):
        estimate_training_memory(model, 0, 1024, "fp32")
    with pytest.raises(ValueError):
        estimate_training_memory(model, 8, 1024, "fp32", overhead_factor=0.9)


# --- compute and time -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tokens", "plan_flops"),
    [("syrinx-50m", 1.0e9, 0.37e18), ("syrinx-125m", 2.5e9, 2.15e18)],
)
def test_train_flops_match_plan(name: str, tokens: float, plan_flops: float) -> None:
    """flops_per_token * planned tokens reproduces PLAN.md §1 to its 2-3 digits."""
    total = flops_per_token(_model(name), 1024) * tokens
    assert math.isclose(total, plan_flops, rel_tol=0.01)


def test_flops_per_token_formula() -> None:
    """Tiny config, T=4: 6N + 12*L*T*d, with per block 4*64 + 3*8*16 + 2*8 = 656
    and N = 10*8 + 2*656 + 8 = 1400."""
    assert flops_per_token(_tiny(), 4) == 6 * 1400 + 12 * 2 * 4 * 8


def test_train_days_matches_exp000() -> None:
    """2.5e9 tokens at the proxy's 6,674 tok/s is ~4.34 days (EXP-000 arithmetic)."""
    assert math.isclose(estimate_train_days(2.5e9, 6674), 4.335, rel_tol=1e-3)


@pytest.mark.parametrize("tps", [0.0, -1.0])
def test_train_days_rejects_nonpositive(tps: float) -> None:
    """Non-positive throughput raises ValueError."""
    with pytest.raises(ValueError, match="tokens_per_sec"):
        estimate_train_days(1e9, tps)


# --- script ---------------------------------------------------------------------------


def test_estimate_memory_script(capsys: pytest.CaptureFixture[str]) -> None:
    """The script prints the parameter count, a total and a time for a given tok/s."""
    script = _load_script("estimate_memory")
    code = script.main([str(ROOT / "configs" / "syrinx-125m.yaml"), "--tok-per-sec", "6674"])
    out = capsys.readouterr().out
    assert code == 0
    assert "124,275,456 parameters" in out
    assert "total x1.25" in out
    assert "days" in out


def test_estimate_memory_script_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    """--micro-batch and --precision override the config values."""
    script = _load_script("estimate_memory")
    script.main(
        [str(ROOT / "configs" / "syrinx-10m.yaml"), "--micro-batch", "4", "--precision", "fp32"]
    )
    out = capsys.readouterr().out
    assert "micro-batch 4" in out and "precision fp32" in out
    assert "days" not in out
