"""Tests for ``syrinx.utils.precision``."""

import pytest
import torch
import torch.nn.functional as F

from syrinx.utils import precision
from syrinx.utils.precision import _make_probe_inputs, _merge_heads, probe_bf16


def test_merge_heads_layout() -> None:
    """``out[b, t, h*D:(h+1)*D] == x[b, h, t, :]`` for every b, h, t; shape (B, T, H*D)."""
    batch, heads, seq, head_dim = 2, 3, 5, 4
    x = torch.arange(batch * heads * seq * head_dim, dtype=torch.float32).reshape(
        batch, heads, seq, head_dim
    )
    out = _merge_heads(x)
    assert out.shape == (batch, seq, heads * head_dim)
    for b in range(batch):
        for h in range(heads):
            for t in range(seq):
                assert torch.equal(out[b, t, h * head_dim : (h + 1) * head_dim], x[b, h, t])


def test_make_probe_inputs_shapes_and_seed() -> None:
    """Shapes are as documented, leaves require grad, same seed gives equal tensors,
    and the global RNG state is unchanged."""
    before = torch.get_rng_state()
    first = _make_probe_inputs(torch.device("cpu"), seed=7)
    second = _make_probe_inputs(torch.device("cpu"), seed=7)
    assert torch.equal(before, torch.get_rng_state())

    q, k, v, weight, targets = first
    for t in (q, k, v):
        assert t.shape == (2, 4, 64, 32) and t.requires_grad
    assert weight.shape == (128, 1024) and weight.requires_grad
    assert targets.shape == (128,) and targets.dtype == torch.int64
    assert not targets.requires_grad
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_probe_cpu_supported() -> None:
    """On cpu, bf16 autocast works: supported, reason non-empty, rel_err <= rtol."""
    result = probe_bf16(torch.device("cpu"))
    assert result.supported, result.reason
    assert result.rel_err is not None and result.rel_err <= 3e-2


@pytest.mark.mps
def test_probe_mps_supported(mps_device: torch.device) -> None:
    """On this Mac, bf16 autocast fwd+bwd works on MPS."""
    result = probe_bf16(mps_device)
    assert result.supported, result.reason
    assert result.rel_err is not None and result.rel_err <= 3e-2


def test_probe_is_deterministic() -> None:
    """Two probes with the same seed give the same rel_err."""
    cpu = torch.device("cpu")
    assert probe_bf16(cpu, seed=3).rel_err == probe_bf16(cpu, seed=3).rel_err


def test_probe_restores_grad_mode() -> None:
    """Called under no_grad, the probe still runs backward, and grad stays disabled after."""
    with torch.no_grad():
        result = probe_bf16(torch.device("cpu"))
        assert not torch.is_grad_enabled()
    assert result.supported, result.reason
    assert torch.is_grad_enabled()


def test_probe_leaves_no_autocast_state() -> None:
    """Autocast is disabled after the probe."""
    probe_bf16(torch.device("cpu"))
    assert not torch.is_autocast_enabled("cpu")


def test_probe_reports_failure_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an op raises, the result is unsupported and the reason carries the message."""

    def broken_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("sdpa exploded")

    monkeypatch.setattr(F, "scaled_dot_product_attention", broken_sdpa)
    result = probe_bf16(torch.device("cpu"))
    assert not result.supported
    assert "sdpa exploded" in result.reason
    assert result.rel_err is None


def test_probe_flags_autocast_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """If autocast silently doesn't cast, the probe reports it instead of passing."""

    class NoOpAutocast:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(precision.torch, "autocast", NoOpAutocast)
    result = probe_bf16(torch.device("cpu"))
    assert not result.supported
    assert "float32" in result.reason


def test_probe_rejects_when_tolerance_too_tight() -> None:
    """With an unrealistically tight rtol, the probe reports the mismatch and rel_err."""
    result = probe_bf16(torch.device("cpu"), rtol=1e-12)
    assert not result.supported
    assert result.rel_err is not None and result.rel_err > 1e-12
    assert "rtol" in result.reason
