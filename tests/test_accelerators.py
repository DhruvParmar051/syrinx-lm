"""Accelerator checks, run once per GPU backend (MPS here, CUDA on a rented GPU).

Each test is parametrized by the ``accelerator`` fixture, so the same assertions that pass
on MPS are what a CUDA machine has to pass before any training run there (see
``docs/cuda-dry-run.md``). Backends that are unavailable are skipped, never faked.
"""

from pathlib import Path

import pytest
import torch

from syrinx.config import ModelConfig, load_config
from syrinx.model import SyrinxLM, lm_loss
from syrinx.training.checkpoint import load_checkpoint, save_checkpoint
from syrinx.training.step import train_step
from syrinx.utils.device import memory_stats, reset_peak_memory, synchronize
from syrinx.utils.precision import probe_bf16
from syrinx.utils.timing import Timer

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20

TINY = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_ff=96,
    max_seq_len=16,
    rope_theta=10000.0,
    norm_eps=1e-5,
)


def _tiny(device: torch.device | str = "cpu", seed: int = 0) -> SyrinxLM:
    torch.manual_seed(seed)
    return SyrinxLM(TINY).to(device)


def _batch(seed: int, device: torch.device | str = "cpu", b: int = 4, t: int = 12):  # type: ignore[no-untyped-def]
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, TINY.vocab_size, (b, t + 1), generator=gen).to(device)
    return ids[:, :-1], ids[:, 1:]


def _step(model: SyrinxLM, opt: torch.optim.Optimizer, seed: int, device, precision: str = "fp32"):  # type: ignore[no-untyped-def]
    x, y = _batch(seed, device)
    return train_step(
        model, opt, lambda: (x, y), grad_accum_steps=1, grad_clip=1.0, precision=precision
    )


def test_bf16_probe_passes(accelerator: torch.device) -> None:
    """bf16 autocast forward + backward (SDPA, matmul, cross-entropy) works and matches fp32."""
    result = probe_bf16(accelerator)
    assert result.supported, result.reason


def test_real_model_matches_cpu_fp32(accelerator: torch.device) -> None:
    """The 10m config gives the same logits and loss on the accelerator as on CPU (fp32)."""
    cfg = load_config(ROOT / "configs" / "syrinx-10m.yaml").model
    torch.manual_seed(0)
    model = SyrinxLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 257), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        cpu_logits = model(ids[:, :-1])
        cpu_loss = lm_loss(cpu_logits, ids[:, 1:])
        model.to(accelerator)
        assert model.rotary.cos.device.type == accelerator.type
        acc_logits = model(ids[:, :-1].to(accelerator))
        acc_loss = lm_loss(acc_logits, ids[:, 1:].to(accelerator))
    torch.testing.assert_close(acc_logits.cpu(), cpu_logits, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(acc_loss.cpu(), cpu_loss, rtol=1e-4, atol=1e-4)


def test_bf16_train_step_tracks_cpu_fp32(accelerator: torch.device) -> None:
    """One bf16-autocast step on the accelerator: finite, and loss and grad norm within bf16
    tolerance of the same step in fp32 on CPU."""
    cpu_model = _tiny("cpu")
    ref = _step(cpu_model, torch.optim.SGD(cpu_model.parameters(), lr=0.0), 7, "cpu")
    model = _tiny(accelerator)
    got = _step(model, torch.optim.SGD(model.parameters(), lr=0.0), 7, accelerator, "bf16-autocast")
    assert torch.isfinite(torch.tensor([got.loss, got.grad_norm])).all()
    assert got.loss == pytest.approx(ref.loss, rel=3e-2)
    assert got.grad_norm == pytest.approx(ref.grad_norm, rel=5e-2)


def test_checkpoint_from_cpu_resumes_on_accelerator(
    accelerator: torch.device, tmp_path: Path
) -> None:
    """A checkpoint written on one device resumes on another: the AdamW state follows the
    parameters to the accelerator, and the next step matches continuing on CPU (fp32)."""
    model = _tiny("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    for seed in (1, 2):
        _step(model, opt, seed, "cpu")
    path = tmp_path / "step_0000002.pt"
    save_checkpoint(path, model, opt, 2, {"note": "test"}, {"name": "tiny"})
    _step(model, opt, 3, "cpu")

    moved = _tiny(accelerator, seed=99)
    moved_opt = torch.optim.AdamW(moved.parameters(), lr=1e-2, weight_decay=0.1)
    ckpt = load_checkpoint(path, moved, moved_opt)
    assert ckpt.step == 2
    states = [s for s in moved_opt.state.values() if "exp_avg" in s]
    assert states and all(s["exp_avg"].device.type == accelerator.type for s in states)
    _step(moved, moved_opt, 3, accelerator)

    for (name, p), q in zip(model.named_parameters(), moved.parameters(), strict=True):
        torch.testing.assert_close(q.detach().cpu(), p.detach(), rtol=1e-4, atol=1e-5, msg=name)


def test_timer_and_memory_stats(accelerator: torch.device) -> None:
    """Timing waits for queued work, and memory stats see an allocation and keep their
    invariants (peak >= allocated, driver >= allocated, a positive ceiling)."""
    reset_peak_memory(accelerator)
    before = memory_stats(accelerator).allocated
    a = torch.randn(2048, 2048, device=accelerator)
    with Timer(accelerator) as timer:
        for _ in range(4):
            a = a @ a / 2048
    synchronize(accelerator)
    stats = memory_stats(accelerator)
    assert timer.elapsed > 0
    assert stats.device_type == accelerator.type
    assert stats.allocated - before >= 16 * MIB
    assert stats.peak_allocated is not None and stats.peak_allocated >= stats.allocated
    assert stats.driver_allocated is not None and stats.driver_allocated >= stats.allocated
    assert stats.recommended_max is not None and stats.recommended_max > 0
    del a
