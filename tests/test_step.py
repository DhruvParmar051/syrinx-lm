"""Tests for ``syrinx.training.step``."""

import contextlib

import pytest
import torch

from syrinx.config import ModelConfig
from syrinx.model import SyrinxLM, lm_loss
from syrinx.training.step import StepStats, autocast_context, evaluate, train_step

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


def _model(seed: int = 0, device: torch.device | str = "cpu") -> SyrinxLM:
    torch.manual_seed(seed)
    return SyrinxLM(TINY).to(device)


def _tokens(b: int, t: int = 12, seed: int = 0, device: torch.device | str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY.vocab_size, (b, t + 1), generator=gen).to(device)


def _feeder(batches: list[tuple[torch.Tensor, torch.Tensor]]):  # type: ignore[no-untyped-def]
    it = iter(batches)
    calls = {"n": 0}

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        calls["n"] += 1
        return next(it)

    return get_batch, calls


def _split(tokens: torch.Tensor, parts: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(c[:, :-1], c[:, 1:]) for c in tokens.chunk(parts)]


def _sgd(model: torch.nn.Module, lr: float = 1.0) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=lr)


def _params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# --- autocast_context -----------------------------------------------------------------


def test_autocast_context_fp32_is_noop() -> None:
    ctx = autocast_context("cpu", "fp32")
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


def test_autocast_context_bf16(each_device: torch.device) -> None:
    a = torch.ones(2, 2, device=each_device)
    with autocast_context(each_device.type, "bf16-autocast"):
        assert (a @ a).dtype == torch.bfloat16


def test_autocast_context_unknown_raises() -> None:
    with pytest.raises(ValueError, match="fp16"):
        autocast_context("cpu", "fp16")


# --- train_step -----------------------------------------------------------------------


def test_accumulation_equals_full_batch(each_device: torch.device) -> None:
    """4 micro-batches of 2 give the same update as 1 batch of 8 (loss / accum)."""
    tokens = _tokens(8, device=each_device)
    full, accum = _model(device=each_device), _model(device=each_device)
    get_full, _ = _feeder(_split(tokens, 1))
    get_accum, _ = _feeder(_split(tokens, 4))
    s_full = train_step(full, _sgd(full, 0.5), get_full, 1, grad_clip=1e9, precision="fp32")
    s_acc = train_step(accum, _sgd(accum, 0.5), get_accum, 4, grad_clip=1e9, precision="fp32")
    assert s_acc.loss == pytest.approx(s_full.loss, rel=1e-5)
    assert s_acc.grad_norm == pytest.approx(s_full.grad_norm, rel=1e-4)
    for (name, p), q in zip(full.named_parameters(), accum.parameters(), strict=True):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6, msg=name)


def test_reports_mean_loss_norm_and_tokens() -> None:
    tokens = _tokens(6, t=10, seed=1)
    batches = _split(tokens, 3)
    ref = _model()
    with torch.no_grad():
        losses = [lm_loss(ref(x), y).item() for x, y in batches]
    ref_loss = sum(lm_loss(ref(x), y) for x, y in batches) / 3
    ref_loss.backward()
    ref_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in ref.parameters()])
    ).item()

    model = _model()
    get_batch, calls = _feeder(batches)
    stats = train_step(model, _sgd(model, 0.0), get_batch, 3, grad_clip=1e9, precision="fp32")
    assert isinstance(stats, StepStats)
    assert calls["n"] == 3
    assert stats.loss == pytest.approx(sum(losses) / 3, rel=1e-6)
    assert stats.grad_norm == pytest.approx(ref_norm, rel=1e-5)
    assert stats.tokens == 6 * 10
    assert isinstance(stats.loss, float) and isinstance(stats.grad_norm, float)


def test_clipping_limits_update_norm() -> None:
    """SGD with lr=1: the update equals the clipped gradient, whose norm is grad_clip."""
    model = _model()
    before = _params(model)
    get_batch, _ = _feeder(_split(_tokens(4), 1))
    stats = train_step(model, _sgd(model, 1.0), get_batch, 1, grad_clip=1e-3, precision="fp32")
    assert stats.grad_norm > 1e-3  # reported norm is the pre-clip norm
    delta = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(p.detach() - before[n]) for n, p in model.named_parameters()]
        )
    ).item()
    assert delta == pytest.approx(1e-3, rel=1e-3)


def test_no_clipping_below_threshold() -> None:
    model = _model()
    before = _params(model)
    get_batch, _ = _feeder(_split(_tokens(4), 1))
    stats = train_step(model, _sgd(model, 1.0), get_batch, 1, grad_clip=1e9, precision="fp32")
    delta = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(p.detach() - before[n]) for n, p in model.named_parameters()]
        )
    ).item()
    assert delta == pytest.approx(stats.grad_norm, rel=1e-4)


def test_stale_gradients_are_cleared() -> None:
    """Gradients left from before the step must not leak into it."""
    tokens = _tokens(4)
    clean, dirty = _model(), _model()
    for p in dirty.parameters():
        p.grad = torch.full_like(p, 5.0)
    for model in (clean, dirty):
        get_batch, _ = _feeder(_split(tokens, 1))
        train_step(model, _sgd(model, 0.1), get_batch, 1, grad_clip=1e9, precision="fp32")
    for p, q in zip(clean.parameters(), dirty.parameters(), strict=True):
        torch.testing.assert_close(q, p)


def test_two_steps_do_not_accumulate_across_steps() -> None:
    """Step 2's gradient is from step 2's batch only (zero_grad at the start of each step)."""
    a, b = _tokens(4, seed=1), _tokens(4, seed=2)
    model = _model()
    opt = _sgd(model, 0.1)
    get_batch, _ = _feeder(_split(a, 1) + _split(b, 1))
    train_step(model, opt, get_batch, 1, grad_clip=1e9, precision="fp32")
    after_first = _params(model)
    train_step(model, opt, get_batch, 1, grad_clip=1e9, precision="fp32")

    ref = _model()
    with torch.no_grad():
        for p, v in zip(ref.parameters(), after_first.values(), strict=True):
            p.copy_(v)
    get_ref, _ = _feeder(_split(b, 1))
    train_step(ref, _sgd(ref, 0.1), get_ref, 1, grad_clip=1e9, precision="fp32")
    for p, q in zip(ref.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(q, p)


def test_sets_train_mode() -> None:
    model = _model().eval()
    get_batch, _ = _feeder(_split(_tokens(2), 1))
    train_step(model, _sgd(model, 0.0), get_batch, 1, grad_clip=1.0, precision="fp32")
    assert model.training


def test_non_finite_gradient_raises_before_update() -> None:
    model = _model()
    with torch.no_grad():
        model.layers[0].mlp.up_proj.weight[0, 0] = float("nan")
    before = _params(model)
    get_batch, _ = _feeder(_split(_tokens(2), 1))
    with pytest.raises(FloatingPointError, match="nan"):
        train_step(model, _sgd(model, 0.1), get_batch, 1, grad_clip=1.0, precision="fp32")
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name]) or torch.isnan(before[name]).any(), name
    assert torch.equal(model.norm.weight.detach(), before["norm.weight"])


def test_bad_accum_steps_raise_before_drawing() -> None:
    model = _model()
    get_batch, calls = _feeder(_split(_tokens(2), 1))
    with pytest.raises(ValueError):
        train_step(model, _sgd(model), get_batch, 0, grad_clip=1.0, precision="fp32")
    assert calls["n"] == 0


def test_bf16_step_runs(each_device: torch.device) -> None:
    """bf16 autocast: finite loss close to fp32, fp32 gradients and weights."""
    tokens = _tokens(4, device=each_device)
    fp32, bf16 = _model(device=each_device), _model(device=each_device)
    get_a, _ = _feeder(_split(tokens, 2))
    get_b, _ = _feeder(_split(tokens, 2))
    s32 = train_step(fp32, _sgd(fp32, 0.0), get_a, 2, grad_clip=1.0, precision="fp32")
    s16 = train_step(bf16, _sgd(bf16, 0.0), get_b, 2, grad_clip=1.0, precision="bf16-autocast")
    assert abs(s16.loss - s32.loss) < 0.05
    assert all(
        p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in bf16.parameters()
    )


def test_loss_decreases_on_repeated_batch(each_device: torch.device) -> None:
    """Sanity: AdamW on one fixed batch lowers the loss within a few steps."""
    tokens = _tokens(4, device=each_device)
    model = _model(device=each_device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        get_batch, _ = _feeder(_split(tokens, 1))
        losses.append(train_step(model, opt, get_batch, 1, grad_clip=1.0, precision="fp32").loss)
    assert losses[-1] < losses[0] - 1.0


# --- evaluate -------------------------------------------------------------------------


def test_evaluate_weights_by_tokens() -> None:
    """Unequal batch sizes: the per-token mean, not the mean of batch means."""
    model = _model()
    tokens = _tokens(5, seed=3)
    batches = [(tokens[:4, :-1], tokens[:4, 1:]), (tokens[4:, :-1], tokens[4:, 1:])]
    x_all, y_all = tokens[:, :-1], tokens[:, 1:]
    with torch.no_grad():
        want = lm_loss(model(x_all), y_all).item()
    assert evaluate(model, batches, "fp32") == pytest.approx(want, rel=1e-5)


def test_evaluate_no_grad_and_restores_mode() -> None:
    model = _model()
    batches = _split(_tokens(2), 1)
    model.train()
    evaluate(model, batches, "fp32")
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    model.eval()
    evaluate(model, batches, "fp32")
    assert not model.training


def test_evaluate_runs_in_eval_mode_without_graph() -> None:
    """eval() mode, and no autograd graph is built (no_grad: activations are freed)."""
    model = _model()
    seen: list[tuple[bool, bool]] = []
    model.norm.register_forward_hook(lambda m, i, o: seen.append((m.training, o.requires_grad)))
    evaluate(model.train(), _split(_tokens(2), 1), "fp32")
    assert seen == [(False, False)]


def test_evaluate_restores_mode_on_error() -> None:
    model = _model().train()

    def bad_batches():  # type: ignore[no-untyped-def]
        yield _split(_tokens(2), 1)[0]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        evaluate(model, bad_batches(), "fp32")
    assert model.training


def test_evaluate_empty_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(_model(), [], "fp32")


def test_evaluate_on_device(each_device: torch.device) -> None:
    model = _model(device=each_device)
    batches = _split(_tokens(4, device=each_device), 2)
    loss = evaluate(model, batches, "bf16-autocast")
    assert isinstance(loss, float) and 3.0 < loss < 5.5  # ~ln(64) = 4.16 at init
