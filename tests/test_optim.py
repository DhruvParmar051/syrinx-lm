"""Tests for ``syrinx.training.optim``."""

import dataclasses
import itertools
import math

import pytest
import torch

from syrinx.config import ModelConfig, TrainConfig
from syrinx.model import SyrinxLM
from syrinx.training.optim import (
    ADAM_EPS,
    build_optimizer,
    lr_at,
    param_groups,
    set_lr,
)

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

TRAIN = TrainConfig(
    micro_batch_size=2,
    grad_accum_steps=1,
    lr=1e-3,
    min_lr=1e-4,
    warmup_steps=10,
    max_steps=110,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    precision="fp32",
    seed=0,
)


def _names(model: torch.nn.Module, params: list[torch.nn.Parameter]) -> set[str]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {by_id[id(p)] for p in params}


# --- param_groups ---------------------------------------------------------------------


def test_param_groups_split_tied() -> None:
    model = SyrinxLM(TINY)
    decay, no_decay = param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    want_decay = {
        f"layers.{i}.{sub}.{w}_proj.weight"
        for i in range(TINY.n_layers)
        for sub, ws in (("self_attn", "qkvo"), ("mlp", ("gate", "up", "down")))
        for w in ws
    }
    want_no_decay = {"embed_tokens.weight", "norm.weight"} | {
        f"layers.{i}.{n}.weight"
        for i in range(TINY.n_layers)
        for n in ("input_layernorm", "post_attention_layernorm")
    }
    assert _names(model, decay["params"]) == want_decay
    assert _names(model, no_decay["params"]) == want_no_decay


def test_param_groups_cover_every_parameter_once() -> None:
    model = SyrinxLM(TINY)
    groups = param_groups(model, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert sum(p.numel() for g in groups for p in g["params"]) == model.num_parameters()


def test_param_groups_keep_named_parameters_order() -> None:
    model = SyrinxLM(TINY)
    order = [n for n, _ in model.named_parameters()]
    for group in param_groups(model, 0.1):
        names = [n for n, p in model.named_parameters() if any(p is q for q in group["params"])]
        assert names == sorted(names, key=order.index)
        assert [id(p) for p in group["params"]] == [
            id(p) for n, p in model.named_parameters() if n in names
        ]


def test_param_groups_untied_head_is_decayed() -> None:
    model = SyrinxLM(dataclasses.replace(TINY, tie_embeddings=False))
    decay, no_decay = param_groups(model, 0.1)
    assert "lm_head.weight" in _names(model, decay["params"])
    assert "embed_tokens.weight" in _names(model, no_decay["params"])


def test_param_groups_skip_frozen() -> None:
    model = SyrinxLM(TINY)
    model.norm.weight.requires_grad_(False)
    model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
    names = set().union(*(_names(model, g["params"]) for g in param_groups(model, 0.1)))
    assert "norm.weight" not in names and "layers.0.self_attn.q_proj.weight" not in names


# --- build_optimizer ------------------------------------------------------------------


def test_build_optimizer_settings() -> None:
    model = SyrinxLM(TINY)
    opt = build_optimizer(model, TRAIN)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    assert [g["weight_decay"] for g in opt.param_groups] == [TRAIN.weight_decay, 0.0]
    for group in opt.param_groups:
        assert group["lr"] == TRAIN.lr
        assert tuple(group["betas"]) == TRAIN.betas
        assert group["eps"] == ADAM_EPS


def test_decay_applies_only_to_decayed_group() -> None:
    """With zero gradients AdamW's Adam update is 0, so one step only applies decoupled
    decay: decayed weights scale by (1 - lr * wd); the rest stay unchanged."""
    torch.manual_seed(0)
    model = SyrinxLM(TINY)
    opt = build_optimizer(model, TRAIN)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    decayed = _names(model, opt.param_groups[0]["params"])
    for name, p in model.named_parameters():
        factor = 1 - TRAIN.lr * TRAIN.weight_decay if name in decayed else 1.0
        torch.testing.assert_close(p.detach(), before[name] * factor, rtol=0, atol=1e-8)


# --- lr_at ----------------------------------------------------------------------------


def _lr(step: int, warmup: int = 10, max_steps: int = 110) -> float:
    return lr_at(step, peak_lr=1e-3, min_lr=1e-4, warmup_steps=warmup, max_steps=max_steps)


def test_lr_warmup_is_linear_from_first_step() -> None:
    assert _lr(0) == pytest.approx(1e-4)
    assert _lr(4) == pytest.approx(5e-4)
    assert _lr(9) == pytest.approx(1e-3)
    warm = [_lr(s) for s in range(10)]
    assert all(b > a for a, b in itertools.pairwise(warm))


def test_lr_cosine_decay() -> None:
    assert _lr(10) == pytest.approx(1e-3)
    assert _lr(60) == pytest.approx(5.5e-4)  # halfway: (peak + min) / 2
    assert _lr(35) == pytest.approx(1e-4 + 0.45e-3 * (1 + math.cos(math.pi * 0.25)))
    decay = [_lr(s) for s in range(10, 111)]
    assert all(b < a for a, b in itertools.pairwise(decay))
    assert _lr(109) > 1e-4


def test_lr_after_max_steps_is_min() -> None:
    for step in (110, 111, 10_000):
        assert _lr(step) == pytest.approx(1e-4)


def test_lr_without_warmup_starts_at_peak() -> None:
    assert _lr(0, warmup=0) == pytest.approx(1e-3)
    assert _lr(1, warmup=0) < 1e-3


def test_lr_negative_step_raises() -> None:
    with pytest.raises(ValueError):
        _lr(-1)


def test_lr_matches_config_endpoints() -> None:
    """With the training config's own values the schedule peaks at lr and ends at min_lr."""
    t = TRAIN
    values = [lr_at(s, t.lr, t.min_lr, t.warmup_steps, t.max_steps) for s in range(t.max_steps)]
    assert max(values) == pytest.approx(t.lr)
    assert min(values[t.warmup_steps :]) > t.min_lr  # decay never undershoots min_lr
    assert lr_at(t.max_steps, t.lr, t.min_lr, t.warmup_steps, t.max_steps) == pytest.approx(
        t.min_lr
    )


# --- set_lr ---------------------------------------------------------------------------


def test_set_lr_updates_every_group() -> None:
    opt = build_optimizer(SyrinxLM(TINY), TRAIN)
    set_lr(opt, 3.5e-4)
    assert [g["lr"] for g in opt.param_groups] == [3.5e-4, 3.5e-4]
