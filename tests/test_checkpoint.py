"""Tests for ``syrinx.training.checkpoint``, including a CPU resume-equivalence test
(Gate 4 ladder (c) at small scale)."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from syrinx.config import ModelConfig, TrainConfig
from syrinx.data.loader import RandomWindowLoader
from syrinx.model import SyrinxLM
from syrinx.training.checkpoint import (
    CHECKPOINT_VERSION,
    Checkpoint,
    capture_rng_state,
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
)
from syrinx.training.optim import build_optimizer, lr_at, set_lr
from syrinx.training.step import train_step

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
    grad_accum_steps=2,
    lr=3e-3,
    min_lr=3e-4,
    warmup_steps=2,
    max_steps=8,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    precision="fp32",
    seed=0,
)
CONFIG = {"name": "tiny", "train": {"max_steps": 8}}


def _shards() -> list[np.ndarray]:
    gen = np.random.default_rng(0)
    return [gen.integers(0, TINY.vocab_size, size=n).astype(np.uint16) for n in (500, 300)]


def _setup(seed: int = 0) -> tuple[SyrinxLM, torch.optim.Optimizer, RandomWindowLoader]:
    torch.manual_seed(seed)
    model = SyrinxLM(TINY)
    opt = build_optimizer(model, TRAIN)
    loader = RandomWindowLoader(_shards(), batch_size=2, seq_len=12, seed=7)
    return model, opt, loader


def _run(
    model: SyrinxLM, opt: Any, loader: RandomWindowLoader, start: int, stop: int
) -> list[float]:
    losses = []
    for step in range(start, stop):
        set_lr(opt, lr_at(step, TRAIN.lr, TRAIN.min_lr, TRAIN.warmup_steps, TRAIN.max_steps))
        stats = train_step(model, opt, loader.next_batch, TRAIN.grad_accum_steps, 1.0, "fp32")
        losses.append(stats.loss)
    return losses


def _trained(steps: int = 2) -> tuple[SyrinxLM, torch.optim.Optimizer, RandomWindowLoader]:
    model, opt, loader = _setup()
    _run(model, opt, loader, 0, steps)
    return model, opt, loader


# --- paths ----------------------------------------------------------------------------


def test_checkpoint_path(tmp_path: Path) -> None:
    assert checkpoint_path(tmp_path, 0) == tmp_path / "step_0000000.pt"
    assert checkpoint_path(tmp_path, 1234) == tmp_path / "step_0001234.pt"
    with pytest.raises(ValueError):
        checkpoint_path(tmp_path, -1)


def test_list_and_latest(tmp_path: Path) -> None:
    assert list_checkpoints(tmp_path / "missing") == []
    assert latest_checkpoint(tmp_path) is None
    for name in (
        "step_0000020.pt",
        "step_0000003.pt",
        "step_0000100.pt",
        "step_0000200.pt.tmp",
        "notes.txt",
        "step_12.pt",
        "step_0000050.pth",
    ):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "step_0000003.pt",
        "step_0000020.pt",
        "step_0000100.pt",
    ]
    assert latest_checkpoint(tmp_path) == tmp_path / "step_0000100.pt"


def test_prune_keeps_last_and_milestones(tmp_path: Path) -> None:
    for step in (100, 200, 250, 300, 400, 450, 500):
        checkpoint_path(tmp_path, step).write_bytes(b"")
    removed = prune_checkpoints(tmp_path, keep_last=2, keep_every=200)
    assert [p.name for p in removed] == ["step_0000100.pt", "step_0000250.pt", "step_0000300.pt"]
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "step_0000200.pt",
        "step_0000400.pt",
        "step_0000450.pt",
        "step_0000500.pt",
    ]


def test_prune_without_milestones(tmp_path: Path) -> None:
    for step in (1, 2, 3):
        checkpoint_path(tmp_path, step).write_bytes(b"")
    (tmp_path / "keep.txt").write_bytes(b"")
    prune_checkpoints(tmp_path, keep_last=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt", "step_0000003.pt"]
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, keep_last=0)


# --- RNG ------------------------------------------------------------------------------


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def test_rng_capture_restore() -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = capture_rng_state()
    first = _draw()
    _draw()
    restore_rng_state(state)
    again = _draw()
    assert first[0] == again[0] and first[1] == again[1]
    assert torch.equal(first[2], again[2])


def test_rng_state_is_plain_data() -> None:
    state = capture_rng_state()
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_mps"}
    assert isinstance(state["numpy"][1], list)
    assert (state["torch_mps"] is None) == (not torch.backends.mps.is_available())


@pytest.mark.mps
def test_mps_rng_restored() -> None:
    torch.mps.manual_seed(5)
    state = capture_rng_state()
    first = torch.rand(4, device="mps").cpu()
    restore_rng_state(state)
    assert torch.equal(torch.rand(4, device="mps").cpu(), first)


# --- save / load ----------------------------------------------------------------------


def test_save_writes_atomically_and_refuses_overwrite(tmp_path: Path) -> None:
    model, opt, loader = _trained()
    path = checkpoint_path(tmp_path / "run", 2)
    save_checkpoint(path, model, opt, 2, loader.state_dict(), CONFIG)
    assert path.is_file()
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no temp file left
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, opt, 3, loader.state_dict(), CONFIG)
    assert path.read_bytes() == before


def test_payload_contents(tmp_path: Path) -> None:
    model, opt, loader = _trained()
    path = checkpoint_path(tmp_path, 2)
    save_checkpoint(path, model, opt, 2, loader.state_dict(), CONFIG)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"version", "step", "model", "optimizer", "loader", "rng", "config"}
    assert payload["version"] == CHECKPOINT_VERSION and payload["step"] == 2
    assert payload["config"] == CONFIG
    assert set(payload["model"]) == set(model.state_dict())


def test_roundtrip_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    model, opt, loader = _trained()
    path = checkpoint_path(tmp_path, 2)
    save_checkpoint(path, model, opt, 2, loader.state_dict(), CONFIG)

    fresh, fresh_opt, fresh_loader = _setup(seed=99)
    ckpt = load_checkpoint(path, fresh, fresh_opt)
    assert isinstance(ckpt, Checkpoint)
    assert ckpt.step == 2 and ckpt.config == CONFIG
    for (name, p), q in zip(model.state_dict().items(), fresh.state_dict().values(), strict=True):
        assert torch.equal(p, q), name
    saved, loaded = opt.state_dict(), fresh_opt.state_dict()
    assert saved["param_groups"] == loaded["param_groups"]
    for key, st in saved["state"].items():
        for field, value in st.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(loaded["state"][key][field]))
    fresh_loader.load_state_dict(ckpt.loader_state)
    x1, _ = loader.next_batch()
    x2, _ = fresh_loader.next_batch()
    assert torch.equal(x1, x2)


def test_load_rejects_other_version(tmp_path: Path) -> None:
    model, opt, loader = _trained(1)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(path, model, opt, 1, loader.state_dict(), CONFIG)
    payload = torch.load(path, weights_only=True)
    payload["version"] = CHECKPOINT_VERSION + 1
    bad = tmp_path / "step_0000002.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError):
        load_checkpoint(bad, *_setup()[:2])


def test_load_rejects_mismatched_model(tmp_path: Path) -> None:
    model, opt, loader = _trained(1)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(path, model, opt, 1, loader.state_dict(), CONFIG)
    other = SyrinxLM(ModelConfig(**{**TINY.__dict__, "d_ff": 64}))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, other, build_optimizer(other, TRAIN))


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    """A tied checkpoint has no lm_head.weight; an untied model must not load it with that
    tensor silently left at its random init."""
    model, opt, loader = _trained(1)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(path, model, opt, 1, loader.state_dict(), CONFIG)
    untied = SyrinxLM(ModelConfig(**{**TINY.__dict__, "tie_embeddings": False}))
    with pytest.raises(RuntimeError, match="lm_head"):
        load_checkpoint(path, untied, build_optimizer(untied, TRAIN))


def test_crash_during_write_leaves_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If torch.save dies halfway, no file with the checkpoint's name exists (so
    latest_checkpoint never picks a truncated file)."""
    model, opt, loader = _trained(1)
    path = checkpoint_path(tmp_path, 1)

    def dying_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
        Path(f).write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(path, model, opt, 1, loader.state_dict(), CONFIG)
    assert not path.exists()
    assert latest_checkpoint(tmp_path) is None


def test_load_does_not_touch_rng(tmp_path: Path) -> None:
    model, opt, loader = _trained(1)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(path, model, opt, 1, loader.state_dict(), CONFIG)
    fresh, fresh_opt, _ = _setup()
    torch.manual_seed(123)
    state = torch.get_rng_state()
    load_checkpoint(path, fresh, fresh_opt)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.mps
def test_load_onto_mps_model(tmp_path: Path) -> None:
    """A checkpoint saved from MPS loads into an MPS model; optimizer state follows it."""
    model, opt, loader = _setup()
    model.to("mps")
    opt = build_optimizer(model, TRAIN)
    loader = RandomWindowLoader(_shards(), batch_size=2, seq_len=12, seed=7, device="mps")
    _run(model, opt, loader, 0, 2)
    path = checkpoint_path(tmp_path, 2)
    save_checkpoint(path, model, opt, 2, loader.state_dict(), CONFIG)
    fresh = SyrinxLM(TINY).to("mps")
    fresh_opt = build_optimizer(fresh, TRAIN)
    load_checkpoint(path, fresh, fresh_opt)
    assert all(p.device.type == "mps" for p in fresh.parameters())
    for st in fresh_opt.state.values():
        assert st["exp_avg"].device.type == "mps"
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p.cpu(), q.cpu())


# --- resume equivalence ---------------------------------------------------------------


def test_resume_reproduces_uninterrupted_run(tmp_path: Path) -> None:
    """Train 8 steps straight vs 4 steps, checkpoint, rebuild everything from scratch,
    resume for 4: identical losses and bit-identical final weights (CPU, fixed seed)."""
    model, opt, loader = _setup()
    straight = _run(model, opt, loader, 0, 8)

    first, first_opt, first_loader = _setup()
    head = _run(first, first_opt, first_loader, 0, 4)
    path = checkpoint_path(tmp_path, 4)
    save_checkpoint(path, first, first_opt, 4, first_loader.state_dict(), CONFIG)
    torch.rand(10)  # disturb global RNG between save and resume
    del first, first_opt, first_loader

    resumed, resumed_opt, resumed_loader = _setup(seed=1234)
    ckpt = load_checkpoint(path, resumed, resumed_opt)
    resumed_loader.load_state_dict(ckpt.loader_state)
    restore_rng_state(ckpt.rng_state)
    tail = _run(resumed, resumed_opt, resumed_loader, ckpt.step, 8)

    assert head + tail == straight
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q)
