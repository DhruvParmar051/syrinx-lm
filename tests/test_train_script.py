"""Tests for ``scripts/train.py`` on a tiny model and synthetic shards (CPU)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from syrinx.config import config_from_dict, config_to_dict, load_config, save_config
from syrinx.data.shards import ShardWriter
from syrinx.training.checkpoint import list_checkpoints
from syrinx.training.optim import lr_at
from syrinx.utils.metrics import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_split(directory: Path, prefix: str, docs: int, seed: int) -> None:
    gen = np.random.default_rng(seed)
    with ShardWriter(directory, prefix, shard_tokens=2_000) as writer:
        for _ in range(docs):
            writer.add_document(gen.integers(2, 64, size=int(gen.integers(20, 80))).tolist())


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    _write_split(data / "train", "train", docs=150, seed=0)
    _write_split(data / "val", "val", docs=30, seed=1)
    raw = {
        "name": "tiny",
        "model": {
            "vocab_size": 64,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 96,
            "max_seq_len": 16,
            "rope_theta": 10000.0,
            "norm_eps": 1.0e-5,
            "tie_embeddings": True,
        },
        "train": {
            "micro_batch_size": 4,
            "grad_accum_steps": 2,
            "lr": 3.0e-3,
            "min_lr": 3.0e-4,
            "warmup_steps": 2,
            "max_steps": 12,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "precision": "fp32",
            "seed": 0,
        },
        "data": {"train_path": str(data / "train"), "val_path": str(data / "val"), "seq_len": 16},
    }
    path = tmp_path / "tiny.yaml"
    save_config(config_from_dict(raw), path)
    return path


def _args(script: ModuleType, config_path: Path, out: Path, *extra: str) -> Any:
    base = ["--config", str(config_path), "--out-dir", str(out), "--device", "cpu"]
    small = ["--log-interval", "1", "--eval-interval", "4", "--eval-batches", "2"]
    return script.parse_args(base + small + ["--ckpt-interval", "4", *extra])


def _train(script: ModuleType, config_path: Path, out: Path, *extra: str) -> dict[str, Any]:
    return script.train(load_config(config_path), _args(script, config_path, out, *extra), CPU)


def _records(out: Path, kind: str) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(out / "metrics.jsonl") if r["kind"] == kind]


# --- parse_args -----------------------------------------------------------------------


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--config", "c.yaml", "--out-dir", "runs/x"])
    assert args.config == Path("c.yaml") and args.out_dir == Path("runs/x")
    assert args.device is None and args.resume is False and args.stop_at_step is None
    assert (args.log_interval, args.eval_interval, args.eval_batches) == (10, 250, 20)
    assert (args.ckpt_interval, args.keep_last, args.milestone_every) == (500, 3, None)
    assert args.overfit_batch is False and args.train_token_limit is None


def test_parse_args_values(script: ModuleType) -> None:
    args = script.parse_args(
        [
            "--config",
            "c",
            "--out-dir",
            "o",
            "--train-token-limit",
            "2.5e5",
            "--resume",
            "--stop-at-step",
            "40",
            "--milestone-every",
            "1000",
            "--overfit-batch",
        ]
    )
    assert args.train_token_limit == 250_000 and isinstance(args.train_token_limit, int)
    assert args.resume and args.stop_at_step == 40 and args.milestone_every == 1000
    assert args.overfit_batch


@pytest.mark.parametrize(
    "flag",
    ["--log-interval", "--eval-interval", "--eval-batches", "--ckpt-interval", "--keep-last"],
)
def test_parse_args_rejects_non_positive(script: ModuleType, flag: str) -> None:
    with pytest.raises(SystemExit):
        script.parse_args(["--config", "c", "--out-dir", "o", flag, "0"])


# --- helpers --------------------------------------------------------------------------


def test_limit_tokens(script: ModuleType) -> None:
    shards = [np.arange(10), np.arange(10, 25), np.arange(25, 30)]
    assert script.limit_tokens(shards, None) == shards
    got = script.limit_tokens(shards, 18)
    assert [len(s) for s in got] == [10, 8]
    np.testing.assert_array_equal(np.concatenate(got), np.arange(18))
    assert [len(s) for s in script.limit_tokens(shards, 100)] == [10, 15, 5]
    with pytest.raises(ValueError):
        script.limit_tokens(shards, 0)


def test_make_batch_fn_overfit_repeats_first_batch(script: ModuleType) -> None:
    from syrinx.data.loader import RandomWindowLoader

    shards = [np.arange(200, dtype=np.uint16)]
    loader = RandomWindowLoader(shards, batch_size=2, seq_len=8, seed=0)
    fixed = script.make_batch_fn(loader, overfit=True)
    a, b = fixed(), fixed()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    fresh = RandomWindowLoader(shards, batch_size=2, seq_len=8, seed=0)
    assert torch.equal(a[0], fresh.next_batch()[0])  # it is the loader's first batch
    moving = script.make_batch_fn(RandomWindowLoader(shards, 2, 8, seed=0), overfit=False)
    assert not torch.equal(moving()[0], moving()[0])


# --- a fresh run ----------------------------------------------------------------------


def test_fresh_run_outputs(script: ModuleType, config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    summary = _train(script, config_path, out)
    config = load_config(config_path)
    assert load_config(out / "config.yaml") == config

    train = _records(out, "train")
    assert [r["step"] for r in train] == list(range(1, 13))
    for r in train:
        assert set(r) >= {
            "loss",
            "lr",
            "grad_norm",
            "tokens_per_sec",
            "tokens_seen",
            "step_time_s",
            "mem_allocated",
            "mem_peak",
        }
        t = config.train
        assert r["lr"] == pytest.approx(
            lr_at(r["step"] - 1, t.lr, t.min_lr, t.warmup_steps, t.max_steps)
        )
        assert r["tokens_seen"] == r["step"] * 4 * 2 * 16
        assert r["tokens_per_sec"] > 0 and np.isfinite(r["loss"])
    val = _records(out, "val")
    assert [r["step"] for r in val] == [4, 8, 12]
    assert all(np.isfinite(r["val_loss"]) for r in val)

    assert [p.name for p in list_checkpoints(out / "checkpoints")] == [
        "step_0000004.pt",
        "step_0000008.pt",
        "step_0000012.pt",
    ]
    assert summary["start_step"] == 0 and summary["end_step"] == 12
    assert summary["train_loss"] == train[-1]["loss"] and summary["val_loss"] == val[-1]["val_loss"]


def test_optimizer_uses_scheduled_lr(
    script: ModuleType, config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LR the optimizer actually steps with follows lr_at (not just the logged value)."""
    used: list[list[float]] = []
    real = script.train_step

    def spy(model: Any, optimizer: Any, *a: Any, **k: Any) -> Any:
        used.append([g["lr"] for g in optimizer.param_groups])
        return real(model, optimizer, *a, **k)

    monkeypatch.setattr(script, "train_step", spy)
    _train(script, config_path, tmp_path / "run")
    t = load_config(config_path).train
    want = [lr_at(s, t.lr, t.min_lr, t.warmup_steps, t.max_steps) for s in range(t.max_steps)]
    assert [g[0] for g in used] == pytest.approx(want)
    assert all(len(set(g)) == 1 for g in used)  # every group gets the same LR


def test_log_interval_and_last_step(script: ModuleType, config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    args = script.parse_args(
        [
            "--config",
            str(config_path),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--log-interval",
            "5",
            "--eval-interval",
            "100",
            "--eval-batches",
            "1",
            "--ckpt-interval",
            "100",
        ]
    )
    script.train(load_config(config_path), args, CPU)
    assert [r["step"] for r in _records(out, "train")] == [5, 10, 12]
    assert [r["step"] for r in _records(out, "val")] == [12]  # always evaluated at the end
    assert [p.name for p in list_checkpoints(out / "checkpoints")] == ["step_0000012.pt"]


def test_keep_last_prunes(script: ModuleType, config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    _train(
        script,
        config_path,
        out,
        "--ckpt-interval",
        "2",
        "--keep-last",
        "2",
        "--milestone-every",
        "6",
    )
    assert [p.name for p in list_checkpoints(out / "checkpoints")] == [
        "step_0000006.pt",
        "step_0000010.pt",
        "step_0000012.pt",
    ]


def test_refuses_existing_run_without_resume(
    script: ModuleType, config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    _train(script, config_path, out, "--stop-at-step", "2")
    before = (out / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        _train(script, config_path, out)
    assert (out / "metrics.jsonl").read_bytes() == before


def test_resume_without_checkpoint_raises(
    script: ModuleType, config_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        _train(script, config_path, tmp_path / "empty", "--resume")


def test_resume_with_changed_config_raises(
    script: ModuleType, config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    _train(script, config_path, out, "--stop-at-step", "4")
    raw = config_to_dict(load_config(config_path))
    raw["train"]["lr"] = 1.0e-2
    changed = tmp_path / "changed.yaml"
    save_config(config_from_dict(raw), changed)
    with pytest.raises(ValueError):
        _train(script, changed, out, "--resume")


# --- resume equivalence (ladder (c)) --------------------------------------------------


def test_interrupted_and_resumed_run_matches_straight_run(
    script: ModuleType, config_path: Path, tmp_path: Path
) -> None:
    straight, split = tmp_path / "straight", tmp_path / "split"
    _train(script, config_path, straight)
    first = _train(script, config_path, split, "--stop-at-step", "6")
    assert first["end_step"] == 6
    assert [p.name for p in list_checkpoints(split / "checkpoints")][-1] == "step_0000006.pt"
    torch.rand(5)  # disturb global RNG between the two processes' worth of work
    second = _train(script, config_path, split, "--resume")
    assert second["start_step"] == 6 and second["end_step"] == 12

    def by_step(out: Path, kind: str, key: str) -> dict[int, float]:
        return {r["step"]: r[key] for r in _records(out, kind)}

    assert by_step(split, "train", "loss") == by_step(straight, "train", "loss")
    assert by_step(split, "train", "grad_norm") == by_step(straight, "train", "grad_norm")
    assert by_step(split, "val", "val_loss") == by_step(straight, "val", "val_loss")
    a = torch.load(straight / "checkpoints" / "step_0000012.pt", weights_only=True)["model"]
    b = torch.load(split / "checkpoints" / "step_0000012.pt", weights_only=True)["model"]
    for name in a:
        assert torch.equal(a[name], b[name]), name


# --- ladder (a) and token limit -------------------------------------------------------


def test_overfit_batch_drives_loss_down(
    script: ModuleType, config_path: Path, tmp_path: Path
) -> None:
    raw = config_to_dict(load_config(config_path))
    raw["train"].update(
        {"max_steps": 60, "warmup_steps": 5, "lr": 1.0e-2, "min_lr": 1.0e-2, "weight_decay": 0.0}
    )
    cfg_path = tmp_path / "overfit.yaml"
    save_config(config_from_dict(raw), cfg_path)
    out = tmp_path / "run"
    args = script.parse_args(
        [
            "--config",
            str(cfg_path),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--overfit-batch",
            "--log-interval",
            "1",
            "--eval-interval",
            "1000",
            "--eval-batches",
            "1",
            "--ckpt-interval",
            "1000",
        ]
    )
    script.train(load_config(cfg_path), args, CPU)
    losses = [r["loss"] for r in _records(out, "train")]
    assert losses[0] > 3.5 and losses[-1] < 0.5, (losses[0], losses[-1])


def test_train_token_limit_restricts_data(
    script: ModuleType, config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = script.RandomWindowLoader

    def spy(shards: list[np.ndarray], *a: Any, **k: Any) -> Any:
        seen["tokens"] = sum(len(s) for s in shards)
        return real(shards, *a, **k)

    monkeypatch.setattr(script, "RandomWindowLoader", spy)
    _train(
        script, config_path, tmp_path / "run", "--train-token-limit", "1000", "--stop-at-step", "1"
    )
    assert seen["tokens"] == 1000


# --- main -----------------------------------------------------------------------------


def test_main_prints_summary(
    script: ModuleType, config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code = script.main(
        [
            "--config",
            str(config_path),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--stop-at-step",
            "2",
            "--eval-batches",
            "1",
        ]
    )
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    summary = json.loads(lines[-1])
    assert summary["end_step"] == 2 and summary["out_dir"] == str(out)
