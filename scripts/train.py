"""Train a Syrinx model from a YAML config (Phase 4+).

Writes into ``--out-dir``: ``config.yaml`` (snapshot), ``metrics.jsonl`` (train and val
records) and ``checkpoints/step_NNNNNNN.pt``. ``--resume`` continues from the newest
checkpoint there and must reproduce the uninterrupted run exactly.

Usage:
    uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/10m-a
    uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/10m-a --resume
    uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/overfit \\
        --overfit-batch                                   # ladder (a)
    uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/memorize \\
        --train-token-limit 2.5e5                         # ladder (b), ~1 MB of text
"""

import argparse
import itertools
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from syrinx.config import SyrinxConfig, config_to_dict, load_config, save_config
from syrinx.data.loader import RandomWindowLoader, load_split, val_batches
from syrinx.model import SyrinxLM
from syrinx.training.checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
)
from syrinx.training.optim import build_optimizer, lr_at, set_lr
from syrinx.training.step import Batch, evaluate, train_step
from syrinx.utils.device import memory_stats, reset_peak_memory, resolve_device
from syrinx.utils.log import setup_logging
from syrinx.utils.metrics import JsonlWriter
from syrinx.utils.seed import seed_everything
from syrinx.utils.timing import Timer

logger = logging.getLogger(__name__)

METRICS_FILE = "metrics.jsonl"
CONFIG_FILE = "config.yaml"
CHECKPOINT_DIR = "checkpoints"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Options and defaults: ``--config`` (required, ``Path``), ``--out-dir`` (required,
    ``Path``), ``--device`` (None = ``resolve_device()`` default), ``--resume`` (flag),
    ``--stop-at-step`` (int, default None: stop after this many optimizer steps without
    changing the LR schedule, to simulate an interruption), ``--log-interval`` (10),
    ``--eval-interval`` (250), ``--eval-batches`` (20), ``--ckpt-interval`` (500),
    ``--keep-last`` (3), ``--milestone-every`` (int, default None), ``--overfit-batch``
    (flag), ``--train-token-limit`` (default None; accepts ``2.5e5``, stored as int).

    The five interval/count options must be positive: ``parser.error(...)`` otherwise
    (exits with status 2).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--milestone-every", type=int, default=None)
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument("--train-token-limit", type=lambda s: int(float(s)), default=None)
    args = parser.parse_args(argv)
    for name in ("log_interval", "eval_interval", "eval_batches", "ckpt_interval", "keep_last"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive, got {getattr(args, name)}")
    return args


def limit_tokens(shards: Sequence[np.ndarray], limit: int | None) -> list[np.ndarray]:
    """The first ``limit`` tokens of the concatenated shards, as views (``s[:k]``), in
    order; shards past the limit are dropped. ``None`` returns ``list(shards)``.

    Raises:
        ValueError: If ``limit`` is not None and not positive.
    """
    if limit is None:
        return list(shards)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}.")
    limited: list[np.ndarray] = []
    remaining = limit
    for shard in shards:
        if remaining <= 0:
            break
        part = shard[:remaining]
        limited.append(part)
        remaining -= len(part)
    return limited


def make_batch_fn(loader: RandomWindowLoader, overfit: bool) -> Callable[[], Batch]:
    """``loader.next_batch``, or with ``overfit`` a function that always returns the
    loader's first batch (drawn once, here)."""
    if not overfit:
        return loader.next_batch
    first = loader.next_batch()

    def same_batch() -> Batch:
        return first

    return same_batch


def train(config: SyrinxConfig, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Run (or resume) training; return a summary.

    Setup:
      1. Fresh run: raise FileExistsError if ``out_dir`` already has ``metrics.jsonl`` or a
         checkpoint; else create ``out_dir`` and ``save_config`` to ``CONFIG_FILE``.
         Resume: ``latest_checkpoint(out_dir / CHECKPOINT_DIR)``; FileNotFoundError if None.
      2. ``seed_everything(train.seed)``; model ``SyrinxLM(config.model).to(device)``;
         ``build_optimizer``; train shards ``limit_tokens(load_split(train_path, "train"),
         args.train_token_limit)``; val shards ``load_split(val_path, "val")``; loader
         ``RandomWindowLoader(train_shards, micro_batch_size, seq_len, train.seed, device)``.
      3. Resume only: ``load_checkpoint``; ValueError if its config differs from
         ``config_to_dict(config)``; ``loader.load_state_dict``; ``restore_rng_state``
         (last, after everything is built); start at ``checkpoint.step`` (else 0).
      4. ``get_batch = make_batch_fn(loader, args.overfit_batch)``;
         ``end = max_steps``, or ``min(max_steps, stop_at_step)``; ``reset_peak_memory``.

    Loop, ``JsonlWriter(metrics, mode="a" if resume else "x")``, for ``step`` in
    ``range(start, end)`` with ``done = step + 1``:
      - ``lr = lr_at(step, ...)``; ``set_lr``; ``train_step`` inside ``Timer(device)``.
      - Train record if ``done % log_interval == 0`` or ``done == end``: ``{"kind":
        "train", "step": done, "loss", "lr", "grad_norm", "tokens_per_sec" (stats.tokens /
        elapsed), "tokens_seen" (done * config.tokens_per_step), "step_time_s",
        "mem_allocated", "mem_peak"}`` (memory from ``memory_stats(device)``).
      - Val record if ``done % eval_interval == 0`` or ``done == max_steps``: ``{"kind":
        "val", "step": done, "val_loss": evaluate(model, first eval_batches of
        val_batches(val_shards, micro_batch_size, seq_len, device), precision)}``.
      - Checkpoint if ``done % ckpt_interval == 0`` or ``done == end``:
        ``save_checkpoint(checkpoint_path(...), model, optimizer, done,
        loader.state_dict(), config_to_dict(config))``, then ``prune_checkpoints(...,
        keep_last, milestone_every)``.

    Returns:
        ``{"out_dir": str, "start_step", "end_step", "train_loss" (last train record's
        loss, or None), "val_loss" (last val record's, or None)}``.
    """
    out_dir = args.out_dir
    metrics_path = out_dir / METRICS_FILE
    ckpt_dir = out_dir / CHECKPOINT_DIR
    t = config.train

    if args.resume:
        latest = latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(f"--resume: no checkpoint in {ckpt_dir}.")
    else:
        if metrics_path.exists() or latest_checkpoint(ckpt_dir) is not None:
            raise FileExistsError(
                f"{out_dir} already holds a run; pass --resume or a new --out-dir."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, out_dir / CONFIG_FILE)

    seed_everything(t.seed)
    model = SyrinxLM(config.model).to(device)
    optimizer = build_optimizer(model, t)
    train_shards = limit_tokens(load_split(config.data.train_path, "train"), args.train_token_limit)
    val_shards = load_split(config.data.val_path, "val")
    loader = RandomWindowLoader(
        train_shards, t.micro_batch_size, config.data.seq_len, t.seed, device
    )

    start = 0
    if args.resume:
        checkpoint = load_checkpoint(latest, model, optimizer)
        if checkpoint.config != config_to_dict(config):
            raise ValueError(f"{latest} was saved with a different config; refusing to resume.")
        loader.load_state_dict(checkpoint.loader_state)
        restore_rng_state(checkpoint.rng_state)
        start = checkpoint.step

    get_batch = make_batch_fn(loader, args.overfit_batch)
    end = t.max_steps if args.stop_at_step is None else min(t.max_steps, args.stop_at_step)
    last: dict[str, dict[str, Any]] = {}
    reset_peak_memory(device)

    with JsonlWriter(metrics_path, mode="a" if args.resume else "x") as writer:
        for step in range(start, end):
            done = step + 1
            lr = lr_at(step, t.lr, t.min_lr, t.warmup_steps, t.max_steps)
            set_lr(optimizer, lr)
            with Timer(device) as timer:
                stats = train_step(
                    model, optimizer, get_batch, t.grad_accum_steps, t.grad_clip, t.precision
                )

            if done % args.log_interval == 0 or done == end:
                memory = memory_stats(device)
                record = {
                    "kind": "train",
                    "step": done,
                    "loss": stats.loss,
                    "lr": lr,
                    "grad_norm": stats.grad_norm,
                    "tokens_per_sec": stats.tokens / timer.elapsed,
                    "tokens_seen": done * config.tokens_per_step,
                    "step_time_s": timer.elapsed,
                    "mem_allocated": memory.allocated,
                    "mem_peak": memory.peak_allocated,
                }
                writer.write(record)
                last["train"] = record
                logger.info("step %d loss %.4f lr %.2e", done, stats.loss, lr)

            if done % args.eval_interval == 0 or done == t.max_steps:
                batches = val_batches(val_shards, t.micro_batch_size, config.data.seq_len, device)
                val_loss = evaluate(
                    model, itertools.islice(batches, args.eval_batches), t.precision
                )
                record = {"kind": "val", "step": done, "val_loss": val_loss}
                writer.write(record)
                last["val"] = record
                logger.info("step %d val_loss %.4f", done, val_loss)

            if done % args.ckpt_interval == 0 or done == end:
                save_checkpoint(
                    checkpoint_path(ckpt_dir, done),
                    model,
                    optimizer,
                    done,
                    loader.state_dict(),
                    config_to_dict(config),
                )
                prune_checkpoints(ckpt_dir, args.keep_last, args.milestone_every)

    return {
        "out_dir": str(out_dir),
        "start_step": start,
        "end_step": end,
        "train_loss": last.get("train", {}).get("loss"),
        "val_loss": last.get("val", {}).get("val_loss"),
    }


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; ``load_config``; ``resolve_device(args.device)``;
    ``train``; print the summary as one JSON line (sort_keys). Returns 0. Call
    ``train`` by its module-level name (tests replace it)."""
    setup_logging()
    args = parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(args.device)
    summary = train(config, args, device)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
