"""Evaluate a checkpoint on a whole validation split: loss, perplexity, bits per byte.

Windows are ``val_batches``' non-overlapping ``seq_len`` windows (the tail of each shard
shorter than one window is not evaluated), so early positions in a window have short
context; report the numbers as "non-overlapping, ctx <seq_len>".

Usage:
    uv run python scripts/eval_checkpoint.py --checkpoint runs/10m-a/checkpoints/step_0009000.pt
    uv run python scripts/eval_checkpoint.py --checkpoint ... --precision fp32 \\
        --out experiments/eval-10m-a-step9000.json
"""

import argparse
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

from syrinx.data.loader import load_split, val_batches
from syrinx.data.shards import EOS_ID
from syrinx.evaluation import evaluate_split, load_model, token_byte_lengths
from syrinx.tokenizer import VOCAB_SIZE, file_sha256, load_tokenizer
from syrinx.utils.device import resolve_device
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--checkpoint`` (required, ``Path``), ``--tokenizer``
    (DEFAULT_TOKENIZER), ``--val-path`` (``Path``, None = the checkpoint config's
    ``data.val_path``), ``--batch-size`` (int, None = the config's ``micro_batch_size``),
    ``--precision`` (choices ``fp32``/``bf16-autocast``, None = the config's),
    ``--max-batches`` (int, None = all), ``--device`` (None), ``--out`` (``Path``, None)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16-autocast"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; ``resolve_device(args.device)``; ``load_model``;
    tokenizer via ``load_tokenizer(args.tokenizer, VOCAB_SIZE)``; resolve ``val_path``,
    ``batch_size`` and ``precision`` (argument, else the checkpoint config); batches from
    ``val_batches(load_split(val_path, "val"), batch_size, config.data.seq_len, device)``,
    cut to ``--max-batches`` with ``itertools.islice`` if given; ``evaluate_split`` with
    ``token_byte_lengths(tokenizer)`` and ``EOS_ID``; time it with ``time.perf_counter``.

    One record: ``{"checkpoint" (str), "step", "val_path" (str), "seq_len", "batch_size",
    "precision", "max_batches", "windows", "tokens", "bytes", "eos_targets", "loss",
    "perplexity", "bpb", "seconds", "tokenizer_sha256"}``. If ``--out`` is given, write it
    as indented JSON (parents created; mode ``"x"``, never overwrite). Print it as ONE JSON
    line. Returns 0.
    """
    setup_logging()
    args = parse_args(argv=argv)
    device = resolve_device(args.device)
    model, config, step = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer, VOCAB_SIZE)
    val_path = args.val_path if args.val_path is not None else Path(config.data.val_path)
    batch_size = args.batch_size if args.batch_size is not None else config.train.micro_batch_size
    precision = args.precision if args.precision is not None else config.train.precision
    batches = val_batches(
        load_split(val_path, "val"),
        batch_size=batch_size,
        seq_len=config.data.seq_len,
        device=device,
    )
    if args.max_batches is not None:
        batches = itertools.islice(batches, args.max_batches)
    start = time.perf_counter()
    result = evaluate_split(
        model,
        batches=batches,
        byte_lengths=token_byte_lengths(tokenizer=tokenizer),
        precision=precision,
        eos_id=EOS_ID,
    )
    seconds = time.perf_counter() - start
    record: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "val_path": str(val_path),
        "seq_len": config.data.seq_len,
        "batch_size": batch_size,
        "precision": precision,
        "max_batches": args.max_batches,
        "windows": result.windows,
        "tokens": result.tokens,
        "bytes": result.bytes,
        "eos_targets": result.eos_targets,
        "loss": result.loss,
        "perplexity": result.perplexity,
        "bpb": result.bpb,
        "seconds": seconds,
        "tokenizer_sha256": file_sha256(args.tokenizer),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
