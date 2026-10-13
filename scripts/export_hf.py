"""Export a checkpoint as a Hugging Face Llama folder and check logit parity (Phase 9).

Writes ``--out-dir`` (must not exist) with ``syrinx.export.export_checkpoint``, reloads it
with ``AutoModelForCausalLM`` / ``AutoTokenizer`` exactly as a user would, and compares:
logits on random ids (fp32, CPU) and token ids on fixed texts. The result goes to
``<out-dir>/parity.json``; the exit code is 1 if parity fails.

Usage:
    uv run python scripts/export_hf.py --checkpoint runs/10m-a/checkpoints/step_0009000.pt \\
        --out-dir exports/syrinx-10m
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from syrinx.export import PARITY_TOLERANCE, export_checkpoint, max_logit_diff
from syrinx.tokenizer import encode_document, load_tokenizer
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
PARITY_TEXTS = (
    "Once upon a time, there was a little girl named Lily.",
    "The mitochondria is the powerhouse of the cell; ATP is made in 3 steps.",
    "café naïve — “quotes”, emoji 😀, tabs\tand\r\nnewlines\n\n",
    "日本語のテキスト 12345",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--checkpoint`` (required, ``Path``), ``--out-dir``
    (required, ``Path``), ``--tokenizer`` (DEFAULT_TOKENIZER), ``--parity-batch`` (2),
    ``--parity-seq-len`` (int, None = the model's ``max_seq_len``), ``--seed`` (0)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--parity-batch", type=int, default=2)
    parser.add_argument("--parity-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; ``export_checkpoint``; reload with
    ``AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32)`` (eval mode) and
    ``AutoTokenizer.from_pretrained(out_dir)``.

    Logit parity: ids = ``torch.randint(0, vocab_size, (parity_batch, T),
    generator=torch.Generator().manual_seed(seed))`` with ``T = parity_seq_len`` or the
    model's ``max_seq_len``; ``max_logit_diff``. Token parity: for every text in
    ``PARITY_TEXTS``, HF ``tokenizer(text)["input_ids"]`` equals ``encode_document(...,
    append_eos=False)`` and HF decode round-trips the text.

    Writes ``out_dir / "parity.json"`` (indent 2): ``{"checkpoint" (str), "step",
    "out_dir" (str), "transformers_version", "batch" and "seq_len" (the shape of the ids
    actually compared), "max_abs_logit_diff",
    "tolerance" (PARITY_TOLERANCE), "logits_ok", "tokenizer_ok", "passed"}`` and prints
    it as one JSON line.

    Returns:
        0 if ``passed`` (logits within tolerance and tokenizer ok), else 1.
    """
    setup_logging()
    args = parse_args(argv)
    model, step = export_checkpoint(args.checkpoint, args.tokenizer, args.out_dir)
    hf = AutoModelForCausalLM.from_pretrained(args.out_dir, dtype=torch.float32).eval()
    hf_tok = AutoTokenizer.from_pretrained(args.out_dir)
    seq_len = args.parity_seq_len or model.config.max_seq_len
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, model.config.vocab_size, (args.parity_batch, seq_len), generator=gen)
    diff = max_logit_diff(model, hf, ids)
    ours_tok = load_tokenizer(args.tokenizer)
    tokenizer_ok = all(
        hf_tok(text)["input_ids"] == encode_document(ours_tok, text, append_eos=False)
        and hf_tok.decode(hf_tok(text)["input_ids"]) == text
        for text in PARITY_TEXTS
    )
    logits_ok = diff <= PARITY_TOLERANCE
    record: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "out_dir": str(args.out_dir),
        "transformers_version": transformers.__version__,
        "batch": ids.shape[0],
        "seq_len": ids.shape[1],
        "max_abs_logit_diff": diff,
        "tolerance": PARITY_TOLERANCE,
        "logits_ok": logits_ok,
        "tokenizer_ok": tokenizer_ok,
        "passed": logits_ok and tokenizer_ok,
    }
    (args.out_dir / "parity.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
