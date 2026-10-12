"""Print text samples from a checkpoint (Gate 4 ladder (d) evidence).

Each prompt is treated as the START of a document: it is normalized like training text
and preceded by ``<|endoftext|>``, since in training every document follows an EOS.

Usage:
    uv run python scripts/sample.py --checkpoint runs/10m-a/checkpoints/step_0009000.pt
    uv run python scripts/sample.py --checkpoint ... --prompt "Once upon a time" \\
        --temperature 0 --out experiments/samples.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from syrinx.config import SyrinxConfig, config_from_dict
from syrinx.data.documents import normalize_text
from syrinx.generate import generate
from syrinx.model import SyrinxLM
from syrinx.tokenizer import EOS_TOKEN, VOCAB_SIZE, decode, encode_document, load_tokenizer
from syrinx.utils.device import resolve_device
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
DEFAULT_PROMPTS = (
    "Once upon a time",
    "Lily and Ben went to the park.",
    "The little dog was sad because",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--checkpoint`` (required, ``Path``), ``--tokenizer``
    (DEFAULT_TOKENIZER), ``--prompt`` (repeatable, ``action="append"``; if none given,
    ``list(DEFAULT_PROMPTS)``), ``--max-new-tokens`` (200), ``--temperature`` (0.8),
    ``--top-k`` (50), ``--seed`` (0), ``--device`` (None), ``--out`` (``Path``, None)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.prompt is None:
        args.prompt = list(DEFAULT_PROMPTS)
    return args


def load_model(path: Path, device: torch.device) -> tuple[SyrinxLM, SyrinxConfig, int]:
    """Rebuild the model from a training checkpoint alone.

    ``torch.load(path, map_location="cpu", weights_only=True)``; config from
    ``config_from_dict(payload["config"])``; ``SyrinxLM(config.model)``;
    ``load_state_dict(payload["model"], strict=True)``; move to ``device``; ``eval()``.
    The optimizer state in the file is ignored.

    Returns:
        ``(model, config, payload["step"])``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = config_from_dict(payload["config"])
    model = SyrinxLM(config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    return model, config, payload["step"]


def sample_texts(
    model: SyrinxLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    device: torch.device,
) -> list[str]:
    """One sample per prompt: prompt text + continuation.

    For prompt ``i``: ids = ``[EOS] + encode_document(tokenizer, normalize_text(prompt),
    append_eos=False)``; ``generate(..., eos_id=EOS, generator=torch.Generator(device=
    device).manual_seed(seed + i))``; keep the new tokens up to (not including) the first
    EOS; return ``decode(tokenizer, prompt_ids_without_the_leading_EOS + new)``.
    """
    eos = tokenizer.token_to_id(EOS_TOKEN)
    texts: list[str] = []
    for i, prompt in enumerate(prompts):
        prompt_ids = encode_document(tokenizer, normalize_text(prompt), append_eos=False)
        ids = torch.tensor([[eos, *prompt_ids]], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(seed + i)
        out = generate(
            model,
            ids,
            max_new_tokens,
            max_seq_len,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos,
            generator=generator,
        )
        new = out[0, ids.shape[1] :].tolist()
        if eos in new:
            new = new[: new.index(eos)]
        texts.append(decode(tokenizer, prompt_ids + new))
    return texts


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; ``resolve_device``; ``load_tokenizer(args.tokenizer,
    VOCAB_SIZE)``; ``load_model``; ``sample_texts`` with ``config.model.max_seq_len``.

    One record per prompt: ``{"checkpoint" (str), "step", "prompt", "text", "temperature",
    "top_k", "seed" (seed + i), "max_new_tokens"}``. If ``--out`` is given, write the
    records as JSON lines to it (parents created; mode ``"x"``, never overwrite;
    ``ensure_ascii=False``). Print each record as one JSON line. Returns 0.
    """
    setup_logging()
    args = parse_args(argv)
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer, VOCAB_SIZE)
    model, config, step = load_model(args.checkpoint, device)
    texts = sample_texts(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        config.model.max_seq_len,
        args.temperature,
        args.top_k,
        args.seed,
        device,
    )
    records: list[dict[str, Any]] = [
        {
            "checkpoint": str(args.checkpoint),
            "step": step,
            "prompt": prompt,
            "text": text,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed + i,
            "max_new_tokens": args.max_new_tokens,
        }
        for i, (prompt, text) in enumerate(zip(args.prompt, texts, strict=True))
    ]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
