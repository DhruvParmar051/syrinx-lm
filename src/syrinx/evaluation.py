"""Full-split evaluation of a trained checkpoint: loss, perplexity and bits per byte.

Bits per byte (bpb) divides the summed loss by the UTF-8 bytes of the predicted text, so it
does not depend on how many bytes the tokenizer packs into a token; it is the only number
here that is comparable across tokenizers. Bytes come from a per-token table, not from
decoding windows, because a window edge can split a multi-byte character.

Conventions: every target token (EOS included) counts in the loss; special tokens (EOS,
PAD) count 0 bytes.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn

from syrinx.config import SyrinxConfig, config_from_dict
from syrinx.model import SyrinxLM
from syrinx.training.step import Batch, autocast_context


def token_byte_lengths(tokenizer: Tokenizer) -> torch.Tensor:
    """UTF-8 bytes each token id stands for, as a (V,) int64 tensor on the CPU.

    Byte-level BPE writes every raw byte as exactly one character of the token string, so a
    normal token's byte count is ``len(tokenizer.id_to_token(id))``. Special tokens
    (``tokenizer.get_added_tokens_decoder()`` entries with ``special=True``) are 0.

    Args:
        tokenizer: Trained byte-level BPE tokenizer.

    Returns:
        (V,) int64 with ``V = tokenizer.get_vocab_size()``.
    """
    vocab = tokenizer.get_vocab_size()
    lengths = [len(tokenizer.id_to_token(i)) for i in range(vocab)]
    for token_id, token in tokenizer.get_added_tokens_decoder().items():
        if token.special:
            lengths[token_id] = 0
    return torch.tensor(lengths, dtype=torch.int64)


@dataclass(frozen=True)
class EvalResult:
    """Sums over every evaluated target token."""

    loss_sum: float
    """Summed negative log-likelihood, in nats."""
    tokens: int
    """Target tokens evaluated."""
    bytes: int
    """UTF-8 bytes of those targets (special tokens count 0)."""
    eos_targets: int
    """Targets equal to ``eos_id`` (in the loss, 0 bytes)."""
    windows: int
    """Sequences (rows) evaluated."""

    @property
    def loss(self) -> float:
        """Mean nats per target token."""
        return self.loss_sum / self.tokens

    @property
    def perplexity(self) -> float:
        """``exp(loss)``."""
        return math.exp(self.loss)

    @property
    def bpb(self) -> float:
        """Bits per byte: ``loss_sum / (ln 2 * bytes)``."""
        return self.loss_sum / (math.log(2) * self.bytes)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    batches: Iterable[Batch],
    byte_lengths: torch.Tensor,
    precision: str,
    eos_id: int,
) -> EvalResult:
    """Sum loss, tokens and bytes over every batch (forward only).

    Per batch ``(x, y)``: logits under ``autocast_context(x.device.type, precision)``;
    per-token NLL ``F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1),
    reduction="sum")``; bytes ``byte_lengths[y].sum()`` (targets, not inputs). Sums are
    accumulated as Python numbers. Runs in ``eval()`` mode and restores the previous mode
    afterwards, even on error.

    Args:
        model: Maps (B, T) ids to (B, T, V) logits.
        batches: ``(x, y)`` pairs, e.g. from ``val_batches``.
        byte_lengths: From :func:`token_byte_lengths`.
        precision: ``"fp32"`` or ``"bf16-autocast"``.
        eos_id: Id counted in ``eos_targets``.

    Returns:
        The sums.

    Raises:
        ValueError: If ``batches`` is empty, or the evaluated targets have 0 bytes.
    """
    was_training = model.training
    model.eval()
    loss_sum, tokens, n_bytes, eos, windows = 0.0, 0, 0, 0, 0
    try:
        for x, y in batches:
            lengths = byte_lengths.to(y.device)
            with autocast_context(x.device.type, precision=precision):
                logits = model(x)
            vocab = logits.shape[-1]
            loss = F.cross_entropy(
                logits.float().reshape(-1, vocab), y.reshape(-1), reduction="sum"
            )
            loss_sum += loss.item()
            tokens += y.numel()
            n_bytes += int(lengths[y].sum().item())
            eos += int((y == eos_id).sum().item())
            windows += y.shape[0]
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("evaluate_split() got no batches.")
    if n_bytes == 0:
        raise ValueError("The evaluated targets have 0 bytes; cannot compute bits per byte.")
    return EvalResult(
        loss_sum=loss_sum, tokens=tokens, bytes=n_bytes, eos_targets=eos, windows=windows
    )


def load_model(path: Path, device: torch.device) -> tuple[SyrinxLM, SyrinxConfig, int]:
    """Rebuild a model from a training checkpoint alone (as ``scripts/sample.py`` does).

    ``torch.load(path, map_location="cpu", weights_only=True)``; ``config_from_dict(
    payload["config"])``; ``SyrinxLM(config.model)``; strict ``load_state_dict(
    payload["model"])``; move to ``device``; ``eval()``.

    Returns:
        ``(model, config, payload["step"])``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = config_from_dict(payload["config"])
    model = SyrinxLM(config=config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device)
    model.eval()
    return model, config, payload["step"]
