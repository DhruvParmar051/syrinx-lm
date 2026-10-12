"""Autoregressive sampling from a trained model (greedy, temperature, top-k).

No KV cache: every new token re-runs the model on the (cropped) context. At 10M-125M and
a few hundred tokens that is fast enough for samples, and it reuses the exact forward
pass the model was trained and tested with.
"""

import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend ``prompt`` by up to ``max_new_tokens`` tokens.

    Each step: run the model on the last ``max_seq_len`` tokens (context crop, so prompts
    and outputs may be longer than the context), take the LAST position's logits (B, V)
    in float32, then pick the next token:

    - ``temperature == 0``: greedy, ``argmax``.
    - else: divide logits by ``temperature``; if ``top_k`` is set, set every logit below
      the k-th largest to ``-inf`` (k capped at V); softmax; one draw with
      ``torch.multinomial(probs, 1, generator=generator)``.

    With ``eos_id``: once a row has produced ``eos_id``, every later token in that row is
    ``eos_id``; stop early when every row has produced it. Runs in ``eval()`` mode and
    restores the previous mode afterwards, even on error.

    Args:
        model: Maps (B, T) int64 ids to (B, T, V) logits.
        prompt: (B, T0) int64, ``T0 >= 1``, on the model's device.
        max_new_tokens: Tokens to add; ``>= 0``.
        max_seq_len: Context length of the model.
        temperature: ``>= 0``.
        top_k: None, or ``>= 1``.
        eos_id: Stop token, or None.
        generator: RNG for sampling (on the prompt's device), for reproducible samples.

    Returns:
        (B, T0 + n) int64 with ``n <= max_new_tokens``; the first T0 columns are ``prompt``.

    Raises:
        ValueError: If ``prompt`` is not 2-D or empty, ``max_new_tokens < 0``,
            ``temperature < 0``, or ``top_k < 1``.
    """
    if prompt.ndim != 2 or prompt.numel() == 0:
        raise ValueError(
            f"prompt must be a non-empty (B, T) tensor, got shape {tuple(prompt.shape)}"
        )
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")

    was_training = model.training
    model.eval()
    ids = prompt
    done = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -max_seq_len:])[:, -1, :].float()
            if temperature == 0:
                next_ids = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            if eos_id is not None:
                next_ids = torch.where(done, torch.full_like(next_ids, eos_id), next_ids)
                done = done | (next_ids == eos_id)
            ids = torch.cat([ids, next_ids.unsqueeze(-1)], dim=1)
            if eos_id is not None and bool(done.all()):
                break
    finally:
        model.train(was_training)
    return ids
