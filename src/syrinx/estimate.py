"""Analytic training memory and compute estimates.

Estimates are made BEFORE a run and compared to measured values AFTER it
(``experiments/log.md`` records the gap). Sizes are in bytes.

Memory model (PLAN.md §1): fp32 weights + grads + AdamW moments (16 B/param),
bf16 autocast weight copies (2 B/param, bf16 only), activations saved for backward
assuming unfused attention, and logits with their loss temporaries; the sum is
multiplied by an overhead factor for allocator slack and short-lived buffers.
These are approximations: the first proxy run measured 20.1 GiB driver-allocated
where this model predicts ~16.4 GiB (EXP-000).
"""

import logging
from dataclasses import dataclass

from syrinx.config import ModelConfig, num_parameters

logger = logging.getLogger(__name__)

DEFAULT_OVERHEAD = 1.25
"""Multiplier on the component sum for fragmentation and temporaries."""

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class MemoryEstimate:
    """Breakdown of estimated peak training memory, in bytes.

    Attributes:
        params: fp32 weights.
        grads: fp32 gradients.
        optimizer: AdamW first and second moments (fp32).
        autocast_copies: bf16 weight copies made by autocast (0 for fp32).
        activations: Activations saved for backward across all layers.
        logits: Logits plus the cross-entropy temporaries.
        overhead_factor: Multiplier applied to the component sum.
    """

    params: int
    grads: int
    optimizer: int
    autocast_copies: int
    activations: int
    logits: int
    overhead_factor: float

    @property
    def components(self) -> dict[str, int]:
        """The byte components by name, in display order."""
        return {
            "params": self.params,
            "grads": self.grads,
            "optimizer": self.optimizer,
            "autocast_copies": self.autocast_copies,
            "activations": self.activations,
            "logits": self.logits,
        }

    @property
    def total(self) -> int:
        """Sum of components times ``overhead_factor``, rounded to whole bytes."""
        return round(sum(self.components.values()) * self.overhead_factor)


def _activation_bytes_per_element(precision: str) -> int:
    """Bytes per saved activation: 4 for ``"fp32"``, 2 for ``"bf16-autocast"``.

    Raises:
        ValueError: If ``precision`` is not recognized.
    """
    if precision == "fp32":
        return 4
    if precision == "bf16-autocast":
        return 2
    raise ValueError(f"Unknown precision {precision!r}; expected 'fp32' or 'bf16-autocast'.")


def parameter_state_bytes(num_params: int, precision: str) -> dict[str, int]:
    """Bytes for weights, grads, optimizer moments and autocast copies.

    Args:
        num_params: Total parameter count N.
        precision: ``"fp32"`` or ``"bf16-autocast"``.

    Returns:
        ``params`` = 4N, ``grads`` = 4N, ``optimizer`` = 8N (two fp32 moments),
        ``autocast_copies`` = 2N under bf16 autocast, else 0.

    Raises:
        ValueError: If ``precision`` is not recognized or ``num_params`` is negative.
    """
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}.")
    _activation_bytes_per_element(precision)  # validates precision
    return {
        "params": 4 * num_params,
        "grads": 4 * num_params,
        "optimizer": 8 * num_params,
        "autocast_copies": 2 * num_params if precision == "bf16-autocast" else 0,
    }


def _per_layer_activation_elements(model: ModelConfig, micro_batch_size: int, seq_len: int) -> int:
    """Elements saved for backward by ONE block, assuming unfused attention.

    Per token (C = d_model, F = d_ff, H = n_heads, T = seq_len):
      8C  = attn-norm input + output, q, k, v, attention output (pre o-proj),
            mlp-norm input + output
      4F  = gate, up, silu(gate), gate*up (input to the down projection)
      2HT = attention scores and softmax probabilities, (B, H, T, T) each
    """
    c, f, h = model.d_model, model.d_ff, model.n_heads
    per_token = 8 * c + 4 * f + 2 * h * seq_len
    return micro_batch_size * seq_len * per_token


def activation_bytes(
    model: ModelConfig, micro_batch_size: int, seq_len: int, precision: str
) -> int:
    """Activation memory for all blocks plus the embedding output and final norm.

    Args:
        model: Architecture config.
        micro_batch_size: B.
        seq_len: T.
        precision: Determines bytes per element.

    Returns:
        Estimated bytes.
    """
    per_element = _activation_bytes_per_element(precision)
    blocks = model.n_layers * _per_layer_activation_elements(model, micro_batch_size, seq_len)
    outside_blocks = micro_batch_size * seq_len * 2 * model.d_model
    return (blocks + outside_blocks) * per_element


def logits_bytes(vocab_size: int, micro_batch_size: int, seq_len: int, precision: str) -> int:
    """Memory for (B, T, V) logits and the cross-entropy temporaries.

    Per logit element: the logits themselves (2 B bf16 / 4 B fp32), the fp32 upcast
    for the loss (4 B, bf16 only), the fp32 log-softmax (4 B) and the fp32 gradient
    of the logits (4 B): 14 B under bf16 autocast, 12 B in fp32.

    Args:
        vocab_size: V.
        micro_batch_size: B.
        seq_len: T.
        precision: ``"fp32"`` or ``"bf16-autocast"``.

    Returns:
        Estimated bytes.
    """
    logits = _activation_bytes_per_element(precision)
    upcast = 4 if precision == "bf16-autocast" else 0
    per_element = logits + upcast + 4 + 4
    return micro_batch_size * seq_len * vocab_size * per_element


def estimate_training_memory(
    model: ModelConfig,
    micro_batch_size: int,
    seq_len: int,
    precision: str,
    overhead_factor: float = DEFAULT_OVERHEAD,
) -> MemoryEstimate:
    """Estimate peak training memory for one micro-batch step.

    Args:
        model: Architecture config.
        micro_batch_size: B.
        seq_len: T.
        precision: ``"fp32"`` or ``"bf16-autocast"``.
        overhead_factor: Multiplier on the component sum; must be >= 1.

    Returns:
        The component breakdown and total.

    Raises:
        ValueError: If a size is not positive or ``overhead_factor < 1``.
    """
    if micro_batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            f"micro_batch_size and seq_len must be positive, got {micro_batch_size}, {seq_len}."
        )
    if overhead_factor < 1:
        raise ValueError(f"overhead_factor must be >= 1, got {overhead_factor}.")
    state = parameter_state_bytes(num_parameters(model), precision)
    return MemoryEstimate(
        params=state["params"],
        grads=state["grads"],
        optimizer=state["optimizer"],
        autocast_copies=state["autocast_copies"],
        activations=activation_bytes(model, micro_batch_size, seq_len, precision),
        logits=logits_bytes(model.vocab_size, micro_batch_size, seq_len, precision),
        overhead_factor=overhead_factor,
    )


def flops_per_token(model: ModelConfig, seq_len: int) -> float:
    """Training FLOPs per token: ``6 * N + 12 * L * T * d_model``.

    ``6N`` covers forward (2N) and backward (4N) of every weight matmul, using the
    total N so the tied output projection (a real V x d matmul) is counted once.
    ``12 * L * T * d`` covers the QK^T and attention-weighted-sum matmuls
    (2 * 2 * T * d per token per layer forward, times 3 for backward), as in the
    PaLM appendix. This reproduces PLAN.md's 2.15e18 FLOPs for 125M on 2.5B tokens.

    Args:
        model: Architecture config.
        seq_len: Context length T.

    Returns:
        FLOPs per trained token.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    return 6.0 * num_parameters(model) + 12.0 * model.n_layers * seq_len * model.d_model


def estimate_train_days(total_tokens: float, tokens_per_sec: float) -> float:
    """Wall-clock days to train ``total_tokens`` at a sustained throughput.

    Args:
        total_tokens: Tokens to train on.
        tokens_per_sec: Measured sustained throughput.

    Returns:
        Days.

    Raises:
        ValueError: If ``tokens_per_sec`` is not positive or ``total_tokens`` is negative.
    """
    if tokens_per_sec <= 0:
        raise ValueError(f"tokens_per_sec must be positive, got {tokens_per_sec}.")
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}.")
    return total_tokens / tokens_per_sec / SECONDS_PER_DAY
