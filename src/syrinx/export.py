"""Export a Syrinx checkpoint as a Hugging Face ``LlamaForCausalLM`` folder (no remote code).

``SyrinxLM`` already uses ``LlamaModel``'s module names, so the weights only gain the
``model.`` prefix; the output head is tied to the embedding and is not stored. The config
must set the values HF would otherwise default differently: ``rms_norm_eps`` (HF 1e-6, ours
1e-5) and the special token ids (HF bos 1 / eos 2; ours EOS 0, PAD 1).

Decisions (docs/decisions.md, 2026-10-13): weights in fp32; ``<|endoftext|>`` is BOS and EOS
but is not prepended automatically; literal special-token strings in text follow HF's
default (they become special tokens).
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from syrinx.config import ModelConfig
from syrinx.data.shards import EOS_ID
from syrinx.evaluation import load_model
from syrinx.model import SyrinxLM
from syrinx.tokenizer import EOS_TOKEN, PAD_TOKEN, load_tokenizer

PAD_ID = 1
"""Id of ``<|pad|>`` (reserved, unused in training)."""
PARITY_TOLERANCE = 1e-4
"""Max abs logit difference allowed between Syrinx and HF (fp32, CPU)."""
HEAD_KEY = "lm_head.weight"


def llama_config(model: ModelConfig) -> LlamaConfig:
    """The ``LlamaConfig`` equivalent of ``model``.

    Shapes from ``model``; ``num_key_value_heads = n_heads`` (MHA);
    ``max_position_embeddings = max_seq_len``; ``rms_norm_eps = norm_eps``;
    ``rope_theta = rope_theta``; ``hidden_act="silu"``; ``attention_bias=False``,
    ``mlp_bias=False``; ``tie_word_embeddings = tie_embeddings``;
    ``bos_token_id = eos_token_id = EOS_ID``; ``pad_token_id = PAD_ID``.
    """
    return LlamaConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        num_hidden_layers=model.n_layers,
        num_attention_heads=model.n_heads,
        num_key_value_heads=model.n_heads,
        intermediate_size=model.d_ff,
        max_position_embeddings=model.max_seq_len,
        rms_norm_eps=model.norm_eps,
        rope_theta=model.rope_theta,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=model.tie_embeddings,
        bos_token_id=EOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )


def to_hf_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename Syrinx weights to ``LlamaForCausalLM`` names.

    Every key gains the ``model.`` prefix except ``lm_head.weight`` (present only when the
    head is untied), which keeps its name. Tensors are detached and made contiguous.

    Raises:
        ValueError: If a key already starts with ``model.``.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("model."):
            raise ValueError(f"unexpected key {key!r}: already has the 'model.' prefix")
        name = key if key == HEAD_KEY else f"model.{key}"
        out[name] = tensor.detach().contiguous()
    return out


def to_hf_model(model: SyrinxLM, config: ModelConfig) -> LlamaForCausalLM:
    """A ``LlamaForCausalLM`` (fp32, CPU, eval mode) holding ``model``'s weights.

    Loads :func:`to_hf_state_dict` with ``strict=False`` and then checks: no unexpected
    keys; missing keys are exactly ``["lm_head.weight"]`` when tied (none when untied);
    when tied, ``lm_head.weight`` is the same tensor as ``model.embed_tokens.weight``.

    Raises:
        ValueError: If the keys do not line up or the head is not tied as configured.
    """
    hf = LlamaForCausalLM(llama_config(config)).to(dtype=torch.float32)
    weights = {
        k: v.to(device="cpu", dtype=torch.float32)
        for k, v in to_hf_state_dict(model.state_dict()).items()
    }
    result = hf.load_state_dict(weights, strict=False)
    expected_missing = [HEAD_KEY] if config.tie_embeddings else []
    if list(result.unexpected_keys) or sorted(result.missing_keys) != expected_missing:
        raise ValueError(
            f"key mismatch: missing {sorted(result.missing_keys)}, "
            f"unexpected {sorted(result.unexpected_keys)}"
        )
    if config.tie_embeddings:
        hf.tie_weights()
        if hf.lm_head.weight.data_ptr() != hf.model.embed_tokens.weight.data_ptr():
            raise ValueError("lm_head is not tied to embed_tokens")
    return hf.eval()


def hf_tokenizer(path: Path, max_seq_len: int) -> PreTrainedTokenizerFast:
    """Our tokenizer as a ``PreTrainedTokenizerFast``.

    ``tokenizer_object = load_tokenizer(path)``; ``bos_token = eos_token = EOS_TOKEN``;
    ``pad_token = PAD_TOKEN``; ``model_max_length = max_seq_len``;
    ``clean_up_tokenization_spaces=False`` (lossless decode). No BOS is added on encode.
    """
    tokenizer: Tokenizer = load_tokenizer(path=path)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=max_seq_len,
        clean_up_tokenization_spaces=False,
    )


def generation_config(max_seq_len: int) -> GenerationConfig:
    """Sampling defaults matching ``scripts/sample.py``: ``do_sample=True``,
    ``temperature=0.8``, ``top_k=50``, ``max_length=max_seq_len``; ``bos_token_id =
    eos_token_id = EOS_ID``; ``pad_token_id = PAD_ID``."""
    return GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=50,
        max_length=max_seq_len,
        bos_token_id=EOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )


@torch.no_grad()
def max_logit_diff(ours: torch.nn.Module, hf: LlamaForCausalLM, ids: torch.Tensor) -> float:
    """Max abs difference between ``ours(ids)`` and ``hf(input_ids=ids).logits`` (both
    fp32, on ``ids.device``; ``ids`` is (B, T) int64)."""
    a = ours(ids).float()
    b = hf(input_ids=ids).logits.float()
    return float((a - b).abs().max())


def export_checkpoint(
    checkpoint: Path, tokenizer_path: Path, out_dir: Path
) -> tuple[SyrinxLM, int]:
    """Write ``out_dir`` as an HF folder: ``config.json`` + ``model.safetensors`` (fp32,
    ``save_pretrained``), the tokenizer files and ``generation_config.json``.

    Loads with ``syrinx.evaluation.load_model`` on the CPU.

    Returns:
        ``(our model on CPU in eval mode, checkpoint step)`` for the caller's parity check.

    Raises:
        FileExistsError: If ``out_dir`` exists.
    """
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; choose a new --out-dir")
    model, config, step = load_model(checkpoint, torch.device("cpu"))
    hf = to_hf_model(model=model, config=config.model)
    out_dir.mkdir(parents=True)
    hf.save_pretrained(out_dir, safe_serialization=True)
    hf_tokenizer(path=tokenizer_path, max_seq_len=config.model.max_seq_len).save_pretrained(out_dir)
    generation_config(config.model.max_seq_len).save_pretrained(out_dir)
    return model, step
