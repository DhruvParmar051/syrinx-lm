"""Byte-level BPE tokenizer for Syrinx (32,768 tokens, HF ``tokenizers``).

Lossless by construction: text -> UTF-8 bytes -> BPE merges -> ids, and decoding
reverses it exactly. No normalizer runs inside the tokenizer; text normalization
(NFC, control characters) belongs to the data pipeline, so the tokenizer never
changes the text it is given.

Special tokens take the first ids: ``<|endoftext|>`` = 0, ``<|pad|>`` = 1.
"""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

logger = logging.getLogger(__name__)

VOCAB_SIZE = 32768
"""Total vocabulary, including the 256 byte symbols and the special tokens."""

EOS_TOKEN = "<|endoftext|>"
"""Document separator; id 0."""

PAD_TOKEN = "<|pad|>"
"""Padding for batched evaluation; id 1. Random-window training never pads."""

SPECIAL_TOKENS: tuple[str, ...] = (EOS_TOKEN, PAD_TOKEN)
"""Special tokens in id order."""


class TokenizerError(ValueError):
    """A tokenizer that doesn't meet the Syrinx contract (size, special ids, settings)."""


def build_tokenizer() -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer with the Syrinx settings.

    Model BPE; pre-tokenizer ``Sequence`` of, in order:
      1. ``Digits(individual_digits=True)``: every digit is its own pre-token, so no
         merge ever joins digits (numbers tokenize consistently, as in Llama).
      2. ``ByteLevel`` with ``add_prefix_space=False`` (the default True prepends a
         space and breaks exact round-trips) and ``use_regex=True`` (GPT-2 pattern).
    Decoder ByteLevel; no normalizer.

    Returns:
        An untrained tokenizer.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
    length: int | None = None,
) -> Tokenizer:
    """Train the BPE merges on ``texts`` (TRAIN split only).

    The trainer gets the full 256-symbol ByteLevel alphabet as its initial alphabet
    and the special tokens first, so they receive ids 0 and 1.

    Args:
        texts: Training documents. Must come from the train split: tokenizer training
            on validation text is leakage.
        vocab_size: Target vocabulary including bytes and special tokens.
        min_frequency: Minimum pair count for a merge.
        length: Number of texts, if known (progress reporting only).

    Returns:
        The trained tokenizer.

    Raises:
        TokenizerError: If the trained vocabulary is not exactly ``vocab_size``
            (the trainer silently stops early when it runs out of merges).
    """
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)
    validate_tokenizer(tokenizer=tokenizer, vocab_size=vocab_size)

    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer, vocab_size: int = VOCAB_SIZE) -> None:
    """Check the Syrinx contract: vocab size and special-token ids.

    Args:
        tokenizer: Tokenizer to check.
        vocab_size: Expected vocabulary size.

    Raises:
        TokenizerError: Naming what is wrong (size, missing/misplaced special token).
    """
    actual = tokenizer.get_vocab_size()
    if actual != vocab_size:
        raise TokenizerError(f"Vocab size is {actual}, expected {vocab_size}.")

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise TokenizerError(f"{token} has id {actual_id}, expected {expected_id}.")


def save_tokenizer(tokenizer: Tokenizer, path: Path, vocab_size: int = VOCAB_SIZE) -> str:
    """Validate and write ``tokenizer.json`` to a new file.

    Args:
        tokenizer: Trained tokenizer.
        path: Destination file; must not exist (parents are created).
        vocab_size: Expected vocabulary size, passed to :func:`validate_tokenizer`.

    Returns:
        The SHA-256 hex digest of the written file (record it in experiments/log.md).

    Raises:
        FileExistsError: If ``path`` exists.
        TokenizerError: If validation fails.
    """
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    validate_tokenizer(tokenizer=tokenizer, vocab_size=vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return file_sha256(path)


def load_tokenizer(path: Path, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Load ``tokenizer.json`` and validate it.

    Args:
        path: File written by :func:`save_tokenizer`.
        vocab_size: Expected vocabulary size.

    Returns:
        The tokenizer.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TokenizerError: If validation fails.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    validate_tokenizer(tokenizer=tokenizer, vocab_size=vocab_size)
    return tokenizer


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_document(tokenizer: Tokenizer, text: str, append_eos: bool = True) -> list[int]:
    """Encode one document's text, treating special-token strings as plain text.

    A literal ``"<|endoftext|>"`` inside a document must NOT become the EOS id, or
    documents can inject fake boundaries. Encode with special-token matching off
    (the ``encode_special_tokens`` setting), then append the real EOS id yourself.
    Restore the setting afterwards so the function has no lasting side effect.

    Args:
        tokenizer: Trained tokenizer.
        text: Document text (already normalized by the data pipeline).
        append_eos: Append the EOS id after the document.

    Returns:
        Token ids; every id is < 65536 so it fits uint16 shards.
    """
    previous = tokenizer.encode_special_tokens
    try:
        tokenizer.encode_special_tokens = True
        ids = tokenizer.encode(text).ids
    finally:
        tokenizer.encode_special_tokens = previous
    if append_eos:
        ids.append(tokenizer.token_to_id(EOS_TOKEN))
    return ids


def encode_documents(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Encode many documents at once (multi-threaded), WITHOUT appending EOS.

    Must return exactly ``[encode_document(tokenizer, t, append_eos=False) for t in
    texts]``, only faster: use ``tokenizer.encode_batch`` with ``encode_special_tokens``
    switched on for the call and restored afterwards (even if encoding raises), exactly
    like :func:`encode_document`. The shard writer appends EOS itself.

    Args:
        tokenizer: Trained tokenizer.
        texts: Document texts (already normalized).

    Returns:
        One id list per document, in input order (``[]`` for no documents).
    """
    previous = tokenizer.encode_special_tokens
    try:
        tokenizer.encode_special_tokens = True
        encodings = tokenizer.encode_batch(texts)
    finally:
        tokenizer.encode_special_tokens = previous
    return [list(encoding.ids) for encoding in encodings]


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """Decode ids to text, KEEPING special tokens (``skip_special_tokens=False``).

    The library default drops them, which would make EOS boundaries invisible in
    shard round-trip checks.

    Args:
        tokenizer: Trained tokenizer.
        ids: Token ids.

    Returns:
        The decoded text.
    """
    return tokenizer.decode(ids, skip_special_tokens=False)


def bytes_per_token(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    """Compression: total UTF-8 bytes / total tokens over ``texts`` (no EOS counted).

    Used for the Phase 1 comparison with GPT-2 and to convert per-token loss into
    bits per byte: ``bpb = loss_nats_per_token / (ln 2 * bytes_per_token)``.

    Args:
        tokenizer: Trained tokenizer.
        texts: Held-out documents (validation split).

    Returns:
        Bytes per token.

    Raises:
        ValueError: If ``texts`` contains no tokens.
    """
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(encode_document(tokenizer=tokenizer, text=text, append_eos=False))

    if total_tokens == 0:
        raise ValueError("No tokens in texts. cannot compute bytes per token.")

    return total_bytes / total_tokens
