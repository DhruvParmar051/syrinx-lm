"""Write a model card from a manifest and its evidence (Phase 10).

Loads ``configs/cards/<name>.yaml``, checks that all evidence describes the same model, renders the
card and writes it twice: ``<export_dir>/README.md`` (shipped with the weights) and
``<card_dir>/<name>.md`` (committed, so the text is reviewable). Neither file is ever overwritten.

Usage:
    uv run python scripts/make_model_card.py --manifest configs/cards/syrinx-10m.yaml
"""

import argparse
import json
import logging
from pathlib import Path

from syrinx.model_card import (
    check_consistency,
    file_sha256,
    load_evidence,
    load_manifest,
    render_card,
)
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CARD_DIR = Path("docs/model-cards")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--manifest`` (required, ``Path``), ``--card-dir``
    (DEFAULT_CARD_DIR)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--card-dir", type=Path, default=DEFAULT_CARD_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; ``load_manifest``; ``load_evidence``; ``check_consistency``
    (before anything is written); ``render_card``. Refuse (``FileExistsError``) if either target
    exists, checking both before writing either; then write ``export_dir / "README.md"`` and
    ``card_dir / f"{name}.md"`` (parents created; mode ``"x"``). Print ONE JSON line
    ``{"manifest", "card", "copy", "sha256"}`` (paths as str, sha256 of the card text file). Returns 0.
    """
    setup_logging()
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    evidence = load_evidence(manifest)
    check_consistency(manifest, evidence)
    text = render_card(manifest, evidence)
    card = manifest.export_dir / "README.md"
    copy = args.card_dir / f"{manifest.name}.md"
    for target in (card, copy):
        if target.exists():
            raise FileExistsError(f"{target} exists; remove it first to regenerate the card")
    copy.parent.mkdir(parents=True, exist_ok=True)
    for target in (card, copy):
        with target.open("x", encoding="utf-8") as fh:
            fh.write(text)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "card": str(card),
                "copy": str(copy),
                "sha256": file_sha256(card),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
