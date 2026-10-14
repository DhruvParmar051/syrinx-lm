"""Tests for ``scripts/make_model_card.py``."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from syrinx.model_card import file_sha256

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "card_test_helpers", Path(__file__).with_name("test_model_card.py")
)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
make_evidence = _helpers.make_evidence


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_model_card_script", ROOT / "scripts" / "make_model_card.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args(script: ModuleType) -> None:
    args = script.parse_args(["--manifest", "m.yaml"])
    assert args.manifest == Path("m.yaml") and args.card_dir == script.DEFAULT_CARD_DIR
    with pytest.raises(SystemExit):
        script.parse_args([])


def test_main_writes_card_and_copy(script: ModuleType, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    manifest = make_evidence(tmp_path)
    cards = tmp_path / "cards"
    assert script.main(["--manifest", str(manifest), "--card-dir", str(cards)]) == 0
    card, copy = tmp_path / "export" / "README.md", cards / "syrinx-test.md"
    assert card.read_text(encoding="utf-8") == copy.read_text(encoding="utf-8")
    assert card.read_text(encoding="utf-8").startswith("---\nlicense: apache-2.0")
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert json.loads(line[0]) == {
        "manifest": str(manifest),
        "card": str(card),
        "copy": str(copy),
        "sha256": file_sha256(card),
    }


def test_main_never_overwrites_and_writes_nothing_on_conflict(
    script: ModuleType, tmp_path: Path
) -> None:
    manifest = make_evidence(tmp_path)
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "syrinx-test.md").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        script.main(["--manifest", str(manifest), "--card-dir", str(cards)])
    assert not (tmp_path / "export" / "README.md").exists()
    assert (cards / "syrinx-test.md").read_text(encoding="utf-8") == "old"


def test_main_refuses_inconsistent_evidence(script: ModuleType, tmp_path: Path) -> None:
    manifest = make_evidence(tmp_path)
    parity = json.loads((tmp_path / "parity.json").read_text())
    parity["passed"] = False
    (tmp_path / "parity.json").write_text(json.dumps(parity))
    with pytest.raises(ValueError, match="parity"):
        script.main(["--manifest", str(manifest), "--card-dir", str(tmp_path / "cards")])
    assert not (tmp_path / "export" / "README.md").exists() and not (tmp_path / "cards").exists()


def test_main_rejects_bad_manifest(script: ModuleType, tmp_path: Path) -> None:
    manifest = make_evidence(tmp_path)
    raw = yaml.safe_load(manifest.read_text())
    del raw["summary"]
    manifest.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="summary"):
        script.main(["--manifest", str(manifest), "--card-dir", str(tmp_path / "cards")])
