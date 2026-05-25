"""Unit tests for parallel lexicon parsing, Unicode tokenization, and spelling variants."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_lexicon import (  # noqa: E402
    expand_lexicon_variants,
    split_lexicon_cell,
)
from src.political_lexicon import tokenize  # noqa: E402
from src.v4_lexicon import (  # noqa: E402
    get_pairs_registry,
    load_pairs_from_v4_csv,
    score_pair_framing,
)


def test_split_lexicon_cell_semicolons() -> None:
    """Function summary: semicolon-separated cells split into lemmas."""
    assert split_lexicon_cell("a; b; c") == ["a", "b", "c"]
    assert split_lexicon_cell("forza italia") == ["forza italia"]


def test_tokenize_unicode_umlaut() -> None:
    """Function summary: Unicode letters stay intact in tokens."""
    assert tokenize("Überfremdung") == ["überfremdung"]
    assert tokenize("perché perché") == ["perché", "perché"]


def test_german_variant_expansion() -> None:
    """Function summary: ue/ä alternates generated for German."""
    variants = expand_lexicon_variants("ueberfremdung", "de")
    assert "ueberfremdung" in variants
    assert "überfremdung" in variants


def test_italian_accent_variant() -> None:
    """Function summary: accented and plain Italian vowels both match."""
    variants = expand_lexicon_variants("perché", "it")
    assert "perché" in variants
    assert "perche" in variants


def test_v4_pairs_load_from_csv() -> None:
    """Function summary: pairs section loads from italian_political_lexicon_v4.csv."""
    path = ROOT / "data/raw/italian_political_lexicon_v4.csv"
    pairs = load_pairs_from_v4_csv(path)
    assert len(pairs) >= 60
    assert any(p.term_a and p.term_b for p in pairs)


def test_pair_framing_left_only() -> None:
    """Function summary: only left-pole term contributes +1 net."""
    from src.v4_lexicon import PairEntry

    pairs = [
        PairEntry(
            pair_id="m1",
            topic="migration",
            term_a="migranti",
            term_b="clandestini",
            pole_a="left",
            pole_b="right",
            polarized="yes",
            tokens_a=("migranti",),
            tokens_b=("clandestini",),
        )
    ]
    out = score_pair_framing("solo migranti qui", pairs, "strict", 3, tokens=("solo", "migranti", "qui"))
    assert out["pair_framing_net_strict"] == 1.0


def test_get_pairs_registry_cached() -> None:
    """Function summary: get_pairs_registry returns stable list from CSV."""
    a = get_pairs_registry(ROOT)
    b = get_pairs_registry(ROOT)
    assert a is b
    assert len(a) >= 60
