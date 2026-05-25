"""Unit tests for dominant-side assignment and pair-framing scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4_lexicon import (  # noqa: E402
    PairEntry,
    dominant_side_from_uses,
    get_pairs_registry,
    score_pair_framing,
)


def test_migranti_left_center_tie_goes_left() -> None:
    """Function summary: left+center tie resolves to left (migranti)."""
    side, rule = dominant_side_from_uses("yes", "yes", "rarely")
    assert side == "left"
    assert rule == "left_center_tie"


def test_decoro_urbano_right_center_tie_goes_right() -> None:
    """Function summary: right+center tie resolves to right."""
    side, rule = dominant_side_from_uses("some", "yes", "yes")
    assert side == "right"
    assert rule == "right_center_tie"


def test_pair_xor_left_only() -> None:
    """Function summary: only left-pole term contributes +1 net."""
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
    scored = score_pair_framing("discorso sui migranti oggi", pairs, "strict", n_words=10)
    assert scored["pair_framing_net_strict"] == 1.0
    assert scored["pair_left_only_strict"] == 1.0
    assert scored["pair_right_only_strict"] == 0.0


def test_pair_both_hits() -> None:
    """Function summary: both terms hit increments pair_both not net."""
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
    scored = score_pair_framing("migranti e clandestini", pairs, "strict", n_words=5)
    assert scored["pair_framing_net_strict"] == 0.0
    assert scored["pair_both_strict"] == 1.0


def test_registry_caches_return_same_object() -> None:
    """Function summary: pairs loader reuses in-process cache."""
    assert get_pairs_registry(ROOT) is get_pairs_registry(ROOT)
    assert len(get_pairs_registry(ROOT)) >= 60


def test_polarization_row_requires_scorer_keys() -> None:
    """Function summary: _polarization_score_row raises if scorer omits a column."""
    from scripts.features._enriched_shard_runner import (  # noqa: E402
        POLARIZATION_COMMENT_COLUMNS,
        _polarization_score_row,
    )

    root = ROOT
    cfg = {"lang_match_filter": False, "negation_window_tokens": 3, "eps": 1.0e-6}
    row = _polarization_score_row("migranti clandestini destra sinistra", "it", "it", cfg, root)
    for col in POLARIZATION_COMMENT_COLUMNS:
        assert col in row
