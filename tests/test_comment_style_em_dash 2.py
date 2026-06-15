"""Unit tests for extended em-dash counting on comment style shards."""

from __future__ import annotations

from pathlib import Path

from src.comment_style import count_em_dash_extended, resolve_em_dash_count, score_comment_style

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_count_em_dash_extended_ascii_spaced_hyphen() -> None:
    """Function summary: spaced ASCII hyphens count as em dashes."""
    text = "one -- two --- three"
    assert count_em_dash_extended(text) == 2


def test_score_comment_style_em_dash_matches_extended() -> None:
    """Function summary: shard style pass stores extended count in em_dash_count."""
    body = "Hello -- world"
    scored = score_comment_style(
        body, "it", PROJECT_ROOT, enable_phrase_lexicons=False
    )
    assert scored["em_dash_count"] == 1
    assert scored["em_dash_extended_count"] == 1


def test_resolve_em_dash_count_prefers_extended_on_legacy_shards() -> None:
    """Function summary: validation uses max of narrow and extended columns."""
    assert resolve_em_dash_count(0.0, 2.0) == 2.0
    assert resolve_em_dash_count(1.0, None) == 1.0
