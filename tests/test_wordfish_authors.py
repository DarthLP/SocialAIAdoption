"""Tests for author-level Wordfish helpers and panel logic."""

from __future__ import annotations

import pandas as pd

from src.wordfish import (
    assign_primary_language,
    compute_change_outcomes,
    normalize_lexicon_code,
    zscore_preban,
)


def test_normalize_lexicon_eu_to_en() -> None:
    """eu/uk/us map to en for assignment."""
    assert normalize_lexicon_code("eu") == "en"
    assert normalize_lexicon_code("it") == "it"


def test_assign_primary_language_priority() -> None:
    """it > de > en assigns bilingual authors to it then de."""
    assert assign_primary_language({"it", "en"}, ["it", "de", "en"]) == "it"
    assert assign_primary_language({"en", "de"}, ["it", "de", "en"]) == "de"
    assert assign_primary_language({"en"}, ["it", "de", "en"]) == "en"


def test_zscore_preban_author_extremity() -> None:
    """Pre-ban extremity mean/sd used for cross-language z-scoring."""
    ext = [1.0, 2.0, 3.0, 4.0]
    bins = ["2023-03-01", "2023-03-10", "2023-03-20", "2023-04-05"]
    mu, sd = zscore_preban(ext, bins, "2023-03-31")
    assert mu == 2.0
    assert abs(sd - (2.0 / 3.0) ** 0.5) < 1e-9


def test_compute_change_author_grouping() -> None:
    """Rolling change computed within author, not across authors."""
    ext = pd.DataFrame(
        {
            "author": ["a1", "a1", "a2", "a2"],
            "primary_lexicon": ["it"] * 4,
            "time_bin": ["week"] * 4,
            "bin_start": ["2023-03-03", "2023-03-10", "2023-03-03", "2023-03-10"],
            "extremity": [1.0, 3.0, 10.0, 20.0],
        }
    )
    out = compute_change_outcomes(ext, "2023-03-31", window_days=2, group_col="author")
    a1 = out[out["author"] == "a1"].sort_values("bin_start")
    assert pd.isna(a1.iloc[0]["change"])
    assert a1.iloc[1]["change"] == 2.0
    assert "change_z" in out.columns


def test_balanced_filter_logic() -> None:
    """Balanced authors need pre and ban bins among kept docs."""
    anchor = "2023-03-31"
    docs_bins = [
        ("u1", "2023-03-01"),
        ("u1", "2023-04-01"),
        ("u2", "2023-03-01"),
        ("u2", "2023-03-15"),
    ]
    pre = {a for a, b in docs_bins if b < anchor}
    ban = {a for a, b in docs_bins if b >= anchor}
    balanced = pre & ban
    assert balanced == {"u1"}
