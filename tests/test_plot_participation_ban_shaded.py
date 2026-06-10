"""Tests for participation ban-window descriptives arm assignment and masking."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd


def _load_mod():
    """Function summary: load plot_participation_ban_shaded with Agg matplotlib backend."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/plot_participation_ban_shaded.py"
    )
    spec = importlib.util.spec_from_file_location("plot_participation_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec.loader.exec_module(mod)
    return mod


def test_assign_arm_topic_family_uses_shard_labels() -> None:
    """Shard topic_family values in DID_ARMS are kept; no fallback subreddits."""
    mod = _load_mod()
    df = pd.DataFrame(
        {
            "subreddit": ["italy", "AnimeItaly", "de"],
            "topic_family": ["it_political", "it_others", "de"],
            "author": ["a1", "a2", "a3"],
            "date_utc": ["2023-03-15"] * 3,
            "lang_comment": ["it", "it", "de"],
        }
    )
    family_map = {"italy": "it_others", "AnimeItaly": "it_others", "de": "de"}
    out, fallback = mod.assign_arm_topic_family(df, family_map)
    assert fallback == []
    assert set(out["topic_family"]) == {"it_political", "it_others", "de"}
    assert len(out) == 3


def test_assign_arm_topic_family_falls_back_per_subreddit() -> None:
    """Missing or invalid shard labels fall back to YAML map per subreddit."""
    mod = _load_mod()
    df = pd.DataFrame(
        {
            "subreddit": ["italy", "AnimeItaly", "de"],
            "topic_family": [pd.NA, "it_pure_political", "de"],
            "author": ["a1", "a2", "a3"],
            "date_utc": ["2023-03-15"] * 3,
            "lang_comment": ["it", "it", "de"],
        }
    )
    family_map = {"italy": "it_others", "AnimeItaly": "it_others", "de": "de"}
    out, fallback = mod.assign_arm_topic_family(df, family_map)
    assert set(fallback) == {"AnimeItaly", "italy"}
    assert out.loc[out["subreddit"] == "italy", "topic_family"].iloc[0] == "it_others"
    assert out.loc[out["subreddit"] == "de", "topic_family"].iloc[0] == "de"
    assert len(out) == 3


def test_assign_arm_topic_family_no_column_full_fallback() -> None:
    """Without topic_family column, all subreddits use YAML fallback."""
    mod = _load_mod()
    df = pd.DataFrame(
        {
            "subreddit": ["italy", "de"],
            "author": ["a1", "a2"],
            "date_utc": ["2023-03-15"] * 2,
            "lang_comment": ["it", "de"],
        }
    )
    family_map = {"italy": "it_others", "de": "de"}
    out, fallback = mod.assign_arm_topic_family(df, family_map)
    assert set(fallback) == {"de", "italy"}
    assert set(out["topic_family"]) == {"de", "it_others"}


def test_apply_burn_in_mask_unchanged() -> None:
    """Burn-in masks entry/return metrics only; churn tail left intact."""
    mod = _load_mod()
    panel = pd.DataFrame(
        {
            "date_utc": ["2023-03-01", "2023-03-15", "2023-04-24"],
            "new_authors": [10.0, 20.0, 30.0],
            "returning_author_comment_share": [0.5, 0.6, 0.7],
            "churned_authors": [5.0, 6.0, float("nan")],
        }
    )
    masked = mod.apply_burn_in_mask(panel, "2023-03-01", burn_in_days=14)
    assert pd.isna(masked.loc[0, "new_authors"])
    assert pd.isna(masked.loc[0, "returning_author_comment_share"])
    assert masked.loc[1, "new_authors"] == 20.0
    assert masked.loc[2, "churned_authors"] != masked.loc[2, "churned_authors"]  # NaN


def test_language_panel_independent_of_arm_assignment() -> None:
    """Language aggregation is identical whether or not topic_family exists."""
    mod = _load_mod()
    base = pd.DataFrame(
        {
            "subreddit": ["italy", "italy", "de"],
            "author": ["a1", "a2", "a3"],
            "date_utc": ["2023-03-15", "2023-03-15", "2023-03-15"],
            "lang_comment": ["it", "it", "de"],
            "comment_in_political_universe": [True, True, False],
        }
    )
    with_arm = base.assign(topic_family=["it_political", "it_others", "de"])
    panel_no_arm = mod.build_participation_panel(base, "2023-03-01", "2023-03-02")
    panel_with_arm = mod.build_participation_panel(with_arm, "2023-03-01", "2023-03-02")
    pd.testing.assert_frame_equal(
        panel_no_arm.sort_values(["universe_slice", "language", "date_utc"]).reset_index(
            drop=True
        ),
        panel_with_arm.sort_values(["universe_slice", "language", "date_utc"]).reset_index(
            drop=True
        ),
    )
