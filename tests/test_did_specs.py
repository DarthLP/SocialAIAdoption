"""Tests for DiD calendar and treatment specs."""

from __future__ import annotations

import pandas as pd

from src.did.specs import (
    ITALY_FAMILIES,
    StrategySpec,
    apply_post_window,
    filter_strategy_sample,
    is_author_strategy,
    rel_day_from_date,
    strategy_label,
)


def test_rel_day_launch() -> None:
    """Function summary: rel_day is 0 on ban date."""
    dates = pd.Series(["2023-03-30", "2023-03-31", "2023-04-01"])
    rel = rel_day_from_date(dates, "2023-03-31")
    assert list(rel) == [-1, 0, 1]


def test_early_ban_post_window() -> None:
    """Function summary: early-ban post is 1 only for rel_day 0..6."""
    df = pd.DataFrame({"rel_day": [-1, 0, 6, 7, 14], "post": 1})
    out = apply_post_window(df, "early_ban_7d", "")
    assert list(out["post"]) == [0, 1, 1, 0, 0]


def test_filter_it_political_vs_controls() -> None:
    """Function summary: treated_family it_political keeps IT political + controls."""
    panel = pd.DataFrame(
        {
            "subreddit": ["a", "b", "c"],
            "topic_family": ["it_political", "it_others", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    strat = StrategySpec("x", treated_family="it_political")
    out = filter_strategy_sample(panel, strat)
    assert set(out["subreddit"]) == {"a", "c"}
    assert out.loc[out["subreddit"] == "a", "treat"].iloc[0] == 1
    assert out.loc[out["subreddit"] == "c", "treat"].iloc[0] == 0


def test_filter_treated_topic() -> None:
    """Function summary: treated_topic keeps topic subs + controls."""
    panel = pd.DataFrame(
        {
            "subreddit": ["a", "b", "c"],
            "topic": ["it_pure_political", "it_others", "de"],
            "topic_family": ["it_political", "it_others", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    strat = StrategySpec("t", treated_topic="it_pure_political")
    out = filter_strategy_sample(panel, strat)
    assert set(out["subreddit"]) == {"a", "c"}
    assert out.loc[out["subreddit"] == "a", "treat"].iloc[0] == 1


def test_author_it_ban_treat_constant() -> None:
    """Function summary: author_it_ban sets treat=1 for IT cohort."""
    panel = pd.DataFrame(
        {
            "author": ["u1", "u2"],
            "primary_lexicon": ["it", "it"],
            "rel_day": [0, 1],
            "post": [1, 1],
            "IT": [1, 1],
        }
    )
    strat = StrategySpec("author_it_ban")
    out = filter_strategy_sample(panel, strat)
    assert (out["treat"] == 1).all()


def test_strategy_label_known() -> None:
    """Function summary: strategy_label returns readable text."""
    assert "Italian" in strategy_label("cross_country_all")
    assert "triple" in strategy_label("within_italy_ddd").lower()


def test_italy_families_constant() -> None:
    """Function summary: Italy families include political and others."""
    assert "it_political" in ITALY_FAMILIES
    assert "it_others" in ITALY_FAMILIES


def test_is_author_strategy() -> None:
    """Function summary: author strategy ids detected."""
    assert is_author_strategy("author_it_ban")
    assert not is_author_strategy("cross_country_all")
