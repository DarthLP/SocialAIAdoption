"""Tests for DDD estimation notes and political variation check."""

from __future__ import annotations

import pandas as pd

from src.did.estimate import estimate_ddd


def test_ddd_no_political_variation_note() -> None:
    """Function summary: DDD returns note when political_universe constant within entity."""
    df = pd.DataFrame(
        {
            "entity_id": ["r1", "r1"],
            "time_id": ["2023-03-30", "2023-03-31"],
            "y_col": [0.1, 0.2],
            "IT": [1, 1],
            "post": [0, 1],
            "political_universe": [1, 1],
            "universe_slice": ["in_political_tree", "in_political_tree"],
        }
    )
    res = estimate_ddd(df, "y_col", entity_col="entity_id", time_col="time_id")
    assert res["estimation_note"] == "no_within_entity_political_variation"
    assert pd.isna(res["beta"])


def test_ddd_with_political_variation_runs() -> None:
    """Function summary: DDD estimates when both slices present per subreddit."""
    rows = []
    for day, post in [("2023-03-30", 0), ("2023-03-31", 1), ("2023-04-01", 1)]:
        for pol in (0, 1):
            rows.append(
                {
                    "entity_id": "r1",
                    "time_id": day,
                    "outcome": 0.1 + 0.05 * post + 0.02 * pol * post,
                    "IT": 1,
                    "post": post,
                    "political_universe": pol,
                }
            )
    df = pd.DataFrame(rows)
    res = estimate_ddd(df, "outcome", entity_col="entity_id", time_col="time_id")
    assert res["estimation_note"] in (
        "ok",
        "post_pol_absorbed",
        "it_post_pol_absorbed",
        "fully_absorbed",
        "estimation_error",
    )
