"""Tests for comment-level DiD panel helpers and estimation."""

from __future__ import annotations

import pandas as pd

from scripts.diagnostics.prepare_did_comment_panel import (
    _annotate_comments,
    build_author_day_panel,
)
from src.did.estimate import estimate_comment_feols, run_strategy_twfe
from src.did.specs import StrategySpec, filter_strategy_sample


def test_annotate_comments_calendar() -> None:
    """Function summary: comment rows get rel_day, post, IT, universe_slice."""
    raw = pd.DataFrame(
        {
            "author": ["u1"],
            "date_utc": ["2023-03-31"],
            "topic_family": ["it_political"],
            "comment_in_political_universe": [True],
        }
    )
    out = _annotate_comments(raw, "2023-03-31", "2023-05-01")
    assert out["rel_day"].iloc[0] == 0
    assert out["post"].iloc[0] == 1
    assert out["IT"].iloc[0] == 1
    assert out["universe_slice"].iloc[0] == "in_political_tree"


def test_build_author_day_panel() -> None:
    """Function summary: author-day panel aggregates comments with weighted means."""
    comments = pd.DataFrame(
        {
            "author": ["u1", "u1"],
            "time_id": ["2023-03-30", "2023-03-30"],
            "date_utc": ["2023-03-30", "2023-03-30"],
            "rel_day": [-1, -1],
            "post": [0, 0],
            "IT": [1, 1],
            "topic_family": ["it_political", "it_political"],
            "n_words": [10, 30],
            "net_ideology": [0.0, 1.0],
        }
    )
    day = build_author_day_panel([comments])
    assert len(day) == 1
    assert day["n_comments"].iloc[0] == 2
    assert abs(day["net_ideology"].iloc[0] - 0.75) < 1e-6


def test_comment_feols_cross_country() -> None:
    """Function summary: pyfixest returns finite beta on synthetic comment panel."""
    rows = []
    for i in range(12):
        treat = int(i >= 6)
        for day, post in [
            ("2023-03-28", 0),
            ("2023-03-29", 0),
            ("2023-03-30", 0),
            ("2023-03-31", 1),
            ("2023-04-01", 1),
        ]:
            for _ in range(2):
                rows.append(
                    {
                        "author": f"u{i}",
                        "time_id": day,
                        "treat": treat,
                        "post": post,
                        "rel_day": 0,
                        "net_ideology": 0.1 + 0.2 * treat * post + 0.01 * i,
                    }
                )
    df = pd.DataFrame(rows)
    res = estimate_comment_feols(df, "net_ideology", entity_only=False)
    assert res["estimation_note"] == "ok"
    assert res["n_clusters"] == 12


def test_comment_italy_only_entity_fe() -> None:
    """Function summary: Italy-only comment filter and entity-FE-only estimation."""
    rows = []
    for i in range(12):
        for day, post in [("2023-03-30", 0), ("2023-03-31", 1), ("2023-04-02", 1)]:
            rows.append(
                {
                    "author": f"it{i}",
                    "time_id": day,
                    "topic_family": "it_political",
                    "IT": 1,
                    "treat": 1,
                    "post": post,
                    "rel_day": 0,
                    "net_ideology": 0.1 + 0.05 * post,
                }
            )
    panel = pd.DataFrame(rows)
    sample = filter_strategy_sample(panel, StrategySpec("italy_only_post"))
    assert (sample["treat"] == 1).all()
    res = run_strategy_twfe(
        sample,
        StrategySpec("italy_only_post"),
        "net_ideology",
        panel_kind="comment",
    )
    assert res["estimation_note"] == "ok_entity_fe_only"
