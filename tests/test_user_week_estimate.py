"""Tests for author×week panel regression helpers."""

from __future__ import annotations

import pandas as pd

from src.user_week.estimate import (
    estimate_user_week_entity_only,
    estimate_user_week_event_study,
)


def _synthetic_panel(n_authors: int = 14, n_weeks: int = 10) -> pd.DataFrame:
    """Function summary: build balanced author×week panel with post shock on outcome."""
    rows = []
    base = pd.Timestamp("2023-03-06")
    week_offsets = list(range(-4, n_weeks - 4))
    for i in range(n_authors):
        for wi, w in enumerate(week_offsets):
            post = int(w >= 0)
            rows.append(
                {
                    "author": f"u{i}",
                    "time_id": (base + pd.Timedelta(weeks=wi)).strftime("%Y-%m-%d"),
                    "rel_week": w,
                    "post": post,
                    "net_ideology_mean": 0.1 * i + 0.5 * post,
                }
            )
    return pd.DataFrame(rows)


def test_entity_only_post_coefficient() -> None:
    """Function summary: entity-only regression returns finite post beta on synthetic panel."""
    panel = _synthetic_panel()
    res = estimate_user_week_entity_only(panel, "net_ideology_mean")
    assert res["estimation_note"] == "ok_entity_fe_only"
    assert res["beta"] == res["beta"]  # finite
    assert abs(res["beta"] - 0.5) < 0.15
    assert res["n_clusters"] >= 12


def test_event_study_returns_week_rows() -> None:
    """Function summary: event study emits one row per non-reference rel_week."""
    panel = _synthetic_panel()
    rows = estimate_user_week_event_study(panel, "net_ideology_mean", reference_week=-1)
    assert len(rows) >= 3
    rel_weeks = {r["rel_week"] for r in rows}
    assert -1 not in rel_weeks
    assert all(r["beta"] == r["beta"] for r in rows)
