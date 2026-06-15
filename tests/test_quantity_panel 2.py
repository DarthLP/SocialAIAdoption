"""Tests for zero-filled quantity DiD panel builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_prepare_mod():
    """Function summary: load prepare_did_subreddit_panel module."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/prepare_did_subreddit_panel.py"
    )
    spec = importlib.util.spec_from_file_location("prep_sub_panel_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_quantity_panel_zero_fills_active_forums() -> None:
    """Active forums get full calendar grid with zero-comment days."""
    mod = _load_prepare_mod()
    sub = pd.DataFrame(
        {
            "subreddit": ["a", "a", "b", "b"],
            "topic_family": ["it_political", "it_political", "de", "de"],
            "topic": ["it_political", "it_political", "de", "de"],
            "date_utc": ["2023-03-01", "2023-04-01", "2023-03-02", "2023-04-02"],
            "n_comments": [5, 3, 1, 2],
            "n_authors": [2, 1, 1, 1],
        }
    )
    out = mod.build_quantity_panel(sub, "2023-03-01", "2023-03-04", "2023-03-31")
    assert len(out) == 6  # 2 forums x 3 days
    zero_row = out[(out["subreddit"] == "a") & (out["date_utc"] == "2023-03-02")]
    assert int(zero_row["n_comments"].iloc[0]) == 0
    assert float(zero_row["log_n_comments"].iloc[0]) == 0.0


def test_build_quantity_panel_skips_inactive_forum() -> None:
    """Forums without comments in both March and April are excluded."""
    mod = _load_prepare_mod()
    sub = pd.DataFrame(
        {
            "subreddit": ["a", "b"],
            "topic_family": ["it_political", "de"],
            "topic": ["it_political", "de"],
            "date_utc": ["2023-03-01", "2023-04-01"],
            "n_comments": [1, 1],
            "n_authors": [1, 1],
        }
    )
    out = mod.build_quantity_panel(sub, "2023-03-01", "2023-04-03", "2023-03-31")
    assert out.empty
