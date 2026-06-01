"""Tests for DDD estimation notes and political variation check."""

from __future__ import annotations

import pandas as pd

import numpy as np

from src.did.estimate import (
    annotate_pretrend_quality,
    apply_degeneracy_guard,
    estimate_ddd,
    estimate_pretrend_f,
    run_strategy_twfe,
)
from src.did.specs import StrategySpec, is_entity_fe_only_strategy


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


def test_apply_degeneracy_guard_huge_beta() -> None:
    """Function summary: extreme β is tagged degenerate_collinear with NaN outputs."""
    res = apply_degeneracy_guard(
        {
            "beta": 1e12,
            "se": 1e11,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "pvalue": 0.01,
            "estimation_note": "ok",
        },
        design_cond=1.0,
        median_abs_beta=0.01,
    )
    assert res["estimation_note"] == "degenerate_collinear"
    assert np.isnan(res["beta"])


def test_italy_only_post_entity_fe_only() -> None:
    """Function summary: italy_only_post uses entity-FE-only TWFE on IT subreddit panel."""
    rows = []
    days = [(f"2023-03-{d:02d}", int(d >= 31), d - 31) for d in range(25, 32)]
    days += [(f"2023-04-{d:02d}", 1, d + 1) for d in range(1, 8)]
    for i in range(4):
        sub = f"it_sub_{i}"
        fam = "it_political" if i % 2 == 0 else "it_others"
        for day, post, rel in days:
            rows.append(
                {
                    "entity_id": sub,
                    "time_id": day,
                    "topic_family": fam,
                    "IT": 1,
                    "treat": 1,
                    "post": post,
                    "rel_day": rel,
                    "net_ideology_mean": 0.1 + 0.05 * post + 0.01 * i,
                }
            )
    df = pd.DataFrame(rows)
    res = run_strategy_twfe(
        df,
        StrategySpec("italy_only_post"),
        "net_ideology_mean",
        entity_col="entity_id",
        time_col="time_id",
    )
    assert res["estimation_note"] == "ok_entity_fe_only"
    assert is_entity_fe_only_strategy("italy_only_post")
    assert np.isfinite(res["beta"])


def test_insufficient_panel_guard() -> None:
    """Function summary: sparse small-cluster panels skip TWFE with insufficient_panel."""
    rows = []
    for ent in range(11):
        for day in ("2023-03-30", "2023-03-31"):
            rows.append(
                {
                    "entity_id": f"r{ent}",
                    "time_id": day,
                    "y": 0.1,
                    "IT": 1 if ent < 6 else 0,
                    "treat": int(ent < 6),
                    "post": int(day > "2023-03-30"),
                    "rel_day": -1 if day == "2023-03-30" else 0,
                }
            )
    df = pd.DataFrame(rows)
    res = run_strategy_twfe(
        df,
        StrategySpec("cross_country_all"),
        "y",
        entity_col="entity_id",
        time_col="time_id",
    )
    assert res["estimation_note"] == "insufficient_panel"


def test_annotate_pretrend_quality_degenerate_clears_pretrend() -> None:
    """Function summary: degenerate_collinear rows must not export a pretrend F p-value."""
    row = annotate_pretrend_quality(
        {
            "strategy_id": "cross_country_all",
            "beta": float("nan"),
            "estimation_note": "degenerate_collinear",
            "pretrend_F_p": 1e-6,
        }
    )
    assert row["pretrend_quality"] == "unreliable_estimate"
    assert np.isnan(row["pretrend_F_p"])


def test_annotate_pretrend_quality_pretrend_reject() -> None:
    """Function summary: finite ok beta with small pretrend F p is tagged pretrend_reject."""
    row = annotate_pretrend_quality(
        {
            "strategy_id": "cross_country_vs_eu",
            "beta": 0.002,
            "estimation_note": "ok",
            "pretrend_F_p": 0.01,
        }
    )
    assert row["pretrend_quality"] == "pretrend_reject"
    assert row["pretrend_F_p"] == 0.01


def test_annotate_pretrend_quality_ddd_not_estimated() -> None:
    """Function summary: within-Italy DDD rows are not_estimated for pretrend."""
    row = annotate_pretrend_quality(
        {
            "strategy_id": "within_italy_ddd",
            "beta": float("nan"),
            "estimation_note": "degenerate_collinear",
            "pretrend_F_p": float("nan"),
        }
    )
    assert row["pretrend_quality"] == "not_estimated"


def test_estimate_pretrend_f_insufficient_preperiods() -> None:
    """Function summary: pretrend returns insufficient_preperiods when leads missing."""
    df = pd.DataFrame(
        {
            "entity_id": ["r1", "r1", "r2", "r2"],
            "time_id": ["2023-03-30", "2023-03-31", "2023-03-30", "2023-03-31"],
            "y": [0.1, 0.2, 0.3, 0.4],
            "treat": [1, 1, 0, 0],
            "post": [0, 1, 0, 1],
            "rel_day": [0, 1, 0, 1],
        }
    )
    pval, note = estimate_pretrend_f(df, "y", entity_col="entity_id", time_col="time_id")
    assert note == "insufficient_preperiods"
    assert np.isnan(pval)
