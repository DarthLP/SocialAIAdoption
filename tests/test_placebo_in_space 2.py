"""Tests for placebo-in-space inference (FIX 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.did.inference import assign_entity_country_series, placebo_in_space_p
from src.did.specs import ITALY_FAMILIES, StrategySpec


def _synthetic_cross_country_panel(n_entities: int = 8, n_days: int = 10) -> pd.DataFrame:
    """Function summary: subreddit-day panel with four control countries + Italy."""
    rows = []
    countries = ["it_political", "de", "eu", "uk", "us"]
    for i, fam in enumerate(countries):
        for d in range(n_days):
            rows.append(
                {
                    "entity_id": f"e_{fam}",
                    "time_id": f"2023-03-{d+1:02d}",
                    "date_utc": f"2023-03-{d+1:02d}",
                    "rel_day": d - 5,
                    "topic_family": fam,
                    "post": int(d >= 5),
                    "y": float(i + 1) + 0.1 * d + np.random.default_rng(i + d).normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_placebo_flips_treat_vs_baseline() -> None:
    """Function summary: each placebo country changes treat relative to all-zero controls."""
    panel = _synthetic_cross_country_panel()
    sample = panel[~panel["topic_family"].isin(ITALY_FAMILIES)].copy()
    sample["treat"] = 0
    baseline = sample["treat"].astype(int).tolist()
    ent = assign_entity_country_series(sample, "entity_id")
    for c in ("de", "eu", "uk", "us"):
        treat = (ent == c).astype(int)
        assert treat.tolist() != baseline
        assert treat.nunique() == 2


def test_stratum_permutation_is_degenerate() -> None:
    """Function summary: old within-topic_family treat shuffle never changes labels."""
    entities = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c"],
            "topic_family": ["it_political", "de", "eu"],
        }
    )
    entities["treat"] = entities["topic_family"].isin(ITALY_FAMILIES).astype(int)
    original = entities["treat"].copy()
    shuffled = []
    for _, grp in entities.groupby("topic_family"):
        g = grp.copy()
        if len(g) > 1:
            g["treat"] = np.random.default_rng(0).permutation(g["treat"].values)
        shuffled.append(g)
    out = pd.concat(shuffled, ignore_index=True)
    merged = entities[["entity_id", "treat"]].merge(
        out[["entity_id", "treat"]], on="entity_id", suffixes=("_orig", "_perm")
    )
    assert (merged["treat_orig"] == merged["treat_perm"]).all()


def test_placebo_p_floor_four_controls() -> None:
    """Function summary: pooled cross-country placebo p-value floor is 1/5."""
    panel = _synthetic_cross_country_panel()
    res = placebo_in_space_p(
        panel,
        StrategySpec("cross_country_all"),
        "y",
        entity_col="entity_id",
        time_col="time_id",
    )
    assert res.p_floor == pytest.approx(0.2)
    assert res.n_placebo_draws == 4
    assert np.isfinite(res.p) or np.isnan(res.p)
    assert res.p >= res.p_floor or np.isnan(res.p)


def test_placebo_non_cross_country_returns_nan() -> None:
    """Function summary: within-Italy DDD does not use placebo-in-space."""
    panel = _synthetic_cross_country_panel()
    res = placebo_in_space_p(
        panel,
        StrategySpec("within_italy_ddd"),
        "y",
    )
    assert np.isnan(res.p)
