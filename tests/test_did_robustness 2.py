"""Tests for DiD robustness grid date-placebo pre-period guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.did.robustness import MIN_PRE_DAYS, _n_pre_period_days, run_robustness_grid
from src.did.specs import StrategySpec


def _synthetic_panel(start: str = "2023-03-01", n_days: int = 45) -> pd.DataFrame:
    """Function summary: minimal subreddit-day panel for robustness grid tests."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for d in dates:
        rel = int((d - pd.Timestamp("2023-03-31")).days)
        for sub, treat, fam in [("it_a", 1, "it_political"), ("de_a", 0, "de")]:
            rows.append(
                {
                    "date_utc": d.strftime("%Y-%m-%d"),
                    "rel_day": rel,
                    "post": int(rel >= 0),
                    "treat": treat,
                    "topic_family": fam,
                    "entity_id": sub,
                    "time_id": d.strftime("%Y-%m-%d"),
                    "y": float(treat) + 0.01 * rel + np.random.default_rng(0).normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_n_pre_period_days_panel_start() -> None:
    """Function summary: placebo on panel start date has zero pre-period days."""
    panel = _synthetic_panel()
    assert _n_pre_period_days(panel, "2023-03-01") == 0
    assert _n_pre_period_days(panel, "2023-02-14") == 0
    assert _n_pre_period_days(panel, "2023-03-16") == 15


def test_robustness_skips_insufficient_pre() -> None:
    """Function summary: early placebo dates skipped when pre-period shorter than MIN_PRE_DAYS."""
    panel = _synthetic_panel()
    strat = StrategySpec("cross_country_all")
    rows = run_robustness_grid(panel, strat, "y", "2023-03-31")
    by_check = {r["check"]: r for r in rows}

    for early in ("placebo_2023-02-14", "placebo_2023-03-01"):
        row = by_check[early]
        assert row["estimation_note"] == "skipped_insufficient_pre"
        assert row["n_pre_days"] < MIN_PRE_DAYS
        assert np.isnan(row["beta"])
        assert np.isnan(row["se"])
        assert np.isnan(row["pvalue"])

    ok_row = by_check["placebo_2023-03-16"]
    assert ok_row["estimation_note"] != "skipped_insufficient_pre"
    assert ok_row["n_pre_days"] >= MIN_PRE_DAYS
