"""Tests for descriptives_util rolling helpers."""

from __future__ import annotations

import pandas as pd

from scripts.diagnostics.descriptives_util import grouped_trailing_daily_rolling


def test_rolling_remasks_nan_tail() -> None:
    """Rolling mean must not bridge NaN raw values (churn right-censoring)."""
    df = pd.DataFrame(
        {
            "language": ["it"] * 5,
            "date_utc": ["2023-04-20", "2023-04-21", "2023-04-22", "2023-04-23", "2023-04-24"],
            "value": [10.0, 10.0, 10.0, float("nan"), float("nan")],
        }
    )
    rolled = grouped_trailing_daily_rolling(
        df, group_col="language", rolling_window_days=7, date_col="date_utc"
    )
    tail = rolled.sort_values("date_utc").tail(2)["value"]
    assert tail.isna().all()


def test_rolling_preserves_finite_values() -> None:
    """Finite raw days still receive smoothed values."""
    df = pd.DataFrame(
        {
            "language": ["it"] * 3,
            "date_utc": ["2023-03-01", "2023-03-02", "2023-03-03"],
            "value": [1.0, 3.0, 5.0],
        }
    )
    rolled = grouped_trailing_daily_rolling(
        df, group_col="language", rolling_window_days=2, date_col="date_utc"
    )
    last = float(rolled.sort_values("date_utc").iloc[-1]["value"])
    assert last == 4.0
