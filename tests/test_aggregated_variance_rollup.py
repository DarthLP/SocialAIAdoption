"""Tests for NaN-safe and pooled variance rollups in aggregated DiD panels."""

from __future__ import annotations

import pandas as pd

from scripts.diagnostics.descriptives_util import bin_lexical_daily_panel, pooled_variance, weighted_mean_nan
from src.did.aggregated import _rollup_weighted_outcomes


def test_weighted_mean_nan_ignores_missing_subgroups() -> None:
    """One NaN subreddit-day must not zero out the hub-day mean."""
    grp = pd.DataFrame(
        {
            "n_comments": [100.0, 200.0, 50.0],
            "sem_axis_ideology_mean": [0.1, float("nan"), 0.3],
        }
    )
    out = weighted_mean_nan(grp["sem_axis_ideology_mean"], grp["n_comments"])
    assert abs(out - (0.1 * 100 + 0.3 * 50) / 150) < 1e-9


def test_pooled_variance_between_subgroup_means() -> None:
    """Averaging subgroup variances misses between-group spread; pooling does not."""
    means = pd.Series([-1.0, 1.0])
    variances = pd.Series([0.0, 0.0])
    weights = pd.Series([100.0, 100.0])
    assert abs(pooled_variance(means, variances, weights) - 1.0) < 1e-9


def test_rollup_weighted_outcomes_it_style_day() -> None:
    """Many subreddit rows with a few NaN variances still yield a finite hub variance."""
    grp = pd.DataFrame(
        {
            "n_comments": [100.0, 80.0, 60.0, 40.0],
            "sem_axis_ideology_mean": [0.0, 0.2, -0.1, 0.15],
            "sem_axis_ideology_var": [0.01, float("nan"), 0.02, 0.005],
        }
    )
    out = _rollup_weighted_outcomes(
        grp,
        ["sem_axis_ideology_mean", "sem_axis_ideology_var"],
        grp["n_comments"],
    )
    assert pd.notna(out["sem_axis_ideology_mean"])
    assert pd.notna(out["sem_axis_ideology_var"])
    assert out["sem_axis_ideology_var"] > 0.01


def test_bin_lexical_3d_pools_variance_across_days() -> None:
    """Launch-aligned bins combine daily subgroup variances with pooled formula."""
    launch = "2023-03-31"
    daily = pd.DataFrame(
        {
            "language_hub": ["it", "it"],
            "date_utc": ["2023-03-31", "2023-04-01"],
            "n_comments": [100, 100],
            "sem_axis_ideology_mean": [0.0, 1.0],
            "sem_axis_ideology_var": [0.0, 0.0],
        }
    )
    binned = bin_lexical_daily_panel(daily, ("language_hub",), 3, launch)
    assert len(binned) == 1
    assert binned.iloc[0]["period_start"] == "2023-03-31"
    assert abs(float(binned.iloc[0]["sem_axis_ideology_var"]) - 0.25) < 1e-9
