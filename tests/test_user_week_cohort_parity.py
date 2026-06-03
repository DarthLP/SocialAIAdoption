"""Cohort author sets must match between analyze audit and regression prep."""

from __future__ import annotations

import pandas as pd

from src.user_week.cohorts import CohortThresholds, build_audit_df, panel_cohort_authors
from src.user_week.ideology_buckets import label_pre_post_weeks
from src.user_week.panel_prep import prepare_regression_sample


def test_regression_sample_authors_match_audit() -> None:
    """Function summary: panel_cohort_authors equals authors in regression sample."""
    panel = pd.DataFrame(
        {
            "author": ["a"] * 5 + ["b"] * 3,
            "iso_week_start": [
                "2023-03-06",
                "2023-03-13",
                "2023-04-03",
                "2023-04-10",
                "2023-04-17",
                "2023-03-06",
                "2023-03-13",
                "2023-04-03",
            ],
            "n_words": [200.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0],
            "net_ideology_mean": [0.1, 0.2, 0.3, 0.0, 0.1, 0.0, 0.1, 0.2],
        }
    )
    launch = "2023-03-27"
    labelled = label_pre_post_weeks(panel, launch, drop_ban_week=False)
    th = CohortThresholds("strict", 100, 2, 2, 400, 400)
    audit_authors = set(panel_cohort_authors(labelled, th))
    reg = prepare_regression_sample(panel, th, launch, drop_ban_week=False)
    reg_authors = set(reg["author"].astype(str).unique())
    assert audit_authors == reg_authors
    assert "a" in audit_authors
    assert "b" not in audit_authors


def test_build_audit_df_panel_category() -> None:
    """Function summary: author with enough pre/post weeks is panel cohort."""
    panel = pd.DataFrame(
        {
            "author": ["u1"] * 4,
            "period": ["pre", "pre", "post", "post"],
            "n_words": [250.0, 250.0, 250.0, 250.0],
        }
    )
    th = CohortThresholds("strict", 100, 2, 2, 400, 400)
    audit = build_audit_df(panel, th)
    assert audit.loc[0, "audit_category"] == "panel"
