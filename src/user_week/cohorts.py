"""
Cohort thresholds and panel-author selection for user-week analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class CohortThresholds:
    """Function summary: cohort threshold bundle for weekly and pooled gating."""

    label: str
    min_words_per_week: int
    min_pre_weeks: int
    min_post_weeks: int
    min_total_words_pre: int
    min_total_words_post: int


def default_cohort_thresholds() -> List[CohortThresholds]:
    """Function summary: default strict and loose cohort definitions (matches analyze CLI defaults).

    Returns:
    - List with strict then loose CohortThresholds.
    """
    return [
        CohortThresholds("strict", 100, 4, 4, 400, 400),
        CohortThresholds("loose", 30, 2, 2, 100, 100),
    ]


def cohort_thresholds_by_label(label: str) -> CohortThresholds:
    """Function summary: resolve one cohort label to thresholds.

    Parameters:
    - label: strict or loose.

    Returns:
    - Matching CohortThresholds.

    Raises:
    - ValueError when label is unknown.
    """
    for th in default_cohort_thresholds():
        if th.label == label:
            return th
    raise ValueError(f"Unknown cohort label: {label}")


def build_audit_df(panel: pd.DataFrame, thresholds: CohortThresholds) -> pd.DataFrame:
    """Function summary: per-author audit categories (panel / pre_only / post_only / below_thresholds).

    Parameters:
    - panel: labelled user-week panel with period and n_words.
    - thresholds: cohort gates.

    Returns:
    - One row per author with audit_category.
    """
    if panel.empty:
        return pd.DataFrame()
    n_words = panel["n_words"].astype(float)
    is_pre = panel["period"].values == "pre"
    is_post = panel["period"].values == "post"
    is_good = (n_words >= float(thresholds.min_words_per_week)).values
    df = pd.DataFrame(
        {
            "author": panel["author"].astype(str).values,
            "n_pre_any_w": is_pre.astype(int),
            "n_post_any_w": is_post.astype(int),
            "n_good_pre_w": (is_pre & is_good).astype(int),
            "n_good_post_w": (is_post & is_good).astype(int),
            "good_pre_words": np.where(is_pre & is_good, n_words.values, 0.0),
            "good_post_words": np.where(is_post & is_good, n_words.values, 0.0),
        }
    )
    user = df.groupby("author", sort=False).sum().reset_index()
    user.rename(
        columns={
            "n_pre_any_w": "n_pre_weeks_any",
            "n_post_any_w": "n_post_weeks_any",
            "n_good_pre_w": "n_good_pre_weeks",
            "n_good_post_w": "n_good_post_weeks",
            "good_pre_words": "pre_words_total_good",
            "good_post_words": "post_words_total_good",
        },
        inplace=True,
    )
    is_panel = (
        (user["n_good_pre_weeks"] >= thresholds.min_pre_weeks)
        & (user["n_good_post_weeks"] >= thresholds.min_post_weeks)
        & (user["pre_words_total_good"] >= thresholds.min_total_words_pre)
        & (user["post_words_total_good"] >= thresholds.min_total_words_post)
    )
    is_pre_only = (user["n_pre_weeks_any"] > 0) & (user["n_post_weeks_any"] == 0) & ~is_panel
    is_post_only = (user["n_pre_weeks_any"] == 0) & (user["n_post_weeks_any"] > 0) & ~is_panel
    user["audit_category"] = "below_thresholds"
    user.loc[is_panel, "audit_category"] = "panel"
    user.loc[is_pre_only, "audit_category"] = "pre_only"
    user.loc[is_post_only, "audit_category"] = "post_only"
    return user


def panel_cohort_authors(panel: pd.DataFrame, thresholds: CohortThresholds) -> List[str]:
    """Function summary: authors passing strict/loose panel cohort gates.

    Parameters:
    - panel: labelled panel with period column.
    - thresholds: cohort definition.

    Returns:
    - Author id strings in the panel cohort.
    """
    audit = build_audit_df(panel, thresholds)
    if audit.empty:
        return []
    return audit.loc[audit["audit_category"] == "panel", "author"].astype(str).tolist()
