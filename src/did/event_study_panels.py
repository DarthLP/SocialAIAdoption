"""
Prepare subreddit and slice panels for aggregated DiD event studies (incl. 3d outcome binning).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import pandas as pd

from scripts.diagnostics.descriptives_util import bin_lexical_daily_panel, event_dates_from_config
from src.did.panels import (
    load_subreddit_panel,
    load_subreddit_slice_panel,
    merge_semantic_axis,
)
from src.did.specs import rel_day_from_date


def prepare_subreddit_event_study_panel(
    daily: pd.DataFrame,
    config: Dict[str, Any],
    bin_days: int,
    *,
    entity_cols: Tuple[str, ...] = ("subreddit",),
) -> pd.DataFrame:
    """Function summary: daily or launch-aligned binned panel with rel_day/rel_period for ES.

    Parameters:
    - daily: subreddit-day (or subreddit×slice-day) panel with date_utc and outcomes.
    - config: project config (launch window).
    - bin_days: 1 keeps daily outcomes; 3 bins outcomes into launch-aligned 3-day blocks.
    - entity_cols: grouping keys for bin_lexical_daily_panel.

    Returns:
    - Panel with entity_id, time_id, rel_day, rel_period (when bin_days=3), treat/post preserved.
    """
    if daily.empty:
        return daily.copy()
    _, _, launch, end_excl = event_dates_from_config(config)
    bd = int(bin_days)
    work = daily.copy()
    if "n_comments" not in work.columns:
        work["n_comments"] = 1
    if bd <= 1:
        date_col = "date_utc" if "date_utc" in work.columns else "period_start"
        work["period_start"] = work[date_col].astype(str)
    else:
        work = bin_lexical_daily_panel(work, entity_cols, bd, launch)
    work = work[work["period_start"].astype(str) < str(end_excl)]
    work["rel_day"] = rel_day_from_date(work["period_start"], launch)
    work["rel_period"] = (work["rel_day"] // bd).astype(int)
    work["time_id"] = work["period_start"].astype(str)
    if len(entity_cols) == 1:
        work["entity_id"] = work[entity_cols[0]].astype(str)
    else:
        work["entity_id"] = (
            work[list(entity_cols)].astype(str).agg("|".join, axis=1)
        )
    return work


def load_subreddit_event_study_panel_binned(
    config: Dict[str, Any], bin_days: int
) -> pd.DataFrame:
    """Function summary: subreddit panel prepared for aggregated language event studies."""
    daily = merge_semantic_axis(load_subreddit_panel(config), config)
    return prepare_subreddit_event_study_panel(
        daily, config, bin_days, entity_cols=("subreddit",)
    )


def load_subreddit_slice_event_study_panel_binned(
    config: Dict[str, Any], bin_days: int
) -> pd.DataFrame:
    """Function summary: subreddit×universe_slice panel with true 3d outcome binning when needed."""
    daily = merge_semantic_axis(load_subreddit_slice_panel(config), config)
    return prepare_subreddit_event_study_panel(
        daily,
        config,
        bin_days,
        entity_cols=("subreddit", "universe_slice"),
    )
