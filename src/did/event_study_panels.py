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

# Entity-constant DiD annotation columns that bin_lexical_daily_panel drops
# (it keeps only numeric outcome columns). Restored post-binning so
# filter_strategy_sample sees the same metadata at 3d as at 1d; losing
# topic_family silently collapsed cross_country_all to an Italian-only
# sample (zero treat variation -> collinear event-study fits).
ES_PANEL_META_COLS: Tuple[str, ...] = (
    "topic",
    "topic_family",
    "primary_lexicon",
    "language_hub",
    "IT",
    "IT_political",
    "IT_others",
    "is_control",
    "control_de",
    "control_eu",
    "control_uk",
    "control_us",
    "political_universe",
)


def restore_entity_meta_after_binning(
    binned: pd.DataFrame,
    daily: pd.DataFrame,
    entity_cols: Sequence[str],
) -> pd.DataFrame:
    """Function summary: merge entity-constant metadata back onto a binned panel.

    Parameters:
    - binned: output of bin_lexical_daily_panel (numeric columns only).
    - daily: pre-binning daily panel carrying the metadata columns.
    - entity_cols: grouping keys (first value per key is taken).

    Returns:
    - Binned panel with missing ES_PANEL_META_COLS restored via left merge.
    """
    keys = list(entity_cols)
    missing = [
        c for c in ES_PANEL_META_COLS if c in daily.columns and c not in binned.columns
    ]
    if not missing:
        return binned
    meta = daily.groupby(keys, observed=True)[missing].first().reset_index()
    return binned.merge(meta, on=keys, how="left")


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
    # event_dates_from_config returns (start, end_exclusive, launch, lift).
    # A previous unpacking grabbed lift as end_excl, silently truncating the
    # subreddit/slice panels at the lift date (2023-04-28) and dropping the
    # post-lift bin (rel_period +10 / rel_day 28-30) that the pooled panels keep.
    _, end_excl, launch, _ = event_dates_from_config(config)
    bd = int(bin_days)
    work = daily.copy()
    if "n_comments" not in work.columns:
        work["n_comments"] = 1
    if bd <= 1:
        date_col = "date_utc" if "date_utc" in work.columns else "period_start"
        work["period_start"] = work[date_col].astype(str)
    else:
        pre_bin = work
        work = bin_lexical_daily_panel(work, entity_cols, bd, launch)
        work = restore_entity_meta_after_binning(work, pre_bin, entity_cols)
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
