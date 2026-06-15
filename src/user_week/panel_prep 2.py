"""
Prepare labelled author×week panels for user-week regression.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.user_week.cohorts import CohortThresholds, panel_cohort_authors
from src.user_week.ideology_buckets import label_pre_post_weeks

# Event-study window (ISO weeks relative to launch Monday).
EVENT_STUDY_REL_WEEK_MIN = -8
EVENT_STUDY_REL_WEEK_MAX = 8
EVENT_STUDY_REFERENCE_WEEK = -1

HEADLINE_OUTCOMES: tuple[str, ...] = (
    "net_ideology",
    "extremity",
    "pole_share",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
)


def launch_iso_week_from_config(config: Dict[str, Any]) -> str:
    """Function summary: ISO Monday date string for ban week from event_window.launch_day_utc.

    Parameters:
    - config: study YAML dict.

    Returns:
    - YYYY-MM-DD string (Monday of launch week).
    """
    raw = str(config["event_window"]["launch_day_utc"])
    launch_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if launch_dt.tzinfo is None:
        launch_dt = launch_dt.replace(tzinfo=timezone.utc)
    weekday = launch_dt.weekday()
    monday = (launch_dt - timedelta(days=weekday)).date()
    return monday.isoformat()


def outcome_panel_column(feature: str, panel_columns: Sequence[str]) -> str:
    """Function summary: resolve display column for a feature on the user-week panel.

    Parameters:
    - feature: logical outcome name (e.g. net_ideology).
    - panel_columns: columns present on the panel.

    Returns:
    - Column name to use as dependent variable.
    """
    cols = set(panel_columns)
    if feature in cols:
        return feature
    mean_alias = f"{feature}_mean"
    if mean_alias in cols:
        return mean_alias
    return feature


def feature_track(feature: str) -> str:
    """Function summary: classify outcome as lexical, semantic, or style for reporting.

    Parameters:
    - feature: outcome name.

    Returns:
    - Track label: lexical, semantic, or style.
    """
    if feature.startswith("sem_axis"):
        return "semantic"
    style_markers = ("ai_style", "semicolon", "em_dash", "hedging")
    if any(m in feature for m in style_markers):
        return "style"
    return "lexical"


def add_calendar_fields(
    panel: pd.DataFrame,
    launch_iso_week: str,
    drop_ban_week: bool,
) -> pd.DataFrame:
    """Function summary: add period, post, rel_week, and time_id for panel regressions.

    Parameters:
    - panel: raw user-week panel.
    - launch_iso_week: ban ISO Monday YYYY-MM-DD.
    - drop_ban_week: exclude launch week rows.

    Returns:
    - Copy with period, post, rel_week, time_id.
    """
    out = label_pre_post_weeks(panel, launch_iso_week, drop_ban_week)
    launch_dt = datetime.fromisoformat(launch_iso_week + "T00:00:00+00:00")
    if launch_dt.tzinfo is None:
        launch_dt = launch_dt.replace(tzinfo=timezone.utc)
    launch_ts = pd.Timestamp(launch_dt)
    if launch_ts.tzinfo is not None:
        launch_ts = launch_ts.tz_localize(None)
    week_starts = pd.to_datetime(out["iso_week_start"].astype(str))
    rel_days = (week_starts - launch_ts).dt.days
    out["rel_week"] = (rel_days // 7).astype(int)
    out["post"] = (out["period"] == "post").astype(int)
    out["time_id"] = out["iso_week_start"].astype(str)
    return out


def prepare_regression_sample(
    panel: pd.DataFrame,
    thresholds: CohortThresholds,
    launch_iso_week: str,
    drop_ban_week: bool,
    outcomes: Optional[Sequence[str]] = None,
    rel_week_min: int = EVENT_STUDY_REL_WEEK_MIN,
    rel_week_max: int = EVENT_STUDY_REL_WEEK_MAX,
) -> pd.DataFrame:
    """Function summary: labelled panel restricted to cohort authors and pre/post weeks.

    Parameters:
    - panel: user_week_panel.parquet contents.
    - thresholds: cohort gates (same as analyze_user_pre_post_shift).
    - launch_iso_week: ban ISO Monday.
    - drop_ban_week: drop launch week.
    - outcomes: optional feature list (unused for filtering; for API symmetry).
    - rel_week_min, rel_week_max: event-study support window.

    Returns:
    - Filtered panel with calendar fields.
    """
    _ = outcomes
    labelled = add_calendar_fields(panel, launch_iso_week, drop_ban_week)
    authors = panel_cohort_authors(labelled, thresholds)
    if not authors:
        return labelled.iloc[0:0].copy()
    sub = labelled[labelled["author"].astype(str).isin(authors)].copy()
    sub = sub[sub["period"].isin(["pre", "post"])].copy()
    sub = sub[(sub["rel_week"] >= rel_week_min) & (sub["rel_week"] <= rel_week_max)].copy()
    return sub


def resolve_outcome_list(mode: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Function summary: headline or all default_features outcomes for regression.

    Parameters:
    - mode: headline or all.
    - config: study YAML when mode is all.

    Returns:
    - List of outcome feature names.
    """
    if mode == "headline":
        return list(HEADLINE_OUTCOMES)
    if config is None:
        return list(HEADLINE_OUTCOMES)
    from src.config_utils import user_week_default_features

    feats = user_week_default_features(config)
    return list(feats) if feats else list(HEADLINE_OUTCOMES)
