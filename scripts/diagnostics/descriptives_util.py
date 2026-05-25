"""
Script summary:
Shared helpers for polarization and lexicon descriptives (prepare/plot scripts).

Functionality:
- weighted_mean, ban_phase, symmetric ban-window masks, trailing rolling, min-n filters.
- Mandatory dominant_v1 config assert and metrics-notes stamping.

How to apply/run:
- Imported by prepare_lexicon_descriptives.py, plot_lexicon_descriptives.py, plot_polarization_descriptives.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config_utils import load_polarization_config, require_dominant_v1_ideology_scoring


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Function summary: compute weighted mean with zero-weight guard.

    Parameters:
    - series: values.
    - weights: weights.

    Returns:
    - Weighted mean or NaN.
    """
    w = weights.astype(float)
    if w.sum() <= 0:
        return float("nan")
    return float((series.astype(float) * w).sum() / w.sum())


def ban_phase(date_utc: str, launch: str, lift: str) -> str:
    """Function summary: assign pre/ban/post label from calendar date.

    Parameters:
    - date_utc: YYYY-MM-DD.
    - launch: ban start date.
    - lift: ban end date (first post day).

    Returns:
    - Phase label.
    """
    if date_utc < launch:
        return "pre"
    if date_utc < lift:
        return "ban"
    return "post"


def event_dates_from_config(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Function summary: parse event window start, end, launch, lift from study config.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Tuple (start, end_exclusive, launch, lift) as YYYY-MM-DD strings.
    """
    from src.config_utils import utc_ts

    ew = config["event_window"]
    start = datetime.fromtimestamp(utc_ts(ew["start_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
    end_excl = datetime.fromtimestamp(utc_ts(ew["end_utc_exclusive"]), tz=timezone.utc).strftime("%Y-%m-%d")
    launch = datetime.fromtimestamp(utc_ts(ew["launch_day_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
    refs = config.get("plot_reference_dates_utc") or []
    lift = "2023-04-29"
    if isinstance(refs, list) and len(refs) >= 2:
        lift = datetime.fromisoformat(str(refs[1]).replace("Z", "+00:00")).strftime("%Y-%m-%d")
    return start, end_excl, launch, lift


def ban_window_masks(
    dates: pd.Series,
    anchor: str,
    week_index: int,
) -> Tuple[pd.Series, pd.Series]:
    """Function summary: boolean masks for symmetric pre/post week windows around anchor.

    Parameters:
    - dates: date_utc strings or datetimes.
    - anchor: YYYY-MM-DD anchor (launch or lift).
    - week_index: 0=W0 ±7d, 1=W1 next 7d outward, etc.

    Returns:
    - Tuple (pre_mask, post_mask).
    """
    anchor_dt = datetime.strptime(anchor, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pre_start = anchor_dt - timedelta(days=7 * (week_index + 1))
    pre_end = anchor_dt - timedelta(days=7 * week_index)
    post_start = anchor_dt + timedelta(days=7 * week_index)
    post_end = anchor_dt + timedelta(days=7 * (week_index + 1))
    dts = pd.to_datetime(dates)
    pre = (dts >= pd.Timestamp(pre_start.date())) & (dts < pd.Timestamp(pre_end.date()))
    post = (dts >= pd.Timestamp(post_start.date())) & (dts < pd.Timestamp(post_end.date()))
    return pre, post


def grouped_trailing_daily_rolling(
    df_daily: pd.DataFrame,
    group_col: str,
    rolling_window_days: int,
    date_col: str = "date_utc",
) -> pd.DataFrame:
    """Function summary: trailing calendar-day rolling means by group (past-only).

    Parameters:
    - df_daily: daily aggregate table.
    - group_col: grouping column.
    - rolling_window_days: window length in days.
    - date_col: date column name.

    Returns:
    - Copy with numeric columns smoothed.
    """
    if df_daily.empty:
        return pd.DataFrame()
    if rolling_window_days <= 1:
        return df_daily.copy()
    exclude_cols = {
        date_col,
        "topic_family",
        "topic",
        "country_panel",
        "subreddit",
        "universe_slice",
        "series_id",
        group_col,
        "n_comments",
        "share_of_panel_comments",
    }
    d = df_daily.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([group_col, date_col])
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    if not numeric_cols:
        return d
    out_parts: List[pd.DataFrame] = []
    for _, grp in d.groupby(group_col, sort=True):
        g = grp.sort_values(date_col).copy()
        g_indexed = g.set_index(date_col)
        original_cols = g_indexed.columns
        rolled_numeric = g_indexed.loc[:, numeric_cols].rolling(
            window=f"{int(rolling_window_days)}D", min_periods=1
        ).mean()
        non_numeric = g_indexed.drop(columns=numeric_cols, errors="ignore")
        g_indexed = pd.concat([non_numeric, rolled_numeric], axis=1).reindex(columns=original_cols)
        g = g_indexed.reset_index()
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, date_col]).reset_index(drop=True)


def apply_min_n_filter(df: pd.DataFrame, min_n: int, n_col: str = "n_comments") -> pd.DataFrame:
    """Function summary: drop rows below minimum comment count per day/group.

    Parameters:
    - df: daily table.
    - min_n: minimum n_comments.
    - n_col: count column.

    Returns:
    - Filtered copy.
    """
    if df.empty or n_col not in df.columns:
        return df
    return df[df[n_col].astype(float) >= float(min_n)].copy()


def stamp_metrics_notes(
    notes_path: Path,
    config: Dict[str, Any],
    extra_sections: Optional[List[str]] = None,
) -> None:
    """Function summary: write dominant_v1 stamp and optional sections to metrics notes file.

    Parameters:
    - notes_path: output text path.
    - config: study config.
    - extra_sections: additional markdown-ish lines.

    Returns:
    - None.
    """
    require_dominant_v1_ideology_scoring(config)
    pol = load_polarization_config(config)
    primary = pol.get("primary_outcomes", {})
    lines = [
        "# Polarization metrics notes (auto-stamped)",
        f"ideology_scoring: {pol.get('ideology_scoring')}",
        f"stamped_at_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Pre-registered primary outcomes",
        f"- Ideology: {primary.get('ideology', 'net_ideology')} (dominant ideology_it.txt)",
        f"- Pair framing: {primary.get('pair_framing', 'pair_framing_net_strict')}",
        f"- Window: {primary.get('window', 'launch_w0')} symmetric around launch",
        f"- Weighting: {primary.get('weighting', 'comment_weighted')}",
        f"- Primary view: {primary.get('rolling_days', 7)}d trailing rolling",
        "",
        "## Threats to validity",
        "- Dominant v4 export applies to Italian (it) lexicons only; EN/DE/ES ideology lists remain hand-curated.",
        "- Within-arm Italy pre/post comparisons are identified if the export is frozen before window contrasts.",
        "- Cross-country level comparisons of net_ideology are not directly comparable without harmonized exports.",
        "- Pair/stance/valence metrics are zero outside Italian-primary shards by design.",
        "- Post+~7d after launch may reflect VPN circumvention; lift-window tables are appendix only.",
        "",
    ]
    if extra_sections:
        lines.extend(extra_sections)
        lines.append("")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines), encoding="utf-8")
