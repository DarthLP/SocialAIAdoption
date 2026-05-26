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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def assign_period_start(
    dates: pd.Series,
    bin_days: int,
    launch: str,
) -> pd.Series:
    """Function summary: map calendar dates to period_start for daily or launch-aligned bins.

    Parameters:
    - dates: YYYY-MM-DD strings.
    - bin_days: 1 (calendar day), 3, or 7 (launch-aligned).
    - launch: launch anchor YYYY-MM-DD.

    Returns:
    - period_start as YYYY-MM-DD strings.
    """
    dt = pd.to_datetime(dates.astype(str))
    if bin_days <= 1:
        return dt.dt.strftime("%Y-%m-%d")
    launch_dt = datetime.strptime(launch, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    launch_ts = pd.Timestamp(launch_dt.date())
    days_from_launch = (dt - launch_ts).dt.days
    bin_index = np.floor(days_from_launch.astype(float) / float(bin_days)).astype(int)
    period_dt = launch_ts + pd.to_timedelta(bin_index * bin_days, unit="D")
    return period_dt.dt.strftime("%Y-%m-%d")


def bin_lexical_daily_panel(
    panel: pd.DataFrame,
    entity_cols: Sequence[str],
    bin_days: int,
    launch: str,
) -> pd.DataFrame:
    """Function summary: rollup daily lexical country panels to launch-aligned period bins.

    Parameters:
    - panel: daily table with date_utc, n_comments, and outcome metrics.
    - entity_cols: grouping keys (e.g. country_panel or country_panel + universe_slice).
    - bin_days: 1, 3, or 7.
    - launch: ban launch YYYY-MM-DD for post flag and bin alignment.

    Returns:
    - Binned panel with period_start, n_days_in_bin, is_partial_bin, post, bin_days.
    """
    if panel.empty:
        return panel.copy()
    work = panel.copy()
    if "date_utc" not in work.columns:
        raise ValueError("bin_lexical_daily_panel requires date_utc column")
    bd = int(bin_days)
    if bd <= 1:
        out = work.rename(columns={"date_utc": "period_start"})
        out["n_days_in_bin"] = 1
        out["is_partial_bin"] = False
        out["bin_days"] = 1
        out["post"] = (out["period_start"].astype(str) >= str(launch)).astype(int)
        return out

    work["period_start"] = assign_period_start(work["date_utc"], bd, launch)
    group_cols = list(entity_cols) + ["period_start"]
    skip = set(entity_cols) | {"date_utc", "period_start", "n_comments"}
    outcome_cols = [
        c
        for c in work.columns
        if c not in skip and pd.api.types.is_numeric_dtype(work[c])
    ]
    records: List[Dict[str, Any]] = []
    for key_vals, grp in work.groupby(group_cols, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        w = grp["n_comments"].astype(float)
        row: Dict[str, Any] = dict(zip(group_cols, key_vals))
        row["n_comments"] = int(grp["n_comments"].sum())
        row["n_days_in_bin"] = int(grp["date_utc"].astype(str).nunique())
        row["is_partial_bin"] = bool(row["n_days_in_bin"] < bd)
        row["bin_days"] = bd
        row["post"] = int(str(row["period_start"]) >= str(launch))
        for col in outcome_cols:
            row[col] = weighted_mean(grp[col], w)
        records.append(row)
    return pd.DataFrame(records)


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
        "## Circumvention / DiD",
        "- Google Trends vpn_interest is within-geo over time only; do not compare levels across countries.",
        "- Tor Metrics user counts have sparse calendar days; missing days stay NaN (not zero-filled).",
        "- circumvention_panel_by_geo: treated=IT tests ban effect on VPN/Tor (first stage).",
        "- Lexical DiD: did_country_panel_{1,3,7}d (+ by_universe_slice); geo-matched vpn_interest/tor_* per country_panel row.",
        "- Semantic DiD: did_semantic_{topic_family,language,language_universe}_{1,3,7}d; intensity interactions use vpn_interest_it / tor_*_it only (not geo-matched columns).",
        "- Cross-country semantic arms: topic_family (six arms, mirrors country_panel IT split); language (it/en/de); political universe via universe_slice on language_universe panel.",
        "- Launch-aligned 3d/7d bins: n_days_in_bin and is_partial_bin on panels; weight partial endpoint bins by n_comments and/or n_days_in_bin/bin_days.",
        "- EU_hub_en has no Trends geo in country_panel_geo_map; VPN columns are omitted for that arm.",
        "",
        "## Semantic axis (embedding outcomes)",
        "- DiD outcomes: use sem_axis_{ideology,emotion,aggression}_mean within language/arm; expect null treatment effects.",
        "- Do not compare raw pole-share levels across languages (separate FastText spaces).",
        "- Pole buckets: per-lexicon abs thresholds + p10/p90 percentile columns; tau50/tau75 removed.",
        "- sem_axis_coverage_mean is saturated (~1); use share_unscored and seed OOV tables instead.",
        "- Ideology axis must pass ideology_axis_orientation_report.csv vs net_ideology before substantive claims.",
        "- did_semantic_topic_family_* must include all six topic_family arms (us, eu, de, uk, it_*).",
        "- Forum-level panel: semantic_axis_panel_by_forum_1d.csv (no separate semantic_axis_panel.csv alias).",
        "",
    ]
    if extra_sections:
        lines.extend(extra_sections)
        lines.append("")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines), encoding="utf-8")
