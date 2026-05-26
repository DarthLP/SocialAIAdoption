"""
Script summary:
Load and reshape Tor Metrics + Google Trends VPN circumvention proxies for DiD panels.

Functionality:
- Read combined circumvention CSVs from data/raw/circumvention/.
- Build geo-day and geo-period panels with post/treated indicators and optional transforms.
- Merge onto Reddit outcome panels (country geo or Italy-anchored by period_start).

How to apply/run:
- Imported by prepare_circumvention_descriptives.py, prepare_did_merged_panels.py,
  and prepare_semantic_axis_descriptives.py; not run standalone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config_utils import load_circumvention_config

# Italy national proxies (broadcast by period_start) for pooled cross-arm semantic DiD.
ITALY_INTENSITY_COLS: tuple[str, ...] = (
    "vpn_interest_it",
    "tor_relay_users_it",
    "tor_bridge_users_it",
    "tor_relay_frac_it",
    "tor_bridge_frac_it",
    "log1p_tor_bridge_users_it",
    "log1p_tor_relay_users_it",
    "vpn_interest_z_it",
)

# Geo-matched columns from merge_circumvention_by_geo (within-geo scale only).
GEO_MATCHED_INTENSITY_COLS: tuple[str, ...] = (
    "vpn_interest",
    "tor_relay_users",
    "tor_bridge_users",
    "tor_relay_frac",
    "tor_bridge_frac",
    "log1p_tor_bridge_users",
    "log1p_tor_relay_users",
    "vpn_interest_z",
)


def circumvention_raw_dir(project_root: Path, circ_cfg: Mapping[str, Any]) -> Path:
    """Function summary: resolve circumvention raw directory path.

    Parameters:
    - project_root: repository root.
    - circ_cfg: circumvention config block.

    Returns:
    - Absolute Path to data/raw/circumvention (or configured raw_dir).
    """
    raw = Path(str(circ_cfg.get("raw_dir", "data/raw/circumvention")))
    if raw.is_absolute():
        return raw
    return project_root / raw


def load_circumvention_daily(
    project_root: Path,
    config: Dict[str, Any],
    *,
    start: str | None = None,
    end_exclusive: str | None = None,
) -> pd.DataFrame:
    """Function summary: load merged VPN + Tor daily series by geo.

    Parameters:
    - project_root: repository root.
    - config: full study YAML.
    - start: optional YYYY-MM-DD clip start (inclusive).
    - end_exclusive: optional YYYY-MM-DD clip end (exclusive).

    Returns:
    - DataFrame with date_utc, geo, vpn_interest, tor_relay_users, tor_bridge_users, etc.
    """
    circ_cfg = load_circumvention_config(config)
    base = circumvention_raw_dir(project_root, circ_cfg)

    gt_path = base / str(circ_cfg.get("google_trends_combined", "google_trends_vpn_by_country.csv"))
    relay_path = base / str(circ_cfg.get("tor_relay_combined", "tor_relay_users_by_country.csv"))
    bridge_path = base / str(circ_cfg.get("tor_bridge_combined", "tor_bridge_users_by_country.csv"))

    if not gt_path.is_file():
        raise FileNotFoundError(f"Google Trends combined CSV not found: {gt_path}")

    gt = pd.read_csv(gt_path)
    gt["date_utc"] = pd.to_datetime(gt["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    gt["geo"] = gt["geo"].astype(str).str.upper()
    gt = gt.rename(columns={"vpn_interest": "vpn_interest"})
    gt_cols = ["date_utc", "geo", "vpn_interest"]
    for c in ("trends_query_type", "trends_mid"):
        if c in gt.columns:
            gt_cols.append(c)
    gt = gt[gt_cols].drop_duplicates(subset=["date_utc", "geo"], keep="last")

    daily = gt.copy()

    for path, prefix in ((relay_path, "tor_relay"), (bridge_path, "tor_bridge")):
        if not path.is_file():
            continue
        tor = pd.read_csv(path)
        tor["date_utc"] = pd.to_datetime(tor["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        geo_col = "query_country" if "query_country" in tor.columns else "country"
        tor["geo"] = tor[geo_col].astype(str).str.upper()
        keep = ["date_utc", "geo", "users"]
        if "frac" in tor.columns:
            keep.append("frac")
        tor = tor[keep].rename(
            columns={
                "users": f"{prefix}_users",
                "frac": f"{prefix}_frac",
            }
        )
        daily = daily.merge(tor, on=["date_utc", "geo"], how="outer")

    if start:
        daily = daily[daily["date_utc"] >= start]
    if end_exclusive:
        daily = daily[daily["date_utc"] < end_exclusive]

    treated_geo = str(circ_cfg.get("treated_geo", "IT")).upper()
    daily["treated"] = (daily["geo"] == treated_geo).astype(int)
    return daily.sort_values(["geo", "date_utc"]).reset_index(drop=True)


def _add_transforms(grp: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add log and within-geo z-scored VPN for one geo group."""
    out = grp.copy()
    if "tor_bridge_users" in out.columns:
        out["log1p_tor_bridge_users"] = np.log1p(out["tor_bridge_users"].astype(float))
    if "tor_relay_users" in out.columns:
        out["log1p_tor_relay_users"] = np.log1p(out["tor_relay_users"].astype(float))
    if "vpn_interest" in out.columns:
        v = out["vpn_interest"].astype(float)
        mu = v.mean()
        sd = v.std()
        out["vpn_interest_z"] = (v - mu) / sd if sd and sd > 0 else float("nan")
    return out


def build_circumvention_geo_panel(
    daily: pd.DataFrame,
    launch: str,
    bin_days: int,
    *,
    assign_period_start,
) -> pd.DataFrame:
    """Function summary: aggregate daily circumvention to geo x period_start bins.

    Parameters:
    - daily: output of load_circumvention_daily.
    - launch: YYYY-MM-DD launch anchor.
    - bin_days: 1, 3, or 7.
    - assign_period_start: callable(dates, bin_days, launch) -> period_start series.

    Returns:
    - Panel with geo, period_start, post, treated, mean VPN/Tor metrics.
    """
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["period_start"] = assign_period_start(work["date_utc"], int(bin_days), launch)
    work["post"] = (work["period_start"].astype(str) >= launch).astype(int)

    value_cols = [
        c
        for c in work.columns
        if c
        in (
            "vpn_interest",
            "tor_relay_users",
            "tor_bridge_users",
            "tor_relay_frac",
            "tor_bridge_frac",
            "log1p_tor_bridge_users",
            "log1p_tor_relay_users",
            "vpn_interest_z",
        )
    ]
    if not value_cols:
        value_cols = [c for c in ("vpn_interest", "tor_relay_users", "tor_bridge_users") if c in work.columns]

    agg_spec = {c: "mean" for c in value_cols}
    agg_spec["post"] = "max"
    agg_spec["treated"] = "max"
    agg_spec["date_utc"] = "count"

    panel = (
        work.groupby(["geo", "period_start"], as_index=False)
        .agg(agg_spec)
        .rename(columns={"date_utc": "n_days_in_bin"})
    )
    panel["bin_days"] = int(bin_days)
    return panel.sort_values(["geo", "period_start"]).reset_index(drop=True)


def italy_circumvention_by_period(
    panel: pd.DataFrame,
    it_geo: str = "IT",
) -> pd.DataFrame:
    """Function summary: one row per period_start with Italy VPN/Tor for left joins.

    Parameters:
    - panel: circumvention geo panel (any bin_days).
    - it_geo: treated country code.

    Returns:
    - DataFrame keyed by period_start with *_it suffix columns.
    """
    geo = str(it_geo).upper()
    it = panel[panel["geo"].astype(str).str.upper() == geo].copy()
    if it.empty:
        return pd.DataFrame()
    rename_map = {
        "vpn_interest": "vpn_interest_it",
        "tor_relay_users": "tor_relay_users_it",
        "tor_bridge_users": "tor_bridge_users_it",
        "tor_relay_frac": "tor_relay_frac_it",
        "tor_bridge_frac": "tor_bridge_frac_it",
        "log1p_tor_bridge_users": "log1p_tor_bridge_users_it",
        "log1p_tor_relay_users": "log1p_tor_relay_users_it",
        "vpn_interest_z": "vpn_interest_z_it",
    }
    cols = ["period_start"] + [c for c in rename_map if c in it.columns]
    out = it[cols].rename(columns={k: v for k, v in rename_map.items() if k in it.columns})
    return out.drop_duplicates(subset=["period_start"], keep="last")


def merge_circumvention_by_geo(
    panel: pd.DataFrame,
    circumvention: pd.DataFrame,
    geo_map: Mapping[str, str],
    *,
    panel_geo_col: str = "country_panel",
    date_col: str = "period_start",
) -> pd.DataFrame:
    """Function summary: left-join geo-matched VPN/Tor onto a Reddit country panel.

    Parameters:
    - panel: outcome panel with country_panel and period_start or date_utc.
    - circumvention: circumvention_panel_by_geo (same bin_days).
    - geo_map: country_panel label -> ISO geo (e.g. it_political -> IT).
    - panel_geo_col: column holding country panel id.
    - date_col: date key on panel (period_start or date_utc).

    Returns:
    - Merged panel with geo, vpn_interest, tor_* , post_circ, treated_circ.
    """
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    out["geo"] = out[panel_geo_col].astype(str).map(dict(geo_map))
    circ = circumvention.copy()
    circ = circ.rename(
        columns={
            "post": "post_circ",
            "treated": "treated_circ",
        }
    )
    join_cols = ["geo", date_col]
    merge_on_left = date_col
    if date_col not in circ.columns and "period_start" in circ.columns:
        circ = circ.rename(columns={"period_start": date_col})
    out = out.merge(
        circ,
        left_on=["geo", merge_on_left],
        right_on=["geo", date_col],
        how="left",
        suffixes=("", "_circ_dup"),
    )
    drop_dup = [c for c in out.columns if c.endswith("_circ_dup")]
    if drop_dup:
        out = out.drop(columns=drop_dup)
    return out


def attach_italy_circumvention_columns(
    panel: pd.DataFrame,
    italy_by_period: pd.DataFrame,
    *,
    panel_level: str | None = None,
    primary_lexicon_col: str = "primary_lexicon",
) -> pd.DataFrame:
    """Function summary: left-join Italy VPN/Tor on period_start; NaN for non-it language rows.

    Parameters:
    - panel: semantic or forum panel with period_start.
    - italy_by_period: from italy_circumvention_by_period.
    - panel_level: when language, only fill IT lexicon rows.
    - primary_lexicon_col: lexicon column name.

    Returns:
    - Panel with vpn_interest_it, tor_*_it, post (from circumvention if missing).
    """
    if panel.empty or italy_by_period.empty:
        return panel.copy()
    out = panel.merge(italy_by_period, on="period_start", how="left", suffixes=("", "_itdup"))
    dup = [c for c in out.columns if c.endswith("_itdup")]
    if dup:
        out = out.drop(columns=dup)
    if panel_level == "language" and primary_lexicon_col in out.columns:
        not_it = out[primary_lexicon_col].astype(str).str.lower() != "it"
        for col in (
            "vpn_interest_it",
            "tor_bridge_users_it",
            "tor_relay_users_it",
            "log1p_tor_bridge_users_it",
            "vpn_interest_z_it",
        ):
            if col in out.columns:
                out.loc[not_it, col] = float("nan")
    return out


def enrich_daily_with_transforms(daily: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add log1p and within-geo z-scores to daily circumvention frame.

    Parameters:
    - daily: load_circumvention_daily output.

    Returns:
    - Copy with transform columns per geo.
    """
    if daily.empty:
        return daily.copy()
    parts = [_add_transforms(grp) for _, grp in daily.groupby("geo", sort=True)]
    return pd.concat(parts, ignore_index=True)
