"""
Script summary:
Exploratory participation/volume descriptives by comment language with ban-window shading.

Functionality:
- Scans enriched Parquet shards for quantity margins (no text): comments/day, authors/day,
  comments/author, author entry (new_authors), returning comment share, and churn exit proxy.
- Groups by lang_comment (it/en/de) and by DiD arm (topic_family); splits all / political / non_political.
- Burn-in masks new_authors and returning_author_comment_share before start+burn_in_days (default 14).
- Plots daily lines with 7-day trailing rolling means (NaN-preserving) and ban shading.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_participation_ban_shaded.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_participation_ban_shaded.py --config config/italy_polarization_setup.yaml --burn-in-days 14
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
CHURN_LOOKAHEAD_DAYS = 7
LANGUAGES = ("it", "en", "de")
UNIVERSE_SLICES = ("all", "political", "non_political")

METRIC_COLUMNS = (
    "n_comments",
    "n_authors",
    "comments_per_author",
    "new_authors",
    "returning_author_comment_share",
    "churned_authors",
)

METRIC_LABELS: Dict[str, str] = {
    "n_comments": "Comments per day",
    "n_authors": "Unique authors per day",
    "comments_per_author": "Comments per author",
    "new_authors": "New authors (entry)",
    "returning_author_comment_share": "Share of comments from returning authors",
    "churned_authors": "Churned authors (no activity in next 7d)",
}

LANGUAGE_STYLE: Dict[str, Dict[str, object]] = {
    "it": {"color": "#c1121f", "linewidth": 2.5, "label": "Italian (IT)"},
    "en": {"color": "#457b9d", "linewidth": 1.5, "label": "English (EN)"},
    "de": {"color": "#2a9d8f", "linewidth": 1.5, "label": "German (DE)"},
}

SLICE_TITLES: Dict[str, str] = {
    "all": "All comments",
    "political": "Political universe",
    "non_political": "Non-political universe",
}

BASE_READ_COLUMNS = ("author", "date_utc", "lang_comment", "topic_family")


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute path to repository root.
    """
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from scripts.diagnostics.descriptives_util import (  # noqa: E402
    event_dates_from_config,
    grouped_trailing_daily_rolling,
)
from scripts.diagnostics.prepare_polarization_descriptives import (  # noqa: E402
    COUNTRY_PANEL_FAMILIES,
)
from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    resolve_primary_subreddits,
    subreddit_family_map,
    tables_subdir,
)

BURN_IN_METRICS = ("new_authors", "returning_author_comment_share")
DID_ARMS = tuple(COUNTRY_PANEL_FAMILIES.keys())

ARM_STYLE: Dict[str, Dict[str, object]] = {
    "it_political": {"color": "#c1121f", "linewidth": 2.5, "label": "Italy political"},
    "it_others": {"color": "#e76f51", "linewidth": 2.0, "label": "Italy others"},
    "de": {"color": "#2a9d8f", "linewidth": 1.5, "label": "Germany"},
    "eu": {"color": "#457b9d", "linewidth": 1.5, "label": "EU hub (EN)"},
    "uk": {"color": "#6a4c93", "linewidth": 1.5, "label": "UK"},
    "us": {"color": "#264653", "linewidth": 1.5, "label": "US"},
}


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for participation ban-window descriptives.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Ban-window participation/volume descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=7)
    parser.add_argument(
        "--burn-in-days",
        type=int,
        default=14,
        help="Days from window start where new_authors and returning_author_comment_share are NaN.",
    )
    return parser.parse_args()


def _shard_read_columns(path: Path) -> List[str]:
    """Function summary: project only columns present in one Parquet shard.

    Parameters:
    - path: path to a monthly shard file.

    Returns:
    - Column list safe for pd.read_parquet(columns=...).
    """
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in BASE_READ_COLUMNS if c in available]
    if "comment_in_political_universe" in available:
        cols.append("comment_in_political_universe")
    return cols


def load_comment_frame(shard_root: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load author/date/language rows from enriched monthly shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks directory.
    - subreddits: primary subreddit names from config.

    Returns:
    - Combined comment frame with optional political-universe flag and topic_family.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        for shard in sorted(shard_dir.glob("*.parquet")):
            try:
                cols = _shard_read_columns(shard)
                if "author" not in cols or "date_utc" not in cols or "lang_comment" not in cols:
                    continue
                chunk = pd.read_parquet(shard, columns=cols)
            except Exception:
                continue
            chunk["subreddit"] = sub
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "comment_in_political_universe" not in out.columns:
        out["comment_in_political_universe"] = pd.NA
    return out


def _slice_mask(df: pd.DataFrame, universe_slice: str) -> pd.Series:
    """Function summary: boolean mask for a universe slice.

    Parameters:
    - df: comment frame with comment_in_political_universe.
    - universe_slice: all, political, or non_political.

    Returns:
    - Boolean Series aligned to df.
    """
    if universe_slice == "all":
        return pd.Series(True, index=df.index)
    flag = df["comment_in_political_universe"]
    if universe_slice == "political":
        return flag.astype("boolean") == True  # noqa: E712
    return flag.astype("boolean") == False  # noqa: E712


def _author_active_dates(df: pd.DataFrame) -> DefaultDict[str, Set[str]]:
    """Function summary: map author id to set of active calendar dates.

    Parameters:
    - df: filtered comment rows for one language x slice.

    Returns:
    - DefaultDict author -> set of YYYY-MM-DD strings.
    """
    out: DefaultDict[str, Set[str]] = defaultdict(set)
    for author, day in zip(df["author"].astype(str), df["date_utc"].astype(str)):
        if author and day:
            out[author].add(day)
    return out


def _calendar_days(start: str, end_exclusive: str) -> List[str]:
    """Function summary: enumerate inclusive-start exclusive-end UTC calendar days.

    Parameters:
    - start: YYYY-MM-DD window start (inclusive).
    - end_exclusive: YYYY-MM-DD window end (exclusive).

    Returns:
    - Sorted list of date strings.
    """
    cur = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end_exclusive, "%Y-%m-%d")
    days: List[str] = []
    while cur < end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def daily_participation_table(
    df: pd.DataFrame,
    language: str,
    universe_slice: str,
    window_days: List[str],
) -> pd.DataFrame:
    """Function summary: compute daily participation metrics for one language x slice.

    Parameters:
    - df: event-window comments with lang_comment and political flag.
    - language: it, en, or de.
    - universe_slice: all, political, or non_political.
    - window_days: full calendar day list for the event window.

    Returns:
    - Daily metrics DataFrame (raw, unsmoothed).
    """
    lang_df = df[df["lang_comment"].astype(str).str.lower() == language].copy()
    lang_df = lang_df[_slice_mask(lang_df, universe_slice)]
    if lang_df.empty:
        return pd.DataFrame()

    author_dates = _author_active_dates(lang_df)
    first_seen: Dict[str, str] = {}
    for author, dates in author_dates.items():
        first_seen[author] = min(dates)

    last_churn_day = (
        datetime.strptime(window_days[-1], "%Y-%m-%d") - timedelta(days=CHURN_LOOKAHEAD_DAYS)
    ).strftime("%Y-%m-%d")

    rows: List[Dict[str, object]] = []
    for day in window_days:
        day_df = lang_df[lang_df["date_utc"].astype(str) == day]
        n_comments = int(len(day_df))
        authors_today = set(day_df["author"].astype(str))
        n_authors = len(authors_today)
        comments_per_author = (
            float(n_comments) / float(n_authors) if n_authors > 0 else float("nan")
        )

        new_authors = sum(1 for a in authors_today if first_seen.get(a) == day)
        if n_comments > 0:
            returning_comments = sum(
                1
                for a in day_df["author"].astype(str)
                if any(d < day for d in author_dates.get(a, set()))
            )
            returning_share = float(returning_comments) / float(n_comments)
        else:
            returning_share = float("nan")

        if day <= last_churn_day:
            churned = 0
            for author in authors_today:
                dates = author_dates.get(author, set())
                future = {
                    d
                    for d in dates
                    if d > day
                    and d <= (
                        datetime.strptime(day, "%Y-%m-%d") + timedelta(days=CHURN_LOOKAHEAD_DAYS)
                    ).strftime("%Y-%m-%d")
                }
                if not future:
                    churned += 1
            churned_authors = churned
        else:
            churned_authors = float("nan")

        rows.append(
            {
                "language": language,
                "universe_slice": universe_slice,
                "date_utc": day,
                "n_comments": n_comments,
                "n_authors": n_authors,
                "comments_per_author": comments_per_author,
                "new_authors": new_authors,
                "returning_author_comment_share": returning_share,
                "churned_authors": churned_authors,
            }
        )
    return pd.DataFrame(rows)


def _burn_in_cutoff(start: str, burn_in_days: int) -> str:
    """Function summary: first calendar day where entry/return metrics are reported.

    Parameters:
    - start: event window start YYYY-MM-DD (inclusive).
    - burn_in_days: days from start before metrics are left-censored.

    Returns:
    - YYYY-MM-DD cutoff (inclusive first valid day).
    """
    return (
        datetime.strptime(start, "%Y-%m-%d") + timedelta(days=int(burn_in_days))
    ).strftime("%Y-%m-%d")


def apply_burn_in_mask(panel: pd.DataFrame, start: str, burn_in_days: int) -> pd.DataFrame:
    """Function summary: NaN entry/return metrics before burn-in cutoff (left-censoring).

    Parameters:
    - panel: daily participation table with date_utc.
    - start: event window start YYYY-MM-DD.
    - burn_in_days: burn-in length in days (default 14 -> cutoff 2023-03-15).

    Returns:
    - Copy with new_authors and returning_author_comment_share masked before cutoff.
    """
    if panel.empty or burn_in_days <= 0:
        return panel.copy()
    out = panel.copy()
    cutoff = _burn_in_cutoff(start, burn_in_days)
    mask = out["date_utc"].astype(str) < cutoff
    for col in BURN_IN_METRICS:
        if col in out.columns:
            out.loc[mask, col] = float("nan")
    return out


def build_participation_panel(df: pd.DataFrame, start: str, end_exclusive: str) -> pd.DataFrame:
    """Function summary: assemble long daily panel for all languages and universe slices.

    Parameters:
    - df: filtered comment frame in event window.
    - start: YYYY-MM-DD inclusive.
    - end_exclusive: YYYY-MM-DD exclusive.

    Returns:
    - Long daily participation table.
    """
    window_days = _calendar_days(start, end_exclusive)
    parts: List[pd.DataFrame] = []
    for language in LANGUAGES:
        for universe_slice in UNIVERSE_SLICES:
            part = daily_participation_table(df, language, universe_slice, window_days)
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["universe_slice", "language", "date_utc"]
    )


def daily_participation_table_by_arm(
    df: pd.DataFrame,
    topic_family: str,
    universe_slice: str,
    window_days: List[str],
) -> pd.DataFrame:
    """Function summary: daily participation metrics for one DiD arm x universe slice.

    Parameters:
    - df: event-window comments with topic_family and political flag.
    - topic_family: DiD arm id (it_political, de, us, ...).
    - universe_slice: all, political, or non_political.
    - window_days: full calendar day list for the event window.

    Returns:
    - Daily metrics DataFrame (raw, unsmoothed).
    """
    arm_df = df[df["topic_family"].astype(str) == topic_family].copy()
    arm_df = arm_df[_slice_mask(arm_df, universe_slice)]
    if arm_df.empty:
        return pd.DataFrame()

    author_dates = _author_active_dates(arm_df)
    first_seen: Dict[str, str] = {}
    for author, dates in author_dates.items():
        first_seen[author] = min(dates)

    last_churn_day = (
        datetime.strptime(window_days[-1], "%Y-%m-%d") - timedelta(days=CHURN_LOOKAHEAD_DAYS)
    ).strftime("%Y-%m-%d")
    country_panel = COUNTRY_PANEL_FAMILIES.get(topic_family, topic_family)

    rows: List[Dict[str, object]] = []
    for day in window_days:
        day_df = arm_df[arm_df["date_utc"].astype(str) == day]
        n_comments = int(len(day_df))
        authors_today = set(day_df["author"].astype(str))
        n_authors = len(authors_today)
        comments_per_author = (
            float(n_comments) / float(n_authors) if n_authors > 0 else float("nan")
        )

        new_authors = sum(1 for a in authors_today if first_seen.get(a) == day)
        if n_comments > 0:
            returning_comments = sum(
                1
                for a in day_df["author"].astype(str)
                if any(d < day for d in author_dates.get(a, set()))
            )
            returning_share = float(returning_comments) / float(n_comments)
        else:
            returning_share = float("nan")

        if day <= last_churn_day:
            churned = 0
            for author in authors_today:
                dates = author_dates.get(author, set())
                future = {
                    d
                    for d in dates
                    if d > day
                    and d <= (
                        datetime.strptime(day, "%Y-%m-%d") + timedelta(days=CHURN_LOOKAHEAD_DAYS)
                    ).strftime("%Y-%m-%d")
                }
                if not future:
                    churned += 1
            churned_authors = churned
        else:
            churned_authors = float("nan")

        rows.append(
            {
                "topic_family": topic_family,
                "country_panel": country_panel,
                "universe_slice": universe_slice,
                "date_utc": day,
                "n_comments": n_comments,
                "n_authors": n_authors,
                "comments_per_author": comments_per_author,
                "new_authors": new_authors,
                "returning_author_comment_share": returning_share,
                "churned_authors": churned_authors,
            }
        )
    return pd.DataFrame(rows)


def assign_arm_topic_family(
    df: pd.DataFrame,
    family_map: Dict[str, str],
    did_arms: Iterable[str] = DID_ARMS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Function summary: assign DiD arm labels from shard topic_family with YAML fallback.

    Parameters:
    - df: comment frame with subreddit and optional topic_family from shards.
    - family_map: subreddit -> topic_family from config (fallback when shard label missing).
    - did_arms: valid DiD arm ids to retain.

    Returns:
    - Tuple of (filtered frame with topic_family, sorted subreddits that used fallback).
    """
    did_arms_set = set(did_arms)
    out = df.copy()
    total = len(out)

    if "topic_family" not in out.columns:
        out["topic_family"] = out["subreddit"].map(family_map)
        fallback_subs = sorted(out["subreddit"].astype(str).unique())
        from_shard = 0
    else:
        shard_col = out["topic_family"].astype(str)
        valid_shard = shard_col.notna() & (shard_col != "") & (shard_col != "nan")
        valid_shard &= shard_col.isin(did_arms_set)
        from_shard = int(valid_shard.sum())
        need_fallback = ~valid_shard
        if need_fallback.any():
            out.loc[need_fallback, "topic_family"] = out.loc[
                need_fallback, "subreddit"
            ].map(family_map)
        fallback_subs = sorted(
            out.loc[need_fallback, "subreddit"].astype(str).unique().tolist()
        )

    out = out[out["topic_family"].astype(str).isin(did_arms_set)].copy()
    print(
        f"[plot_participation_ban_shaded] Arm assignment: {from_shard}/{total} comments "
        f"from shard topic_family; {len(fallback_subs)} subreddits used YAML fallback: "
        f"{fallback_subs if fallback_subs else '(none — all arms from shards)'}",
        flush=True,
    )
    return out, fallback_subs


def build_participation_panel_by_arm(
    df: pd.DataFrame,
    start: str,
    end_exclusive: str,
) -> pd.DataFrame:
    """Function summary: assemble long daily panel for all DiD arms and universe slices.

    Parameters:
    - df: filtered comment frame in event window with topic_family.
    - start: YYYY-MM-DD inclusive.
    - end_exclusive: YYYY-MM-DD exclusive.

    Returns:
    - Long daily participation table by topic_family.
    """
    window_days = _calendar_days(start, end_exclusive)
    parts: List[pd.DataFrame] = []
    for topic_family in DID_ARMS:
        for universe_slice in UNIVERSE_SLICES:
            part = daily_participation_table_by_arm(
                df, topic_family, universe_slice, window_days
            )
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["universe_slice", "topic_family", "date_utc"]
    )


def _plot_metric_slice(
    panel: pd.DataFrame,
    metric: str,
    universe_slice: str,
    out_path: Path,
    rolling_window: int,
) -> None:
    """Function summary: plot one metric for one universe slice across languages.

    Parameters:
    - panel: full participation panel (raw values).
    - metric: column to plot on y-axis.
    - universe_slice: all, political, or non_political.
    - out_path: PNG destination.
    - rolling_window: trailing rolling window in days.

    Returns:
    - None; writes PNG when data exist.
    """
    sub = panel[panel["universe_slice"] == universe_slice].copy()
    if sub.empty or metric not in sub.columns:
        return

    roll = grouped_trailing_daily_rolling(
        sub.rename(columns={metric: "value"}),
        group_col="language",
        rolling_window_days=rolling_window,
        date_col="date_utc",
    )
    roll["date_utc"] = pd.to_datetime(roll["date_utc"])
    roll = roll.rename(columns={"value": metric})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for language in LANGUAGES:
        lang = roll[roll["language"] == language]
        if lang.empty:
            continue
        style = LANGUAGE_STYLE[language]
        ax.plot(
            lang["date_utc"],
            lang[metric],
            color=style["color"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
    ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
    ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="--", linewidth=0.9)
    ax.set_title(f"{METRIC_LABELS[metric]} — {SLICE_TITLES[universe_slice]}")
    ax.set_xlabel("Date (UTC)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_slice_by_arm(
    panel: pd.DataFrame,
    metric: str,
    universe_slice: str,
    out_path: Path,
    rolling_window: int,
) -> None:
    """Function summary: plot one metric for one universe slice across DiD arms.

    Parameters:
    - panel: full arm participation panel (raw values).
    - metric: column to plot on y-axis.
    - universe_slice: all, political, or non_political.
    - out_path: PNG destination.
    - rolling_window: trailing rolling window in days.

    Returns:
    - None; writes PNG when data exist.
    """
    sub = panel[panel["universe_slice"] == universe_slice].copy()
    if sub.empty or metric not in sub.columns:
        return

    roll = grouped_trailing_daily_rolling(
        sub.rename(columns={metric: "value"}),
        group_col="topic_family",
        rolling_window_days=rolling_window,
        date_col="date_utc",
    )
    roll["date_utc"] = pd.to_datetime(roll["date_utc"])
    roll = roll.rename(columns={"value": metric})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for topic_family in DID_ARMS:
        arm = roll[roll["topic_family"] == topic_family]
        if arm.empty:
            continue
        style = ARM_STYLE.get(
            topic_family,
            {"color": "0.4", "linewidth": 1.5, "label": topic_family},
        )
        ax.plot(
            arm["date_utc"],
            arm[metric],
            color=style["color"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
    ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
    ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="--", linewidth=0.9)
    ax.set_title(f"{METRIC_LABELS[metric]} — {SLICE_TITLES[universe_slice]} (by DiD arm)")
    ax.set_xlabel("Date (UTC)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, burn_in_days: int = 14, start: str = "2023-03-01") -> None:
    """Function summary: write exploratory framing and coverage caveats.

    Parameters:
    - path: output text file path.

    Returns:
    - None.
    """
    lines = [
        "Participation margins descriptives (exploratory).",
        "",
        "Framing: quantity outcomes are two-sided (substitution up vs. attention elsewhere);",
        "expect modest ban-window signal (no large volume spike in prior descriptives).",
        "",
        "Grouping: comment language (lang_comment in it/en/de), not forum.",
        "Universe slices: all | political (comment_in_political_universe=True) |",
        "non_political (flag=False). Shards without the flag contribute to all only.",
        "",
        "Entry: new_authors = first-seen date within window equals day.",
        "Composition: returning_author_comment_share = share of day's comments from authors",
        "active on strictly earlier days.",
        "Exit proxy: churned_authors = active on day d with no activity in (d, d+7];",
        f"NaN for final {CHURN_LOOKAHEAD_DAYS} calendar days (right-censoring).",
        "",
        f"Burn-in: new_authors and returning_author_comment_share are NaN before "
        f"{_burn_in_cutoff(start, burn_in_days)} ({burn_in_days} days from window start) "
        "because first_seen is computed inside the Mar–Apr window only (left-censoring).",
        "",
        "Arm grouping: topic_family (six DiD arms) mirrors cross_country_all panels; "
        "prefers enrich-time topic_family on shards; YAML subreddit_family_map is fallback only; "
        "see daily_participation_by_arm.csv and ban_window/participation_by_arm/.",
        "",
        "Figures use 7-day trailing rolling means; CSV stores raw daily values.",
        "Rolling means re-mask NaN raw days (no bridging across right-censored churn tail).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: scan shards, write participation table, and export ban-window figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    tables_out = tables_subdir(config, "descriptives")
    figures_out = figures_subdir(config, "descriptives") / "ban_window" / "participation"
    tables_out.mkdir(parents=True, exist_ok=True)

    start, end_excl, _, _ = event_dates_from_config(config)
    subs = resolve_primary_subreddits(config)

    df = load_comment_frame(shard_root, subs)
    if df.empty:
        print("[plot_participation_ban_shaded] no parquet data found", flush=True)
        return

    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    df["lang_comment"] = df["lang_comment"].astype(str).str.lower()
    df = df[df["lang_comment"].isin(LANGUAGES)].copy()

    panel = build_participation_panel(df, start, end_excl)
    panel = apply_burn_in_mask(panel, start, args.burn_in_days)
    table_path = tables_out / "daily_participation_by_language.csv"
    panel.to_csv(table_path, index=False)
    write_notes(
        tables_out / "participation_margins_notes.txt",
        burn_in_days=args.burn_in_days,
        start=start,
    )
    print(f"[plot_participation_ban_shaded] wrote {table_path}", flush=True)

    family_map = subreddit_family_map(config)
    df_arm, _fallback_subs = assign_arm_topic_family(df.copy(), family_map)
    panel_arm = build_participation_panel_by_arm(df_arm, start, end_excl)
    panel_arm = apply_burn_in_mask(panel_arm, start, args.burn_in_days)
    arm_table_path = tables_out / "daily_participation_by_arm.csv"
    panel_arm.to_csv(arm_table_path, index=False)
    print(f"[plot_participation_ban_shaded] wrote {arm_table_path}", flush=True)

    if panel.empty and panel_arm.empty:
        print("[plot_participation_ban_shaded] empty participation panels", flush=True)
        return

    n_plots = 0
    for universe_slice in UNIVERSE_SLICES:
        for metric in METRIC_COLUMNS:
            out_path = figures_out / universe_slice / f"{metric}.png"
            if not panel.empty:
                _plot_metric_slice(panel, metric, universe_slice, out_path, args.rolling_window)
            if out_path.is_file():
                n_plots += 1
                print(
                    f"[plot_participation_ban_shaded] {universe_slice}/{metric}.png",
                    flush=True,
                )

    figures_arm_out = figures_subdir(config, "descriptives") / "ban_window" / "participation_by_arm"
    for universe_slice in UNIVERSE_SLICES:
        for metric in METRIC_COLUMNS:
            out_path = figures_arm_out / universe_slice / f"{metric}.png"
            if not panel_arm.empty:
                _plot_metric_slice_by_arm(
                    panel_arm, metric, universe_slice, out_path, args.rolling_window
                )
            if out_path.is_file():
                n_plots += 1
                print(
                    f"[plot_participation_ban_shaded] participation_by_arm/"
                    f"{universe_slice}/{metric}.png",
                    flush=True,
                )
    print(f"[plot_participation_ban_shaded] exported {n_plots} figures", flush=True)


if __name__ == "__main__":
    main()
