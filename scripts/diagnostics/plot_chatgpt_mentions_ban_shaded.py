"""
Script summary:
Daily ChatGPT/AI keyword mention rate with Italy vs control ban-window shading.

Functionality:
- Scans enriched comment Parquet shards for multi-language ChatGPT/AI-related terms.
- Aggregates chatgpt_mention_rate_100w by topic_family and country panel.
- Plots 2x2 Italy vs control panels with 7-day trailing rolling means and ban shading.
- Plots single-panel Italy vs word-weighted pooled controls (de+eu+uk+us) with optional
  3-day trailing smoothing and faint raw daily points.
- Plots Italy vs pooled controls plus a min–max band across individual control panels
  (`chatgpt_mention_rate_100w_pooled_range.png`).

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_chatgpt_mentions_ban_shaded.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_chatgpt_mentions_ban_shaded.py --config config/italy_polarization_setup.yaml --pooled-smoothing 3
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
CONTROL_PANELS = ("Germany", "EU_hub_en", "UK", "US_political")
ITALY_PANELS = ("Italy_political", "Italy_others")
ITALY_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})
CONTROL_TOPIC_FAMILIES = frozenset({"de", "eu", "uk", "us"})
POOL_GROUP_LABELS = ("Italy", "Controls")
OUTCOME_ID = "chatgpt_mention_rate_100w"
POOLED_OUTCOME_ID = "chatgpt_mention_rate_100w_pooled"
POOLED_RANGE_OUTCOME_ID = "chatgpt_mention_rate_100w_pooled_range"
OUTCOME_TITLE = "ChatGPT/AI mention rate (per 100 words)"

SANITY_TOLERANCE_RTOL = 0.15
SANITY_BENCHMARKS: Dict[str, Tuple[str, str, str, float]] = {
    "it_pre": ("Italy", "2023-03-01", "2023-03-30", 0.033),
    "it_ban_week1": ("Italy", "2023-03-31", "2023-04-06", 0.18),
    "ctrl_pre": ("Controls", "2023-03-01", "2023-03-30", 0.015),
    "ctrl_ban_week1": ("Controls", "2023-03-31", "2023-04-06", 0.039),
}
SANITY_IT_PEAK_DATE = "2023-04-01"
SANITY_IT_PEAK_RATE = 0.36

CONTROL_PANEL_DISPLAY: Dict[str, str] = {
    "Germany": "Germany",
    "EU_hub_en": "EU_hub_en",
    "UK": "UK",
    "US_political": "US",
}

# Case-insensitive, word-boundary phrases (unambiguous product/field terms).
CHATGPT_KEYWORDS_CI: Tuple[str, ...] = (
    r"chat\s*gpt",
    r"gpt",
    r"openai",
    r"open\s*ai",
    r"\bllm\b",
    r"large\s+language\s+model",
    r"chatbot",
    r"artificial\s+intelligence",
    r"intelligenza\s+artificiale",
    r"intelligence\s+artificielle",
    r"k[uü]nstliche\s+intelligenz",
    r"machine\s+learning",
    r"modell[oi]\s+linguistic[oi]",
    r"generative\s+ai",
)

# Uppercase acronyms only (avoids Italian preposition false positives like "ia"/"ai").
CHATGPT_KEYWORDS_CS: Tuple[str, ...] = (
    r"\bAI\b",
    r"\bA\.I\.\b",
    r"\bIA\b",
    r"\bKI\b",
)

READ_COLUMNS = ("body", "n_words", "date_utc", "topic_family", "subreddit")


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
from src.plotting.thesis_theme import (  # noqa: E402
    THESIS_CONTROL,
    THESIS_CONTROL_BAND,
    THESIS_ITALY,
    XLABEL_CALENDAR,
    shade_ban_window,
)

POOL_GROUP_STYLE: Dict[str, Dict[str, object]] = {
    "Italy": {"color": THESIS_ITALY, "linewidth": 2.2, "label": "Italy"},
    "Controls": {"color": THESIS_CONTROL, "linewidth": 1.5, "label": "Pooled controls"},
}


def _compile_keyword_patterns() -> Tuple[List[Pattern[str]], List[Pattern[str]]]:
    """Function summary: compile case-insensitive and case-sensitive mention regexes.

    Returns:
    - Tuple of (ci_patterns, cs_patterns) ready for findall.
    """
    ci = [re.compile(rf"(?:{pat})", re.IGNORECASE) for pat in CHATGPT_KEYWORDS_CI]
    cs = [re.compile(pat) for pat in CHATGPT_KEYWORDS_CS]
    return ci, cs


CI_PATTERNS, CS_PATTERNS = _compile_keyword_patterns()


def count_chatgpt_hits(text: str) -> int:
    """Function summary: count ChatGPT/AI keyword matches in one comment body.

    Parameters:
    - text: comment body string.

    Returns:
    - Total regex match count across all lexicon patterns.
    """
    if not text or not isinstance(text, str):
        return 0
    total = 0
    for pat in CI_PATTERNS:
        total += len(pat.findall(text))
    for pat in CS_PATTERNS:
        total += len(pat.findall(text))
    return total


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for ChatGPT mention ban-window plot.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Ban-window ChatGPT/AI mention descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=7)
    parser.add_argument(
        "--pooled-smoothing",
        type=int,
        default=3,
        help="Trailing rolling window (days) for pooled single-panel figure; 1 disables smoothing.",
    )
    return parser.parse_args()


def _control_display(panel_id: str) -> str:
    """Function summary: human-readable control label for titles and legends.

    Parameters:
    - panel_id: canonical country_panel id.

    Returns:
    - Display label string.
    """
    return CONTROL_PANEL_DISPLAY.get(panel_id, panel_id)


def load_comment_frame(shard_root: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load comment bodies and metadata from enriched Parquet shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks directory.
    - subreddits: primary subreddit list from config.

    Returns:
    - Combined comment dataframe with chatgpt_hits column.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        for shard in sorted(shard_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(shard)
            except Exception:
                continue
            cols = [c for c in READ_COLUMNS if c in df.columns]
            if "body" not in cols:
                continue
            chunk = df[cols].copy()
            chunk["subreddit"] = sub
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["chatgpt_hits"] = out["body"].map(count_chatgpt_hits)
    return out


def daily_topic_family_table(df: pd.DataFrame, family_map: Dict[str, str]) -> pd.DataFrame:
    """Function summary: aggregate daily ChatGPT mention rate by topic_family.

    Parameters:
    - df: comment-level frame with chatgpt_hits, n_words, date_utc.
    - family_map: subreddit -> topic_family fallback map.

    Returns:
    - Daily table with chatgpt_mention_rate_100w and n_comments.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "topic_family" not in work.columns:
        work["topic_family"] = work["subreddit"].map(family_map)
    work = work[work["topic_family"].notna()]
    rows: List[Dict[str, object]] = []
    for (family, day), grp in work.groupby(["topic_family", "date_utc"], sort=True):
        total_words = float(grp["n_words"].astype(float).sum())
        total_hits = int(grp["chatgpt_hits"].sum())
        rate = 100.0 * total_hits / total_words if total_words > 0 else float("nan")
        rows.append(
            {
                "topic_family": family,
                "date_utc": day,
                "n_comments": len(grp),
                "chatgpt_hits": total_hits,
                "n_words": total_words,
                "chatgpt_mention_rate_100w": rate,
            }
        )
    return pd.DataFrame(rows)


def country_panel_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Function summary: map topic_family to country_panel for plotting.

    Parameters:
    - daily: topic_family x date table from daily_topic_family_table.

    Returns:
    - Long series with country_panel, date_utc, value.
    """
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["country_panel"] = work["topic_family"].astype(str).map(COUNTRY_PANEL_FAMILIES)
    work = work[work["country_panel"].notna()]
    out = work[["country_panel", "date_utc", "chatgpt_mention_rate_100w"]].rename(
        columns={"chatgpt_mention_rate_100w": "value"}
    )
    out["date_utc"] = pd.to_datetime(out["date_utc"])
    return out.sort_values(["country_panel", "date_utc"])


def _pool_group_for_family(topic_family: str) -> Optional[str]:
    """Function summary: map topic_family to Italy or Controls pool label.

    Parameters:
    - topic_family: DiD arm id (e.g. it_political, de).

    Returns:
    - Pool group name, or None when family is outside IT/control arms.
    """
    if topic_family in ITALY_TOPIC_FAMILIES:
        return "Italy"
    if topic_family in CONTROL_TOPIC_FAMILIES:
        return "Controls"
    return None


def word_weighted_pool_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Function summary: word-weighted daily ChatGPT rate for Italy vs pooled controls.

    Parameters:
    - daily: topic_family x date table from daily_topic_family_table.

    Returns:
    - Long frame with pool_group, date_utc, value, chatgpt_hits, n_words.
    """
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["pool_group"] = work["topic_family"].astype(str).map(_pool_group_for_family)
    work = work[work["pool_group"].notna()]
    if work.empty:
        return pd.DataFrame()
    agg = (
        work.groupby(["pool_group", "date_utc"], as_index=False)
        .agg(
            chatgpt_hits=("chatgpt_hits", "sum"),
            n_words=("n_words", "sum"),
        )
        .sort_values(["pool_group", "date_utc"])
    )
    agg["value"] = agg.apply(
        lambda r: 100.0 * r["chatgpt_hits"] / r["n_words"] if r["n_words"] > 0 else float("nan"),
        axis=1,
    )
    agg["date_utc"] = pd.to_datetime(agg["date_utc"])
    return agg


def _window_weighted_rate(
    pooled: pd.DataFrame,
    pool_group: str,
    start: str,
    end: str,
) -> float:
    """Function summary: word-weighted mention rate over a calendar window.

    Parameters:
    - pooled: output of word_weighted_pool_daily.
    - pool_group: Italy or Controls.
    - start: inclusive window start (YYYY-MM-DD).
    - end: inclusive window end (YYYY-MM-DD).

    Returns:
    - 100 * sum(hits) / sum(words), or NaN when no words in window.
    """
    sub = pooled[
        (pooled["pool_group"] == pool_group)
        & (pooled["date_utc"] >= pd.Timestamp(start))
        & (pooled["date_utc"] <= pd.Timestamp(end))
    ]
    total_words = float(sub["n_words"].sum())
    if total_words <= 0:
        return float("nan")
    return 100.0 * float(sub["chatgpt_hits"].sum()) / total_words


def _within_tolerance(observed: float, expected: float, rtol: float = SANITY_TOLERANCE_RTOL) -> bool:
    """Function summary: check observed rate is within relative tolerance of expected.

    Parameters:
    - observed: computed benchmark rate.
    - expected: target benchmark rate.
    - rtol: relative tolerance (default SANITY_TOLERANCE_RTOL).

    Returns:
    - True when both finite and |observed - expected| <= rtol * |expected|.
    """
    if not (pd.notna(observed) and pd.notna(expected)):
        return False
    return abs(float(observed) - float(expected)) <= rtol * abs(float(expected))


def sanity_check_pooled_rates(pooled: pd.DataFrame) -> None:
    """Function summary: verify word-weighted pooled rates match ban-window benchmarks.

    Parameters:
    - pooled: output of word_weighted_pool_daily (unsmoothed daily rates).

    Returns:
    - None when all checks pass; otherwise prints report and exits with code 1.
    """
    if pooled.empty:
        print("[plot_chatgpt_mentions_ban_shaded] sanity check failed: empty pooled series", flush=True)
        sys.exit(1)

    failures: List[str] = []
    rows: List[Dict[str, object]] = []

    for key, (group, start, end, expected) in SANITY_BENCHMARKS.items():
        observed = _window_weighted_rate(pooled, group, start, end)
        ok = _within_tolerance(observed, expected)
        rows.append(
            {
                "check": key,
                "pool_group": group,
                "window": f"{start}..{end}",
                "expected": expected,
                "observed": round(observed, 4) if pd.notna(observed) else None,
                "pass": ok,
            }
        )
        if not ok:
            failures.append(
                f"{key}: {group} {start}..{end} expected≈{expected}, observed={observed:.4f}"
            )

    it_peak_row = pooled[
        (pooled["pool_group"] == "Italy") & (pooled["date_utc"] == pd.Timestamp(SANITY_IT_PEAK_DATE))
    ]
    if it_peak_row.empty:
        peak_observed = float("nan")
    else:
        peak_observed = float(it_peak_row["value"].iloc[0])
    peak_ok = _within_tolerance(peak_observed, SANITY_IT_PEAK_RATE)
    rows.append(
        {
            "check": "it_peak_apr1",
            "pool_group": "Italy",
            "window": SANITY_IT_PEAK_DATE,
            "expected": SANITY_IT_PEAK_RATE,
            "observed": round(peak_observed, 4) if pd.notna(peak_observed) else None,
            "pass": peak_ok,
        }
    )
    if not peak_ok:
        failures.append(
            f"it_peak_apr1: Italy {SANITY_IT_PEAK_DATE} expected≈{SANITY_IT_PEAK_RATE}, "
            f"observed={peak_observed:.4f}"
        )

    report = pd.DataFrame(rows)
    print("[plot_chatgpt_mentions_ban_shaded] pooled sanity checks (word-weighted, unsmoothed):", flush=True)
    print(report.to_string(index=False), flush=True)

    if failures:
        print("[plot_chatgpt_mentions_ban_shaded] SANITY CHECK FAILED — stopping before pooled PNG:", flush=True)
        for msg in failures:
            print(f"  - {msg}", flush=True)
        sys.exit(1)


def control_panel_band_daily(series: pd.DataFrame, smoothing_days: int) -> pd.DataFrame:
    """Function summary: min–max band across control country panels per date.

    Parameters:
    - series: long daily series (country_panel, date_utc, value).
    - smoothing_days: trailing rolling window applied per panel before min/max.

    Returns:
    - Frame with date_utc, ctrl_min, ctrl_max; empty when no control panels present.
    """
    ctrl = series[series["country_panel"].isin(CONTROL_PANELS)].copy()
    if ctrl.empty:
        return pd.DataFrame()
    roll = grouped_trailing_daily_rolling(
        ctrl.rename(columns={"value": OUTCOME_ID}),
        group_col="country_panel",
        rolling_window_days=smoothing_days,
        date_col="date_utc",
    )
    roll = roll.rename(columns={OUTCOME_ID: "value"})
    wide = roll.pivot(index="date_utc", columns="country_panel", values="value")
    if wide.empty:
        return pd.DataFrame()
    band = wide.assign(
        ctrl_min=wide.min(axis=1, skipna=True),
        ctrl_max=wide.max(axis=1, skipna=True),
    )[["ctrl_min", "ctrl_max"]].reset_index()
    return band.sort_values("date_utc")


def plot_ban_window_pooled(
    pooled: pd.DataFrame,
    out_path: Path,
    smoothing_days: int,
) -> None:
    """Function summary: single-panel Italy vs pooled controls with ban shading.

    Parameters:
    - pooled: word-weighted daily series (pool_group, date_utc, value).
    - out_path: PNG output path.
    - smoothing_days: trailing rolling window; 1 plots raw lines only.

    Returns:
    - None; writes PNG to out_path.
    """
    if pooled.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    roll = grouped_trailing_daily_rolling(
        pooled.rename(columns={"value": OUTCOME_ID}),
        group_col="pool_group",
        rolling_window_days=smoothing_days,
        date_col="date_utc",
    )
    roll = roll.rename(columns={OUTCOME_ID: "value"})

    show_smoothing_legend = smoothing_days > 1
    smooth_label_suffix = f" ({smoothing_days}-day trailing mean)" if show_smoothing_legend else ""

    for group in POOL_GROUP_LABELS:
        style = POOL_GROUP_STYLE[group]
        raw = pooled[pooled["pool_group"] == group]
        if raw.empty:
            continue
        if show_smoothing_legend:
            ax.plot(
                raw["date_utc"],
                raw["value"],
                color=str(style["color"]),
                linewidth=0.8,
                alpha=0.25,
                zorder=1,
            )
        smooth = roll[roll["pool_group"] == group]
        if not smooth.empty:
            ax.plot(
                smooth["date_utc"],
                smooth["value"],
                color=str(style["color"]),
                linewidth=float(style["linewidth"]),
                label=f"{style['label']}{smooth_label_suffix}",
                zorder=2,
            )

    ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
    ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="-", linewidth=0.9)
    ax.axvline(pd.Timestamp(BAN_END), color="0.4", linestyle="--", linewidth=0.9)
    ax.set_title("IT vs pooled controls")
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if show_smoothing_legend:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.45",
                linewidth=0.8,
                alpha=0.25,
                label="Daily unsmoothed (faint)",
            )
        )
    ax.legend(legend_handles, legend_labels, fontsize=8)
    fig.suptitle(OUTCOME_TITLE, fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ban_window_pooled_with_control_range(
    pooled: pd.DataFrame,
    series: pd.DataFrame,
    out_path: Path,
    smoothing_days: int,
) -> None:
    """Function summary: pooled IT vs controls with control-panel min–max band.

    Parameters:
    - pooled: word-weighted daily series (pool_group, date_utc, value).
    - series: country-panel daily series for building the control band.
    - out_path: PNG output path.
    - smoothing_days: trailing rolling window; 1 plots raw lines only.

    Returns:
    - None; writes PNG to out_path.
    """
    if pooled.empty or series.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    roll = grouped_trailing_daily_rolling(
        pooled.rename(columns={"value": OUTCOME_ID}),
        group_col="pool_group",
        rolling_window_days=smoothing_days,
        date_col="date_utc",
    )
    roll = roll.rename(columns={OUTCOME_ID: "value"})
    band = control_panel_band_daily(series, smoothing_days)

    show_smoothing_legend = smoothing_days > 1
    smooth_label_suffix = f" ({smoothing_days}-day trailing mean)" if show_smoothing_legend else ""
    control_color = str(POOL_GROUP_STYLE["Controls"]["color"])

    if not band.empty:
        ax.fill_between(
            band["date_utc"],
            band["ctrl_min"],
            band["ctrl_max"],
            color=THESIS_CONTROL_BAND,
            alpha=0.45,
            label="Controls (panel range)",
            zorder=2,
        )

    shade_ban_window(ax, mode="calendar", ban_start=BAN_START, ban_end=BAN_END, zorder=0)

    # Faint raw daily series: Italy only (controls are summarized by the
    # pooled line + range band; their raw daily lines read as clutter).
    if show_smoothing_legend:
        italy_raw = pooled[pooled["pool_group"] == "Italy"]
        if not italy_raw.empty:
            ax.plot(
                italy_raw["date_utc"],
                italy_raw["value"],
                linestyle="none",
                marker=".",
                markersize=3,
                color=str(POOL_GROUP_STYLE["Italy"]["color"]),
                alpha=0.35,
                label="Italy (raw daily rate)",
                zorder=1,
            )
    for group in POOL_GROUP_LABELS:
        style = POOL_GROUP_STYLE[group]
        smooth = roll[roll["pool_group"] == group]
        if not smooth.empty:
            ax.plot(
                smooth["date_utc"],
                smooth["value"],
                color=str(style["color"]),
                linewidth=float(style["linewidth"]),
                label=f"{style['label']}{smooth_label_suffix}",
                zorder=4,
            )

    ax.set_xlabel(XLABEL_CALENDAR)
    ax.set_ylabel("Mentions per 100 words")
    ax.set_title("ChatGPT / AI mention rate")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _italy_daily(roll: pd.DataFrame) -> pd.DataFrame:
    """Function summary: mean across Italian topic-family panels per date.

    Parameters:
    - roll: rolled long series with country_panel and value.

    Returns:
    - Italy pooled daily means.
    """
    it = roll[roll["country_panel"].isin(ITALY_PANELS)]
    if it.empty:
        return pd.DataFrame()
    return it.groupby("date_utc", as_index=False)["value"].mean()


def plot_ban_window(series: pd.DataFrame, out_path: Path, rolling_window: int) -> None:
    """Function summary: four-panel IT vs control lines with ban shading.

    Parameters:
    - series: long daily series (country_panel, date_utc, value).
    - out_path: PNG output path.
    - rolling_window: trailing rolling window in days.

    Returns:
    - None; writes PNG to out_path.
    """
    if series.empty:
        return
    roll = grouped_trailing_daily_rolling(
        series.rename(columns={"value": OUTCOME_ID}),
        group_col="country_panel",
        rolling_window_days=rolling_window,
        date_col="date_utc",
    )
    roll = roll.rename(columns={OUTCOME_ID: "value"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    it_daily = _italy_daily(roll)
    for ax, ctrl in zip(axes.flatten(), CONTROL_PANELS):
        ct = roll[roll["country_panel"] == ctrl]
        ctrl_label = _control_display(ctrl)
        if not it_daily.empty:
            ax.plot(
                it_daily["date_utc"],
                it_daily["value"],
                color="#c1121f",
                linewidth=2,
                label="Italy",
            )
        if not ct.empty:
            ax.plot(
                ct["date_utc"],
                ct["value"],
                color="#457b9d",
                linewidth=1.5,
                label=ctrl_label,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.45",
            )
        ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
        ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="--", linewidth=0.9)
        ax.set_title(f"IT vs {ctrl_label}")
        ax.legend(fontsize=8)
    fig.suptitle(OUTCOME_TITLE, fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: scan shards, write daily table, and export ban-window figure."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    tables_out = tables_subdir(config, "descriptives")
    figures_out = figures_subdir(config, "descriptives") / "ban_window"
    tables_out.mkdir(parents=True, exist_ok=True)

    start, end_excl, _, _ = event_dates_from_config(config)
    subs = resolve_primary_subreddits(config)
    family_map = subreddit_family_map(config)

    df = load_comment_frame(shard_root, subs)
    if df.empty:
        print("[plot_chatgpt_mentions_ban_shaded] no parquet data found", flush=True)
        return

    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    daily = daily_topic_family_table(df, family_map)
    table_path = tables_out / "daily_chatgpt_mentions_by_topic_family.csv"
    daily.to_csv(table_path, index=False)
    print(f"[plot_chatgpt_mentions_ban_shaded] wrote {table_path}", flush=True)

    series = country_panel_series(daily)
    if series.empty:
        print("[plot_chatgpt_mentions_ban_shaded] empty country-panel series", flush=True)
        return

    fig_path = figures_out / f"{OUTCOME_ID}.png"
    plot_ban_window(series, fig_path, args.rolling_window)
    print(f"[plot_chatgpt_mentions_ban_shaded] {OUTCOME_ID} -> {fig_path}", flush=True)

    pooled = word_weighted_pool_daily(daily)
    if pooled.empty:
        print("[plot_chatgpt_mentions_ban_shaded] empty pooled series", flush=True)
        return

    sanity_check_pooled_rates(pooled)
    pooled_path = figures_out / f"{POOLED_OUTCOME_ID}.png"
    plot_ban_window_pooled(pooled, pooled_path, args.pooled_smoothing)
    print(f"[plot_chatgpt_mentions_ban_shaded] {POOLED_OUTCOME_ID} -> {pooled_path}", flush=True)

    pooled_range_path = figures_out / f"{POOLED_RANGE_OUTCOME_ID}.png"
    plot_ban_window_pooled_with_control_range(
        pooled, series, pooled_range_path, args.pooled_smoothing
    )
    print(
        f"[plot_chatgpt_mentions_ban_shaded] {POOLED_RANGE_OUTCOME_ID} -> {pooled_range_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
