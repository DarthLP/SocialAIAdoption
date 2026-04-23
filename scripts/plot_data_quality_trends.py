"""
Script summary:
This script builds and visualizes pre-cleaning data-quality indicators from the
daily Reddit NDJSON chunks. It computes day-level metrics per subreddit and in
aggregate, validates totals against existing filtering audits, and saves trend
plots around the ChatGPT launch anchor.

Functionality:
- Reads `data/raw/political_forums/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`.
- Computes counts for removed/deleted placeholders, deleted authors, AutoModerator,
  stickied comments, and an exploratory bot-name heuristic.
- Computes percentage rates relative to daily `rows_total`.
- Writes tidy outputs to `results/tables/data_quality_trends/`.
- Generates percentage trend plots in
  `results/figures/data_quality_trends/`.
- Annotates AutoModerator plots with the fixed total note requested by project policy.
- Validates `rows_total` against `results/tables/filtering/dump_filter_counts_by_day.csv`.

How to apply/run:
- `.venv/bin/python scripts/plot_data_quality_trends.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts


BASE_COUNT_METRICS = [
    "rows_total",
    "body_removed_count",
    "body_deleted_count",
    "author_deleted_count",
    "automod_author_count",
    "stickied_count",
    "bot_name_heuristic_count",
]

RATE_METRICS = [
    "body_removed_rate_pct",
    "body_deleted_rate_pct",
    "author_deleted_rate_pct",
    "automod_author_rate_pct",
    "stickied_rate_pct",
    "bot_name_heuristic_rate_pct",
]

PLOT_RATE_METRICS = [
    "body_removed_rate_pct",
    "body_deleted_rate_pct",
    "author_deleted_rate_pct",
    "automod_author_rate_pct",
    "stickied_rate_pct",
    "bot_name_heuristic_rate_pct",
]

METRIC_LABELS = {
    "body_removed_rate_pct": "Body == [removed] (% of daily rows)",
    "body_deleted_rate_pct": "Body == [deleted] (% of daily rows)",
    "author_deleted_rate_pct": "Author == [deleted] (% of daily rows)",
    "automod_author_rate_pct": "Author == AutoModerator (% of daily rows)",
    "stickied_rate_pct": "Stickied == true (% of daily rows)",
    "bot_name_heuristic_rate_pct": "Bot-name heuristic (% of daily rows, exploratory)",
}


@dataclass
class RuntimePaths:
    """Function summary: store resolved input/output locations for this script run."""

    raw_daily_chunks_dir: Path
    tables_subdir: Path
    figures_subdir: Path
    baseline_counts_path: Path


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for config path and output naming."""
    parser = argparse.ArgumentParser(description="Build and plot data-quality trends.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/political_forums_setup.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_paths(config: Dict[str, Any]) -> RuntimePaths:
    """Function summary: resolve and create output subfolders required by this workflow."""
    raw_dir = Path(config["paths"]["raw_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    tables_subdir = tables_dir / "data_quality_trends"
    figures_subdir = figures_dir / "data_quality_trends"
    tables_subdir.mkdir(parents=True, exist_ok=True)
    figures_subdir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        raw_daily_chunks_dir=raw_dir / "daily_chunks",
        tables_subdir=tables_subdir,
        figures_subdir=figures_subdir,
        baseline_counts_path=tables_dir / "filtering" / "dump_filter_counts_by_day.csv",
    )


def is_bot_name_heuristic(author: str) -> bool:
    """Function summary: flag exploratory bot-like usernames using conservative substring rules."""
    name = (author or "").strip().lower()
    if not name:
        return False
    if name == "automoderator":
        return False
    return ("bot" in name) or ("moderatorbot" in name)


def iter_daily_files(raw_daily_chunks_dir: Path, subreddits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Function summary: iterate existing per-subreddit daily NDJSON files in sorted date order."""
    for subreddit in sorted(subreddits):
        subreddit_dir = raw_daily_chunks_dir / subreddit
        if not subreddit_dir.exists():
            continue
        for ndjson_path in sorted(subreddit_dir.glob("*.ndjson")):
            yield subreddit, ndjson_path


def compute_daily_metrics(raw_daily_chunks_dir: Path, subreddits: list[str]) -> pd.DataFrame:
    """Function summary: scan daily files and compute configured per-day count metrics."""
    rows: list[Dict[str, Any]] = []
    for subreddit, file_path in iter_daily_files(raw_daily_chunks_dir, subreddits):
        date_utc = file_path.stem
        counter = {metric: 0 for metric in BASE_COUNT_METRICS}
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                body = (record.get("body") or "").strip()
                author = (record.get("author") or "").strip()
                counter["rows_total"] += 1
                if body == "[removed]":
                    counter["body_removed_count"] += 1
                if body == "[deleted]":
                    counter["body_deleted_count"] += 1
                if author == "[deleted]":
                    counter["author_deleted_count"] += 1
                if author == "AutoModerator":
                    counter["automod_author_count"] += 1
                if bool(record.get("stickied")):
                    counter["stickied_count"] += 1
                if is_bot_name_heuristic(author):
                    counter["bot_name_heuristic_count"] += 1
        rows.append({"subreddit": subreddit, "date_utc": date_utc, **counter})
    if not rows:
        return pd.DataFrame(columns=["subreddit", "date_utc", *BASE_COUNT_METRICS])
    df = pd.DataFrame(rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True)
    return df


def add_time_and_rate_columns(df: pd.DataFrame, event_ts: int) -> pd.DataFrame:
    """Function summary: add UTC date columns, event-day offsets, and percentage rate metrics."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date_utc"], utc=True).dt.tz_convert(None)
    event_date = datetime.fromtimestamp(event_ts, tz=timezone.utc).date()
    out["days_from_event"] = (
        out["date"].dt.date.apply(lambda d: (d - event_date).days).astype(int)
    )
    denominator = out["rows_total"].replace(0, pd.NA)
    out["body_removed_rate_pct"] = (out["body_removed_count"] / denominator) * 100.0
    out["body_deleted_rate_pct"] = (out["body_deleted_count"] / denominator) * 100.0
    out["author_deleted_rate_pct"] = (out["author_deleted_count"] / denominator) * 100.0
    out["automod_author_rate_pct"] = (out["automod_author_count"] / denominator) * 100.0
    out["stickied_rate_pct"] = (out["stickied_count"] / denominator) * 100.0
    out["bot_name_heuristic_rate_pct"] = (out["bot_name_heuristic_count"] / denominator) * 100.0
    return out


def build_overall_daily(df: pd.DataFrame, event_ts: int) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit daily metrics into one overall daily series."""
    grouped = (
        df.groupby("date_utc", as_index=False)[BASE_COUNT_METRICS]
        .sum()
        .sort_values("date_utc")
        .reset_index(drop=True)
    )
    grouped["subreddit"] = "ALL"
    grouped = grouped[["subreddit", "date_utc", *BASE_COUNT_METRICS]]
    return add_time_and_rate_columns(grouped, event_ts)


def validate_against_baseline(df: pd.DataFrame, baseline_counts_path: Path) -> pd.DataFrame:
    """Function summary: compare computed rows_total against existing filter audit day counts."""
    computed = (
        df.groupby(["subreddit", "date_utc"], as_index=False)["rows_total"]
        .sum()
        .rename(columns={"rows_total": "rows_total_computed"})
    )
    baseline = pd.read_csv(baseline_counts_path).rename(columns={"rows": "rows_total_baseline"})
    merged = computed.merge(baseline, on=["subreddit", "date_utc"], how="outer")
    merged["rows_total_computed"] = merged["rows_total_computed"].fillna(0).astype(int)
    merged["rows_total_baseline"] = merged["rows_total_baseline"].fillna(0).astype(int)
    merged["delta_rows_total"] = merged["rows_total_computed"] - merged["rows_total_baseline"]
    return merged.sort_values(["subreddit", "date_utc"]).reset_index(drop=True)


def write_tables(
    per_subreddit_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tables_subdir: Path,
) -> None:
    """Function summary: write trend and validation tables to the dedicated output subfolder."""
    per_subreddit_df.to_csv(tables_subdir / "daily_quality_metrics_by_subreddit.csv", index=False)
    overall_df.to_csv(tables_subdir / "daily_quality_metrics_overall.csv", index=False)
    validation_df.to_csv(tables_subdir / "daily_quality_metrics_validation_vs_filter_counts.csv", index=False)


def add_launch_marker(ax: Any, event_date: datetime) -> None:
    """Function summary: draw the launch-day vertical reference line on one plot axis."""
    ax.axvline(x=event_date, color="red", linestyle="--", linewidth=1)


def format_date_axis(ax: Any) -> None:
    """Function summary: format date ticks to prevent overlap with dense daily labels."""
    locator = mdates.DayLocator(interval=3)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def add_automod_annotation(fig: Any, metric_name: str) -> None:
    """Function summary: attach required AutoModerator total note on relevant plots."""
    if metric_name.startswith("automod_author_"):
        fig.text(
            0.01,
            0.01,
            'Note: author == "AutoModerator" total = 8602 in this analysis window.',
            ha="left",
            va="bottom",
            fontsize=9,
        )


def plot_overall(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    event_date: datetime,
) -> None:
    """Function summary: plot one overall daily trend line with launch marker and annotation."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df.sort_values("date"), x="date", y=metric, marker="o", ax=ax)
    add_launch_marker(ax, event_date)
    format_date_axis(ax)
    ax.set_title(f"Overall Daily Trend: {METRIC_LABELS[metric]}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(METRIC_LABELS[metric])
    add_automod_annotation(fig, metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_by_subreddit(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    event_date: datetime,
) -> None:
    """Function summary: plot one daily trend metric by subreddit with shared launch marker."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df.sort_values(["subreddit", "date"]),
        x="date",
        y=metric,
        hue="subreddit",
        marker="o",
        ax=ax,
    )
    add_launch_marker(ax, event_date)
    format_date_axis(ax)
    ax.set_title(f"Per-Subreddit Daily Trend: {METRIC_LABELS[metric]}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(title="Subreddit", loc="best")
    add_automod_annotation(fig, metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_plots(
    per_subreddit_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    figures_subdir: Path,
    event_ts: int,
) -> None:
    """Function summary: generate percentage-only plot sets for overall and subreddit views."""
    event_date = datetime.fromtimestamp(event_ts, tz=timezone.utc).replace(tzinfo=None)
    for metric in PLOT_RATE_METRICS:
        plot_overall(
            overall_df,
            metric,
            figures_subdir / f"overall_{metric}.png",
            event_date,
        )
        plot_by_subreddit(
            per_subreddit_df,
            metric,
            figures_subdir / f"by_subreddit_{metric}.png",
            event_date,
        )


def write_metadata_note(
    per_subreddit_df: pd.DataFrame,
    tables_subdir: Path,
) -> None:
    """Function summary: write concise metadata note including requested equivalence statement."""
    automod_total = int(per_subreddit_df["automod_author_count"].sum())
    note_path = tables_subdir / "quality_trends_notes.txt"
    lines = [
        "Data Quality Trends Notes",
        "=========================",
        "",
        "Primary moderation automation series uses author == 'AutoModerator'.",
        "Near-equivalence check documented from prior audit:",
        "- automod_author_count and moderator_distinguished_count differ by 1 row.",
        "- Exception row: author == 'AutoModerator' with distinguished == null.",
        "",
        f"Computed author == 'AutoModerator' total from this run: {automod_total}",
        "Policy note for figure captions: author == 'AutoModerator' total = 8602.",
        "The bot-name heuristic metric is exploratory and not a cleaning rule.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: execute full trend-table, validation, and plotting workflow."""
    sns.set_theme(style="whitegrid")
    args = parse_args()
    config = load_config(args.config)
    paths = build_paths(config)
    subreddits = list(config["subreddits"]["primary"])
    event_ts = utc_ts(config["event_window"]["launch_day_utc"])

    per_subreddit_counts = compute_daily_metrics(paths.raw_daily_chunks_dir, subreddits)
    per_subreddit_df = add_time_and_rate_columns(per_subreddit_counts, event_ts)
    overall_df = build_overall_daily(per_subreddit_counts, event_ts)
    validation_df = validate_against_baseline(per_subreddit_counts, paths.baseline_counts_path)

    write_tables(per_subreddit_df, overall_df, validation_df, paths.tables_subdir)
    write_metadata_note(per_subreddit_df, paths.tables_subdir)
    make_plots(per_subreddit_df, overall_df, paths.figures_subdir, event_ts)


if __name__ == "__main__":
    main()
