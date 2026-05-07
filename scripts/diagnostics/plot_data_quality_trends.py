"""
Script summary:
This script builds and visualizes pre-cleaning data-quality indicators from the
daily Reddit NDJSON chunks. It computes day-level metrics per subreddit, topic,
and family (config-driven), validates totals against existing filtering audits,
and saves trend plots around the ChatGPT launch anchor.

Functionality:
- Reads `data/raw/political_forums/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`.
- Computes counts for removed/deleted placeholders, deleted authors, AutoModerator,
  stickied comments, and an exploratory bot-name heuristic.
- Enforces configured event window bounds before writing trend tables and figures.
- Computes percentage rates relative to daily `rows_total`.
- Aggregates per-subreddit daily counts into per-family daily series using
  `topics` + `topic_families` in config (subreddits not assigned to a family
  are skipped from family plots and surfaced via a warning).
- Writes tidy outputs to `results/tables/data_quality_trends/`:
  per-subreddit (granular audit), per-family, and pooled overall tables.
- Generates percentage trend plots in `results/figures/data_quality_trends/`:
  one `overall_<metric>.png`, one `by_family_<metric>.png`, one
  `by_subreddit_by_family/<family>/<metric>.png`, and one
  `by_topic_by_family/by_topic_by_family_<metric>.png`.
- Uses a non-interactive plotting backend by default for terminal-safe rendering.
- Logs per-metric `plot_progress` markers so long plotting runs show progress.
- Annotates AutoModerator plots with the AutoModerator row total summed for the current window.
- Validates `rows_total` against `results/tables/filtering/dump_filter_counts_by_day.csv`.

How to apply/run:
- `.venv/bin/python scripts/diagnostics/plot_data_quality_trends.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, Iterable

import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

def _resolve_project_root() -> Path:
    """Load scripts/_project_root.py and return the repository root Path."""
    _scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod",
        _scripts_dir / "_project_root.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import (
    load_config,
    subreddit_family_map,
    subreddit_topic_map,
    topic_groups,
    topic_families,
    utc_ts,
)


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


def filter_to_event_window(df: pd.DataFrame, start_ts: int, end_ts_exclusive: int) -> pd.DataFrame:
    """Function summary: keep only rows whose UTC day falls within configured event window timestamps."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date_utc"], utc=True)
    start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    end_exclusive = pd.Timestamp(end_ts_exclusive, unit="s", tz="UTC")
    mask = (out["date"] >= start) & (out["date"] < end_exclusive)
    filtered = out.loc[mask].drop(columns=["date"])
    return filtered.reset_index(drop=True)


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


def build_family_daily(
    df: pd.DataFrame,
    event_ts: int,
    sub_to_family: Dict[str, str],
) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit daily counts into per-family daily series using config-driven mapping.

    Parameters:
    - df: per-subreddit daily counts (BASE_COUNT_METRICS columns) within the event window.
    - event_ts: launch-day UTC timestamp used to compute days_from_event.
    - sub_to_family: mapping from subreddit name to family name.

    Returns:
    - DataFrame with one row per (topic_family, date_utc), summed counts, and recomputed rate
      columns. Subreddits not assigned to a family are dropped and surfaced via a printed warning.
    """
    if df.empty:
        return df.copy()
    d = df.copy()
    d["topic_family"] = d["subreddit"].map(sub_to_family)
    unmapped = sorted(d.loc[d["topic_family"].isna(), "subreddit"].dropna().unique())
    if unmapped:
        print(f"warning unmapped_subreddits_in_family_view={','.join(unmapped)}")
    d = d.dropna(subset=["topic_family"])
    if d.empty:
        return pd.DataFrame(columns=["topic_family", "date_utc", *BASE_COUNT_METRICS])
    grouped = (
        d.groupby(["topic_family", "date_utc"], as_index=False)[BASE_COUNT_METRICS]
        .sum()
        .sort_values(["topic_family", "date_utc"])
        .reset_index(drop=True)
    )
    return add_time_and_rate_columns(grouped, event_ts)


def build_topic_family_daily(
    df: pd.DataFrame,
    event_ts: int,
    sub_to_topic: Dict[str, str],
    topic_to_family: Dict[str, str],
) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit daily counts into per-topic-per-family daily series."""
    if df.empty:
        return df.copy()
    d = df.copy()
    d["topic_group"] = d["subreddit"].map(sub_to_topic)
    d = d.dropna(subset=["topic_group"])
    d["topic_family"] = d["topic_group"].map(topic_to_family)
    d = d.dropna(subset=["topic_family"])
    if d.empty:
        return pd.DataFrame(columns=["topic_family", "topic_group", "date_utc", *BASE_COUNT_METRICS])
    grouped = (
        d.groupby(["topic_family", "topic_group", "date_utc"], as_index=False)[BASE_COUNT_METRICS]
        .sum()
        .sort_values(["topic_family", "topic_group", "date_utc"])
        .reset_index(drop=True)
    )
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
    per_family_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tables_subdir: Path,
) -> None:
    """Function summary: write trend and validation tables to the dedicated output subfolder.

    The per-subreddit table is preserved as the granular audit; the per-family and overall
    tables drive the figures and are written alongside.
    """
    per_subreddit_df.to_csv(tables_subdir / "daily_quality_metrics_by_subreddit.csv", index=False)
    per_family_df.to_csv(tables_subdir / "daily_quality_metrics_by_family.csv", index=False)
    overall_df.to_csv(tables_subdir / "daily_quality_metrics_overall.csv", index=False)
    validation_df.to_csv(tables_subdir / "daily_quality_metrics_validation_vs_filter_counts.csv", index=False)


def add_release_markers(ax: Any, release_dates: list[datetime]) -> None:
    """Function summary: draw vertical reference lines for key ChatGPT release dates on one plot axis."""
    for release_date in release_dates:
        ax.axvline(x=release_date, color="red", linestyle=":", linewidth=1.2)


def date_span_days(overall_df: pd.DataFrame) -> int:
    """Function summary: return inclusive number of UTC calendar days covered by overall_df for tick spacing."""
    if overall_df.empty or "date" not in overall_df.columns:
        return 1
    dmin = pd.Timestamp(overall_df["date"].min()).normalize()
    dmax = pd.Timestamp(overall_df["date"].max()).normalize()
    return max(1, int((dmax - dmin).days) + 1)


def format_date_axis(ax: Any, span_days: int) -> None:
    """Function summary: format date ticks at month starts to keep all date-based plots consistent."""
    _ = span_days
    locator = mdates.MonthLocator(bymonthday=1)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def add_automod_annotation(fig: Any, metric_name: str, automod_total: int) -> None:
    """Function summary: attach AutoModerator total note on automod rate plots using the current-window sum."""
    if metric_name.startswith("automod_author_"):
        fig.text(
            0.01,
            0.01,
            f'Note: author == "AutoModerator" total = {automod_total} in this analysis window.',
            ha="left",
            va="bottom",
            fontsize=9,
        )


def plot_overall(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    release_dates: list[datetime],
    *,
    span_days: int,
    automod_total: int,
) -> None:
    """Function summary: plot one overall daily trend line with launch marker and annotation."""
    if df.empty:
        return
    ordered = df.sort_values("date").copy()
    metric_values = ordered[metric].fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        ordered["date"],
        metric_values,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    add_release_markers(ax, release_dates)
    format_date_axis(ax, span_days)
    ax.set_title(f"Overall Daily Trend: {METRIC_LABELS[metric]}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(METRIC_LABELS[metric])
    add_automod_annotation(fig, metric, automod_total)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_by_group(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    release_dates: list[datetime],
    *,
    span_days: int,
    automod_total: int,
    group_col: str = "subreddit",
    group_label: str = "Subreddit",
    legend_below: bool = True,
) -> None:
    """Function summary: plot one daily trend metric as one line per group value.

    Parameters:
    - df: long-format daily data containing `group_col`, `date`, and `metric`.
    - metric: rate column name to plot on the y-axis.
    - out_path: destination PNG path.
    - release_dates: vertical reference markers (e.g., ChatGPT/GPT-4 launches).
    - span_days: total day span of the dataset, used for date-axis tick spacing.
    - automod_total: AutoModerator row total annotated only on automod rate plots.
    - group_col: dataframe column whose unique values become individual lines.
    - group_label: user-facing label used in the title and legend (e.g., "Topic").
    """
    if df.empty:
        return
    ordered = df.sort_values([group_col, "date"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = sorted(ordered[group_col].dropna().unique())
    color_map = dict(zip(groups, sns.color_palette("husl", n_colors=max(1, len(groups)))))
    for group_value, group in ordered.groupby(group_col, sort=True):
        series = group[metric].fillna(0.0)
        ax.plot(
            group["date"],
            series,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=group_value,
            color=color_map.get(group_value),
        )
    add_release_markers(ax, release_dates)
    format_date_axis(ax, span_days)
    ax.set_title(f"Per-{group_label} Daily Trend: {METRIC_LABELS[metric]}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(METRIC_LABELS[metric])
    n_legend = len(groups)
    ncol = min(max(1, n_legend), 6)
    if legend_below:
        ax.legend(
            title=group_label,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=ncol,
            frameon=False,
        )
    else:
        ax.legend(
            title=group_label,
            loc="best",
            ncol=1 if n_legend <= 6 else 2,
            frameon=True,
        )
    add_automod_annotation(fig, metric, automod_total)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    per_family_df: pd.DataFrame,
    per_subreddit_df: pd.DataFrame,
    per_topic_family_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    figures_subdir: Path,
    event_ts: int,
    *,
    automod_total: int,
    span_days: int,
    family_topics: Dict[str, list[str]],
    subreddit_to_family: Dict[str, str],
    sub_to_topic: Dict[str, str],
) -> None:
    """Function summary: generate percentage-only plot sets for overall, by-family, and family-faceted subreddit/topic views."""
    _ = event_ts
    release_dates = [
        datetime(2022, 11, 30),
        datetime(2023, 3, 14),
    ]
    total_metrics = len(PLOT_RATE_METRICS)
    for idx, metric in enumerate(PLOT_RATE_METRICS, start=1):
        print(f"plot_progress metric={idx}/{total_metrics} name={metric} stage=overall")
        plot_overall(
            overall_df,
            metric,
            figures_subdir / f"overall_{metric}.png",
            release_dates,
            span_days=span_days,
            automod_total=automod_total,
        )
        print(f"plot_progress metric={idx}/{total_metrics} name={metric} stage=by_family")
        plot_by_group(
            per_family_df,
            metric,
            figures_subdir / f"by_family_{metric}.png",
            release_dates,
            span_days=span_days,
            automod_total=automod_total,
            group_col="topic_family",
            group_label="Family",
            legend_below=False,
        )
        ordered_families = [name for name in family_topics.keys() if name in set(per_family_df["topic_family"].dropna().unique())]
        if not ordered_families:
            ordered_families = list(family_topics.keys())

        by_subreddit_by_family_dir = figures_subdir / "by_subreddit_by_family"
        by_subreddit_by_family_dir.mkdir(parents=True, exist_ok=True)
        for family_name in ordered_families:
            family_out_dir = by_subreddit_by_family_dir / family_name
            family_out_dir.mkdir(parents=True, exist_ok=True)
            topics = list(family_topics.get(family_name, []))
            n_panels = max(1, len(topics))
            ncols = 3
            nrows = (n_panels + ncols - 1) // ncols
            fig_sub, axes_sub = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
            axes_sub_flat = axes_sub.flatten() if hasattr(axes_sub, "flatten") else [axes_sub]
            for ax, topic_name in zip(axes_sub_flat, topics):
                topic_frame = per_subreddit_df[
                    per_subreddit_df["subreddit"].map(subreddit_to_family).eq(family_name)
                ].copy()
                topic_frame = topic_frame[
                    topic_frame["subreddit"].map(sub_to_topic).eq(topic_name)
                ]
                if topic_frame.empty:
                    ax.set_title(f"{topic_name} (no data)")
                    ax.axis("off")
                    continue
                ordered = topic_frame.sort_values(["subreddit", "date"])
                subreddits = sorted(ordered["subreddit"].dropna().unique())
                palette = dict(zip(subreddits, sns.color_palette("husl", n_colors=max(1, len(subreddits)))))
                for subreddit, group in ordered.groupby("subreddit", sort=True):
                    ax.plot(
                        group["date"],
                        group[metric].fillna(0.0),
                        marker="o",
                        markersize=1.8,
                        linewidth=1.0,
                        color=palette.get(subreddit),
                        label=subreddit,
                    )
                add_release_markers(ax, release_dates)
                format_date_axis(ax, span_days)
                ax.set_title(topic_name)
                ax.legend(loc="best", fontsize=7, frameon=True)
            for ax in axes_sub_flat[len(topics):]:
                ax.axis("off")
            fig_sub.suptitle(f"Per-subreddit by topic in {family_name}: {METRIC_LABELS[metric]}", fontsize=12)
            fig_sub.tight_layout(rect=[0, 0, 1, 0.95])
            fig_sub.savefig(family_out_dir / f"{metric}.png", dpi=140)
            plt.close(fig_sub)

        by_topic_by_family_dir = figures_subdir / "by_topic_by_family"
        by_topic_by_family_dir.mkdir(parents=True, exist_ok=True)
        fig_topic, axes_topic = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
        axes_topic_flat = axes_topic.flatten() if hasattr(axes_topic, "flatten") else [axes_topic]
        for ax, family_name in zip(axes_topic_flat, ordered_families):
            family_topic_frame = per_topic_family_df[per_topic_family_df["topic_family"] == family_name].copy()
            if family_topic_frame.empty:
                ax.set_title(f"{family_name} (no data)")
                ax.axis("off")
                continue
            ordered = family_topic_frame.sort_values(["topic_group", "date"])
            topics = sorted(ordered["topic_group"].dropna().unique())
            palette = dict(zip(topics, sns.color_palette("husl", n_colors=max(1, len(topics)))))
            for topic_name, group in ordered.groupby("topic_group", sort=True):
                ax.plot(
                    group["date"],
                    group[metric].fillna(0.0),
                    marker="o",
                    markersize=2.0,
                    linewidth=1.1,
                    color=palette.get(topic_name),
                    label=topic_name,
                )
            add_release_markers(ax, release_dates)
            format_date_axis(ax, span_days)
            ax.set_title(family_name)
            ax.legend(loc="best", fontsize=8, frameon=True)
        for ax in axes_topic_flat[len(ordered_families):]:
            ax.axis("off")
        fig_topic.suptitle(f"Per-topic by family: {METRIC_LABELS[metric]}", fontsize=12)
        fig_topic.tight_layout(rect=[0, 0, 1, 0.95])
        fig_topic.savefig(by_topic_by_family_dir / f"by_topic_by_family_{metric}.png", dpi=140)
        plt.close(fig_topic)
        print(f"plot_progress metric={idx}/{total_metrics} name={metric} stage=done")


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
        "Figure captions for automod series use this same total for the current event window.",
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
    start_ts = utc_ts(config["event_window"]["start_utc"])
    end_ts_exclusive = utc_ts(config["event_window"]["end_utc_exclusive"])
    event_ts = utc_ts(config["event_window"]["launch_day_utc"])
    missing_subreddits = sorted(
        subreddit
        for subreddit in subreddits
        if not (paths.raw_daily_chunks_dir / subreddit).exists()
    )
    if missing_subreddits:
        missing_joined = ",".join(missing_subreddits)
        print(f"warning missing_subreddit_dirs={missing_joined}")

    per_subreddit_counts = compute_daily_metrics(paths.raw_daily_chunks_dir, subreddits)
    per_subreddit_counts = filter_to_event_window(
        per_subreddit_counts,
        start_ts=start_ts,
        end_ts_exclusive=end_ts_exclusive,
    )
    print(f"rows_after_event_window_filter={len(per_subreddit_counts)}")
    sub_to_topic = subreddit_topic_map(config, include_topic_aliases=False)
    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    family_topics = topic_families(config)
    topic_to_family = {
        topic_name: family_name
        for family_name, topic_names in family_topics.items()
        for topic_name in topic_names
    }
    per_subreddit_df = add_time_and_rate_columns(per_subreddit_counts, event_ts)
    per_family_df = build_family_daily(per_subreddit_counts, event_ts, sub_to_family)
    per_topic_family_df = build_topic_family_daily(
        per_subreddit_counts,
        event_ts,
        sub_to_topic=sub_to_topic,
        topic_to_family=topic_to_family,
    )
    overall_df = build_overall_daily(per_subreddit_counts, event_ts)
    validation_df = validate_against_baseline(per_subreddit_counts, paths.baseline_counts_path)

    write_tables(per_subreddit_df, per_family_df, overall_df, validation_df, paths.tables_subdir)
    write_metadata_note(per_subreddit_df, paths.tables_subdir)
    automod_total = int(per_subreddit_df["automod_author_count"].sum())
    span_days = date_span_days(overall_df)
    make_plots(
        per_family_df,
        per_subreddit_df,
        per_topic_family_df,
        overall_df,
        paths.figures_subdir,
        event_ts,
        automod_total=automod_total,
        span_days=span_days,
        family_topics=family_topics,
        subreddit_to_family=sub_to_family,
        sub_to_topic=sub_to_topic,
    )


if __name__ == "__main__":
    main()
