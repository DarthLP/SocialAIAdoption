"""
Script summary:
This script reads daily event-time metric tables and creates event-time line
plots for semicolon rate, comment length, complexity index, AI-likeness,
AI-typical word intensity, style proxies (assistant-tone, list structure,
repetition similarity, formality), extended lexicon rates, and toxicity-related
proxies. It writes pooled figures, per-subreddit multi-line figures, optional
per-topic (daily/weekly/rolling) multi-line figures, one combined strict-10
word graph (pooled), and a pooled multi-panel style overview.

How to apply/run:
- Run after setup pipeline:
  `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
- Optional topic-level views:
  `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml --topic_views --topic_rolling_window 7`
- Figures are saved in view-specific folders:
  - pooled: `results/figures/event_time/pooled/{daily,weekly,rolling_daily}/`
  - by subreddit: `results/figures/event_time/by_subreddit/{daily,weekly,rolling_daily}/`
  - optional by topic: `results/figures/event_time/by_topic/{daily,weekly,rolling_daily}/`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts

LAUNCH_DATE_UTC = pd.Timestamp(datetime(2022, 11, 30))
WORD_WEIGHT_COLS = [
    "semicolon_rate_100w",
    "ai_word_rate_100w",
    "ai_word_extended_rate_100w",
    "toxic_lexicon_rate_100w",
    "contraction_rate_100w",
    "full_form_rate_100w",
    "assistant_tone_rate_100w",
    "formality_balance_100w",
    "passive_rate_100w",
]
COMMENT_WEIGHT_COLS = [
    "comment_length_words",
    "complexity_index",
    "vader_compound_mean",
    "vader_negativity_mean",
    "toxicity_score",
    "list_structure_intensity",
    "repetition_template_similarity",
    "ai_likeness_index",
    "z_ai_word_rate_100w",
    "z_formality_balance_100w",
    "z_assistant_tone_rate_100w",
    "z_list_structure_intensity",
    "z_contraction_rate_100w",
    "detector_primary_human_score",
    "detector_secondary_human_score",
    "hostility_score",
    "emotion_anger",
    "emotion_fear",
    "emotion_sadness",
    "emotion_surprise",
    "perplexity_mean",
    "coverage_detector_primary",
    "coverage_detector_secondary",
    "coverage_perplexity",
    "coverage_hostility",
    "coverage_emotion",
    "detector_low_confidence_share",
]


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI args and return plotting runtime options."""
    parser = argparse.ArgumentParser(description="Plot event-time metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/political_forums_setup.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--topic_views",
        action="store_true",
        help="Generate topic-level figures (daily/weekly/centered-rolling) from by-subreddit table.",
    )
    parser.add_argument(
        "--topic_rolling_window",
        type=int,
        default=7,
        help="Centered rolling window size (in days) for rolling views.",
    )
    return parser.parse_args()


def event_time_xlabel(config: dict) -> str:
    """Function summary: build x-axis label text from config launch_day_utc (UTC date string)."""
    launch_ts = utc_ts(str(config["event_window"]["launch_day_utc"]))
    launch_date = datetime.fromtimestamp(launch_ts, tz=timezone.utc).date().isoformat()
    return f"Event time (days from {launch_date})"


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: ensure a naive UTC datetime date column exists for calendar-date plotting."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=False, errors="coerce")
    if out["date"].isna().all():
        out["date"] = pd.to_datetime(out["date_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    return out.dropna(subset=["date"])


def topic_map() -> dict[str, str]:
    """Function summary: return the fixed subreddit-to-topic mapping aligned with configured primary subreddits."""
    return {
        "AskProgramming": "coding",
        "CodingHelp": "coding",
        "learnprogramming": "coding",
        "coding": "coding",
        "Ask_Politics": "politics",
        "NeutralPolitics": "politics",
        "PoliticalDiscussion": "politics",
        "politics": "politics",
        "moderatepolitics": "politics",
        "cscareerquestions": "career",
        "ITCareerQuestions": "career",
        "csMajors": "career",
        "career": "career",
        "answers": "general_questions",
        "OutOfTheLoop": "general_questions",
        "TooAfraidToAsk": "general_questions",
        "general_questions": "general_questions",
    }


def aggregate_daily_weighted(df: pd.DataFrame, group_col: str, alias_col: str | None = None) -> pd.DataFrame:
    """Function summary: aggregate rows by date and group key using weighted recomputation of rate/mean metrics."""
    if df.empty:
        return pd.DataFrame()
    required = {group_col, "date_utc", "n_comments", "n_words"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    d = ensure_date_column(df.copy())
    if d.empty:
        return pd.DataFrame()
    for col in ["n_comments", "n_words", "strict_ai_word_hits_total", "extended_ai_word_hits_total"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    grouped_rows: list[dict] = []
    for (group_value, date_utc), grp in d.groupby([group_col, "date_utc"], sort=True):
        n_comments = float(grp["n_comments"].sum())
        n_words = float(grp["n_words"].sum())
        if n_comments <= 0:
            continue
        row: dict[str, float | str] = {
            "subreddit": str(group_value),
            group_col: str(group_value),
            "date_utc": str(date_utc),
            "date": grp["date"].min(),
            "n_comments": n_comments,
            "n_words": n_words,
        }
        if alias_col:
            row[alias_col] = str(group_value)
        if {"strict_ai_word_hits_total", "extended_ai_word_hits_total"}.issubset(grp.columns):
            row["strict_ai_word_hits_total"] = float(grp["strict_ai_word_hits_total"].sum())
            row["extended_ai_word_hits_total"] = float(grp["extended_ai_word_hits_total"].sum())
        for col in WORD_WEIGHT_COLS:
            if col in grp.columns:
                numer = (pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * grp["n_words"]).sum()
                row[col] = float(numer / n_words) if n_words > 0 else 0.0
        for col in COMMENT_WEIGHT_COLS:
            if col in grp.columns:
                numer = (pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * grp["n_comments"]).sum()
                row[col] = float(numer / n_comments)
        grouped_rows.append(row)
    if not grouped_rows:
        return pd.DataFrame()
    out = pd.DataFrame(grouped_rows).sort_values([group_col, "date"]).reset_index(drop=True)
    out["event_time_t"] = (out["date"] - LAUNCH_DATE_UTC).dt.days
    return out


def aggregate_topic_daily(df_by_sub: pd.DataFrame) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit daily rows into per-topic daily rows using weighted metric recomputation."""
    if df_by_sub.empty:
        return pd.DataFrame()
    required = {"subreddit", "date_utc", "n_comments", "n_words"}
    if not required.issubset(df_by_sub.columns):
        return pd.DataFrame()
    d = ensure_date_column(df_by_sub.copy())
    d = d[d["subreddit"] != "ALL"].copy()
    d["topic_group"] = d["subreddit"].map(topic_map())
    unknown_subs = sorted(d.loc[d["topic_group"].isna(), "subreddit"].dropna().unique())
    if unknown_subs:
        print(f"[plot_event_time_metrics] skipping unmapped subreddits in topic view: {', '.join(unknown_subs)}", flush=True)
    d = d.dropna(subset=["topic_group"])
    return aggregate_daily_weighted(d, group_col="topic_group", alias_col="topic_group")


def aggregate_weekly_weighted(daily_df: pd.DataFrame, group_col: str, alias_col: str | None = None) -> pd.DataFrame:
    """Function summary: convert daily grouped rows to weekly grouped rows with weighted recomputation."""
    if daily_df.empty:
        return pd.DataFrame()
    d = ensure_date_column(daily_df.copy())
    d["week_start"] = d["date"].dt.to_period("W-MON").dt.start_time
    d["date_utc"] = d["week_start"].dt.strftime("%Y-%m-%dT00:00:00Z")
    d["date"] = d["week_start"]
    return aggregate_daily_weighted(d, group_col=group_col, alias_col=alias_col)


def aggregate_topic_weekly(topic_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: aggregate topic daily rows into weekly bins using weighted recomputation."""
    return aggregate_weekly_weighted(topic_daily_df, group_col="topic_group", alias_col="topic_group")


def grouped_weekly_rolling(df_weekly: pd.DataFrame, group_col: str, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply rolling means within each group for weekly rows while preserving date anchors."""
    if df_weekly.empty:
        return pd.DataFrame()
    if rolling_window <= 1:
        return df_weekly.copy()
    d = df_weekly.sort_values([group_col, "date"]).copy()
    exclude_cols = {"subreddit", "topic_group", "date_utc", "date", group_col}
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    out_parts: list[pd.DataFrame] = []
    for group_value, grp in d.groupby(group_col, sort=True):
        g = grp.copy()
        for col in numeric_cols:
            g[col] = g[col].rolling(window=int(rolling_window), min_periods=1).mean()
        g["subreddit"] = group_value
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, "date"]).reset_index(drop=True)


def grouped_centered_daily_rolling(df_daily: pd.DataFrame, group_col: str, rolling_window_days: int) -> pd.DataFrame:
    """Function summary: smooth daily rows by group using centered day-based rolling windows with edge-aware partial windows."""
    if df_daily.empty:
        return pd.DataFrame()
    if rolling_window_days <= 1:
        return df_daily.copy()
    d = ensure_date_column(df_daily.copy()).sort_values([group_col, "date"])
    exclude_cols = {"subreddit", "topic_group", "date_utc", "date", group_col, "event_time_t"}
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    out_parts: list[pd.DataFrame] = []
    for group_value, grp in d.groupby(group_col, sort=True):
        g = grp.sort_values("date").copy()
        g_indexed = g.set_index("date")
        for col in numeric_cols:
            g_indexed[col] = (
                g_indexed[col]
                .rolling(window=f"{int(rolling_window_days)}D", center=True, min_periods=1)
                .mean()
            )
        g = g_indexed.reset_index()
        g["subreddit"] = group_value
        g["event_time_t"] = (g["date"] - LAUNCH_DATE_UTC).dt.days
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, "date"]).reset_index(drop=True)


def topic_weekly_rolling(topic_weekly_df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply per-topic rolling mean over weekly metrics while preserving weekly date anchors."""
    return grouped_weekly_rolling(topic_weekly_df, group_col="topic_group", rolling_window=rolling_window)


def release_dates() -> list[datetime]:
    """Function summary: return the configured ChatGPT and GPT-4 public release dates used as visual anchors."""
    return [datetime(2022, 11, 30), datetime(2023, 3, 14)]


def add_release_markers(ax: plt.Axes) -> None:
    """Function summary: draw red vertical dotted reference lines at ChatGPT and GPT-4 release dates."""
    for release_date in release_dates():
        ax.axvline(x=release_date, color="red", linestyle=":", linewidth=1.2)


def format_month_start_axis(ax: plt.Axes) -> None:
    """Function summary: force monthly x-axis ticks to the first day of each month for date-based plots."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_metric(
    df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: generate and save one date-based line plot for a chosen pooled metric."""
    _ = event_time_xlabel_text
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=d, x="date", y=y_col, marker=("o" if show_markers else None))
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_metric_by_subreddit(
    df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: plot one metric over calendar dates with one line per subreddit."""
    _ = event_time_xlabel_text
    if df.empty or y_col not in df.columns:
        return
    sub_df = ensure_date_column(df[df["subreddit"] != "ALL"].copy())
    if sub_df.empty:
        return
    subreddits = sorted(sub_df["subreddit"].dropna().unique())
    palette = dict(zip(subreddits, sns.color_palette("tab20", n_colors=max(1, len(subreddits)))))
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=sub_df.sort_values(["subreddit", "date"]),
        x="date",
        y=y_col,
        hue="subreddit",
        palette=palette,
        marker=("o" if show_markers else None),
    )
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_col)
    plt.legend(title="Subreddit", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_metric_by_topic(
    topic_df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: plot one metric over calendar dates with one line per topic group."""
    _ = event_time_xlabel_text
    if topic_df.empty or y_col not in topic_df.columns:
        return
    d = ensure_date_column(topic_df.copy())
    if d.empty:
        return
    topics = sorted(d["topic_group"].dropna().unique())
    palette = dict(zip(topics, sns.color_palette("tab10", n_colors=max(1, len(topics)))))
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=d.sort_values(["topic_group", "date"]),
        x="date",
        y=y_col,
        hue="topic_group",
        palette=palette,
        marker=("o" if show_markers else None),
    )
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_col)
    plt.legend(title="Topic group", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_two_series_same_axes(
    df: pd.DataFrame,
    y_a: str,
    y_b: str,
    label_a: str,
    label_b: str,
    title: str,
    y_label: str,
    out_path: Path,
    *,
    event_time_xlabel_text: str,
    show_markers: bool = True,
) -> None:
    """Function summary: plot two pooled columns on a shared calendar-date axis with legend."""
    _ = event_time_xlabel_text
    if df.empty or y_a not in df.columns or y_b not in df.columns:
        return
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    plt.figure(figsize=(10, 5))
    marker_a = "o" if show_markers else None
    marker_b = "s" if show_markers else None
    plt.plot(d["date"], d[y_a], marker=marker_a, label=label_a)
    plt.plot(d["date"], d[y_b], marker=marker_b, label=label_b)
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_style_panel_pooled(
    df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: save a 2x2 pooled panel of main style proxy metrics on date-based axes."""
    _ = event_time_xlabel_text
    panels = [
        ("assistant_tone_rate_100w", "Assistant-tone phrases (per 100 words)"),
        ("list_structure_intensity", "List-structure intensity (share of comments)"),
        ("repetition_template_similarity", "Repetition / template similarity (mean)"),
        ("ai_word_extended_rate_100w", "Extended AI lexicon (per 100 words)"),
    ]
    if df.empty:
        return
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    for ax, (col, subtitle) in zip(axes_flat, panels):
        if col not in d.columns:
            ax.set_visible(False)
            continue
        ax.plot(d["date"], d[col], marker=("o" if show_markers else None))
        add_release_markers(ax)
        format_month_start_axis(ax)
        ax.set_title(subtitle)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel(col)
    fig.suptitle("Event-time: Style proxies (pooled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ai_likeness_components_pooled(
    df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: plot z-scored AI-likeness input components on one pooled date-based figure."""
    _ = event_time_xlabel_text
    cols = [
        "z_ai_word_rate_100w",
        "z_formality_balance_100w",
        "z_assistant_tone_rate_100w",
        "z_list_structure_intensity",
        "z_contraction_rate_100w",
    ]
    if df.empty or not all(c in df.columns for c in cols):
        return
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    plt.figure(figsize=(11, 6))
    for col in cols:
        plt.plot(d["date"], d[col], marker=("o" if show_markers else None), label=col, linewidth=1.4)
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    plt.title("Event-time: AI-likeness index components (z-scores, pooled)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("z-score")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_ai_word_individual_plus_combined(
    ai_word_long_df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: plot strict individual word rates and strict combined rate on a calendar-date axis."""
    _ = event_time_xlabel_text
    subset = ai_word_long_df[
        (ai_word_long_df["subreddit"] == "ALL")
        & (ai_word_long_df["word_group"].isin(["strict_individual", "strict_combined"]))
    ].copy()
    if subset.empty:
        return
    subset = ensure_date_column(subset)
    if subset.empty:
        return

    plt.figure(figsize=(12, 6))
    plot_df = subset.copy()
    plot_df["series"] = plot_df["word"]
    sns.lineplot(
        data=plot_df,
        x="date",
        y="rate_100w",
        hue="series",
        marker=("o" if show_markers else None),
        palette="tab20",
        linewidth=1.6,
    )

    combined_mask = plot_df["series"] == "strict_10_combined"
    if combined_mask.any():
        combined_df = plot_df[combined_mask].sort_values("date")
        plt.plot(
            combined_df["date"],
            combined_df["rate_100w"],
            linestyle="--",
            linewidth=3.0,
            label="strict_10_combined",
        )

    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title("Event-time: Strict AI Word Rates (10 Individual + Combined)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Rate per 100 words")
    plt.legend(title="Series", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def aggregate_ai_word_long_weekly(ai_word_long_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: aggregate pooled strict-word long table to weekly bins and recompute per-100-word rates."""
    if ai_word_long_df.empty:
        return pd.DataFrame()
    d = ai_word_long_df[
        (ai_word_long_df["subreddit"] == "ALL")
        & (ai_word_long_df["word_group"].isin(["strict_individual", "strict_combined"]))
    ].copy()
    if d.empty:
        return pd.DataFrame()
    d = ensure_date_column(d)
    d["week_start"] = d["date"].dt.to_period("W-MON").dt.start_time
    out = (
        d.groupby(["week_start", "word", "word_group"], as_index=False)[["hits", "n_words"]]
        .sum()
        .rename(columns={"week_start": "date"})
    )
    out["subreddit"] = "ALL"
    out["date_utc"] = out["date"].dt.strftime("%Y-%m-%dT00:00:00Z")
    out["rate_100w"] = 0.0
    mask = out["n_words"] > 0
    out.loc[mask, "rate_100w"] = out.loc[mask, "hits"] / out.loc[mask, "n_words"] * 100.0
    return out.sort_values(["date", "word_group", "word"]).reset_index(drop=True)


def rolling_ai_word_long_weekly(ai_word_weekly_df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply rolling smoothing to weekly pooled strict-word trajectories."""
    if ai_word_weekly_df.empty or rolling_window <= 1:
        return ai_word_weekly_df.copy()
    d = ai_word_weekly_df.sort_values(["word_group", "word", "date"]).copy()
    out_parts: list[pd.DataFrame] = []
    for (_, _), grp in d.groupby(["word_group", "word"], sort=True):
        g = grp.copy()
        g["rate_100w"] = g["rate_100w"].rolling(window=int(rolling_window), min_periods=1).mean()
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values(["date", "word_group", "word"]).reset_index(drop=True)


def rolling_ai_word_long_centered_daily(ai_word_daily_df: pd.DataFrame, rolling_window_days: int) -> pd.DataFrame:
    """Function summary: apply centered day-based rolling smoothing to pooled strict-word daily trajectories."""
    if ai_word_daily_df.empty or rolling_window_days <= 1:
        return ai_word_daily_df.copy()
    d = ai_word_daily_df[
        (ai_word_daily_df["subreddit"] == "ALL")
        & (ai_word_daily_df["word_group"].isin(["strict_individual", "strict_combined"]))
    ].copy()
    if d.empty:
        return pd.DataFrame()
    d = ensure_date_column(d).sort_values(["word_group", "word", "date"])
    out_parts: list[pd.DataFrame] = []
    for (_, _), grp in d.groupby(["word_group", "word"], sort=True):
        g = grp.sort_values("date").copy()
        g_indexed = g.set_index("date")
        g_indexed["rate_100w"] = (
            g_indexed["rate_100w"]
            .rolling(window=f"{int(rolling_window_days)}D", center=True, min_periods=1)
            .mean()
        )
        g = g_indexed.reset_index()
        g["date_utc"] = g["date"].dt.strftime("%Y-%m-%dT00:00:00Z")
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values(["date", "word_group", "word"]).reset_index(drop=True)


def main() -> None:
    """Function summary: load daily data and write pooled and per-subreddit event-time figures."""
    args = parse_args()
    config = load_config(args.config)
    xt = event_time_xlabel(config)
    figures_dir = Path(config["paths"]["figures_dir"]) / "event_time"
    pooled_figures_dir = figures_dir / "pooled"
    by_subreddit_figures_dir = figures_dir / "by_subreddit"
    by_topic_figures_dir = figures_dir / "by_topic"
    pooled_view_dirs = {
        "daily": pooled_figures_dir / "daily",
        "weekly": pooled_figures_dir / "weekly",
        "rolling_daily": pooled_figures_dir / "rolling_daily",
    }
    by_sub_view_dirs = {
        "daily": by_subreddit_figures_dir / "daily",
        "weekly": by_subreddit_figures_dir / "weekly",
        "rolling_daily": by_subreddit_figures_dir / "rolling_daily",
    }
    by_topic_view_dirs = {
        "daily": by_topic_figures_dir / "daily",
        "weekly": by_topic_figures_dir / "weekly",
        "rolling_daily": by_topic_figures_dir / "rolling_daily",
    }
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    pooled_figures_dir.mkdir(parents=True, exist_ok=True)
    by_subreddit_figures_dir.mkdir(parents=True, exist_ok=True)
    by_topic_figures_dir.mkdir(parents=True, exist_ok=True)
    for out_dir in [*pooled_view_dirs.values(), *by_sub_view_dirs.values(), *by_topic_view_dirs.values()]:
        out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = tables_dir / "event_time" / "event_time_daily_metrics_pooled.csv"
    if not daily_path.exists():
        daily_path = tables_dir / "event_time_daily_metrics.csv"
    df_pooled = pd.read_csv(daily_path)
    df_pooled = df_pooled.sort_values("event_time_t")

    by_sub_path = tables_dir / "event_time" / "event_time_daily_metrics_by_subreddit.csv"
    df_by_sub = pd.read_csv(by_sub_path) if by_sub_path.exists() else pd.DataFrame()

    pooled_specs: list[tuple[str, str, str]] = [
        ("semicolon_rate_100w", "Event-time: Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
        ("comment_length_words", "Event-time: Average Comment Length (words)", "event_time_comment_length.png"),
        ("complexity_index", "Event-time: Complexity Index", "event_time_complexity_index.png"),
        ("ai_likeness_index", "Event-time: AI-likeness Index", "event_time_ai_likeness.png"),
        ("ai_word_rate_100w", "Event-time: AI-typical Word Rate (strict 10-word basket)", "event_time_ai_word_rate.png"),
        (
            "ai_word_extended_rate_100w",
            "Event-time: Extended AI Lexicon Rate (per 100 words)",
            "event_time_ai_word_extended_rate.png",
        ),
        (
            "assistant_tone_rate_100w",
            "Event-time: Assistant-tone Phrase Rate (per 100 words)",
            "event_time_assistant_tone_rate.png",
        ),
        (
            "list_structure_intensity",
            "Event-time: List-structure Intensity (share of comments)",
            "event_time_list_structure_intensity.png",
        ),
        (
            "repetition_template_similarity",
            "Event-time: Repetition / Template Similarity (mean Jaccard to recent)",
            "event_time_repetition_template_similarity.png",
        ),
        (
            "formality_balance_100w",
            "Event-time: Formality Balance (full-form minus contraction rate, per 100 words)",
            "event_time_formality_balance.png",
        ),
        ("contraction_rate_100w", "Event-time: Contraction Rate (per 100 words)", "event_time_contraction_rate.png"),
        ("full_form_rate_100w", "Event-time: Full-form Rate (per 100 words)", "event_time_full_form_rate.png"),
        ("vader_compound_mean", "Event-time: VADER Compound Mean (sentiment)", "event_time_vader_compound_mean.png"),
        ("toxicity_score", "Event-time: Toxicity Proxy (VADER negativity mean)", "event_time_toxicity_score.png"),
        ("detector_primary_human_score", "Event-time: Detector Primary Human Score", "event_time_detector_primary_human_score.png"),
        ("detector_secondary_human_score", "Event-time: Detector Secondary Human Score", "event_time_detector_secondary_human_score.png"),
        ("passive_rate_100w", "Event-time: Passive Construction Rate (per 100 words)", "event_time_passive_rate.png"),
        ("perplexity_mean", "Event-time: Perplexity Mean", "event_time_perplexity_mean.png"),
        ("hostility_score", "Event-time: Hostility Score Mean", "event_time_hostility_score.png"),
        ("emotion_anger", "Event-time: Emotion Anger Mean", "event_time_emotion_anger.png"),
        ("emotion_fear", "Event-time: Emotion Fear Mean", "event_time_emotion_fear.png"),
        ("emotion_sadness", "Event-time: Emotion Sadness Mean", "event_time_emotion_sadness.png"),
        ("emotion_surprise", "Event-time: Emotion Surprise Mean", "event_time_emotion_surprise.png"),
        ("coverage_perplexity", "Event-time: Perplexity Coverage Share", "event_time_coverage_perplexity.png"),
        ("coverage_detector_primary", "Event-time: Detector Primary Coverage Share", "event_time_coverage_detector_primary.png"),
    ]
    pooled_daily = aggregate_daily_weighted(df_pooled.copy(), group_col="subreddit")
    pooled_weekly = aggregate_weekly_weighted(pooled_daily, group_col="subreddit")
    pooled_rolling_daily = grouped_centered_daily_rolling(
        pooled_daily, group_col="subreddit", rolling_window_days=int(max(1, args.topic_rolling_window))
    )
    pooled_views: list[tuple[str, pd.DataFrame]] = [
        ("daily", pooled_daily),
        ("weekly", pooled_weekly),
        ("rolling_daily", pooled_rolling_daily),
    ]
    for view_name, view_df in pooled_views:
        print(
            f"[plot_event_time_metrics] pooled_view_start view={view_name} rows={len(view_df)}",
            flush=True,
        )
        view_dir = pooled_view_dirs[view_name]
        show_markers = view_name != "rolling_daily"
        for y_col, title, fname in pooled_specs:
            if y_col in view_df.columns:
                print(f"[plot_event_time_metrics] pooled_metric view={view_name} metric={y_col}", flush=True)
                plot_metric(
                    view_df,
                    y_col,
                    f"{title} ({view_name})",
                    view_dir / fname,
                    event_time_xlabel_text=xt,
                    show_markers=show_markers,
                )

        if "toxic_lexicon_rate_100w" in view_df.columns:
            plot_metric(
                view_df,
                "toxic_lexicon_rate_100w",
                f"Event-time: Toxic Lexicon Incidence (per 100 words) ({view_name})",
                view_dir / "event_time_toxic_lexicon_rate.png",
                event_time_xlabel_text=xt,
                show_markers=show_markers,
            )

        plot_two_series_same_axes(
            view_df,
            "ai_word_rate_100w",
            "ai_word_extended_rate_100w",
            "strict_10 (per 100 words)",
            "extended (per 100 words)",
            f"Event-time: Strict vs Extended AI Lexicon (pooled, {view_name})",
            "Rate per 100 words",
            view_dir / "event_time_ai_lexicon_strict_vs_extended.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        plot_style_panel_pooled(
            view_df,
            view_dir / "event_time_style_proxies_panel.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        plot_ai_likeness_components_pooled(
            view_df,
            view_dir / "event_time_ai_likeness_components_z.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        print(f"[plot_event_time_metrics] pooled_view_done view={view_name}", flush=True)

    if not df_by_sub.empty:
        by_sub_specs: list[tuple[str, str, str]] = [
            ("semicolon_rate_100w", "Per-subreddit: Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
            ("comment_length_words", "Per-subreddit: Average Comment Length (words)", "event_time_comment_length.png"),
            ("complexity_index", "Per-subreddit: Complexity Index", "event_time_complexity_index.png"),
            ("ai_likeness_index", "Per-subreddit: AI-likeness Index", "event_time_ai_likeness.png"),
            ("ai_word_rate_100w", "Per-subreddit: Strict 10-word Rate (per 100 words)", "event_time_ai_word_rate.png"),
            (
                "ai_word_extended_rate_100w",
                "Per-subreddit: Extended AI Lexicon (per 100 words)",
                "event_time_ai_word_extended_rate.png",
            ),
            (
                "assistant_tone_rate_100w",
                "Per-subreddit: Assistant-tone Phrase Rate (per 100 words)",
                "event_time_assistant_tone_rate.png",
            ),
            (
                "list_structure_intensity",
                "Per-subreddit: List-structure Intensity",
                "event_time_list_structure_intensity.png",
            ),
            (
                "repetition_template_similarity",
                "Per-subreddit: Repetition / Template Similarity",
                "event_time_repetition_template_similarity.png",
            ),
            ("formality_balance_100w", "Per-subreddit: Formality Balance (per 100 words)", "event_time_formality_balance.png"),
            ("contraction_rate_100w", "Per-subreddit: Contraction Rate (per 100 words)", "event_time_contraction_rate.png"),
            ("toxicity_score", "Per-subreddit: VADER Negativity Mean", "event_time_toxicity_score.png"),
            ("toxic_lexicon_rate_100w", "Per-subreddit: Toxic Lexicon (per 100 words)", "event_time_toxic_lexicon_rate.png"),
            ("detector_primary_human_score", "Per-subreddit: Detector Primary Human Score", "event_time_detector_primary_human_score.png"),
            ("passive_rate_100w", "Per-subreddit: Passive Construction Rate (per 100 words)", "event_time_passive_rate.png"),
            ("perplexity_mean", "Per-subreddit: Perplexity Mean", "event_time_perplexity_mean.png"),
            ("hostility_score", "Per-subreddit: Hostility Score Mean", "event_time_hostility_score.png"),
            ("emotion_anger", "Per-subreddit: Emotion Anger Mean", "event_time_emotion_anger.png"),
            ("emotion_fear", "Per-subreddit: Emotion Fear Mean", "event_time_emotion_fear.png"),
            ("emotion_sadness", "Per-subreddit: Emotion Sadness Mean", "event_time_emotion_sadness.png"),
            ("emotion_surprise", "Per-subreddit: Emotion Surprise Mean", "event_time_emotion_surprise.png"),
        ]
        by_sub_daily = aggregate_daily_weighted(df_by_sub[df_by_sub["subreddit"] != "ALL"].copy(), group_col="subreddit")
        by_sub_weekly = aggregate_weekly_weighted(by_sub_daily, group_col="subreddit")
        by_sub_rolling_daily = grouped_centered_daily_rolling(
            by_sub_daily, group_col="subreddit", rolling_window_days=int(max(1, args.topic_rolling_window))
        )
        by_sub_views: list[tuple[str, pd.DataFrame]] = [
            ("daily", by_sub_daily),
            ("weekly", by_sub_weekly),
            ("rolling_daily", by_sub_rolling_daily),
        ]
        for view_name, view_df in by_sub_views:
            print(
                f"[plot_event_time_metrics] by_subreddit_view_start view={view_name} rows={len(view_df)}",
                flush=True,
            )
            show_markers = view_name != "rolling_daily"
            for y_col, title, fname in by_sub_specs:
                if y_col in view_df.columns:
                    print(f"[plot_event_time_metrics] by_subreddit_metric view={view_name} metric={y_col}", flush=True)
                    plot_metric_by_subreddit(
                        view_df,
                        y_col,
                        f"{title} ({view_name})",
                        by_sub_view_dirs[view_name] / fname,
                        event_time_xlabel_text=xt,
                        show_markers=show_markers,
                    )
            print(f"[plot_event_time_metrics] by_subreddit_view_done view={view_name}", flush=True)

    if args.topic_views and not df_by_sub.empty:
        topic_specs: list[tuple[str, str, str]] = [
            ("semicolon_rate_100w", "Per-topic ({view}): Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
            ("comment_length_words", "Per-topic ({view}): Average Comment Length (words)", "event_time_comment_length.png"),
            ("complexity_index", "Per-topic ({view}): Complexity Index", "event_time_complexity_index.png"),
            ("ai_likeness_index", "Per-topic ({view}): AI-likeness Index", "event_time_ai_likeness.png"),
            ("ai_word_rate_100w", "Per-topic ({view}): Strict 10-word Rate (per 100 words)", "event_time_ai_word_rate.png"),
            (
                "ai_word_extended_rate_100w",
                "Per-topic ({view}): Extended AI Lexicon (per 100 words)",
                "event_time_ai_word_extended_rate.png",
            ),
            (
                "assistant_tone_rate_100w",
                "Per-topic ({view}): Assistant-tone Phrase Rate (per 100 words)",
                "event_time_assistant_tone_rate.png",
            ),
            ("list_structure_intensity", "Per-topic ({view}): List-structure Intensity", "event_time_list_structure_intensity.png"),
            (
                "repetition_template_similarity",
                "Per-topic ({view}): Repetition / Template Similarity",
                "event_time_repetition_template_similarity.png",
            ),
            ("formality_balance_100w", "Per-topic ({view}): Formality Balance (per 100 words)", "event_time_formality_balance.png"),
            ("contraction_rate_100w", "Per-topic ({view}): Contraction Rate (per 100 words)", "event_time_contraction_rate.png"),
            ("toxicity_score", "Per-topic ({view}): VADER Negativity Mean", "event_time_toxicity_score.png"),
            ("toxic_lexicon_rate_100w", "Per-topic ({view}): Toxic Lexicon (per 100 words)", "event_time_toxic_lexicon_rate.png"),
            ("detector_primary_human_score", "Per-topic ({view}): Detector Primary Human Score", "event_time_detector_primary_human_score.png"),
            ("passive_rate_100w", "Per-topic ({view}): Passive Construction Rate (per 100 words)", "event_time_passive_rate.png"),
            ("perplexity_mean", "Per-topic ({view}): Perplexity Mean", "event_time_perplexity_mean.png"),
            ("hostility_score", "Per-topic ({view}): Hostility Score Mean", "event_time_hostility_score.png"),
            ("emotion_anger", "Per-topic ({view}): Emotion Anger Mean", "event_time_emotion_anger.png"),
            ("emotion_fear", "Per-topic ({view}): Emotion Fear Mean", "event_time_emotion_fear.png"),
            ("emotion_sadness", "Per-topic ({view}): Emotion Sadness Mean", "event_time_emotion_sadness.png"),
            ("emotion_surprise", "Per-topic ({view}): Emotion Surprise Mean", "event_time_emotion_surprise.png"),
        ]
        topic_daily = aggregate_topic_daily(df_by_sub)
        topic_weekly = aggregate_topic_weekly(topic_daily)
        topic_rolling_daily = grouped_centered_daily_rolling(
            topic_daily, group_col="topic_group", rolling_window_days=int(max(1, args.topic_rolling_window))
        )

        topic_views: list[tuple[str, pd.DataFrame]] = [
            ("daily", topic_daily),
            ("weekly", topic_weekly),
            ("rolling_daily", topic_rolling_daily),
        ]
        for view_name, view_df in topic_views:
            if view_df.empty:
                continue
            show_markers = view_name != "rolling_daily"
            print(f"[plot_event_time_metrics] topic_view_start view={view_name} rows={len(view_df)}", flush=True)
            for y_col, title_template, filename_template in topic_specs:
                if y_col in view_df.columns:
                    print(f"[plot_event_time_metrics] topic_metric view={view_name} metric={y_col}", flush=True)
                    plot_metric_by_topic(
                        view_df,
                        y_col,
                        title_template.format(view=view_name),
                        by_topic_view_dirs[view_name] / filename_template,
                        event_time_xlabel_text=xt,
                        show_markers=show_markers,
                    )
            print(f"[plot_event_time_metrics] topic_view_done view={view_name}", flush=True)

    ai_word_long_path = tables_dir / "event_time" / "ai_word_rates_daily_long.csv"
    if ai_word_long_path.exists():
        ai_word_long_df = pd.read_csv(ai_word_long_path)
        ai_word_long_daily = ai_word_long_df.copy()
        ai_word_long_weekly = aggregate_ai_word_long_weekly(ai_word_long_df)
        ai_word_long_rolling_daily = rolling_ai_word_long_centered_daily(
            ai_word_long_daily, rolling_window_days=int(max(1, args.topic_rolling_window))
        )
        ai_word_views: list[tuple[str, pd.DataFrame]] = [
            ("daily", ai_word_long_daily),
            ("weekly", ai_word_long_weekly),
            ("rolling_daily", ai_word_long_rolling_daily),
        ]
        for view_name, view_df in ai_word_views:
            if view_df.empty:
                continue
            print(f"[plot_event_time_metrics] ai_word_view view={view_name} rows={len(view_df)}", flush=True)
            show_markers = view_name != "rolling_daily"
            plot_ai_word_individual_plus_combined(
                view_df,
                pooled_view_dirs[view_name] / "event_time_ai_words_individual_plus_combined.png",
                event_time_xlabel_text=xt,
                show_markers=show_markers,
            )


if __name__ == "__main__":
    main()
