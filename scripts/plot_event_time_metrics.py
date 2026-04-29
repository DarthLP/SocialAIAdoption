"""
Script summary:
This script reads daily event-time metric tables and creates event-time line
plots for semicolon rate, comment length, complexity index, AI-likeness,
AI-typical word intensity, style proxies (assistant-tone, list structure,
repetition similarity, formality), extended lexicon rates, and toxicity-related
proxies. It writes pooled figures, per-subreddit multi-line figures, one
combined strict-10 word graph (pooled), and a pooled multi-panel style overview.

How to apply/run:
- Run after setup pipeline:
  `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
- Figures are saved under `results/figures/event_time/` (pooled) and
  `results/figures/event_time/by_subreddit/`.
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


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI args and return plotting runtime options."""
    parser = argparse.ArgumentParser(description="Plot event-time metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/political_forums_setup.yaml",
        help="Path to YAML configuration file.",
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


def plot_metric(df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str) -> None:
    """Function summary: generate and save one date-based line plot for a chosen pooled metric."""
    _ = event_time_xlabel_text
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=d, x="date", y=y_col, marker="o")
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_metric_by_subreddit(
    df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str
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
        marker="o",
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
) -> None:
    """Function summary: plot two pooled columns on a shared calendar-date axis with legend."""
    _ = event_time_xlabel_text
    if df.empty or y_a not in df.columns or y_b not in df.columns:
        return
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(d["date"], d[y_a], marker="o", label=label_a)
    plt.plot(d["date"], d[y_b], marker="s", label=label_b)
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_style_panel_pooled(df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str) -> None:
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
        ax.plot(d["date"], d[col], marker="o")
        add_release_markers(ax)
        format_month_start_axis(ax)
        ax.set_title(subtitle)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel(col)
    fig.suptitle("Event-time: Style proxies (pooled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ai_likeness_components_pooled(df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str) -> None:
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
        plt.plot(d["date"], d[col], marker="o", label=col, linewidth=1.4)
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
    ai_word_long_df: pd.DataFrame, out_path: Path, *, event_time_xlabel_text: str
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
        marker="o",
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


def main() -> None:
    """Function summary: load daily data and write pooled and per-subreddit event-time figures."""
    args = parse_args()
    config = load_config(args.config)
    xt = event_time_xlabel(config)
    figures_dir = Path(config["paths"]["figures_dir"]) / "event_time"
    by_subreddit_figures_dir = figures_dir / "by_subreddit"
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    by_subreddit_figures_dir.mkdir(parents=True, exist_ok=True)

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
    ]
    for y_col, title, fname in pooled_specs:
        if y_col in df_pooled.columns:
            plot_metric(df_pooled, y_col, title, figures_dir / fname, event_time_xlabel_text=xt)

    if "toxic_lexicon_rate_100w" in df_pooled.columns:
        plot_metric(
            df_pooled,
            "toxic_lexicon_rate_100w",
            "Event-time: Toxic Lexicon Incidence (per 100 words)",
            figures_dir / "event_time_toxic_lexicon_rate.png",
            event_time_xlabel_text=xt,
        )

    plot_two_series_same_axes(
        df_pooled,
        "ai_word_rate_100w",
        "ai_word_extended_rate_100w",
        "strict_10 (per 100 words)",
        "extended (per 100 words)",
        "Event-time: Strict vs Extended AI Lexicon (pooled)",
        "Rate per 100 words",
        figures_dir / "event_time_ai_lexicon_strict_vs_extended.png",
        event_time_xlabel_text=xt,
    )
    plot_style_panel_pooled(df_pooled, figures_dir / "event_time_style_proxies_panel.png", event_time_xlabel_text=xt)
    plot_ai_likeness_components_pooled(
        df_pooled, figures_dir / "event_time_ai_likeness_components_z.png", event_time_xlabel_text=xt
    )

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
        ]
        for y_col, title, fname in by_sub_specs:
            if y_col in df_by_sub.columns:
                plot_metric_by_subreddit(
                    df_by_sub, y_col, title, by_subreddit_figures_dir / fname, event_time_xlabel_text=xt
                )

    ai_word_long_path = tables_dir / "event_time" / "ai_word_rates_daily_long.csv"
    if ai_word_long_path.exists():
        ai_word_long_df = pd.read_csv(ai_word_long_path)
        plot_ai_word_individual_plus_combined(
            ai_word_long_df,
            figures_dir / "event_time_ai_words_individual_plus_combined.png",
            event_time_xlabel_text=xt,
        )


if __name__ == "__main__":
    main()
