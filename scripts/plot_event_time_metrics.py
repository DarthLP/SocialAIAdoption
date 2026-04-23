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
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


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


def plot_metric(df: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    """Function summary: generate and save one event-time line plot for a chosen metric."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="event_time_t", y=y_col, marker="o")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_metric_by_subreddit(df: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    """Function summary: plot one metric over event time with one line per subreddit."""
    if df.empty or y_col not in df.columns:
        return
    sub_df = df[df["subreddit"] != "ALL"].copy()
    if sub_df.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=sub_df.sort_values(["subreddit", "event_time_t"]),
        x="event_time_t",
        y=y_col,
        hue="subreddit",
        marker="o",
    )
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel(y_col)
    plt.legend(title="Subreddit", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_two_series_same_axes(
    df: pd.DataFrame, y_a: str, y_b: str, label_a: str, label_b: str, title: str, y_label: str, out_path: Path
) -> None:
    """Function summary: plot two columns from the same pooled frame on one axes with legend."""
    if df.empty or y_a not in df.columns or y_b not in df.columns:
        return
    d = df.sort_values("event_time_t")
    plt.figure(figsize=(10, 5))
    plt.plot(d["event_time_t"], d[y_a], marker="o", label=label_a)
    plt.plot(d["event_time_t"], d[y_b], marker="s", label=label_b)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_style_panel_pooled(df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: save a 2x2 pooled panel of main style proxy metrics."""
    panels = [
        ("assistant_tone_rate_100w", "Assistant-tone phrases (per 100 words)"),
        ("list_structure_intensity", "List-structure intensity (share of comments)"),
        ("repetition_template_similarity", "Repetition / template similarity (mean)"),
        ("ai_word_extended_rate_100w", "Extended AI lexicon (per 100 words)"),
    ]
    if df.empty:
        return
    d = df.sort_values("event_time_t")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    for ax, (col, subtitle) in zip(axes_flat, panels):
        if col not in d.columns:
            ax.set_visible(False)
            continue
        ax.plot(d["event_time_t"], d[col], marker="o")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax.set_title(subtitle)
        ax.set_xlabel("Event time (days from 2022-11-30)")
        ax.set_ylabel(col)
    fig.suptitle("Event-time: Style proxies (pooled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ai_likeness_components_pooled(df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: plot z-scored AI-likeness input components on one pooled figure."""
    cols = [
        "z_ai_word_rate_100w",
        "z_formality_balance_100w",
        "z_assistant_tone_rate_100w",
        "z_list_structure_intensity",
        "z_contraction_rate_100w",
    ]
    if df.empty or not all(c in df.columns for c in cols):
        return
    d = df.sort_values("event_time_t")
    plt.figure(figsize=(11, 6))
    for col in cols:
        plt.plot(d["event_time_t"], d[col], marker="o", label=col, linewidth=1.4)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    plt.title("Event-time: AI-likeness index components (z-scores, pooled)")
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel("z-score")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_ai_word_individual_plus_combined(ai_word_long_df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: plot strict individual word rates and strict combined rate in one graph."""
    subset = ai_word_long_df[
        (ai_word_long_df["subreddit"] == "ALL")
        & (ai_word_long_df["word_group"].isin(["strict_individual", "strict_combined"]))
    ].copy()
    if subset.empty:
        return

    plt.figure(figsize=(12, 6))
    plot_df = subset.copy()
    plot_df["series"] = plot_df["word"]
    sns.lineplot(
        data=plot_df,
        x="event_time_t",
        y="rate_100w",
        hue="series",
        marker="o",
        linewidth=1.6,
    )

    combined_mask = plot_df["series"] == "strict_10_combined"
    if combined_mask.any():
        combined_df = plot_df[combined_mask].sort_values("event_time_t")
        plt.plot(
            combined_df["event_time_t"],
            combined_df["rate_100w"],
            linestyle="--",
            linewidth=3.0,
            label="strict_10_combined",
        )

    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.title("Event-time: Strict AI Word Rates (10 Individual + Combined)")
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel("Rate per 100 words")
    plt.legend(title="Series", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> None:
    """Function summary: load daily data and write pooled and per-subreddit event-time figures."""
    args = parse_args()
    config = load_config(args.config)
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
            plot_metric(df_pooled, y_col, title, figures_dir / fname)

    if "toxic_lexicon_rate_100w" in df_pooled.columns:
        plot_metric(
            df_pooled,
            "toxic_lexicon_rate_100w",
            "Event-time: Toxic Lexicon Incidence (per 100 words)",
            figures_dir / "event_time_toxic_lexicon_rate.png",
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
    )
    plot_style_panel_pooled(df_pooled, figures_dir / "event_time_style_proxies_panel.png")
    plot_ai_likeness_components_pooled(df_pooled, figures_dir / "event_time_ai_likeness_components_z.png")

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
                plot_metric_by_subreddit(df_by_sub, y_col, title, by_subreddit_figures_dir / fname)

    ai_word_long_path = tables_dir / "event_time" / "ai_word_rates_daily_long.csv"
    if ai_word_long_path.exists():
        ai_word_long_df = pd.read_csv(ai_word_long_path)
        plot_ai_word_individual_plus_combined(
            ai_word_long_df,
            figures_dir / "event_time_ai_words_individual_plus_combined.png",
        )


if __name__ == "__main__":
    main()
