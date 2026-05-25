"""
Script summary:
Plot polarization and AI-use descriptives from prepared CSV tables.

Functionality:
- Family-level daily trends for primary polarization metrics (raw daily and trailing rolling).
- Topic- and country-panel overlays; Italian-topic subset plots.
- Italian ideology bucket rates (left/center/right per 100w) and pole-share comparison figures.
- Vertical ban reference lines from config.
- Rolling view: same figures under descriptives/rolling_daily/{view}/ (7-day trailing default).

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml --rolling_window 7
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import pandas as pd



def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
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

from scripts.diagnostics.descriptives_util import grouped_trailing_daily_rolling  # noqa: E402

from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    plot_reference_dates_calendar_utc,
    require_dominant_v1_ideology_scoring,
    tables_subdir,
)

TOPIC_DISPLAY_LABELS = {
    "it_political": "IT political (soft)",
    "it_pure_political": "IT pure political",
    "it_others": "IT others",
    "us": "US political (EN)",
    "uk_political": "UK political",
    "uk": "UK hub",
    "de": "DE hub",
    "eu": "EU hub (EN)",
}

ITALIAN_TOPICS = frozenset({"it_political", "it_pure_political", "it_others"})

IDEOLOGY_BUCKET_METRICS = (
    ("left_rate_100w_mean", "Left"),
    ("center_rate_100w_mean", "Center"),
    ("right_rate_100w_mean", "Right"),
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot polarization descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=7,
        help="Trailing calendar-day window for rolling_daily figures (past-only).",
    )
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: draw vertical reference dates on an axes.

    Parameters:
    - ax: matplotlib axes.
    - config: study YAML.
    """
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def topic_display_label(topic: str) -> str:
    """Function summary: map internal topic code to plot legend label.

    Parameters:
    - topic: topic id (e.g. it_political).

    Returns:
    - Human-readable label.
    """
    return TOPIC_DISPLAY_LABELS.get(topic, topic.replace("_", " "))


def plot_topic_metric(
    topic_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    config: dict,
    title: str,
    topics: set[str] | None = None,
) -> None:
    """Function summary: line plot of one metric by topic over date.

    Parameters:
    - topic_df: daily_by_topic table.
    - metric: column name.
    - out_path: PNG path.
    - config: YAML for reference lines.
    - title: plot title.
    - topics: optional subset of topic ids to include.
    """
    if topic_df.empty or metric not in topic_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work = topic_df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    if topics is not None:
        work = work[work["topic"].isin(topics)]
    if work.empty:
        plt.close(fig)
        return
    for topic, grp in work.groupby("topic"):
        grp = grp.sort_values("date_utc")
        ax.plot(grp["date_utc"], grp[metric], label=topic_display_label(str(topic)), alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(title)
    ax.set_xlabel("date_utc")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_family_metric(family_df: pd.DataFrame, metric: str, out_path: Path, config: dict, title: str) -> None:
    """Function summary: line plot of one metric by topic_family over date.

    Parameters:
    - family_df: daily_by_topic_family table.
    - metric: column name.
    - out_path: PNG path.
    - config: YAML for reference lines.
    - title: plot title.
    """
    if family_df.empty or metric not in family_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work = family_df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    for family, grp in work.groupby("topic_family"):
        grp = grp.sort_values("date_utc")
        ax.plot(grp["date_utc"], grp[metric], label=family, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(title)
    ax.set_xlabel("date_utc")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_country_panel(
    country_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    config: dict,
    title_suffix: str = "",
) -> None:
    """Function summary: plot metric by country_panel for ban-window comparison.

    Parameters:
    - country_df: daily_country_panel table.
    - metric: column to plot.
    - out_path: PNG output.
    - config: study config.
    - title_suffix: optional suffix for plot title (e.g. ' (7d trailing)').
    """
    if country_df.empty or metric not in country_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work = country_df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    for panel, grp in work.groupby("country_panel"):
        grp = grp.sort_values("date_utc")
        ax.plot(grp["date_utc"], grp[metric], label=panel, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(f"Country panel: {metric}{title_suffix}")
    ax.set_xlabel("date_utc")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ideology_bucket_rates(
    df: pd.DataFrame,
    group_col: str,
    group_value: str,
    out_path: Path,
    config: dict,
    title_suffix: str,
    slice_label: str,
) -> None:
    """Function summary: plot left/center/right ideology rates for one filtered group.

    Parameters:
    - df: daily family or topic table.
    - group_col: column to filter on (topic_family or topic).
    - group_value: value to keep (e.g. it_political).
    - out_path: PNG path.
    - config: study YAML for ban reference lines.
    - title_suffix: appended to title (e.g. rolling window note).
    - slice_label: human-readable slice name for title.

    Returns:
    - None; skips if required columns or rows are missing.
    """
    required = {m[0] for m in IDEOLOGY_BUCKET_METRICS}
    if df.empty or group_col not in df.columns or not required.issubset(df.columns):
        return
    work = df[df[group_col].astype(str) == group_value].copy()
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    work = work.sort_values("date_utc")
    for metric, label in IDEOLOGY_BUCKET_METRICS:
        ax.plot(work["date_utc"], work[metric], label=label, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(f"Ideology bucket rates: {slice_label}{title_suffix}")
    ax.set_xlabel("date_utc")
    ax.set_ylabel("Ideology lexicon hits per 100 words")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pole_share_comparison(
    family_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    out_path: Path,
    config: dict,
    title_suffix: str,
) -> None:
    """Function summary: compare pole share for it_political family vs it_pure_political topic.

    Parameters:
    - family_df: daily_by_topic_family table.
    - topic_df: daily_by_topic table.
    - out_path: PNG path.
    - config: study YAML for ban reference lines.
    - title_suffix: appended to plot title.

    Returns:
    - None; skips if pole_share column or both series are unavailable.
    """
    if "pole_share" not in family_df.columns and (
        topic_df.empty or "pole_share" not in topic_df.columns
    ):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    if not family_df.empty and "pole_share" in family_df.columns and "topic_family" in family_df.columns:
        fam = family_df[family_df["topic_family"].astype(str) == "it_political"].copy()
        if not fam.empty:
            fam["date_utc"] = pd.to_datetime(fam["date_utc"])
            fam = fam.sort_values("date_utc")
            ax.plot(
                fam["date_utc"],
                fam["pole_share"],
                label="it_political (topic family)",
                alpha=0.85,
            )
            plotted = True
    if not topic_df.empty and "pole_share" in topic_df.columns and "topic" in topic_df.columns:
        top = topic_df[topic_df["topic"].astype(str) == "it_pure_political"].copy()
        if not top.empty:
            top["date_utc"] = pd.to_datetime(top["date_utc"])
            top = top.sort_values("date_utc")
            ax.plot(
                top["date_utc"],
                top["pole_share"],
                label="it_pure_political (topic)",
                alpha=0.85,
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    add_ref_lines(ax, config)
    ax.set_title(f"Pole share (L+R)/(L+C+R){title_suffix}")
    ax.set_xlabel("date_utc")
    ax.set_ylabel("Pole share (L+R)/(L+C+R)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_italian_ideology_figures(
    family_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    fig_dir: Path,
    config: dict,
    title_suffix: str,
) -> None:
    """Function summary: write Italian-focused ideology bucket and pole-share PNGs.

    Parameters:
    - family_df: daily_by_topic_family (or rolled) table.
    - topic_df: daily_by_topic (or rolled) table.
    - fig_dir: output directory.
    - config: study YAML.
    - title_suffix: appended to plot titles.

    Returns:
    - None.
    """
    ideology_dir = fig_dir / "ideology"
    ideology_dir.mkdir(parents=True, exist_ok=True)
    plot_ideology_bucket_rates(
        family_df,
        "topic_family",
        "it_political",
        ideology_dir / "it_political_ideology_bucket_rates.png",
        config,
        title_suffix,
        "it_political (topic family)",
    )
    plot_ideology_bucket_rates(
        topic_df,
        "topic",
        "it_pure_political",
        ideology_dir / "it_pure_political_ideology_bucket_rates.png",
        config,
        title_suffix,
        "it_pure_political (topic)",
    )
    plot_pole_share_comparison(
        family_df,
        topic_df,
        ideology_dir / "italian_ideology_pole_share_it_political_vs_pure.png",
        config,
        title_suffix,
    )


def write_all_descriptive_figures(
    family_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    country_df: pd.DataFrame,
    fig_dir: Path,
    config: dict,
    title_suffix: str,
) -> None:
    """Function summary: write family, topic, and country-panel metric figures to fig_dir.

    Parameters:
    - family_df: daily_by_topic_family (or rolled) table.
    - topic_df: daily_by_topic table (may be empty).
    - country_df: daily_country_panel table (may be empty).
    - fig_dir: output directory for PNGs.
    - config: study YAML for reference lines.
    - title_suffix: appended to plot titles (e.g. '' or ' (7d trailing)').

    Returns:
    - None.
    """
    by_family_dir = fig_dir / "by_family"
    by_topic_dir = fig_dir / "by_topic"
    by_topic_italian_dir = fig_dir / "by_topic_italian"
    country_panel_dir = fig_dir / "country_panel"
    for d in (by_family_dir, by_topic_dir, by_topic_italian_dir, country_panel_dir):
        d.mkdir(parents=True, exist_ok=True)
    metric_plots = [
        ("net_ideology_mean", "net_ideology"),
        ("extremity_mean", "extremity"),
        ("other_side_salience_rate_100w_mean", "other_side_salience"),
        ("aggression_rate_100w_mean", "aggression"),
        ("ai_style_rate_100w_mean", "ai_style"),
        ("em_dash_rate_100w", "em_dash_rate"),
        ("exclamation_rate_100w_mean", "exclamation_rate"),
        ("avg_words_per_sentence_mean", "avg_words_per_sentence"),
        ("semicolon_rate_100w", "semicolon_rate"),
        ("colon_rate_100w", "colon_rate"),
        ("hedging_phrase_rate_100w", "hedging_phrase"),
        ("complexity_index", "complexity_index"),
    ]
    for metric, slug in metric_plots:
        plot_family_metric(
            family_df, metric, by_family_dir / f"{slug}.png", config, f"{metric}{title_suffix}"
        )
        if not topic_df.empty:
            plot_topic_metric(
                topic_df, metric, by_topic_dir / f"{slug}.png", config, f"{metric}{title_suffix}"
            )
            plot_topic_metric(
                topic_df,
                metric,
                by_topic_italian_dir / f"{slug}.png",
                config,
                f"{metric} (Italian topics){title_suffix}",
                topics=ITALIAN_TOPICS,
            )

    if not country_df.empty:
        for metric, fname in [
            ("ai_style_rate_100w_mean", "ai_style.png"),
            ("em_dash_rate_100w", "em_dash_rate.png"),
            ("net_ideology_mean", "net_ideology.png"),
            ("esteban_ray_index", "esteban_ray.png"),
        ]:
            plot_country_panel(
                country_df, metric, country_panel_dir / fname, config, title_suffix=title_suffix
            )

    write_italian_ideology_figures(family_df, topic_df, fig_dir, config, title_suffix)


def main() -> None:
    """Function summary: generate descriptives figures from CSV tables (daily and rolling_daily)."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    require_dominant_v1_ideology_scoring(config)
    tables_dir = tables_subdir(config, "descriptives")
    fig_dir = figures_subdir(config, "descriptives", "daily")
    fig_roll_dir = figures_subdir(config, "descriptives", "rolling_daily")
    rolling_days = int(max(1, args.rolling_window))
    roll_title_suffix = f" ({rolling_days}d trailing)"

    family_path = tables_dir / "daily_by_topic_family.csv"
    topic_path = tables_dir / "daily_by_topic.csv"
    country_path = tables_dir / "daily_country_panel.csv"
    if not family_path.is_file():
        print("[plot_polarization_descriptives] run prepare_polarization_descriptives.py first", flush=True)
        return

    family_df = pd.read_csv(family_path)
    topic_df = pd.read_csv(topic_path) if topic_path.is_file() else pd.DataFrame()
    country_df = pd.read_csv(country_path) if country_path.is_file() else pd.DataFrame()

    write_all_descriptive_figures(family_df, topic_df, country_df, fig_dir, config, title_suffix="")

    family_roll = grouped_trailing_daily_rolling(family_df, "topic_family", rolling_days)
    topic_roll = (
        grouped_trailing_daily_rolling(topic_df, "topic", rolling_days) if not topic_df.empty else pd.DataFrame()
    )
    country_roll = (
        grouped_trailing_daily_rolling(country_df, "country_panel", rolling_days)
        if not country_df.empty
        else pd.DataFrame()
    )
    write_all_descriptive_figures(
        family_roll, topic_roll, country_roll, fig_roll_dir, config, title_suffix=roll_title_suffix
    )

    print(
        f"[plot_polarization_descriptives] wrote figures to {fig_dir} and {fig_roll_dir} "
        f"(rolling_window={rolling_days}d)",
        flush=True,
    )


if __name__ == "__main__":
    main()
