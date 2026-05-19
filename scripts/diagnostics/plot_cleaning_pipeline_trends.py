"""
Script summary:
Build family- and topic-focused cleaning pipeline diagnostics from Stage-1 audits,
screening tables, enriched profiles, and political audit outputs.

Functionality:
- Aggregates drop rules, kept volumes, Italian langid shares, and political metrics.
- Writes CSV tables and PNG plots under paths.tables_dir/cleaning_pipeline/ and figures_dir.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_cleaning_pipeline_trends.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import (  # noqa: E402
    load_config,
    plot_reference_dates_calendar_utc,
    subreddit_family_map,
    subreddit_topic_map,
)

STAGE1_RATE_METRICS = [
    ("drop_body_removed", "Body [removed] drop rate (%)"),
    ("drop_body_deleted", "Body [deleted] drop rate (%)"),
    ("drop_url_only", "URL-only drop rate (%)"),
    ("drop_distinguished_moderator", "Moderator distinguished drop rate (%)"),
    ("drop_stickied_true", "Stickied drop rate (%)"),
]


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot cleaning pipeline diagnostics.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def add_reference_lines(ax: plt.Axes, config: Dict[str, Any]) -> None:
    """Function summary: draw vertical reference lines from config plot dates.

    Parameters:
    - ax: matplotlib axes.
    - config: study YAML.
    """
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(dt, color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    ylabel: Optional[str] = None,
) -> None:
    """Function summary: save a labeled bar chart.

    Parameters:
    - df: data.
    - x_col: category column.
    - y_col: value column.
    - title: figure title.
    - out_path: PNG path.
    - ylabel: optional y-axis label (defaults to y_col).
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel or y_col.replace("_", " "))
    ax.set_xlabel(x_col.replace("_", " "))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_audit_by_group(
    audit_df: pd.DataFrame, group_col: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    """Function summary: sum Stage-1 audit counters by family or topic.

    Parameters:
    - audit_df: day-level audit data with subreddit column.
    - group_col: name for grouping column in output.
    - mapping: subreddit -> group label.

    Returns:
    - Aggregated dataframe.
    """
    d = audit_df.copy()
    d[group_col] = d["subreddit"].map(mapping)
    d = d.dropna(subset=[group_col])
    sum_cols = [
        c
        for c in d.columns
        if c.startswith("drop_") or c in {"rows_input", "rows_kept", "rows_dropped_any", "invalid_json_rows"}
    ]
    out = d.groupby(group_col, as_index=False)[sum_cols].sum()
    out["kept_rate_pct"] = 100.0 * out["rows_kept"] / out["rows_input"].replace(0, pd.NA)
    return out


def load_assignment_topics(tables_dir: Path) -> Dict[str, str]:
    """Function summary: load final topic assignments if enrichment has run.

    Parameters:
    - tables_dir: study tables directory.

    Returns:
    - Subreddit -> topic mapping.
    """
    path = tables_dir / "screening" / "subreddit_topic_assignment.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {str(r["subreddit"]): str(r["topic"]) for r in df.to_dict(orient="records")}


def _volume_band_column(pooled: pd.DataFrame) -> pd.Series:
    """Function summary: resolve volume_band from pooled screening (new or legacy columns).

    Parameters:
    - pooled: screening pooled dataframe.

    Returns:
    - Series of volume band labels.
    """
    if "volume_band" in pooled.columns:
        return pooled["volume_band"].astype(str)
    tier = pooled.get("analysis_tier", pd.Series(dtype=str))
    action = pooled.get("action", pd.Series(dtype=str))
    if tier is not None and len(tier):
        return tier.astype(str)
    mapped = action.astype(str).replace(
        {"include": "large_volume", "tier_b_only": "low_volume", "exclude_analysis": "excluded"}
    )
    return mapped


def build_window_summaries(
    screening_pooled: pd.DataFrame,
    sub_to_family: Dict[str, str],
    sub_to_topic: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: build window summaries by subreddit, family, and topic with volume bands.

    Parameters:
    - screening_pooled: pooled screening table.
    - sub_to_family: subreddit -> family.
    - sub_to_topic: subreddit -> topic.

    Returns:
    - Tuple (by_subreddit, by_family, by_topic, by_family_with_bands).
    """
    by_sub = screening_pooled.copy()
    by_sub["volume_band"] = _volume_band_column(by_sub)

    def _agg_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        g = df.dropna(subset=[group_col])
        rows = []
        for label, chunk in g.groupby(group_col, sort=False):
            large = chunk[chunk["volume_band"] == "large_volume"]
            low = chunk[chunk["volume_band"] == "low_volume"]
            excl = chunk[chunk["volume_band"] == "excluded"]
            rows.append(
                {
                    group_col: label,
                    "n_subreddits": int(chunk["subreddit"].nunique()),
                    "n_kept_window": int(chunk["n_kept_window"].sum()),
                    "n_kept_large_volume": int(large["n_kept_window"].sum()),
                    "n_kept_low_volume": int(low["n_kept_window"].sum()),
                    "n_subreddits_large_volume": int(large["subreddit"].nunique()),
                    "n_subreddits_low_volume": int(low["subreddit"].nunique()),
                    "n_excluded": int(excl["subreddit"].nunique()),
                    "mean_italian_share_pooled": float(chunk["italian_share_pooled"].mean()),
                }
            )
        return pd.DataFrame(rows)

    by_family = by_sub.copy()
    by_family["topic_family"] = by_family["subreddit"].map(sub_to_family)
    by_family_df = _agg_group(by_family, "topic_family")

    by_topic = by_sub.copy()
    by_topic["topic"] = by_topic["subreddit"].map(sub_to_topic)
    by_topic_df = _agg_group(by_topic, "topic")

    return by_sub, by_family_df, by_topic_df, by_family_df


def prepare_daily_rates(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add date column and drop-rate percentages to daily audit.

    Parameters:
    - audit_df: clean_daily_chunks_audit_by_day.

    Returns:
    - Dataframe with date_utc and rate columns.
    """
    d = audit_df.copy()
    d["date_utc"] = pd.to_datetime(d["date_utc"], utc=True)
    for col, _ in STAGE1_RATE_METRICS:
        if col in d.columns:
            d[f"{col}_rate_pct"] = 100.0 * d[col] / d["rows_input"].replace(0, pd.NA)
    return d


def plot_timeseries_by_group(
    audit_df: pd.DataFrame,
    group_col: str,
    mapping: Dict[str, str],
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: plot daily Stage-1 metric by family or topic.

    Parameters:
    - audit_df: daily audit with rate columns.
    - group_col: grouping column name.
    - mapping: subreddit -> group.
    - metric_col: rate column to plot.
    - ylabel: y-axis label.
    - title: plot title.
    - out_path: output PNG path.
    - config: study config for reference lines.
    """
    d = audit_df.copy()
    d[group_col] = d["subreddit"].map(mapping)
    d = d.dropna(subset=[group_col, metric_col])
    if d.empty or metric_col not in d.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, chunk in d.groupby(group_col):
        daily = chunk.groupby("date_utc", as_index=False)[metric_col].mean()
        daily["date_utc"] = pd.to_datetime(daily["date_utc"], utc=True)
        ax.plot(daily["date_utc"], daily[metric_col], label=str(label), linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    add_reference_lines(ax, config)
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_langid_by_topic(
    pooled: pd.DataFrame,
    sub_to_topic: Dict[str, str],
    sub_to_family: Dict[str, str],
    out_path: Path,
) -> None:
    """Function summary: bar chart of mean Italian langid share by topic (Italian family only).

    Parameters:
    - pooled: screening pooled table.
    - sub_to_topic: subreddit -> topic.
    - sub_to_family: subreddit -> family.
    - out_path: PNG output path.
    """
    d = pooled.copy()
    d["topic"] = d["subreddit"].map(sub_to_topic)
    d["topic_family"] = d["subreddit"].map(sub_to_family)
    d["volume_band"] = _volume_band_column(d)
    italian = d[(d["topic_family"] == "italian") & (d["volume_band"] != "excluded")]
    if italian.empty:
        return
    by_topic = italian.groupby("topic", as_index=False)["italian_share_pooled"].mean()
    plot_bar(
        by_topic.sort_values("italian_share_pooled"),
        "topic",
        "italian_share_pooled",
        "Mean pooled Italian share by topic",
        out_path,
        ylabel="Italian share (langid)",
    )


def plot_political_scatter(
    audit_df: pd.DataFrame,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: scatter word-weighted political rate vs subreddit from audit CSV.

    Parameters:
    - audit_df: subreddit_topic_political_audit table.
    - out_path: PNG path.
    - config: study config.
    """
    if audit_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for topic, chunk in audit_df.groupby("topic"):
        ax.scatter(
            chunk["subreddit"],
            chunk["word_weighted_political_rate_100w"],
            label=topic,
            alpha=0.8,
        )
    threshold = audit_df["political_threshold"].iloc[0] if "political_threshold" in audit_df.columns else None
    if threshold is not None:
        ax.axhline(float(threshold), color="red", linestyle="--", label="assignment threshold")
    ax.set_title("Word-weighted political rate by subreddit")
    ax.set_ylabel("Political rate per 100 words")
    ax.set_xlabel("Subreddit")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: generate cleaning pipeline tables and figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    out_tables = tables_dir / "cleaning_pipeline"
    out_figures = figures_dir / "cleaning_pipeline"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    assignment_topics = load_assignment_topics(tables_dir)
    sub_to_topic = assignment_topics if assignment_topics else subreddit_topic_map(
        config, include_topic_aliases=False
    )

    audit_path = tables_dir / "cleaning" / "clean_daily_chunks_audit_by_day.csv"
    if audit_path.is_file():
        audit_df = pd.read_csv(audit_path)
        audit_df = prepare_daily_rates(audit_df)
        audit_df.to_csv(out_tables / "daily_metrics_by_subreddit.csv", index=False)
        by_family = aggregate_audit_by_group(audit_df, "topic_family", sub_to_family)
        by_topic = aggregate_audit_by_group(audit_df, "topic", sub_to_topic)
        by_family.to_csv(out_tables / "daily_metrics_by_family.csv", index=False)
        by_topic.to_csv(out_tables / "daily_metrics_by_topic.csv", index=False)
        plot_bar(
            by_family,
            "topic_family",
            "rows_kept",
            "Stage-1 rows kept by family",
            out_figures / "by_family_rows_kept.png",
            ylabel="Rows kept",
        )
        drop_cols = [c for c in by_family.columns if c.startswith("drop_")]
        if drop_cols:
            melt = by_family.melt(id_vars=["topic_family"], value_vars=drop_cols, var_name="rule", value_name="count")
            totals = melt.groupby("topic_family", as_index=False)["count"].sum().rename(columns={"count": "total_drops"})
            plot_bar(
                totals,
                "topic_family",
                "total_drops",
                "Stage-1 total drops by family",
                out_figures / "by_family_total_drops.png",
                ylabel="Dropped rows",
            )

        for drop_col, ylabel in STAGE1_RATE_METRICS:
            rate_col = f"{drop_col}_rate_pct"
            if rate_col not in audit_df.columns:
                continue
            safe = drop_col.replace("drop_", "")
            plot_timeseries_by_group(
                audit_df,
                "topic_family",
                sub_to_family,
                rate_col,
                ylabel,
                f"Stage-1 {ylabel} by family",
                out_figures / f"by_family_{safe}_rate_pct.png",
                config,
            )
            plot_timeseries_by_group(
                audit_df,
                "topic",
                sub_to_topic,
                rate_col,
                ylabel,
                f"Stage-1 {ylabel} by topic",
                out_figures / f"by_topic_{safe}_rate_pct.png",
                config,
            )
            overall = audit_df.groupby("date_utc", as_index=False)[rate_col].mean()
            overall["date_utc"] = pd.to_datetime(overall["date_utc"], utc=True)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(overall["date_utc"], overall[rate_col], linewidth=1.5)
            ax.set_title(f"Stage-1 {ylabel} (overall)")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Date (UTC)")
            add_reference_lines(ax, config)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out_figures / f"overall_{safe}_rate_pct.png", dpi=150)
            plt.close(fig)

    screening_pooled_path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if screening_pooled_path.is_file():
        pooled = pd.read_csv(screening_pooled_path)
        by_sub, by_family, by_topic, by_family_bands = build_window_summaries(
            pooled, sub_to_family, sub_to_topic
        )
        by_sub.to_csv(out_tables / "window_summary_by_subreddit.csv", index=False)
        by_family.to_csv(out_tables / "window_summary_by_family.csv", index=False)
        by_topic.to_csv(out_tables / "window_summary_by_topic.csv", index=False)
        by_family_bands.to_csv(out_tables / "window_summary_by_family_with_volume_bands.csv", index=False)
        plot_bar(
            by_family,
            "topic_family",
            "n_kept_window",
            "Kept comments Mar-Apr by family",
            out_figures / "by_family_n_kept_window.png",
            ylabel="Kept comments",
        )
        plot_langid_by_topic(pooled, sub_to_topic, sub_to_family, out_figures / "italian_langid_share_by_topic.png")

    exclusions_path = tables_dir / "screening" / "subreddit_exclusions.csv"
    if exclusions_path.is_file():
        excl = pd.read_csv(exclusions_path)
        summary = excl.groupby("code", as_index=False).size().rename(columns={"size": "n_subreddits"})
        summary.to_csv(out_tables / "exclusion_summary.csv", index=False)
        plot_bar(
            summary,
            "code",
            "n_subreddits",
            "Excluded subreddits by code",
            out_figures / "exclusion_summary.png",
            ylabel="Number of subreddits",
        )

    profile_path = tables_dir / "screening" / "subreddit_forum_political_profile.csv"
    if profile_path.is_file():
        prof = pd.read_csv(profile_path)
        prof.to_csv(out_tables / "political_metrics_by_subreddit.csv", index=False)
        ww_col = "word_weighted_political_rate_100w"
        if ww_col in prof.columns:
            topic_agg = prof.groupby("topic", as_index=False).agg(
                political_rate_100w_mean=(ww_col, "mean"),
                political_comment_share=("comment_hit_share", "mean"),
                thread_political_share=("thread_political_share", "mean"),
                n_subreddits=("subreddit", "nunique"),
            )
            topic_agg.to_csv(out_tables / "political_metrics_by_topic.csv", index=False)
            prof["topic_family"] = prof["subreddit"].map(sub_to_family)
            family_agg = prof.groupby("topic_family", as_index=False).agg(
                political_rate_100w_mean=(ww_col, "mean"),
                political_comment_share=("comment_hit_share", "mean"),
                thread_political_share=("thread_political_share", "mean"),
            )
            family_agg.to_csv(out_tables / "political_metrics_by_family.csv", index=False)
            plot_bar(
                topic_agg.sort_values("political_rate_100w_mean"),
                "topic",
                "political_rate_100w_mean",
                "Word-weighted political rate by topic",
                out_figures / "by_topic_word_weighted_political_rate.png",
                ylabel="Rate per 100 words (word-weighted mean)",
            )
            plot_bar(
                topic_agg.sort_values("political_comment_share"),
                "topic",
                "political_comment_share",
                "Political comment hit share by topic",
                out_figures / "by_topic_political_comment_hit_share.png",
                ylabel="Share of comments with ≥1 lexicon hit",
            )

    audit_path = tables_dir / "screening" / "subreddit_topic_political_audit.csv"
    if audit_path.is_file():
        plot_political_scatter(pd.read_csv(audit_path), out_figures / "by_subreddit_political_rate_vs_topic.png", config)

    print(f"[plot_cleaning_pipeline_trends] wrote tables={out_tables} figures={out_figures}", flush=True)


if __name__ == "__main__":
    main()
