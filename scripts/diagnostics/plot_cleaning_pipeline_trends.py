"""
Script summary:
Build family- and topic-focused cleaning pipeline diagnostics from Stage-1 audits,
screening tables, enriched profiles, and political audit outputs.

Functionality:
- Aggregates drop rules, kept volumes, Italian langid shares, and political metrics.
- Political audit: topic boxplot, per-subreddit bars, name-axis scatter, and thread-share bubble.
- Writes CSV tables under paths.tables_dir/cleaning_pipeline/ and PNGs under
  figures_dir/cleaning_pipeline/{volume,stage1_drop_rates,political_qa}/.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_cleaning_pipeline_trends.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


import importlib.util
import sys
from pathlib import Path


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

POLITICAL_RATE_COL = "word_weighted_political_rate_100w"
THREAD_SHARE_COL = "thread_political_share"
THRESHOLD_SENSITIVITY_CANDIDATES = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    1.00,
    1.20,
)
THRESHOLD_HIGHLIGHT_SUBREDDITS = frozenset(
    {
        "Italia",
        "ItaliaCareerAdvice",
        "politicaITA",
        "ItaliaMeme",
        "commercialisti",
        "news_and_talk",
        "BancaDelMeme",
        "europe",
        "ukpolitics",
    }
)
POLITICAL_TOPIC_ORDER = [
    "de",
    "eu",
    "it_others",
    "it_political",
    "it_pure_political",
    "uk",
    "uk_political",
    "us",
]
# Italian arms: it_others (red) → it_political (rose) → it_pure_political (purple).
POLITICAL_TOPIC_COLORS: Dict[str, str] = {
    "de": "#1f77b4",
    "eu": "#ff7f0e",
    "it_others": "#d62728",
    "it_political": "#c44e9a",
    "it_pure_political": "#9467bd",
    "uk": "#8c564b",
    "uk_political": "#2ca02c",
    "us": "#7f7f7f",
}


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
    italian = d[(d["topic_family"].isin(["it_political", "it_others"])) & (d["volume_band"] != "excluded")]
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


def _ordered_political_topics(topics: pd.Series) -> List[str]:
    """Function summary: stable topic order for political audit plots.

    Parameters:
    - topics: topic labels from audit data.

    Returns:
    - Ordered topic names (known topics first, then any extras sorted).
    """
    present = set(topics.dropna().astype(str))
    ordered = [t for t in POLITICAL_TOPIC_ORDER if t in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _political_topic_colors(topics: List[str]) -> Dict[str, Any]:
    """Function summary: map each topic to a stable political-QA color.

    Parameters:
    - topics: ordered topic list.

    Returns:
    - Dict topic -> color.
    """
    cmap = plt.get_cmap("tab10")
    out: Dict[str, Any] = {}
    for i, topic in enumerate(topics):
        out[topic] = POLITICAL_TOPIC_COLORS.get(topic, cmap(i % 10))
    return out


def _sort_df_by_political_topic_order(df: pd.DataFrame, topic_col: str = "topic") -> pd.DataFrame:
    """Function summary: sort rows by POLITICAL_TOPIC_ORDER for consistent bar/group ordering.

    Parameters:
    - df: dataframe with a topic column.
    - topic_col: name of topic column.

    Returns:
    - Sorted copy of df.
    """
    order = _ordered_political_topics(df[topic_col])
    cat = pd.Categorical(df[topic_col].astype(str), categories=order, ordered=True)
    return df.assign(_topic_ord=cat).sort_values("_topic_ord").drop(columns=["_topic_ord"])


def plot_political_topic_bar(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    out_path: Path,
    ylabel: Optional[str] = None,
) -> None:
    """Function summary: vertical bar chart by topic using political topic colors and order.

    Parameters:
    - df: one row per topic with topic and y_col.
    - y_col: value column.
    - title: figure title.
    - out_path: PNG path.
    - ylabel: optional y-axis label.
    """
    if df.empty:
        return
    d = _sort_df_by_political_topic_order(df, "topic")
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        d["topic"].astype(str),
        d[y_col],
        color=[colors[str(t)] for t in d["topic"]],
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel or y_col.replace("_", " "))
    ax.set_xlabel("Topic")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _political_assignment_thresholds(audit_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Function summary: read soft and pure assignment thresholds from audit table if present.

    Parameters:
    - audit_df: subreddit_topic_political_audit table.

    Returns:
    - Tuple (soft_threshold for it_political, pure_threshold for it_pure_political).
    """
    if audit_df.empty:
        return None, None
    soft = None
    pure = None
    if "political_soft_threshold" in audit_df.columns:
        soft = float(audit_df["political_soft_threshold"].iloc[0])
    if "political_pure_threshold" in audit_df.columns:
        pure = float(audit_df["political_pure_threshold"].iloc[0])
    elif "political_threshold" in audit_df.columns:
        pure = float(audit_df["political_threshold"].iloc[0])
    if soft is None and pure is not None:
        soft = 0.6
    return soft, pure


def _political_assignment_threshold(audit_df: pd.DataFrame) -> Optional[float]:
    """Function summary: read pure assignment threshold (backward-compatible helper).

    Parameters:
    - audit_df: subreddit_topic_political_audit table.

    Returns:
    - Pure threshold float or None.
    """
    _, pure = _political_assignment_thresholds(audit_df)
    return pure


def _draw_political_threshold_lines(
    ax: plt.Axes,
    audit_df: pd.DataFrame,
    *,
    vertical: bool,
) -> None:
    """Function summary: draw soft (rose) and pure (red dashed) assignment cutoffs on a plot axis.

    Parameters:
    - ax: matplotlib axes.
    - audit_df: audit table with threshold columns.
    - vertical: if True use axvline (horizontal bar charts); else axhline.

    Returns:
    - None.
    """
    soft, pure = _political_assignment_thresholds(audit_df)
    line_fn = ax.axvline if vertical else ax.axhline
    if soft is not None:
        line_fn(soft, color="#c44e9a", linestyle="--", linewidth=1.1, label=f"soft τ={soft:.2f}")
    if pure is not None:
        line_fn(pure, color="red", linestyle="--", linewidth=1.2, label=f"pure τ={pure:.2f}")


def _merge_political_screening(
    audit_df: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    """Function summary: attach screening/profile fields needed for political audit plots.

    Parameters:
    - audit_df: subreddit_topic_political_audit table.
    - tables_dir: results tables root.

    Returns:
    - audit_df with optional action, volume_band, thread_political_share, n_kept_window.
    """
    return _merge_political_plot_data(audit_df, tables_dir)


def _merge_political_plot_data(audit_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Function summary: merge audit with political profile and pooled screening volume.

    Parameters:
    - audit_df: subreddit_topic_political_audit table.
    - tables_dir: results tables root.

    Returns:
    - Enriched audit dataframe for plotting.
    """
    d = audit_df.copy()
    profile_path = tables_dir / "screening" / "subreddit_forum_political_profile.csv"
    if profile_path.is_file():
        profile = pd.read_csv(profile_path)
        want = [
            "subreddit",
            "action",
            "volume_band",
            THREAD_SHARE_COL,
            "comment_hit_share",
            POLITICAL_RATE_COL,
        ]
        merge_cols = [c for c in want if c in profile.columns]
        if "subreddit" in merge_cols:
            extra = [c for c in merge_cols if c != "subreddit" and c not in d.columns]
            if extra:
                d = d.merge(profile[["subreddit"] + extra], on="subreddit", how="left")
    pooled_path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if pooled_path.is_file():
        pooled = pd.read_csv(pooled_path)
        if "subreddit" in pooled.columns and "n_kept_window" in pooled.columns:
            if "n_kept_window" not in d.columns:
                d = d.merge(pooled[["subreddit", "n_kept_window"]], on="subreddit", how="left")
    return d


def _political_bubble_sizes(n_kept: pd.Series) -> np.ndarray:
    """Function summary: map kept-comment window counts to matplotlib marker areas.

    Parameters:
    - n_kept: per-forum n_kept_window series.

    Returns:
    - 1d array of marker sizes (points^2 scale).
    """
    n = n_kept.fillna(1).astype(float).clip(lower=1.0)
    return 30.0 + 280.0 * (np.sqrt(n) / np.sqrt(n.max()))


def _interesting_political_label_mask(
    d: pd.DataFrame, threshold: Optional[float]
) -> pd.Series:
    """Function summary: select subreddits to annotate on the political bubble plot.

    Labels all forums at or above the assignment threshold, audit mismatches, and the
    largest forums per topic (even if below threshold). Does not label small low-rate forums.

    Parameters:
    - d: screened-in audit rows with topic and rate columns.
    - threshold: assignment political-rate threshold if available.

    Returns:
    - Boolean series aligned to d.index.
    """
    label = pd.Series(False, index=d.index)
    if "high_score_non_political" in d.columns:
        label |= d["high_score_non_political"].fillna(False).astype(bool)
    if "low_score_political" in d.columns:
        label |= d["low_score_political"].fillna(False).astype(bool)
    if threshold is not None and POLITICAL_RATE_COL in d.columns:
        label |= d[POLITICAL_RATE_COL].astype(float) >= threshold
    if "n_kept_window" in d.columns and "topic" in d.columns:
        for _, chunk in d.groupby(d["topic"].astype(str)):
            if chunk.empty:
                continue
            n_top = min(3, len(chunk))
            label.loc[chunk.nlargest(n_top, "n_kept_window").index] = True
    return label


def _analysis_sample_political_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep forums that pass screening (exclude action=excluded).

    Parameters:
    - audit_df: audit table with optional action column.

    Returns:
    - Filtered dataframe (unchanged if action column missing).
    """
    if "action" not in audit_df.columns:
        return audit_df
    return audit_df[audit_df["action"].astype(str) != "excluded"].copy()


def plot_political_by_topic_boxplot(
    audit_df: pd.DataFrame,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: boxplot of word-weighted political rate by assigned topic (screened-in forums).

    Parameters:
    - audit_df: subreddit_topic_political_audit table (may include action after merge).
    - out_path: PNG path.
    - config: study config (unused; kept for API consistency).
    """
    del config
    if audit_df.empty or POLITICAL_RATE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df)
    if d.empty:
        return
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    series = [d.loc[d["topic"].astype(str) == topic, POLITICAL_RATE_COL].astype(float).values for topic in topics]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(
        series,
        tick_labels=topics,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, topic in zip(bp["boxes"], topics):
        patch.set_facecolor(colors[topic])
        patch.set_alpha(0.75)
    soft, pure = _political_assignment_thresholds(audit_df)
    _draw_political_threshold_lines(ax, audit_df, vertical=False)
    if soft is not None or pure is not None:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    y_hi = max(
        float(d[POLITICAL_RATE_COL].max()),
        float(pure) if pure is not None else 0.0,
        float(soft) if soft is not None else 0.0,
    ) * 1.08
    ax.set_ylim(0.0, y_hi)
    ax.set_title("Word-weighted political rate by topic (screened-in forums)")
    ax.set_ylabel("Political rate per 100 words")
    ax.set_xlabel("Assigned topic")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_political_by_subreddit_bars(
    audit_df: pd.DataFrame,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: horizontal bar chart of political rate per subreddit, colored by topic.

    Parameters:
    - audit_df: subreddit_topic_political_audit table (screened-in forums only when action present).
    - out_path: PNG path.
    - config: study config (unused; kept for API consistency).
    """
    del config
    if audit_df.empty or POLITICAL_RATE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df)
    if d.empty:
        return
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    topic_ord = {topic: i for i, topic in enumerate(topics)}
    d = d.copy()
    d["topic"] = d["topic"].astype(str)
    d["_topic_ord"] = d["topic"].map(topic_ord)
    d = d.sort_values(["_topic_ord", POLITICAL_RATE_COL]).drop(columns=["_topic_ord"])
    bar_colors = [colors[t] for t in d["topic"]]
    n = len(d)
    row_h = 0.14
    fig_h = max(6.0, row_h * n + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = list(range(n))
    ax.barh(y_pos, d[POLITICAL_RATE_COL], color=bar_colors, height=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["subreddit"].astype(str), fontsize=7)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.margins(x=0.02, y=0)
    rate_max = float(d[POLITICAL_RATE_COL].max())
    soft, pure = _political_assignment_thresholds(audit_df)
    x_hi = max(
        rate_max,
        float(pure) if pure is not None else rate_max,
        float(soft) if soft is not None else rate_max,
    ) * 1.06
    ax.set_xlim(0.0, x_hi)
    _draw_political_threshold_lines(ax, audit_df, vertical=True)
    handles = [Patch(facecolor=colors[t], label=t, edgecolor="none") for t in topics]
    if soft is not None:
        handles.append(Line2D([0], [0], color="#c44e9a", linestyle="--", label=f"soft τ={soft:.2f}"))
    if pure is not None:
        handles.append(Line2D([0], [0], color="red", linestyle="--", label=f"pure τ={pure:.2f}"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9)
    ax.set_title("Word-weighted political rate by subreddit (screened-in forums)")
    ax.set_xlabel("Political rate per 100 words")
    ax.set_ylabel("Subreddit")
    fig.subplots_adjust(left=0.22, right=0.72, top=0.98, bottom=0.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_political_threshold_sensitivity_csv(
    audit_df: pd.DataFrame,
    out_path: Path,
    candidates: tuple[float, ...] = THRESHOLD_SENSITIVITY_CANDIDATES,
) -> None:
    """Function summary: write per-subreddit would-be-pure flags across candidate forum thresholds.

    Parameters:
    - audit_df: merged political audit rows with word_weighted_political_rate_100w.
    - out_path: CSV output path.
    - candidates: forum WW rate cutoffs to evaluate.

    Returns:
    - None.
    """
    if audit_df.empty or POLITICAL_RATE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df).copy()
    if d.empty:
        return
    soft_tau, pure_tau = _political_assignment_thresholds(audit_df)
    rows: List[Dict[str, Any]] = []
    for _, row in d.iterrows():
        ww = float(row[POLITICAL_RATE_COL])
        rows.append(
            {
                "subreddit": str(row["subreddit"]),
                "topic_current": str(row.get("topic", "")),
                "word_weighted_political_rate_100w": ww,
                "would_be_political": int(ww >= (soft_tau if soft_tau is not None else 0.6)),
                "would_be_pure": int(ww >= (pure_tau if pure_tau is not None else 1.2)),
            }
        )
    for tau in candidates:
        for _, row in d.iterrows():
            ww = float(row[POLITICAL_RATE_COL])
            rows.append(
                {
                    "threshold": tau,
                    "subreddit": str(row["subreddit"]),
                    "topic_current": str(row.get("topic", "")),
                    "word_weighted_political_rate_100w": ww,
                    "would_be_pure": int(ww >= tau),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_political_threshold_sensitivity_bars(
    audit_df: pd.DataFrame,
    out_path: Path,
    chosen_threshold: Optional[float] = None,
    candidates: tuple[float, ...] = THRESHOLD_SENSITIVITY_CANDIDATES,
) -> None:
    """Function summary: horizontal bar chart with multiple candidate assignment cutoffs.

    Parameters:
    - audit_df: merged political audit table.
    - out_path: PNG path.
    - chosen_threshold: active assignment threshold (red); others gray dashed.
    - candidates: cutoff values to draw.

    Returns:
    - None.
    """
    if audit_df.empty or POLITICAL_RATE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df)
    if d.empty:
        return
    rate = d[POLITICAL_RATE_COL].astype(float)
    keep = rate >= 0.12
    keep |= d["subreddit"].astype(str).isin(THRESHOLD_HIGHLIGHT_SUBREDDITS)
    d = d.loc[keep].copy()
    if d.empty:
        return
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    topic_ord = {topic: i for i, topic in enumerate(topics)}
    d = d.copy()
    d["topic"] = d["topic"].astype(str)
    d["_topic_ord"] = d["topic"].map(topic_ord)
    d = d.sort_values(["_topic_ord", POLITICAL_RATE_COL]).drop(columns=["_topic_ord"])
    bar_colors = [colors[t] for t in d["topic"]]
    n = len(d)
    row_h = 0.14
    fig_h = max(6.0, row_h * n + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = list(range(n))
    ax.barh(y_pos, d[POLITICAL_RATE_COL], color=bar_colors, height=0.82)
    ax.set_yticks(y_pos)
    labels = []
    for sub in d["subreddit"].astype(str):
        lab = sub
        if sub in THRESHOLD_HIGHLIGHT_SUBREDDITS:
            lab = f"> {sub}"
        labels.append(lab)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.margins(x=0.02, y=0)
    rate_max = float(d[POLITICAL_RATE_COL].max())
    soft, pure = _political_assignment_thresholds(audit_df)
    x_hi = max(
        rate_max,
        max(candidates),
        float(pure) if pure is not None else rate_max,
        float(soft) if soft is not None else rate_max,
    ) * 1.08
    ax.set_xlim(0.0, x_hi)
    _draw_political_threshold_lines(ax, audit_df, vertical=True)
    handles = [Patch(facecolor=colors[t], label=t, edgecolor="none") for t in topics]
    if soft is not None:
        handles.append(Line2D([0], [0], color="#c44e9a", linestyle="--", label=f"soft τ={soft:.2f}"))
    if pure is not None:
        handles.append(Line2D([0], [0], color="red", linestyle="--", label=f"pure τ={pure:.2f}"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9)
    ax.set_title("Political rate by subreddit - threshold sensitivity (v5 salience)")
    ax.set_xlabel("Political rate per 100 words")
    ax.set_ylabel("Subreddit")
    fig.subplots_adjust(left=0.24, right=0.68, top=0.98, bottom=0.01)
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    except RuntimeError:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_political_by_subreddit_scatter(
    audit_df: pd.DataFrame,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: scatter political rate by subreddit on the x-axis (one tick per forum).

    Parameters:
    - audit_df: merged audit table (screened-in forums).
    - out_path: PNG path.
    - config: study config (unused; kept for API consistency).
    """
    del config
    if audit_df.empty or POLITICAL_RATE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df)
    if d.empty:
        return
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    d = d.copy()
    d["topic"] = d["topic"].astype(str)
    if "n_kept_window" in d.columns:
        d = d.sort_values("n_kept_window", ascending=False)
    else:
        d = d.sort_values(POLITICAL_RATE_COL, ascending=False)
    n = len(d)
    x_pos = np.arange(n)
    soft, pure = _political_assignment_thresholds(audit_df)

    fig_w = max(16.0, 0.11 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    for topic in topics:
        mask = d["topic"] == topic
        if not mask.any():
            continue
        ax.scatter(
            x_pos[mask.to_numpy()],
            d.loc[mask, POLITICAL_RATE_COL],
            c=[colors[topic]],
            label=topic,
            alpha=0.82,
            s=28,
            edgecolors="white",
            linewidths=0.4,
        )
    _draw_political_threshold_lines(ax, audit_df, vertical=False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(d["subreddit"].astype(str), rotation=30, ha="right", fontsize=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.01, y=0.04)
    soft, pure = _political_assignment_thresholds(audit_df)
    rate_max = float(d[POLITICAL_RATE_COL].max())
    y_hi = max(
        rate_max,
        float(pure) if pure is not None else rate_max,
        float(soft) if soft is not None else rate_max,
    ) * 1.06
    ax.set_ylim(0.0, y_hi)
    ax.set_title(
        "Word-weighted political rate by subreddit (screened-in forums, ordered by size)"
    )
    ax.set_ylabel("Political rate per 100 words")
    ax.set_xlabel("Subreddit (largest Mar–Apr volume on the left)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.subplots_adjust(bottom=0.22, left=0.06, right=0.98, top=0.94)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_political_rate_vs_thread_share_bubble(
    audit_df: pd.DataFrame,
    out_path: Path,
    config: Dict[str, Any],
) -> None:
    """Function summary: bubble scatter of political rate vs thread political share.

    Parameters:
    - audit_df: merged audit table (screened-in; includes thread share and n_kept_window).
    - out_path: PNG path.
    - config: study config (unused; kept for API consistency).
    """
    del config
    if audit_df.empty:
        return
    if POLITICAL_RATE_COL not in audit_df.columns or THREAD_SHARE_COL not in audit_df.columns:
        return
    d = _analysis_sample_political_audit(audit_df)
    d = d.dropna(subset=[POLITICAL_RATE_COL, THREAD_SHARE_COL])
    if d.empty:
        return
    topics = _ordered_political_topics(d["topic"])
    colors = _political_topic_colors(topics)
    soft, pure = _political_assignment_thresholds(audit_df)
    threshold = pure
    if "n_kept_window" in d.columns:
        sizes = _political_bubble_sizes(d["n_kept_window"])
    else:
        sizes = np.full(len(d), 80.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for topic in topics:
        mask = d["topic"].astype(str) == topic
        if not mask.any():
            continue
        ax.scatter(
            d.loc[mask, THREAD_SHARE_COL],
            d.loc[mask, POLITICAL_RATE_COL],
            s=sizes[mask.to_numpy()],
            c=[colors[topic]],
            label=topic,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
        )
    _draw_political_threshold_lines(ax, audit_df, vertical=False)
    label_mask = _interesting_political_label_mask(d, threshold)
    for row in d.loc[label_mask].itertuples(index=False):
        ax.annotate(
            str(row.subreddit),
            (float(getattr(row, THREAD_SHARE_COL)), float(getattr(row, POLITICAL_RATE_COL))),
            fontsize=7,
            alpha=0.92,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_title(
        "Word-weighted political rate vs thread political share (screened-in forums)"
    )
    ax.set_xlabel("Thread political share")
    ax.set_ylabel("Political rate per 100 words")
    ax.margins(x=0.04, y=0.04)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_political_audit_figures(
    audit_df: pd.DataFrame,
    out_dir: Path,
    config: Dict[str, Any],
    tables_dir: Path,
) -> None:
    """Function summary: write political audit QA figures from audit CSV.

    Parameters:
    - audit_df: subreddit_topic_political_audit table.
    - out_dir: directory for PNG outputs.
    - config: study config.
    - tables_dir: results tables root (for screening/profile merges).
    """
    merged = _merge_political_plot_data(audit_df, tables_dir)
    for plot_fn, out_name in (
        (plot_political_by_topic_boxplot, "by_topic_political_rate_boxplot.png"),
        (plot_political_by_subreddit_bars, "by_subreddit_political_rate_bars.png"),
        (plot_political_by_subreddit_scatter, "by_subreddit_political_rate_vs_topic.png"),
        (plot_political_rate_vs_thread_share_bubble, "by_subreddit_political_rate_vs_thread_share.png"),
    ):
        try:
            plot_fn(merged, out_dir / out_name, config)
        except RuntimeError as exc:
            print(
                f"[plot_cleaning_pipeline_trends] warn skipped {out_name}: {exc}",
                flush=True,
            )
    chosen = _political_assignment_threshold(audit_df)
    sens_csv = tables_dir / "cleaning_pipeline" / "political_threshold_sensitivity.csv"
    write_political_threshold_sensitivity_csv(merged, sens_csv)
    try:
        plot_political_threshold_sensitivity_bars(
            merged,
            out_dir / "by_subreddit_political_rate_threshold_sensitivity.png",
            chosen_threshold=chosen,
        )
    except RuntimeError as exc:
        print(
            f"[plot_cleaning_pipeline_trends] warn sensitivity PNG skipped ({exc}); "
            f"see {sens_csv} and by_subreddit_political_rate_bars.png",
            flush=True,
        )


def run_cleaning_pipeline_plots(config: Dict[str, Any], project_root: Path | None = None) -> None:
    """Function summary: generate cleaning pipeline tables and figures from existing stage 1–3 CSVs.

    Parameters:
    - config: loaded study YAML dict.
    - project_root: repository root (defaults to PROJECT_ROOT).

    Returns:
    - None. Writes under paths.tables_dir/cleaning_pipeline/ and paths.figures_dir/cleaning_pipeline/.
    """
    root = project_root or PROJECT_ROOT
    tables_dir = root / config["paths"]["tables_dir"] if not Path(config["paths"]["tables_dir"]).is_absolute() else Path(config["paths"]["tables_dir"])
    figures_dir = root / config["paths"]["figures_dir"] if not Path(config["paths"]["figures_dir"]).is_absolute() else Path(config["paths"]["figures_dir"])
    out_tables = tables_dir / "cleaning_pipeline"
    out_figures = figures_dir / "cleaning_pipeline"
    out_tables.mkdir(parents=True, exist_ok=True)
    fig_volume = out_figures / "volume"
    fig_stage1 = out_figures / "stage1_drop_rates"
    fig_political = out_figures / "political_qa"
    for d in (fig_volume, fig_stage1, fig_political):
        d.mkdir(parents=True, exist_ok=True)
    print(
        f"[plot_cleaning_pipeline_trends] start tables={out_tables} figures={out_figures}",
        flush=True,
    )

    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    assignment_topics = load_assignment_topics(tables_dir)
    sub_to_topic = assignment_topics if assignment_topics else subreddit_topic_map(
        config, include_topic_aliases=False
    )

    audit_path = tables_dir / "cleaning" / "clean_daily_chunks_audit_by_day.csv"
    if audit_path.is_file():
        print("[plot_cleaning_pipeline_trends] section=stage1_daily_audit", flush=True)
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
            fig_volume / "by_family_rows_kept.png",
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
                fig_volume / "by_family_total_drops.png",
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
                fig_stage1 / f"by_family_{safe}_rate_pct.png",
                config,
            )
            plot_timeseries_by_group(
                audit_df,
                "topic",
                sub_to_topic,
                rate_col,
                ylabel,
                f"Stage-1 {ylabel} by topic",
                fig_stage1 / f"by_topic_{safe}_rate_pct.png",
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
            fig.savefig(fig_stage1 / f"overall_{safe}_rate_pct.png", dpi=150)
            plt.close(fig)

    screening_pooled_path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if screening_pooled_path.is_file():
        print("[plot_cleaning_pipeline_trends] section=screening_pooled", flush=True)
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
            fig_volume / "by_family_n_kept_window.png",
            ylabel="Kept comments",
        )
        plot_langid_by_topic(pooled, sub_to_topic, sub_to_family, fig_political / "italian_langid_share_by_topic.png")

    exclusions_path = tables_dir / "screening" / "subreddit_exclusions.csv"
    if exclusions_path.is_file():
        print("[plot_cleaning_pipeline_trends] section=exclusions", flush=True)
        excl = pd.read_csv(exclusions_path)
        summary = excl.groupby("code", as_index=False).size().rename(columns={"size": "n_subreddits"})
        summary.to_csv(out_tables / "exclusion_summary.csv", index=False)
        plot_bar(
            summary,
            "code",
            "n_subreddits",
            "Excluded subreddits by code",
            fig_volume / "exclusion_summary.png",
            ylabel="Number of subreddits",
        )

    profile_path = tables_dir / "screening" / "subreddit_forum_political_profile.csv"
    if profile_path.is_file():
        print("[plot_cleaning_pipeline_trends] section=political_profile", flush=True)
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
            plot_political_topic_bar(
                topic_agg,
                "political_rate_100w_mean",
                "Word-weighted political rate by topic",
                fig_political / "by_topic_word_weighted_political_rate.png",
                ylabel="Rate per 100 words (word-weighted mean)",
            )
            plot_political_topic_bar(
                topic_agg,
                "political_comment_share",
                "Political comment hit share by topic",
                fig_political / "by_topic_political_comment_hit_share.png",
                ylabel="Share of comments with ≥1 lexicon hit",
            )

    audit_path = tables_dir / "screening" / "subreddit_topic_political_audit.csv"
    if audit_path.is_file():
        print("[plot_cleaning_pipeline_trends] section=political_audit_figures", flush=True)
        plot_political_audit_figures(pd.read_csv(audit_path), fig_political, config, tables_dir)

    print(f"[plot_cleaning_pipeline_trends] wrote tables={out_tables} figures={out_figures}", flush=True)


def main() -> None:
    """Function summary: CLI entrypoint for cleaning pipeline diagnostic plots."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    run_cleaning_pipeline_plots(config, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
