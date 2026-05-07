"""
Script summary:
This script reads daily event-time metric tables and creates event-time line
plots for semicolon rate, comment length, complexity index, AI-likeness,
AI-typical word intensity, style proxies (assistant-tone, list structure,
repetition similarity, formality), extended lexicon rates, typography and
Markdown proxy rates (em/en dash, ASCII double-hyphen, colon, parens, curly
quotes, bold/heading), hedging/polite/signposting phrase rates,
avg words per sentence, and toxicity-related proxies. It writes pooled figures,
optional per-subreddit-by-family topic-panel figures, by-default per-family
(daily/rolling plus optional weekly) multi-line figures, one combined strict-10
word graph
(pooled), and a pooled multi-panel style overview.

How to apply/run:
- Default run (pooled + by-family with 7-day trailing rolling window):
  `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
- Disable per-family outputs:
  `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml --no_topic_views`
- Disable per-subreddit-by-family outputs:
  `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml --no_by_subreddit`
- Figures are saved in view-specific folders:
  - pooled: `results/figures/event_time/pooled/{daily,rolling_daily}/` by default (`weekly/` with `--include_weekly`)
  - by family (default): `results/figures/event_time/by_family/{daily,rolling_daily}/` by default (`weekly/` with `--include_weekly`)
  - by subreddit by family (default): `results/figures/event_time/by_subreddit_by_family/{daily,rolling_daily}/<family>/` by default (`weekly/` with `--include_weekly`)
    with one page per metric and one subplot per topic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys

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

from src.config_utils import load_config, subreddit_family_map, subreddit_topic_map, topic_families, utc_ts

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
    "em_dash_rate_100w",
    "em_dash_extended_rate_100w",
    "en_dash_rate_100w",
    "ascii_double_hyphen_rate_100w",
    "colon_rate_100w",
    "colon_extended_rate_100w",
    "open_paren_rate_100w",
    "curly_quote_rate_100w",
    "quote_all_rate_100w",
    "url_rate_100w",
    "time_expression_rate_100w",
    "markdown_bold_pair_rate_100w",
    "markdown_heading_line_rate_100w",
    "hedging_phrase_rate_100w",
    "polite_closer_rate_100w",
    "signposting_phrase_rate_100w",
]
COMMENT_WEIGHT_COLS = [
    "comment_length_words",
    "avg_words_per_sentence_mean",
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
        dest="topic_views",
        action="store_true",
        help="Generate family-level figures (daily/rolling_daily plus optional weekly) from by-subreddit table (default).",
    )
    parser.add_argument(
        "--no_topic_views",
        dest="topic_views",
        action="store_false",
        help="Disable family-level figures (turned on by default).",
    )
    parser.set_defaults(topic_views=True)
    parser.add_argument(
        "--by_subreddit",
        dest="by_subreddit",
        action="store_true",
        help="Generate per-subreddit-by-family multi-line figures (on by default).",
    )
    parser.add_argument(
        "--no_by_subreddit",
        dest="by_subreddit",
        action="store_false",
        help="Disable per-subreddit-by-family multi-line figures.",
    )
    parser.set_defaults(by_subreddit=True)
    parser.add_argument(
        "--topic_rolling_window",
        type=int,
        default=7,
        help="Trailing rolling window size (in days) for rolling views.",
    )
    parser.add_argument(
        "--include_weekly",
        action="store_true",
        help="Include weekly view plots in addition to default daily and rolling_daily views.",
    )
    parser.add_argument(
        "--min_metric_coverage",
        type=float,
        default=0.8,
        help="Minimum weighted coverage required for plotting a metric point (0.0 keeps legacy behavior).",
    )
    return parser.parse_args()


def active_view_names(include_weekly: bool) -> list[str]:
    """Function summary: return enabled event-time view names with weekly as optional extra."""
    views = ["daily", "rolling_daily"]
    if include_weekly:
        views.insert(1, "weekly")
    return views


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


def weighted_mean_with_coverage(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    """Function summary: compute weighted mean over valid values and return (mean, coverage_share)."""
    value_num = pd.to_numeric(values, errors="coerce")
    weight_num = pd.to_numeric(weights, errors="coerce")
    valid_weights = weight_num.notna() & (weight_num > 0)
    total_weight = float(weight_num.loc[valid_weights].sum()) if bool(valid_weights.any()) else 0.0
    if total_weight <= 0:
        return float("nan"), 0.0
    valid = value_num.notna() & valid_weights
    valid_weight = float(weight_num.loc[valid].sum()) if bool(valid.any()) else 0.0
    coverage = valid_weight / total_weight if total_weight > 0 else 0.0
    if valid_weight <= 0:
        return float("nan"), coverage
    numer = float((value_num.loc[valid] * weight_num.loc[valid]).sum())
    return numer / valid_weight, coverage


def coverage_col_name(metric_col: str) -> str:
    """Function summary: build stable internal column name for metric-level weighted coverage shares."""
    return f"__coverage__{metric_col}"


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
    for col in [
        "n_comments",
        "n_words",
        "strict_ai_word_hits_total",
        "extended_ai_word_hits_total",
        "quote_curly_share_num",
        "quote_curly_share_den",
    ]:
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
        if {"quote_curly_share_num", "quote_curly_share_den"}.issubset(grp.columns):
            quote_num = float(grp["quote_curly_share_num"].sum())
            quote_den = float(grp["quote_curly_share_den"].sum())
            row["quote_curly_share_num"] = quote_num
            row["quote_curly_share_den"] = quote_den
            row["quote_curly_share"] = float(quote_num / quote_den) if quote_den > 0 else float("nan")
            row[coverage_col_name("quote_curly_share")] = 1.0 if quote_den > 0 else 0.0
        for col in WORD_WEIGHT_COLS:
            if col in grp.columns:
                mean_val, coverage = weighted_mean_with_coverage(grp[col], grp["n_words"])
                row[col] = float(mean_val)
                row[coverage_col_name(col)] = float(coverage)
        for col in COMMENT_WEIGHT_COLS:
            if col in grp.columns:
                mean_val, coverage = weighted_mean_with_coverage(grp[col], grp["n_comments"])
                row[col] = float(mean_val)
                row[coverage_col_name(col)] = float(coverage)
        grouped_rows.append(row)
    if not grouped_rows:
        return pd.DataFrame()
    out = pd.DataFrame(grouped_rows).sort_values([group_col, "date"]).reset_index(drop=True)
    out["event_time_t"] = (out["date"] - LAUNCH_DATE_UTC).dt.days
    return out


def aggregate_family_daily(df_by_sub: pd.DataFrame, subreddit_to_family: dict[str, str]) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit daily rows into per-family daily rows using config-driven mapping."""
    if df_by_sub.empty:
        return pd.DataFrame()
    required = {"subreddit", "date_utc", "n_comments", "n_words"}
    if not required.issubset(df_by_sub.columns):
        return pd.DataFrame()
    d = ensure_date_column(df_by_sub.copy())
    d = d[d["subreddit"] != "ALL"].copy()
    d["topic_family"] = d["subreddit"].map(subreddit_to_family)
    unknown_subs = sorted(d.loc[d["topic_family"].isna(), "subreddit"].dropna().unique())
    if unknown_subs:
        print(f"[plot_event_time_metrics] skipping unmapped subreddits in family view: {', '.join(unknown_subs)}", flush=True)
    d = d.dropna(subset=["topic_family"])
    return aggregate_daily_weighted(d, group_col="topic_family", alias_col="topic_family")


def aggregate_weekly_weighted(daily_df: pd.DataFrame, group_col: str, alias_col: str | None = None) -> pd.DataFrame:
    """Function summary: convert daily grouped rows to weekly grouped rows with weighted recomputation."""
    if daily_df.empty:
        return pd.DataFrame()
    d = ensure_date_column(daily_df.copy())
    d["week_start"] = d["date"].dt.to_period("W-MON").dt.start_time
    d["date_utc"] = d["week_start"].dt.strftime("%Y-%m-%dT00:00:00Z")
    d["date"] = d["week_start"]
    return aggregate_daily_weighted(d, group_col=group_col, alias_col=alias_col)


def aggregate_family_weekly(family_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: aggregate family daily rows into weekly bins using weighted recomputation."""
    return aggregate_weekly_weighted(family_daily_df, group_col="topic_family", alias_col="topic_family")


TOPIC_FAMILY_KEY_SEP = "||"


def topic_family_group_key(topic_family: str, topic_group: str) -> str:
    """Function summary: build a stable composite key for topic-family grouping.

    Parameters:
    - topic_family: Family label.
    - topic_group: Topic label within the family.

    Returns:
    - Composite key string joining family and topic.
    """
    return f"{topic_family}{TOPIC_FAMILY_KEY_SEP}{topic_group}"


def split_topic_family_group_key(key: str) -> tuple[str, str]:
    """Function summary: split a composite topic-family key back into family and topic.

    Parameters:
    - key: Composite key built by topic_family_group_key.

    Returns:
    - Tuple of (topic_family, topic_group). Falls back to empty topic if separator missing.
    """
    if TOPIC_FAMILY_KEY_SEP not in key:
        return key, ""
    family_name, topic_name = key.split(TOPIC_FAMILY_KEY_SEP, 1)
    return family_name, topic_name


def aggregate_topic_family_daily(
    df_by_sub: pd.DataFrame,
    subreddit_to_topic: dict[str, str],
    topic_to_family: dict[str, str],
) -> pd.DataFrame:
    """Function summary: aggregate per-subreddit rows into daily topic-within-family trajectories.

    Parameters:
    - df_by_sub: Daily per-subreddit event-time rows.
    - subreddit_to_topic: Mapping from subreddit name to topic label.
    - topic_to_family: Mapping from topic label to family label.

    Returns:
    - Daily weighted table keyed by topic family and topic group.
    """
    if df_by_sub.empty:
        return pd.DataFrame()
    required = {"subreddit", "date_utc", "n_comments", "n_words"}
    if not required.issubset(df_by_sub.columns):
        return pd.DataFrame()
    d = ensure_date_column(df_by_sub.copy())
    d = d[d["subreddit"] != "ALL"].copy()
    d["topic_group"] = d["subreddit"].map(subreddit_to_topic)
    d["topic_family"] = d["topic_group"].map(topic_to_family)
    unknown_topics = sorted(d.loc[d["topic_group"].isna(), "subreddit"].dropna().unique())
    if unknown_topics:
        print(
            f"[plot_event_time_metrics] skipping subreddits with unmapped topics in topic-family view: {', '.join(unknown_topics)}",
            flush=True,
        )
    unknown_families = sorted(d.loc[d["topic_family"].isna(), "topic_group"].dropna().unique())
    if unknown_families:
        print(
            f"[plot_event_time_metrics] skipping topics with unmapped families in topic-family view: {', '.join(unknown_families)}",
            flush=True,
        )
    d = d.dropna(subset=["topic_group", "topic_family"])
    if d.empty:
        return pd.DataFrame()
    d["topic_family_group"] = d.apply(
        lambda row: topic_family_group_key(str(row["topic_family"]), str(row["topic_group"])),
        axis=1,
    )
    out = aggregate_daily_weighted(d, group_col="topic_family_group", alias_col="topic_family_group")
    if out.empty:
        return out
    split_vals = out["topic_family_group"].map(lambda value: split_topic_family_group_key(str(value)))
    out["topic_family"] = split_vals.map(lambda value: value[0])
    out["topic_group"] = split_vals.map(lambda value: value[1])
    out["subreddit"] = out["topic_family_group"]
    return out


def aggregate_topic_family_weekly(topic_family_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: aggregate topic-family daily rows into weekly bins with weighted recomputation.

    Parameters:
    - topic_family_daily_df: Daily topic-family table from aggregate_topic_family_daily.

    Returns:
    - Weekly weighted table preserving topic_family and topic_group labels.
    """
    if topic_family_daily_df.empty:
        return pd.DataFrame()
    weekly = aggregate_weekly_weighted(
        topic_family_daily_df,
        group_col="topic_family_group",
        alias_col="topic_family_group",
    )
    if weekly.empty:
        return weekly
    split_vals = weekly["topic_family_group"].map(lambda value: split_topic_family_group_key(str(value)))
    weekly["topic_family"] = split_vals.map(lambda value: value[0])
    weekly["topic_group"] = split_vals.map(lambda value: value[1])
    weekly["subreddit"] = weekly["topic_family_group"]
    return weekly


def rolling_topic_family_daily(topic_family_daily_df: pd.DataFrame, rolling_window_days: int) -> pd.DataFrame:
    """Function summary: apply trailing-day smoothing for topic-family daily trajectories.

    Parameters:
    - topic_family_daily_df: Daily topic-family table.
    - rolling_window_days: Trailing daily window size.

    Returns:
    - Smoothed daily topic-family table preserving topic_family and topic_group labels.
    """
    if topic_family_daily_df.empty:
        return pd.DataFrame()
    rolled = grouped_trailing_daily_rolling(
        topic_family_daily_df,
        group_col="topic_family_group",
        rolling_window_days=rolling_window_days,
    )
    if rolled.empty:
        return rolled
    split_vals = rolled["topic_family_group"].map(lambda value: split_topic_family_group_key(str(value)))
    rolled["topic_family"] = split_vals.map(lambda value: value[0])
    rolled["topic_group"] = split_vals.map(lambda value: value[1])
    rolled["subreddit"] = rolled["topic_family_group"]
    return rolled


def grouped_weekly_rolling(df_weekly: pd.DataFrame, group_col: str, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply rolling means within each group for weekly rows while preserving date anchors."""
    if df_weekly.empty:
        return pd.DataFrame()
    if rolling_window <= 1:
        return df_weekly.copy()
    d = df_weekly.sort_values([group_col, "date"]).copy()
    exclude_cols = {"subreddit", "topic_family", "date_utc", "date", group_col}
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    out_parts: list[pd.DataFrame] = []
    for group_value, grp in d.groupby(group_col, sort=True):
        g = grp.copy()
        if numeric_cols:
            # Rebuild with concat to avoid fragmented internals from repeated per-column writes.
            rolled_numeric = g.loc[:, numeric_cols].rolling(window=int(rolling_window), min_periods=1).mean()
            non_numeric = g.drop(columns=numeric_cols)
            g = pd.concat([non_numeric, rolled_numeric], axis=1).reindex(columns=grp.columns)
        g["subreddit"] = group_value
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, "date"]).reset_index(drop=True)


def grouped_trailing_daily_rolling(df_daily: pd.DataFrame, group_col: str, rolling_window_days: int) -> pd.DataFrame:
    """Function summary: smooth daily rows by group using trailing day-based rolling windows with edge-aware partial windows.

    Past-only: pandas `.rolling(window="ND", min_periods=1)` on a sorted DatetimeIndex defaults to
    `center=False`, producing a right-aligned/trailing window. Each output point uses only the current
    day plus the prior (N-1) days; no future leakage. Applies identically for `subreddit`, pooled
    `subreddit`, and `topic_family` callers.
    """
    if df_daily.empty:
        return pd.DataFrame()
    if rolling_window_days <= 1:
        return df_daily.copy()
    d = ensure_date_column(df_daily.copy()).sort_values([group_col, "date"])
    exclude_cols = {"subreddit", "topic_family", "date_utc", "date", group_col, "event_time_t"}
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    out_parts: list[pd.DataFrame] = []
    for group_value, grp in d.groupby(group_col, sort=True):
        g = grp.sort_values("date").copy()
        g_indexed = g.set_index("date")
        if numeric_cols:
            original_cols = g_indexed.columns
            # Rebuild with concat so rolling updates do not fragment the underlying DataFrame blocks.
            rolled_numeric = g_indexed.loc[:, numeric_cols].rolling(
                window=f"{int(rolling_window_days)}D", min_periods=1
            ).mean()
            non_numeric = g_indexed.drop(columns=numeric_cols)
            g_indexed = pd.concat([non_numeric, rolled_numeric], axis=1).reindex(columns=original_cols)
        g = g_indexed.reset_index()
        g["subreddit"] = group_value
        g["event_time_t"] = (g["date"] - LAUNCH_DATE_UTC).dt.days
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, "date"]).reset_index(drop=True)


def family_weekly_rolling(family_weekly_df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply per-family rolling mean over weekly metrics while preserving weekly date anchors."""
    return grouped_weekly_rolling(family_weekly_df, group_col="topic_family", rolling_window=rolling_window)


COVERAGE_METRICS_REQUIRE_SIGNAL = {
    "coverage_perplexity",
    "coverage_detector_primary",
    "coverage_detector_secondary",
    "coverage_hostility",
    "coverage_emotion",
}


def metric_has_plotworthy_signal(df: pd.DataFrame, y_col: str) -> bool:
    """Function summary: True only if y_col has at least one finite non-zero value (filters all-NaN and all-zero series)."""
    if df.empty or y_col not in df.columns:
        return False
    series = pd.to_numeric(df[y_col], errors="coerce")
    if not bool(series.notna().any()):
        return False
    nonzero = series.dropna() != 0.0
    return bool(nonzero.any())


def apply_metric_coverage_gate(df: pd.DataFrame, metric_cols: list[str], min_metric_coverage: float) -> pd.DataFrame:
    """Function summary: set metric values to NaN where weighted coverage is below the configured threshold."""
    out = df.copy()
    threshold = float(max(0.0, min(1.0, min_metric_coverage)))
    if threshold <= 0.0:
        return out
    for metric_col in metric_cols:
        cov_col = coverage_col_name(metric_col)
        if metric_col not in out.columns or cov_col not in out.columns:
            continue
        coverage = pd.to_numeric(out[cov_col], errors="coerce")
        low_mask = coverage.notna() & (coverage < threshold)
        out.loc[low_mask, metric_col] = float("nan")
    return out


def collect_metric_coverage_rows(
    df: pd.DataFrame,
    *,
    view_name: str,
    group_kind: str,
    group_col: str,
    metric_cols: list[str],
) -> list[dict[str, float | str]]:
    """Function summary: flatten metric-level coverage columns into long rows for diagnostics CSV output."""
    rows: list[dict[str, float | str]] = []
    if df.empty or group_col not in df.columns:
        return rows
    for metric_col in metric_cols:
        cov_col = coverage_col_name(metric_col)
        if cov_col not in df.columns:
            continue
        subset = df[[group_col, "date_utc", cov_col]].copy()
        subset[cov_col] = pd.to_numeric(subset[cov_col], errors="coerce")
        subset = subset.dropna(subset=[cov_col])
        for item in subset.to_dict(orient="records"):
            rows.append(
                {
                    "view": view_name,
                    "group_kind": group_kind,
                    "group_value": str(item.get(group_col, "")),
                    "date_utc": str(item.get("date_utc", "")),
                    "metric": metric_col,
                    "coverage_share": float(item.get(cov_col, float("nan"))),
                }
            )
    return rows


def write_metric_coverage_table(rows: list[dict[str, float | str]], tables_dir: Path) -> None:
    """Function summary: write long-form metric coverage diagnostics for daily/weekly/rolling views."""
    out_path = tables_dir / "event_time" / "metric_coverage_by_view.csv"
    if not rows:
        pd.DataFrame(columns=["view", "group_kind", "group_value", "date_utc", "metric", "coverage_share"]).to_csv(
            out_path, index=False
        )
        return
    cov_df = pd.DataFrame(rows).sort_values(["view", "group_kind", "group_value", "metric", "date_utc"]).reset_index(drop=True)
    cov_df.to_csv(out_path, index=False)


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
    palette = dict(zip(subreddits, sns.color_palette("husl", n_colors=max(1, len(subreddits)))))
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
    ncol = min(max(1, len(subreddits)), 6)
    plt.legend(
        title="Subreddit",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=ncol,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_metric_by_family(
    family_df: pd.DataFrame, y_col: str, title: str, out_path: Path, *, event_time_xlabel_text: str, show_markers: bool = True
) -> None:
    """Function summary: plot one metric over calendar dates with one line per topic family."""
    _ = event_time_xlabel_text
    if family_df.empty or y_col not in family_df.columns:
        return
    d = ensure_date_column(family_df.copy())
    if d.empty:
        return
    families = sorted(d["topic_family"].dropna().unique())
    palette = dict(zip(families, sns.color_palette("husl", n_colors=max(1, len(families)))))
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=d.sort_values(["topic_family", "date"]),
        x="date",
        y=y_col,
        hue="topic_family",
        palette=palette,
        marker=("o" if show_markers else None),
    )
    add_release_markers(plt.gca())
    format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_col)
    plt.legend(
        title="Topic family",
        loc="best",
        ncol=1 if len(families) <= 6 else 2,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def prepare_top_n_stacked_time_series(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str = "n_comments",
    top_n: int = 10,
    rest_label: str = "Rest",
) -> pd.DataFrame:
    """Function summary: return date-indexed wide table with top-N groups plus Rest based on total volume."""
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    d = ensure_date_column(df.copy())
    d = d.dropna(subset=[group_col, "date"])
    if d.empty:
        return pd.DataFrame()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0)
    totals = (
        d.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    top_groups = totals[group_col].head(max(1, int(top_n))).astype(str).tolist()
    d[group_col] = d[group_col].astype(str)
    d["stack_group"] = d[group_col].where(d[group_col].isin(top_groups), rest_label)
    stacked = (
        d.groupby(["date", "stack_group"], as_index=False)[value_col]
        .sum()
        .pivot(index="date", columns="stack_group", values=value_col)
        .fillna(0.0)
        .sort_index()
    )
    ordered_cols = [g for g in top_groups if g in stacked.columns]
    if rest_label in stacked.columns:
        ordered_cols.append(rest_label)
    if ordered_cols:
        stacked = stacked[ordered_cols]
    return stacked


def plot_stacked_area_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    legend_title: str,
    out_path: Path,
    event_time_xlabel_text: str,
    value_col: str = "n_comments",
    top_n: int = 10,
) -> None:
    """Function summary: plot top-N-plus-Rest stacked area chart for grouped comment volume over time."""
    _ = event_time_xlabel_text
    stacked = prepare_top_n_stacked_time_series(df, group_col=group_col, value_col=value_col, top_n=top_n)
    if stacked.empty:
        return
    series_names = stacked.columns.tolist()
    colors = sns.color_palette("tab20", n_colors=max(1, len(series_names)))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        stacked.index,
        *[stacked[col].to_numpy() for col in series_names],
        labels=series_names,
        colors=colors,
        alpha=0.85,
    )
    add_release_markers(ax)
    format_month_start_axis(ax)
    ax.set_title(title)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Comments")
    ncol = min(max(1, len(series_names)), 6)
    ax.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=ncol,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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


def plot_pooled_quote_dual_axis(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    *,
    event_time_xlabel_text: str,
    show_markers: bool = True,
) -> None:
    """Function summary: pooled dual-axis figure with curly + all-quote rates (left) and curly share (right) on shared date axis."""
    _ = event_time_xlabel_text
    required_cols = {"curly_quote_rate_100w", "quote_all_rate_100w", "quote_curly_share"}
    if df.empty or not required_cols.issubset(df.columns):
        return
    d = ensure_date_column(df).sort_values("date")
    if d.empty:
        return
    rates_have_data = bool(
        pd.to_numeric(d["curly_quote_rate_100w"], errors="coerce").notna().any()
        or pd.to_numeric(d["quote_all_rate_100w"], errors="coerce").notna().any()
    )
    share_have_data = bool(pd.to_numeric(d["quote_curly_share"], errors="coerce").notna().any())
    if not rates_have_data and not share_have_data:
        return
    fig, ax_left = plt.subplots(figsize=(10, 5))
    marker_a = "o" if show_markers else None
    marker_b = "s" if show_markers else None
    marker_share = "^" if show_markers else None
    color_curly = "tab:blue"
    color_all = "tab:orange"
    color_share = "tab:green"
    ax_left.plot(
        d["date"],
        pd.to_numeric(d["curly_quote_rate_100w"], errors="coerce"),
        marker=marker_a,
        color=color_curly,
        label="curly quote rate (per 100w)",
    )
    ax_left.plot(
        d["date"],
        pd.to_numeric(d["quote_all_rate_100w"], errors="coerce"),
        marker=marker_b,
        color=color_all,
        label="all quote characters (per 100w)",
    )
    ax_left.set_xlabel("Date (UTC)")
    ax_left.set_ylabel("Rate per 100 words")
    add_release_markers(ax_left)
    format_month_start_axis(ax_left)
    ax_right = ax_left.twinx()
    ax_right.plot(
        d["date"],
        pd.to_numeric(d["quote_curly_share"], errors="coerce"),
        marker=marker_share,
        color=color_share,
        linestyle="--",
        label="curly share (curly / all quotes)",
    )
    ax_right.set_ylabel("Curly share (0-1)")
    ax_right.set_ylim(0.0, 1.0)
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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
    """Function summary: plot strict individual word rates and strict combined rate on a calendar-date axis; individuals use hue lineplot, combined is a single dashed overlay (avoids duplicate legend entries)."""
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
    individual_df = plot_df[plot_df["word_group"] == "strict_individual"].copy()
    if not individual_df.empty:
        sns.lineplot(
            data=individual_df,
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
    plt.title("Event-time: Strict AI Word Rates (Top-10 Stem-aware + Combined)")
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


def rolling_ai_word_long_trailing_daily(ai_word_daily_df: pd.DataFrame, rolling_window_days: int) -> pd.DataFrame:
    """Function summary: apply trailing day-based rolling smoothing to pooled strict-word daily trajectories."""
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
        # Use single-step assignment to keep internals defragmented before resetting index.
        rolled_rate = g_indexed["rate_100w"].rolling(window=f"{int(rolling_window_days)}D", min_periods=1).mean()
        g_indexed = g_indexed.assign(rate_100w=rolled_rate)
        g = g_indexed.reset_index()
        g["date_utc"] = g["date"].dt.strftime("%Y-%m-%dT00:00:00Z")
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values(["date", "word_group", "word"]).reset_index(drop=True)


def main() -> None:
    """Function summary: load daily data and write pooled and per-subreddit event-time figures."""
    args = parse_args()
    config = load_config(args.config)
    config_family_map = subreddit_family_map(config, include_family_aliases=False)
    config_topic_map = subreddit_topic_map(config, include_topic_aliases=False)
    family_topic_map = topic_families(config)
    topic_to_family = {
        topic_name: family_name
        for family_name, topic_names in family_topic_map.items()
        for topic_name in topic_names
    }
    xt = event_time_xlabel(config)
    figures_dir = Path(config["paths"]["figures_dir"]) / "event_time"
    pooled_figures_dir = figures_dir / "pooled"
    by_subreddit_figures_dir = figures_dir / "by_subreddit_by_family"
    by_family_figures_dir = figures_dir / "by_family"
    by_topic_family_figures_dir = figures_dir / "by_topic_by_family"
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
    by_family_view_dirs = {
        "daily": by_family_figures_dir / "daily",
        "weekly": by_family_figures_dir / "weekly",
        "rolling_daily": by_family_figures_dir / "rolling_daily",
    }
    by_topic_family_view_dirs = {
        "daily": by_topic_family_figures_dir / "daily",
        "weekly": by_topic_family_figures_dir / "weekly",
        "rolling_daily": by_topic_family_figures_dir / "rolling_daily",
    }
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    pooled_figures_dir.mkdir(parents=True, exist_ok=True)
    by_subreddit_figures_dir.mkdir(parents=True, exist_ok=True)
    by_family_figures_dir.mkdir(parents=True, exist_ok=True)
    by_topic_family_figures_dir.mkdir(parents=True, exist_ok=True)
    enabled_views = active_view_names(bool(args.include_weekly))
    for out_dir in [
        *[pooled_view_dirs[name] for name in enabled_views],
        *[by_sub_view_dirs[name] for name in enabled_views],
        *[by_family_view_dirs[name] for name in enabled_views],
        *[by_topic_family_view_dirs[name] for name in enabled_views],
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = tables_dir / "event_time" / "event_time_daily_metrics_pooled.csv"
    if not daily_path.exists():
        daily_path = tables_dir / "event_time_daily_metrics.csv"
    df_pooled = pd.read_csv(daily_path)
    df_pooled = df_pooled.sort_values("event_time_t")

    by_sub_path = tables_dir / "event_time" / "event_time_daily_metrics_by_subreddit.csv"
    df_by_sub = pd.read_csv(by_sub_path) if by_sub_path.exists() else pd.DataFrame()
    if "repetition_template_similarity" in df_pooled.columns and not bool(
        pd.to_numeric(df_pooled["repetition_template_similarity"], errors="coerce").notna().any()
    ):
        print(
            "[plot_event_time_metrics] repetition_metric_unavailable pooled table has no non-null repetition values; "
            "repetition plots will be skipped.",
            flush=True,
        )

    pooled_specs: list[tuple[str, str, str]] = [
        ("semicolon_rate_100w", "Event-time: Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
        ("em_dash_rate_100w", "Event-time: Em Dash Rate (per 100 words)", "event_time_em_dash_rate.png"),
        ("en_dash_rate_100w", "Event-time: En Dash Rate (per 100 words)", "event_time_en_dash_rate.png"),
        (
            "ascii_double_hyphen_rate_100w",
            "Event-time: ASCII Double-hyphen (` -- `) Rate (per 100 words)",
            "event_time_ascii_double_hyphen_rate.png",
        ),
        ("colon_rate_100w", "Event-time: Colon Rate (per 100 words)", "event_time_colon_rate.png"),
        ("open_paren_rate_100w", "Event-time: Open-parenthesis Rate (per 100 words)", "event_time_open_paren_rate.png"),
        ("curly_quote_rate_100w", "Event-time: Curly Quote Characters (per 100 words)", "event_time_curly_quote_rate.png"),
        ("quote_all_rate_100w", "Event-time: All Quote Characters (per 100 words)", "event_time_quote_all_rate.png"),
        ("url_rate_100w", "Event-time: URL Rate (per 100 words)", "event_time_url_rate.png"),
        (
            "time_expression_rate_100w",
            "Event-time: Time-expression Rate (per 100 words)",
            "event_time_time_expression_rate.png",
        ),
        (
            "markdown_bold_pair_rate_100w",
            "Event-time: Markdown Bold Pair Rate (per 100 words)",
            "event_time_markdown_bold_pair_rate.png",
        ),
        (
            "markdown_heading_line_rate_100w",
            "Event-time: Markdown Heading Line Rate (per 100 words)",
            "event_time_markdown_heading_line_rate.png",
        ),
        ("hedging_phrase_rate_100w", "Event-time: Hedging Phrase Rate (per 100 words)", "event_time_hedging_phrase_rate.png"),
        ("polite_closer_rate_100w", "Event-time: Polite-closer Phrase Rate (per 100 words)", "event_time_polite_closer_rate.png"),
        (
            "signposting_phrase_rate_100w",
            "Event-time: Signposting Phrase Rate (per 100 words)",
            "event_time_signposting_phrase_rate.png",
        ),
        (
            "avg_words_per_sentence_mean",
            "Event-time: Mean Words per Sentence",
            "event_time_avg_words_per_sentence.png",
        ),
        ("comment_length_words", "Event-time: Average Comment Length (words)", "event_time_comment_length.png"),
        ("complexity_index", "Event-time: Complexity Index", "event_time_complexity_index.png"),
        ("ai_likeness_index", "Event-time: AI-likeness Index", "event_time_ai_likeness.png"),
        ("ai_word_rate_100w", "Event-time: AI-typical Word Rate (strict top-10 stem-aware basket)", "event_time_ai_word_rate.png"),
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
    coverage_rows: list[dict[str, float | str]] = []
    pooled_metric_cols = [m[0] for m in pooled_specs]
    pooled_daily = aggregate_daily_weighted(df_pooled.copy(), group_col="subreddit")
    pooled_weekly = aggregate_weekly_weighted(pooled_daily, group_col="subreddit")
    pooled_rolling_daily = grouped_trailing_daily_rolling(
        pooled_daily, group_col="subreddit", rolling_window_days=int(max(1, args.topic_rolling_window))
    )
    pooled_views_all: dict[str, pd.DataFrame] = {
        "daily": pooled_daily,
        "weekly": pooled_weekly,
        "rolling_daily": pooled_rolling_daily,
    }
    pooled_views: list[tuple[str, pd.DataFrame]] = [(name, pooled_views_all[name]) for name in enabled_views]
    for view_name, view_df in pooled_views:
        print(
            f"[plot_event_time_metrics] pooled_view_start view={view_name} rows={len(view_df)}",
            flush=True,
        )
        coverage_rows.extend(
            collect_metric_coverage_rows(
                view_df,
                view_name=view_name,
                group_kind="pooled",
                group_col="subreddit",
                metric_cols=pooled_metric_cols,
            )
        )
        view_df = apply_metric_coverage_gate(view_df, pooled_metric_cols, min_metric_coverage=float(args.min_metric_coverage))
        view_dir = pooled_view_dirs[view_name]
        show_markers = view_name != "rolling_daily"
        for y_col, title, fname in pooled_specs:
            if y_col in view_df.columns:
                if not bool(pd.to_numeric(view_df[y_col], errors="coerce").notna().any()):
                    print(
                        f"[plot_event_time_metrics] pooled_metric_skip_all_nan view={view_name} metric={y_col}",
                        flush=True,
                    )
                    continue
                if y_col in COVERAGE_METRICS_REQUIRE_SIGNAL and not metric_has_plotworthy_signal(view_df, y_col):
                    print(
                        f"[plot_event_time_metrics] pooled_metric_skip_no_signal view={view_name} metric={y_col}",
                        flush=True,
                    )
                    continue
                print(f"[plot_event_time_metrics] pooled_metric view={view_name} metric={y_col}", flush=True)
                plot_metric(
                    view_df,
                    y_col,
                    f"{title} ({view_name})",
                    view_dir / fname,
                    event_time_xlabel_text=xt,
                    show_markers=show_markers,
                )

        if "toxic_lexicon_rate_100w" in view_df.columns and bool(pd.to_numeric(view_df["toxic_lexicon_rate_100w"], errors="coerce").notna().any()):
            plot_metric(
                view_df,
                "toxic_lexicon_rate_100w",
                f"Event-time: Toxic Lexicon Incidence (per 100 words) ({view_name})",
                view_dir / "event_time_toxic_lexicon_rate.png",
                event_time_xlabel_text=xt,
                show_markers=show_markers,
            )

        if all(c in view_df.columns and bool(pd.to_numeric(view_df[c], errors="coerce").notna().any()) for c in ["ai_word_rate_100w", "ai_word_extended_rate_100w"]):
            plot_two_series_same_axes(
                view_df,
                "ai_word_rate_100w",
                "ai_word_extended_rate_100w",
                "strict_top10_stem (per 100 words)",
                "extended (per 100 words)",
                f"Event-time: Strict vs Extended AI Lexicon (pooled, {view_name})",
                "Rate per 100 words",
                view_dir / "event_time_ai_lexicon_strict_vs_extended.png",
                event_time_xlabel_text=xt,
                show_markers=show_markers,
            )
        plot_two_series_same_axes(
            view_df,
            "em_dash_rate_100w",
            "em_dash_extended_rate_100w",
            "em dash strict",
            "em dash extended",
            f"Event-time: Strict vs Extended Dash Signal (pooled, {view_name})",
            "Rate per 100 words",
            view_dir / "event_time_em_dash_strict_vs_extended.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        plot_two_series_same_axes(
            view_df,
            "colon_rate_100w",
            "colon_extended_rate_100w",
            "colon strict",
            "colon extended",
            f"Event-time: Strict vs Extended Colon Signal (pooled, {view_name})",
            "Rate per 100 words",
            view_dir / "event_time_colon_strict_vs_extended.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        plot_pooled_quote_dual_axis(
            view_df,
            f"Event-time: Quote rates and curly share (pooled, {view_name})",
            view_dir / "event_time_quote_rates_and_curly_share.png",
            event_time_xlabel_text=xt,
            show_markers=show_markers,
        )
        if any(c in view_df.columns and bool(pd.to_numeric(view_df[c], errors="coerce").notna().any()) for c in ["assistant_tone_rate_100w", "list_structure_intensity", "repetition_template_similarity", "ai_word_extended_rate_100w"]):
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

    if args.by_subreddit and not df_by_sub.empty:
        by_sub_specs: list[tuple[str, str, str]] = [
            ("semicolon_rate_100w", "Per-subreddit: Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
            ("em_dash_rate_100w", "Per-subreddit: Em Dash Rate (per 100 words)", "event_time_em_dash_rate.png"),
            (
                "em_dash_extended_rate_100w",
                "Per-subreddit: Extended Dash Rate (per 100 words)",
                "event_time_em_dash_extended_rate.png",
            ),
            ("en_dash_rate_100w", "Per-subreddit: En Dash Rate (per 100 words)", "event_time_en_dash_rate.png"),
            (
                "ascii_double_hyphen_rate_100w",
                "Per-subreddit: ASCII Double-hyphen Rate (per 100 words)",
                "event_time_ascii_double_hyphen_rate.png",
            ),
            ("colon_rate_100w", "Per-subreddit: Colon Rate (per 100 words)", "event_time_colon_rate.png"),
            (
                "colon_extended_rate_100w",
                "Per-subreddit: Extended Colon Rate (per 100 words)",
                "event_time_colon_extended_rate.png",
            ),
            ("open_paren_rate_100w", "Per-subreddit: Open-parenthesis Rate (per 100 words)", "event_time_open_paren_rate.png"),
            ("curly_quote_rate_100w", "Per-subreddit: Curly Quote Rate (per 100 words)", "event_time_curly_quote_rate.png"),
            ("quote_all_rate_100w", "Per-subreddit: All Quote Characters (per 100 words)", "event_time_quote_all_rate.png"),
            ("quote_curly_share", "Per-subreddit: Curly Quote Share", "event_time_quote_curly_share.png"),
            ("url_rate_100w", "Per-subreddit: URL Rate (per 100 words)", "event_time_url_rate.png"),
            (
                "time_expression_rate_100w",
                "Per-subreddit: Time-expression Rate (per 100 words)",
                "event_time_time_expression_rate.png",
            ),
            (
                "markdown_bold_pair_rate_100w",
                "Per-subreddit: Markdown Bold Pair Rate (per 100 words)",
                "event_time_markdown_bold_pair_rate.png",
            ),
            (
                "markdown_heading_line_rate_100w",
                "Per-subreddit: Markdown Heading Line Rate (per 100 words)",
                "event_time_markdown_heading_line_rate.png",
            ),
            ("hedging_phrase_rate_100w", "Per-subreddit: Hedging Phrase Rate (per 100 words)", "event_time_hedging_phrase_rate.png"),
            ("polite_closer_rate_100w", "Per-subreddit: Polite-closer Rate (per 100 words)", "event_time_polite_closer_rate.png"),
            (
                "signposting_phrase_rate_100w",
                "Per-subreddit: Signposting Phrase Rate (per 100 words)",
                "event_time_signposting_phrase_rate.png",
            ),
            ("avg_words_per_sentence_mean", "Per-subreddit: Mean Words per Sentence", "event_time_avg_words_per_sentence.png"),
            ("comment_length_words", "Per-subreddit: Average Comment Length (words)", "event_time_comment_length.png"),
            ("complexity_index", "Per-subreddit: Complexity Index", "event_time_complexity_index.png"),
            ("ai_likeness_index", "Per-subreddit: AI-likeness Index", "event_time_ai_likeness.png"),
            ("ai_word_rate_100w", "Per-subreddit: Strict Top-10 Stem-aware Rate (per 100 words)", "event_time_ai_word_rate.png"),
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
        by_sub_metric_cols = [m[0] for m in by_sub_specs]
        by_sub_daily = aggregate_daily_weighted(df_by_sub[df_by_sub["subreddit"] != "ALL"].copy(), group_col="subreddit")
        by_sub_weekly = aggregate_weekly_weighted(by_sub_daily, group_col="subreddit")
        by_sub_rolling_daily = grouped_trailing_daily_rolling(
            by_sub_daily, group_col="subreddit", rolling_window_days=int(max(1, args.topic_rolling_window))
        )
        by_sub_views_all: dict[str, pd.DataFrame] = {
            "daily": by_sub_daily,
            "weekly": by_sub_weekly,
            "rolling_daily": by_sub_rolling_daily,
        }
        by_sub_views: list[tuple[str, pd.DataFrame]] = [(name, by_sub_views_all[name]) for name in enabled_views]
        family_names = list(family_topic_map.keys())
        for view_name, view_df in by_sub_views:
            print(
                f"[plot_event_time_metrics] by_subreddit_view_start view={view_name} rows={len(view_df)}",
                flush=True,
            )
            coverage_rows.extend(
                collect_metric_coverage_rows(
                    view_df,
                    view_name=view_name,
                    group_kind="subreddit",
                    group_col="subreddit",
                    metric_cols=by_sub_metric_cols,
                )
            )
            view_df = apply_metric_coverage_gate(view_df, by_sub_metric_cols, min_metric_coverage=float(args.min_metric_coverage))
            show_markers = view_name != "rolling_daily"
            for family_name in family_names:
                family_df = view_df[view_df["subreddit"].map(config_family_map).eq(family_name)].copy()
                if family_df.empty:
                    continue
                family_out_dir = by_sub_view_dirs[view_name] / family_name
                family_out_dir.mkdir(parents=True, exist_ok=True)
                for y_col, title, fname in by_sub_specs:
                    if y_col in family_df.columns:
                        if not bool(pd.to_numeric(family_df[y_col], errors="coerce").notna().any()):
                            print(
                                f"[plot_event_time_metrics] by_subreddit_by_family_metric_skip_all_nan view={view_name} family={family_name} metric={y_col}",
                                flush=True,
                            )
                            continue
                        print(
                            f"[plot_event_time_metrics] by_subreddit_by_family_metric view={view_name} family={family_name} metric={y_col}",
                            flush=True,
                        )
                        family_topics = list(family_topic_map.get(family_name, []))
                        n_panels = max(1, len(family_topics))
                        ncols = 3
                        nrows = (n_panels + ncols - 1) // ncols
                        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
                        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
                        for ax, topic_name in zip(axes_flat, family_topics):
                            topic_df = family_df[family_df["subreddit"].map(config_topic_map).eq(topic_name)].copy()
                            if topic_df.empty:
                                ax.set_title(f"{topic_name} (no data)")
                                ax.axis("off")
                                continue
                            ordered_topic = topic_df.sort_values(["subreddit", "date"])
                            subreddits = sorted(ordered_topic["subreddit"].dropna().unique())
                            palette = dict(zip(subreddits, sns.color_palette("husl", n_colors=max(1, len(subreddits)))))
                            for subreddit, grp in ordered_topic.groupby("subreddit", sort=True):
                                ax.plot(
                                    grp["date"],
                                    grp[y_col].fillna(0.0),
                                    marker=("o" if show_markers else None),
                                    markersize=1.8 if show_markers else 0,
                                    linewidth=1.0,
                                    color=palette.get(subreddit),
                                    label=subreddit,
                                )
                            add_release_markers(ax)
                            format_month_start_axis(ax)
                            ax.set_title(topic_name)
                            ax.legend(loc="best", fontsize=7, frameon=True)
                        for ax in axes_flat[len(family_topics):]:
                            ax.axis("off")
                        fig.suptitle(f"{title} [{family_name}] ({view_name})", fontsize=12)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.savefig(family_out_dir / fname, dpi=140)
                        plt.close(fig)
                plot_stacked_area_by_group(
                    family_df,
                    group_col="subreddit",
                    title=f"Per-subreddit [{family_name}] ({view_name}): Comment volume (top 10 + Rest)",
                    legend_title="Subreddit",
                    out_path=family_out_dir / "event_time_stacked_forum_size.png",
                    event_time_xlabel_text=xt,
                    value_col="n_comments",
                    top_n=10,
                )
            print(f"[plot_event_time_metrics] by_subreddit_view_done view={view_name}", flush=True)

    if args.topic_views and not df_by_sub.empty:
        family_specs: list[tuple[str, str, str]] = [
            ("semicolon_rate_100w", "Per-family ({view}): Semicolon Rate (per 100 words)", "event_time_semicolon_rate.png"),
            ("em_dash_rate_100w", "Per-family ({view}): Em Dash Rate (per 100 words)", "event_time_em_dash_rate.png"),
            (
                "em_dash_extended_rate_100w",
                "Per-family ({view}): Extended Dash Rate (per 100 words)",
                "event_time_em_dash_extended_rate.png",
            ),
            ("en_dash_rate_100w", "Per-family ({view}): En Dash Rate (per 100 words)", "event_time_en_dash_rate.png"),
            (
                "ascii_double_hyphen_rate_100w",
                "Per-family ({view}): ASCII Double-hyphen Rate (per 100 words)",
                "event_time_ascii_double_hyphen_rate.png",
            ),
            ("colon_rate_100w", "Per-family ({view}): Colon Rate (per 100 words)", "event_time_colon_rate.png"),
            (
                "colon_extended_rate_100w",
                "Per-family ({view}): Extended Colon Rate (per 100 words)",
                "event_time_colon_extended_rate.png",
            ),
            ("open_paren_rate_100w", "Per-family ({view}): Open-parenthesis Rate (per 100 words)", "event_time_open_paren_rate.png"),
            ("curly_quote_rate_100w", "Per-family ({view}): Curly Quote Rate (per 100 words)", "event_time_curly_quote_rate.png"),
            ("quote_all_rate_100w", "Per-family ({view}): All Quote Characters (per 100 words)", "event_time_quote_all_rate.png"),
            ("quote_curly_share", "Per-family ({view}): Curly Quote Share", "event_time_quote_curly_share.png"),
            ("url_rate_100w", "Per-family ({view}): URL Rate (per 100 words)", "event_time_url_rate.png"),
            (
                "time_expression_rate_100w",
                "Per-family ({view}): Time-expression Rate (per 100 words)",
                "event_time_time_expression_rate.png",
            ),
            (
                "markdown_bold_pair_rate_100w",
                "Per-family ({view}): Markdown Bold Pair Rate (per 100 words)",
                "event_time_markdown_bold_pair_rate.png",
            ),
            (
                "markdown_heading_line_rate_100w",
                "Per-family ({view}): Markdown Heading Line Rate (per 100 words)",
                "event_time_markdown_heading_line_rate.png",
            ),
            ("hedging_phrase_rate_100w", "Per-family ({view}): Hedging Phrase Rate (per 100 words)", "event_time_hedging_phrase_rate.png"),
            ("polite_closer_rate_100w", "Per-family ({view}): Polite-closer Rate (per 100 words)", "event_time_polite_closer_rate.png"),
            (
                "signposting_phrase_rate_100w",
                "Per-family ({view}): Signposting Phrase Rate (per 100 words)",
                "event_time_signposting_phrase_rate.png",
            ),
            ("avg_words_per_sentence_mean", "Per-family ({view}): Mean Words per Sentence", "event_time_avg_words_per_sentence.png"),
            ("comment_length_words", "Per-family ({view}): Average Comment Length (words)", "event_time_comment_length.png"),
            ("complexity_index", "Per-family ({view}): Complexity Index", "event_time_complexity_index.png"),
            ("ai_likeness_index", "Per-family ({view}): AI-likeness Index", "event_time_ai_likeness.png"),
            ("ai_word_rate_100w", "Per-family ({view}): Strict Top-10 Stem-aware Rate (per 100 words)", "event_time_ai_word_rate.png"),
            (
                "ai_word_extended_rate_100w",
                "Per-family ({view}): Extended AI Lexicon (per 100 words)",
                "event_time_ai_word_extended_rate.png",
            ),
            (
                "assistant_tone_rate_100w",
                "Per-family ({view}): Assistant-tone Phrase Rate (per 100 words)",
                "event_time_assistant_tone_rate.png",
            ),
            ("list_structure_intensity", "Per-family ({view}): List-structure Intensity", "event_time_list_structure_intensity.png"),
            (
                "repetition_template_similarity",
                "Per-family ({view}): Repetition / Template Similarity",
                "event_time_repetition_template_similarity.png",
            ),
            ("formality_balance_100w", "Per-family ({view}): Formality Balance (per 100 words)", "event_time_formality_balance.png"),
            ("contraction_rate_100w", "Per-family ({view}): Contraction Rate (per 100 words)", "event_time_contraction_rate.png"),
            ("toxicity_score", "Per-family ({view}): VADER Negativity Mean", "event_time_toxicity_score.png"),
            ("toxic_lexicon_rate_100w", "Per-family ({view}): Toxic Lexicon (per 100 words)", "event_time_toxic_lexicon_rate.png"),
            ("detector_primary_human_score", "Per-family ({view}): Detector Primary Human Score", "event_time_detector_primary_human_score.png"),
            ("passive_rate_100w", "Per-family ({view}): Passive Construction Rate (per 100 words)", "event_time_passive_rate.png"),
            ("perplexity_mean", "Per-family ({view}): Perplexity Mean", "event_time_perplexity_mean.png"),
            ("hostility_score", "Per-family ({view}): Hostility Score Mean", "event_time_hostility_score.png"),
            ("emotion_anger", "Per-family ({view}): Emotion Anger Mean", "event_time_emotion_anger.png"),
            ("emotion_fear", "Per-family ({view}): Emotion Fear Mean", "event_time_emotion_fear.png"),
            ("emotion_sadness", "Per-family ({view}): Emotion Sadness Mean", "event_time_emotion_sadness.png"),
            ("emotion_surprise", "Per-family ({view}): Emotion Surprise Mean", "event_time_emotion_surprise.png"),
        ]
        family_metric_cols = [m[0] for m in family_specs]
        family_daily = aggregate_family_daily(df_by_sub, config_family_map)
        family_weekly = aggregate_family_weekly(family_daily)
        family_rolling_daily = grouped_trailing_daily_rolling(
            family_daily, group_col="topic_family", rolling_window_days=int(max(1, args.topic_rolling_window))
        )
        topic_family_daily = aggregate_topic_family_daily(
            df_by_sub,
            subreddit_to_topic=config_topic_map,
            topic_to_family=topic_to_family,
        )
        topic_family_weekly = aggregate_topic_family_weekly(topic_family_daily)
        topic_family_rolling_daily = rolling_topic_family_daily(
            topic_family_daily, rolling_window_days=int(max(1, args.topic_rolling_window))
        )

        family_views_all: dict[str, pd.DataFrame] = {
            "daily": family_daily,
            "weekly": family_weekly,
            "rolling_daily": family_rolling_daily,
        }
        family_views: list[tuple[str, pd.DataFrame]] = [(name, family_views_all[name]) for name in enabled_views]
        topic_family_views_all: dict[str, pd.DataFrame] = {
            "daily": topic_family_daily,
            "weekly": topic_family_weekly,
            "rolling_daily": topic_family_rolling_daily,
        }
        topic_family_views: list[tuple[str, pd.DataFrame]] = [
            (name, topic_family_views_all[name]) for name in enabled_views
        ]
        for view_name, view_df in family_views:
            if view_df.empty:
                continue
            coverage_rows.extend(
                collect_metric_coverage_rows(
                    view_df,
                    view_name=view_name,
                    group_kind="family",
                    group_col="topic_family",
                    metric_cols=family_metric_cols,
                )
            )
            view_df = apply_metric_coverage_gate(view_df, family_metric_cols, min_metric_coverage=float(args.min_metric_coverage))
            show_markers = view_name != "rolling_daily"
            print(f"[plot_event_time_metrics] family_view_start view={view_name} rows={len(view_df)}", flush=True)
            for y_col, title_template, filename_template in family_specs:
                if y_col in view_df.columns:
                    if not bool(pd.to_numeric(view_df[y_col], errors="coerce").notna().any()):
                        print(f"[plot_event_time_metrics] family_metric_skip_all_nan view={view_name} metric={y_col}", flush=True)
                        continue
                    print(f"[plot_event_time_metrics] family_metric view={view_name} metric={y_col}", flush=True)
                    plot_metric_by_family(
                        view_df,
                        y_col,
                        title_template.format(view=view_name),
                        by_family_view_dirs[view_name] / filename_template,
                        event_time_xlabel_text=xt,
                        show_markers=show_markers,
                    )
            plot_stacked_area_by_group(
                view_df,
                group_col="topic_family",
                title=f"Per-family ({view_name}): Comment volume (top 10 + Rest)",
                legend_title="Topic family",
                out_path=by_family_view_dirs[view_name] / "event_time_stacked_family_size.png",
                event_time_xlabel_text=xt,
                value_col="n_comments",
                top_n=10,
            )
            print(f"[plot_event_time_metrics] family_view_done view={view_name}", flush=True)

        for view_name, view_df in topic_family_views:
            if view_df.empty:
                continue
            coverage_rows.extend(
                collect_metric_coverage_rows(
                    view_df,
                    view_name=view_name,
                    group_kind="topic_by_family",
                    group_col="topic_family_group",
                    metric_cols=family_metric_cols,
                )
            )
            view_df = apply_metric_coverage_gate(view_df, family_metric_cols, min_metric_coverage=float(args.min_metric_coverage))
            show_markers = view_name != "rolling_daily"
            print(f"[plot_event_time_metrics] topic_by_family_view_start view={view_name} rows={len(view_df)}", flush=True)
            family_set = set(view_df["topic_family"].dropna().unique()) if "topic_family" in view_df.columns else set()
            ordered_families = [name for name in family_topic_map.keys() if name in family_set]
            for y_col, title_template, _ in family_specs:
                if y_col not in view_df.columns:
                    continue
                metric_df = view_df[["topic_family", "topic_group", "date", y_col]].copy()
                if not bool(pd.to_numeric(metric_df[y_col], errors="coerce").notna().any()):
                    print(f"[plot_event_time_metrics] topic_by_family_metric_skip_all_nan view={view_name} metric={y_col}", flush=True)
                    continue
                print(f"[plot_event_time_metrics] topic_by_family_metric view={view_name} metric={y_col}", flush=True)
                n_panels = max(1, len(ordered_families))
                ncols = 3
                nrows = (n_panels + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
                axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
                for ax, family_name in zip(axes_flat, ordered_families):
                    family_frame = metric_df[metric_df["topic_family"] == family_name].copy()
                    if family_frame.empty:
                        ax.set_title(f"{family_name} (no data)")
                        ax.axis("off")
                        continue
                    ordered = family_frame.sort_values(["topic_group", "date"])
                    topics = sorted(ordered["topic_group"].dropna().unique())
                    palette = dict(zip(topics, sns.color_palette("husl", n_colors=max(1, len(topics)))))
                    for topic_name, grp in ordered.groupby("topic_group", sort=True):
                        ax.plot(
                            grp["date"],
                            grp[y_col].fillna(0.0),
                            marker=("o" if show_markers else None),
                            markersize=2.0 if show_markers else 0,
                            linewidth=1.1,
                            color=palette.get(topic_name),
                            label=topic_name,
                        )
                    add_release_markers(ax)
                    format_month_start_axis(ax)
                    ax.set_title(family_name)
                    ax.legend(loc="best", fontsize=8, frameon=True)
                for ax in axes_flat[len(ordered_families):]:
                    ax.axis("off")
                fig.suptitle(title_template.format(view=f"topic-by-family {view_name}"), fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(by_topic_family_view_dirs[view_name] / f"by_topic_by_family_{y_col}.png", dpi=140)
                plt.close(fig)
            print(f"[plot_event_time_metrics] topic_by_family_view_done view={view_name}", flush=True)

    ai_word_long_path = tables_dir / "event_time" / "ai_word_rates_daily_long.csv"
    if ai_word_long_path.exists():
        ai_word_long_df = pd.read_csv(ai_word_long_path)
        ai_word_long_daily = ai_word_long_df.copy()
        ai_word_long_weekly = aggregate_ai_word_long_weekly(ai_word_long_df)
        ai_word_long_rolling_daily = rolling_ai_word_long_trailing_daily(
            ai_word_long_daily, rolling_window_days=int(max(1, args.topic_rolling_window))
        )
        ai_word_views_all: dict[str, pd.DataFrame] = {
            "daily": ai_word_long_daily,
            "weekly": ai_word_long_weekly,
            "rolling_daily": ai_word_long_rolling_daily,
        }
        ai_word_views: list[tuple[str, pd.DataFrame]] = [(name, ai_word_views_all[name]) for name in enabled_views]
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

    write_metric_coverage_table(coverage_rows, tables_dir)


if __name__ == "__main__":
    main()
