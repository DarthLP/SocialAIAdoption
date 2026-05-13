"""
Script summary:
Reads pooled stratified event-time CSVs produced by prepare_event_time_stratified_metrics.py
(event_time_daily_metrics_pooled_by_user_cohort.csv, ..._by_length_bucket.csv, and
event_time_length_bucket_daily_shares_pooled.csv) and writes calendar-date figures with
multiple hues: old vs new vs debut_observed users; short vs medium vs long length buckets;
comment volume; length-bucket share mix; and the same style/toxicity/detector metric suite as
plot_event_time_metrics.py except repetition_template_similarity (Jaccard) is omitted. Writes two
topic trees: results/figures/event_time/stratified_pooled/user_series/{daily,rolling_daily}/
by default (`weekly/` with `--include_weekly`) and
stratified_pooled/length_bucket/{daily,rolling_daily}/ by default (`weekly/` with `--include_weekly`) (length_bucket omits detector,
perplexity, hostility, emotion, and coverage metrics—those are not meaningful when the stratifier
is itself length). Length-bucket daily share mix lives under length_bucket/daily/.

How to apply/run:
- After prepare_event_time_stratified_metrics.py:
  `.venv/bin/python scripts/event_time/plot_event_time_stratified_metrics.py --config config/political_forums_setup.yaml`
- Rolling window (days) matches plot_event_time_metrics default pattern via --rolling_window.
- Stratified plotting reuses `plot_event_time_metrics.add_release_markers`; `main` applies the same `plot_reference_dates_utc` marker set as the non-stratified event-time plotter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
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

import plot_event_time_metrics as pet

from src.config_utils import load_config, plot_reference_dates_calendar_utc, utc_ts

USER_SERIES_LABELS = {
    "old": "Old users (pre-launch first post in forum)",
    "new": "New users (all comments)",
    "debut_observed": "Observed debut comment (all cohorts)",
}
LENGTH_LABELS = {"short": "Short (<20 words)", "medium": "Medium (20–49)", "long": "Long (≥50)"}

# Metrics omitted when hue is length_bucket: stratifying by word-count bucket makes detector,
# perplexity, hostility, emotion, and coverage shares structurally misleading or NaN-heavy.
LENGTH_STRATIFY_OMIT_METRICS: frozenset[str] = frozenset(
    {
        "detector_primary_human_score",
        "detector_secondary_human_score",
        "perplexity_mean",
        "hostility_score",
        "emotion_anger",
        "emotion_fear",
        "emotion_sadness",
        "emotion_surprise",
        "coverage_perplexity",
        "coverage_detector_primary",
        "coverage_detector_secondary",
        "coverage_hostility",
        "coverage_emotion",
    }
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for config path and rolling window size in days."""
    parser = argparse.ArgumentParser(description="Plot stratified pooled event-time metrics.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=7,
        help="Trailing rolling window size (days) for rolling_daily views.",
    )
    parser.add_argument(
        "--include_weekly",
        action="store_true",
        help="Include weekly view plots in addition to default daily and rolling_daily views.",
    )
    return parser.parse_args()


def launch_timestamp_pd(config: dict) -> pd.Timestamp:
    """Function summary: return naive UTC midnight pandas Timestamp for launch_day_utc from config."""
    launch_ts = utc_ts(str(config["event_window"]["launch_day_utc"]))
    dt = datetime.fromtimestamp(launch_ts, tz=timezone.utc).replace(tzinfo=None)
    return pd.Timestamp(dt.date())


def stratified_comment_weight_cols() -> list[str]:
    """Function summary: return COMMENT_WEIGHT_COLS without repetition_template_similarity for weighted re-aggregation."""
    return [c for c in pet.COMMENT_WEIGHT_COLS if c != "repetition_template_similarity"]


def aggregate_daily_weighted_stratified(
    df: pd.DataFrame,
    group_col: str,
    launch_date: pd.Timestamp,
    alias_col: str | None = None,
) -> pd.DataFrame:
    """Function summary: same logic as plot_event_time_metrics.aggregate_daily_weighted with configurable launch and comment columns."""
    if df.empty:
        return pd.DataFrame()
    required = {group_col, "date_utc", "n_comments", "n_words"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    d = pet.ensure_date_column(df.copy())
    if d.empty:
        return pd.DataFrame()
    for col in ["n_comments", "n_words", "strict_ai_word_hits_total", "extended_ai_word_hits_total"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    comment_cols = stratified_comment_weight_cols()
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
        for col in pet.WORD_WEIGHT_COLS:
            if col in grp.columns:
                numer = (pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * grp["n_words"]).sum()
                row[col] = float(numer / n_words) if n_words > 0 else 0.0
        for col in comment_cols:
            if col in grp.columns:
                numer = (pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * grp["n_comments"]).sum()
                row[col] = float(numer / n_comments)
        grouped_rows.append(row)
    if not grouped_rows:
        return pd.DataFrame()
    out = pd.DataFrame(grouped_rows).sort_values([group_col, "date"]).reset_index(drop=True)
    out["event_time_t"] = (out["date"] - launch_date).dt.days
    return out


def aggregate_weekly_weighted_stratified(
    daily_df: pd.DataFrame, group_col: str, launch_date: pd.Timestamp, alias_col: str | None = None
) -> pd.DataFrame:
    """Function summary: bin daily stratified rows to weekly Monday starts then re-weight metrics."""
    if daily_df.empty:
        return pd.DataFrame()
    d = pet.ensure_date_column(daily_df.copy())
    d["week_start"] = d["date"].dt.to_period("W-MON").dt.start_time
    d["date_utc"] = d["week_start"].dt.strftime("%Y-%m-%dT00:00:00Z")
    d["date"] = d["week_start"]
    return aggregate_daily_weighted_stratified(d, group_col=group_col, launch_date=launch_date, alias_col=alias_col)


def grouped_trailing_daily_rolling_stratified(
    df_daily: pd.DataFrame, group_col: str, rolling_window_days: int, launch_date: pd.Timestamp
) -> pd.DataFrame:
    """Function summary: trailing day rolling mean per stratum; refresh event_time_t from config launch."""
    if df_daily.empty:
        return pd.DataFrame()
    if rolling_window_days <= 1:
        return df_daily.copy()
    d = pet.ensure_date_column(df_daily.copy()).sort_values([group_col, "date"])
    exclude_cols = {"subreddit", "topic_group", "date_utc", "date", group_col, "event_time_t"}
    numeric_cols = [c for c in d.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(d[c])]
    out_parts: list[pd.DataFrame] = []
    for group_value, grp in d.groupby(group_col, sort=True):
        g = grp.sort_values("date").copy()
        g_indexed = g.set_index("date")
        for col in numeric_cols:
            g_indexed[col] = (
                g_indexed[col]
                .rolling(window=f"{int(rolling_window_days)}D", min_periods=1)
                .mean()
            )
        g = g_indexed.reset_index()
        g["subreddit"] = group_value
        g["event_time_t"] = (g["date"] - launch_date).dt.days
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True).sort_values([group_col, "date"]).reset_index(drop=True)


def plot_multi_hue_lines(
    df: pd.DataFrame,
    y_col: str,
    hue_col: str,
    hue_order: list[str],
    hue_labels: dict[str, str],
    title: str,
    y_label: str,
    out_path: Path,
    *,
    show_markers: bool = True,
) -> None:
    """Function summary: seaborn lineplot with ordered hue and legend labels; calendar x-axis with release markers."""
    if df.empty or y_col not in df.columns or hue_col not in df.columns:
        return
    d = pet.ensure_date_column(df.copy())
    d = d[d[hue_col].isin(hue_order)].copy()
    if d.empty:
        return
    present = [h for h in hue_order if h in set(d[hue_col].astype(str).unique())]
    if not present:
        return
    d["_hue_label"] = d[hue_col].astype(str).map(lambda x: hue_labels.get(x, x))
    label_order = [hue_labels[h] for h in present]
    plt.figure(figsize=(11, 5.5))
    sns.lineplot(
        data=d.sort_values([hue_col, "date"]),
        x="date",
        y=y_col,
        hue="_hue_label",
        hue_order=label_order,
        marker=("o" if show_markers else None),
    )
    pet.add_release_markers(plt.gca())
    pet.format_month_start_axis(plt.gca())
    plt.title(title)
    plt.xlabel("Date (UTC)")
    plt.ylabel(y_label)
    plt.legend(title="", loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_length_shares_long_form(shares_df: pd.DataFrame, out_path: Path, *, show_markers: bool = True) -> None:
    """Function summary: melt share_* columns and plot three lines for bucket mix over time."""
    if shares_df.empty:
        return
    d = pet.ensure_date_column(shares_df.copy())
    if d.empty:
        return
    long = d.melt(
        id_vars=["date"],
        value_vars=["share_short", "share_medium", "share_long"],
        var_name="bucket",
        value_name="share",
    )
    long["bucket"] = long["bucket"].map(
        {"share_short": "short", "share_medium": "medium", "share_long": "long"}
    )
    long["_label"] = long["bucket"].map(lambda b: LENGTH_LABELS.get(str(b), str(b)))
    order = [LENGTH_LABELS["short"], LENGTH_LABELS["medium"], LENGTH_LABELS["long"]]
    plt.figure(figsize=(11, 5.5))
    sns.lineplot(
        data=long.sort_values(["bucket", "date"]),
        x="date",
        y="share",
        hue="_label",
        hue_order=order,
        marker=("o" if show_markers else None),
    )
    pet.add_release_markers(plt.gca())
    pet.format_month_start_axis(plt.gca())
    plt.title("Pooled: share of comments by length bucket (daily)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Share of comments")
    plt.legend(title="", loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def pooled_stratified_metric_specs() -> list[tuple[str, str, str]]:
    """Function summary: (y_col, title, filename) tuples mirroring plot_event_time_metrics pooled_specs without repetition."""
    return [
        ("semicolon_rate_100w", "Stratified pooled: Semicolon Rate (per 100 words)", "stratified_event_time_semicolon_rate.png"),
        ("em_dash_rate_100w", "Stratified pooled: Em Dash Rate (per 100 words)", "stratified_event_time_em_dash_rate.png"),
        ("en_dash_rate_100w", "Stratified pooled: En Dash Rate (per 100 words)", "stratified_event_time_en_dash_rate.png"),
        (
            "ascii_double_hyphen_rate_100w",
            "Stratified pooled: ASCII Double-hyphen Rate (per 100 words)",
            "stratified_event_time_ascii_double_hyphen_rate.png",
        ),
        ("colon_rate_100w", "Stratified pooled: Colon Rate (per 100 words)", "stratified_event_time_colon_rate.png"),
        ("open_paren_rate_100w", "Stratified pooled: Open-parenthesis Rate (per 100 words)", "stratified_event_time_open_paren_rate.png"),
        ("curly_quote_rate_100w", "Stratified pooled: Curly Quote Rate (per 100 words)", "stratified_event_time_curly_quote_rate.png"),
        (
            "markdown_bold_pair_rate_100w",
            "Stratified pooled: Markdown Bold Pair Rate (per 100 words)",
            "stratified_event_time_markdown_bold_pair_rate.png",
        ),
        (
            "markdown_heading_line_rate_100w",
            "Stratified pooled: Markdown Heading Line Rate (per 100 words)",
            "stratified_event_time_markdown_heading_line_rate.png",
        ),
        ("hedging_phrase_rate_100w", "Stratified pooled: Hedging Phrase Rate (per 100 words)", "stratified_event_time_hedging_phrase_rate.png"),
        ("polite_closer_rate_100w", "Stratified pooled: Polite-closer Rate (per 100 words)", "stratified_event_time_polite_closer_rate.png"),
        (
            "signposting_phrase_rate_100w",
            "Stratified pooled: Signposting Phrase Rate (per 100 words)",
            "stratified_event_time_signposting_phrase_rate.png",
        ),
        ("avg_words_per_sentence_mean", "Stratified pooled: Mean Words per Sentence", "stratified_event_time_avg_words_per_sentence.png"),
        ("comment_length_words", "Stratified pooled: Average Comment Length (words)", "stratified_event_time_comment_length.png"),
        ("complexity_index", "Stratified pooled: Complexity Index", "stratified_event_time_complexity_index.png"),
        ("ai_likeness_index", "Stratified pooled: AI-likeness Index", "stratified_event_time_ai_likeness.png"),
        ("ai_word_rate_100w", "Stratified pooled: AI-typical Word Rate (strict top-10 stem-aware)", "stratified_event_time_ai_word_rate.png"),
        (
            "ai_word_extended_rate_100w",
            "Stratified pooled: Extended AI Lexicon Rate (per 100 words)",
            "stratified_event_time_ai_word_extended_rate.png",
        ),
        (
            "assistant_tone_rate_100w",
            "Stratified pooled: Assistant-tone Phrase Rate (per 100 words)",
            "stratified_event_time_assistant_tone_rate.png",
        ),
        (
            "list_structure_intensity",
            "Stratified pooled: List-structure Intensity",
            "stratified_event_time_list_structure_intensity.png",
        ),
        (
            "formality_balance_100w",
            "Stratified pooled: Formality Balance (per 100 words)",
            "stratified_event_time_formality_balance.png",
        ),
        ("contraction_rate_100w", "Stratified pooled: Contraction Rate (per 100 words)", "stratified_event_time_contraction_rate.png"),
        ("full_form_rate_100w", "Stratified pooled: Full-form Rate (per 100 words)", "stratified_event_time_full_form_rate.png"),
        ("vader_compound_mean", "Stratified pooled: VADER Compound Mean", "stratified_event_time_vader_compound_mean.png"),
        ("toxicity_score", "Stratified pooled: Toxicity Proxy (VADER negativity mean)", "stratified_event_time_toxicity_score.png"),
        (
            "detector_primary_human_score",
            "Stratified pooled: Detector Primary Human Score",
            "stratified_event_time_detector_primary_human_score.png",
        ),
        (
            "detector_secondary_human_score",
            "Stratified pooled: Detector Secondary Human Score",
            "stratified_event_time_detector_secondary_human_score.png",
        ),
        ("passive_rate_100w", "Stratified pooled: Passive Construction Rate (per 100 words)", "stratified_event_time_passive_rate.png"),
        ("perplexity_mean", "Stratified pooled: Perplexity Mean", "stratified_event_time_perplexity_mean.png"),
        ("hostility_score", "Stratified pooled: Hostility Score Mean", "stratified_event_time_hostility_score.png"),
        ("emotion_anger", "Stratified pooled: Emotion Anger Mean", "stratified_event_time_emotion_anger.png"),
        ("emotion_fear", "Stratified pooled: Emotion Fear Mean", "stratified_event_time_emotion_fear.png"),
        ("emotion_sadness", "Stratified pooled: Emotion Sadness Mean", "stratified_event_time_emotion_sadness.png"),
        ("emotion_surprise", "Stratified pooled: Emotion Surprise Mean", "stratified_event_time_emotion_surprise.png"),
        ("coverage_perplexity", "Stratified pooled: Perplexity Coverage Share", "stratified_event_time_coverage_perplexity.png"),
        (
            "coverage_detector_primary",
            "Stratified pooled: Detector Primary Coverage Share",
            "stratified_event_time_coverage_detector_primary.png",
        ),
    ]


def pooled_stratified_metric_specs_for_length() -> list[tuple[str, str, str]]:
    """Function summary: same as pooled_stratified_metric_specs but drop ML/coverage columns unsuitable for length-bucket hue."""
    return [spec for spec in pooled_stratified_metric_specs() if spec[0] not in LENGTH_STRATIFY_OMIT_METRICS]


def main() -> None:
    """Function summary: load stratified tables, build daily/weekly/rolling views, and save all stratified figures."""
    args = parse_args()
    config = load_config(args.config)
    pet.set_calendar_release_dates_for_plotting(plot_reference_dates_calendar_utc(config))
    xt = pet.event_time_xlabel(config)
    launch_pd = launch_timestamp_pd(config)
    tables_dir = Path(config["paths"]["tables_dir"])
    event_dir = tables_dir / "event_time"
    user_path = event_dir / "event_time_daily_metrics_pooled_by_user_cohort.csv"
    len_path = event_dir / "event_time_daily_metrics_pooled_by_length_bucket.csv"
    share_path = event_dir / "event_time_length_bucket_daily_shares_pooled.csv"
    if not user_path.exists():
        raise FileNotFoundError(f"Missing {user_path}; run prepare_event_time_stratified_metrics.py first.")
    if not len_path.exists():
        raise FileNotFoundError(f"Missing {len_path}; run prepare_event_time_stratified_metrics.py first.")

    df_user = pd.read_csv(user_path)
    df_len = pd.read_csv(len_path)
    df_share = pd.read_csv(share_path) if share_path.exists() else pd.DataFrame()

    figures_root = Path(config["paths"]["figures_dir"]) / "event_time" / "stratified_pooled"
    user_series_root = figures_root / "user_series"
    length_bucket_root = figures_root / "length_bucket"
    user_view_dirs = {
        "daily": user_series_root / "daily",
        "weekly": user_series_root / "weekly",
        "rolling_daily": user_series_root / "rolling_daily",
    }
    len_view_dirs = {
        "daily": length_bucket_root / "daily",
        "weekly": length_bucket_root / "weekly",
        "rolling_daily": length_bucket_root / "rolling_daily",
    }
    enabled_views = pet.active_view_names(bool(args.include_weekly))
    for p in [*[user_view_dirs[name] for name in enabled_views], *[len_view_dirs[name] for name in enabled_views]]:
        p.mkdir(parents=True, exist_ok=True)

    user_daily = aggregate_daily_weighted_stratified(df_user, "user_series", launch_pd, alias_col="user_series")
    len_daily = aggregate_daily_weighted_stratified(df_len, "length_bucket", launch_pd, alias_col="length_bucket")
    user_weekly = aggregate_weekly_weighted_stratified(df_user, "user_series", launch_pd, alias_col="user_series")
    len_weekly = aggregate_weekly_weighted_stratified(df_len, "length_bucket", launch_pd, alias_col="length_bucket")
    user_roll = grouped_trailing_daily_rolling_stratified(
        user_daily, "user_series", int(max(1, args.rolling_window)), launch_pd
    )
    len_roll = grouped_trailing_daily_rolling_stratified(
        len_daily, "length_bucket", int(max(1, args.rolling_window)), launch_pd
    )

    user_views_all: dict[str, pd.DataFrame] = {"daily": user_daily, "weekly": user_weekly, "rolling_daily": user_roll}
    len_views_all: dict[str, pd.DataFrame] = {"daily": len_daily, "weekly": len_weekly, "rolling_daily": len_roll}
    user_views = [(name, user_views_all[name]) for name in enabled_views]
    len_views = [(name, len_views_all[name]) for name in enabled_views]
    user_order = ["old", "new", "debut_observed"]
    len_order = ["short", "medium", "long"]

    for view_name, view_df in user_views:
        if view_df.empty:
            continue
        show_markers = view_name != "rolling_daily"
        vdir = user_view_dirs[view_name]
        plot_multi_hue_lines(
            view_df,
            "n_comments",
            "user_series",
            user_order,
            USER_SERIES_LABELS,
            f"Stratified pooled: comment volume by user series ({view_name})",
            "Comments (n)",
            vdir / "stratified_event_time_volume_by_user_series.png",
            show_markers=show_markers,
        )
        for y_col, title, fname in pooled_stratified_metric_specs():
            if y_col in view_df.columns:
                if y_col in pet.COVERAGE_METRICS_REQUIRE_SIGNAL and not pet.metric_has_plotworthy_signal(view_df, y_col):
                    print(
                        f"[plot_event_time_stratified_metrics] user_series_skip_no_signal view={view_name} metric={y_col}",
                        flush=True,
                    )
                    continue
                plot_multi_hue_lines(
                    view_df,
                    y_col,
                    "user_series",
                    user_order,
                    USER_SERIES_LABELS,
                    f"{title} ({view_name})",
                    y_col,
                    vdir / fname,
                    show_markers=show_markers,
                )
        if "toxic_lexicon_rate_100w" in view_df.columns:
            plot_multi_hue_lines(
                view_df,
                "toxic_lexicon_rate_100w",
                "user_series",
                user_order,
                USER_SERIES_LABELS,
                f"Stratified pooled: Toxic Lexicon (per 100 words) ({view_name})",
                "toxic_lexicon_rate_100w",
                vdir / "stratified_event_time_toxic_lexicon_rate.png",
                show_markers=show_markers,
            )
        print(f"[plot_event_time_stratified_metrics] user_series view={view_name} rows={len(view_df)}", flush=True)

    for view_name, view_df in len_views:
        if view_df.empty:
            continue
        show_markers = view_name != "rolling_daily"
        vdir = len_view_dirs[view_name]
        plot_multi_hue_lines(
            view_df,
            "n_comments",
            "length_bucket",
            len_order,
            LENGTH_LABELS,
            f"Stratified pooled: comment volume by length bucket ({view_name})",
            "Comments (n)",
            vdir / "stratified_event_time_volume_by_length_bucket.png",
            show_markers=show_markers,
        )
        for y_col, title, fname in pooled_stratified_metric_specs_for_length():
            if y_col in view_df.columns:
                if y_col in pet.COVERAGE_METRICS_REQUIRE_SIGNAL and not pet.metric_has_plotworthy_signal(view_df, y_col):
                    print(
                        f"[plot_event_time_stratified_metrics] length_bucket_skip_no_signal view={view_name} metric={y_col}",
                        flush=True,
                    )
                    continue
                plot_multi_hue_lines(
                    view_df,
                    y_col,
                    "length_bucket",
                    len_order,
                    LENGTH_LABELS,
                    f"{title} — by length ({view_name})",
                    y_col,
                    vdir / fname,
                    show_markers=show_markers,
                )
        if "toxic_lexicon_rate_100w" in view_df.columns:
            plot_multi_hue_lines(
                view_df,
                "toxic_lexicon_rate_100w",
                "length_bucket",
                len_order,
                LENGTH_LABELS,
                f"Stratified pooled: Toxic Lexicon by length ({view_name})",
                "toxic_lexicon_rate_100w",
                vdir / "stratified_event_time_toxic_lexicon_rate.png",
                show_markers=show_markers,
            )
        print(f"[plot_event_time_stratified_metrics] length_bucket view={view_name} rows={len(view_df)}", flush=True)

    if not df_share.empty:
        plot_length_shares_long_form(
            df_share, len_view_dirs["daily"] / "stratified_event_time_length_bucket_shares.png"
        )
        print("[plot_event_time_stratified_metrics] length shares plot written (daily)", flush=True)

    _ = xt
    print(f"[plot_event_time_stratified_metrics] done -> {figures_root}", flush=True)


if __name__ == "__main__":
    main()
