"""
Script summary:
This script builds a per-author, per-ISO-week style panel from the reusable
`comment_features/` Parquet shards. Each row in the panel represents one user's
writing in one calendar week (UTC ISO week, Monday start) across all configured
forums they posted in.

It is the foundation for the within-user pre/post analysis: rather than
aggregating to subreddit-day (the existing event-time pipeline), we aggregate
to (author, iso_week) so we can later compare each user's post-launch writing
to their own pre-launch baseline.

The panel keeps both display-friendly weekly aggregates (rates per 100 words,
weighted means) AND the precision-preserving raw fields (hit counts, sums,
sums-of-squares, comment counts) needed downstream to compute pooled, volume-
aware standard errors without rereading the original shards.

Functionality:
- Reads `data/interim/political_forums/comment_features/<subreddit>/<YYYY-MM>.parquet`.
- Filters out empty/deleted authors, AutoModerator, and bot-name heuristic accounts.
- Skips rows with missing `created_utc` or `n_words_comment <= 0`.
- Computes per-row `iso_week_start` (Monday in UTC, ISO 8601 date string).
- Aggregates per (author, iso_week_start, subreddit) within each shard, then
  pools intermediate rows across shards and collapses across subreddit to one
  row per (author, iso_week_start). Subreddit mix survives as `top_subreddit`,
  `subreddit_concentration`, and `top_topic`.
- Writes one Parquet shard per month under
  `data/interim/political_forums/user_week_style_panel/<YYYY-MM>.parquet`
  and a merged panel at `results/tables/user_week/user_week_panel.parquet`.
- Optional bounded controls (`--max_total_month_files`, `--max_days_per_month`)
  and a soft post-aggregation filter `--min_words_per_week_for_keep` (default 0
  so the same panel can serve strict and loose downstream cohorts).

How to apply/run:
- Full run:
  `.venv/bin/python scripts/prepare_user_week_style_panel.py --config config/political_forums_setup.yaml`
- Bounded benchmark:
  `.venv/bin/python scripts/prepare_user_week_style_panel.py --config config/political_forums_setup.yaml --max_total_month_files 2 --max_days_per_month 10 --profile`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


# ----------------------------- Topic / hygiene helpers -----------------------------

TOPIC_MAP: Dict[str, str] = {
    "AskProgramming": "coding",
    "CodingHelp": "coding",
    "learnprogramming": "coding",
    "Ask_Politics": "politics",
    "NeutralPolitics": "politics",
    "PoliticalDiscussion": "politics",
    "politics": "politics",
    "moderatepolitics": "politics",
    "cscareerquestions": "career",
    "ITCareerQuestions": "career",
    "csMajors": "career",
    "answers": "general_questions",
    "OutOfTheLoop": "general_questions",
    "TooAfraidToAsk": "general_questions",
}

# Rate features stored as `<base>_rate_100w` per 100 words, with the underlying
# integer `hits` count also persisted so pooled rates can be recomputed.
# Mapping: panel_rate_column_name -> (raw_hits_source_column_in_shards, raw_hits_panel_column).
RATE_FEATURES: Dict[str, tuple[str, str]] = {
    "ai_word_rate_100w": ("strict_ai_word_hits_total", "strict_ai_word_hits_total"),
    "ai_word_extended_rate_100w": ("extended_ai_word_hits_total", "extended_ai_word_hits_total"),
    "assistant_tone_rate_100w": ("assistant_tone_phrase_count", "assistant_tone_phrase_count"),
    "contraction_rate_100w": ("contraction_count", "contraction_count"),
    "full_form_rate_100w": ("full_form_count", "full_form_count"),
    "passive_rate_100w": ("passive_count", "passive_count"),
    "toxic_lexicon_rate_100w": ("toxic_lexicon_hits", "toxic_lexicon_hits"),
    "semicolon_rate_100w": ("semicolon_count", "semicolon_count"),
    "em_dash_rate_100w": ("em_dash_count", "em_dash_count"),
    "en_dash_rate_100w": ("en_dash_count", "en_dash_count"),
    "ascii_double_hyphen_rate_100w": ("ascii_double_hyphen_count", "ascii_double_hyphen_count"),
    "colon_rate_100w": ("colon_count", "colon_count"),
    "open_paren_rate_100w": ("open_paren_count", "open_paren_count"),
    "curly_quote_rate_100w": ("curly_quote_count", "curly_quote_count"),
    "markdown_bold_pair_rate_100w": ("markdown_bold_pair_count", "markdown_bold_pair_count"),
    "markdown_heading_line_rate_100w": ("markdown_heading_line_count", "markdown_heading_line_count"),
    "hedging_phrase_rate_100w": ("hedging_phrase_hits", "hedging_phrase_hits"),
    "polite_closer_rate_100w": ("polite_closer_hits", "polite_closer_hits"),
    "signposting_phrase_rate_100w": ("signposting_phrase_hits", "signposting_phrase_hits"),
}

# Mean features stored as `<feat>_mean`, with `<feat>_sum`, `<feat>_sumsq`,
# `<feat>_n` so pooled mean and pooled SE are recoverable downstream. NaN values
# are skipped per comment so coverage stays honest (n counts only non-NaN).
MEAN_FEATURES_REQUIRED: List[str] = [
    "comment_length_words",
    "avg_words_per_sentence_comment",
]

MEAN_FEATURES_OPTIONAL: List[str] = [
    "vader_compound",
    "toxicity_score_comment",
    "detector_primary_human_score",
    "detector_secondary_human_score",
    "hostility_score",
    "perplexity",
    "log_perplexity",
    "emotion_anger",
    "emotion_fear",
    "emotion_sadness",
    "emotion_surprise",
]

# Complexity is a ratio-of-sums metric (matches `compute_complexity_index` in
# `prepare_event_time_metrics.py`); we keep the three raw totals per week so
# pooled pre/post complexity can be recomputed downstream.
COMPLEXITY_RAW_COLUMNS: List[str] = [
    "n_words_comment",
    "total_word_chars_comment",
    "sentence_count_comment",
]


def is_bot_name_heuristic(author: str) -> bool:
    """Function summary: flag exploratory bot-like usernames using the same conservative substring rule as plot_data_quality_trends.py."""
    name = (author or "").strip().lower()
    if not name:
        return False
    if name == "automoderator":
        return False
    return ("bot" in name) or ("moderatorbot" in name)


# ----------------------------- Data classes -----------------------------


@dataclass
class RuntimePaths:
    """Function summary: store resolved runtime input and output paths for the user-week panel build."""

    comment_features_dir: Path
    interim_panel_dir: Path
    tables_dir: Path
    user_week_tables_dir: Path
    logs_dir: Path


@dataclass
class ProfilingStats:
    """Function summary: store cumulative runtime counters and workload volume for phase-level profiling."""

    phase_read_s: float = 0.0
    phase_aggregate_s: float = 0.0
    phase_write_s: float = 0.0
    files_processed: int = 0
    rows_read: int = 0
    rows_kept: int = 0
    user_week_rows_emitted: int = 0
    months_emitted: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Function summary: return a stable dictionary representation for logs and optional JSON export."""
        return {
            "phase_read_s": round(self.phase_read_s, 4),
            "phase_aggregate_s": round(self.phase_aggregate_s, 4),
            "phase_write_s": round(self.phase_write_s, 4),
            "files_processed": int(self.files_processed),
            "rows_read": int(self.rows_read),
            "rows_kept": int(self.rows_kept),
            "user_week_rows_emitted": int(self.user_week_rows_emitted),
            "months_emitted": int(self.months_emitted),
        }


# ----------------------------- CLI / paths -----------------------------


def parse_args() -> argparse.Namespace:
    """Function summary: parse command line options for config path and bounded benchmark controls."""
    parser = argparse.ArgumentParser(description="Build per-author per-ISO-week style panel from comment_features shards.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--max_total_month_files",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process at most this many monthly files total.",
    )
    parser.add_argument(
        "--max_month_files_per_subreddit",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process at most this many monthly files per subreddit.",
    )
    parser.add_argument(
        "--max_days_per_month",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process only the first N days per monthly file.",
    )
    parser.add_argument(
        "--min_words_per_week_for_keep",
        type=int,
        default=0,
        help="Drop user-week rows with fewer than this many words (default 0 keeps all so downstream picks the cohort).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print phase-level profiling summary at end of run.",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="",
        help="Optional path to write profiling JSON payload.",
    )
    return parser.parse_args()


def build_paths(config: Dict[str, Any]) -> RuntimePaths:
    """Function summary: resolve configured locations and ensure user-week output folders exist."""
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    user_week_tables_dir = tables_dir / "user_week"
    interim_panel_dir = interim_dir / "user_week_style_panel"
    user_week_logs_dir = logs_dir / "user_week"
    user_week_tables_dir.mkdir(parents=True, exist_ok=True)
    interim_panel_dir.mkdir(parents=True, exist_ok=True)
    user_week_logs_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        comment_features_dir=interim_dir / "comment_features",
        interim_panel_dir=interim_panel_dir,
        tables_dir=tables_dir,
        user_week_tables_dir=user_week_tables_dir,
        logs_dir=user_week_logs_dir,
    )


def iter_monthly_files(
    monthly_shards_dir: Path,
    subreddits: Iterable[str],
    max_month_files_per_subreddit: int = 0,
    max_total_month_files: int = 0,
) -> Iterable[tuple[str, Path]]:
    """Function summary: yield existing monthly Parquet paths under each configured subreddit directory."""
    yielded_total = 0
    for subreddit in sorted(subreddits):
        subreddit_dir = monthly_shards_dir / subreddit
        if not subreddit_dir.exists():
            continue
        yielded_for_subreddit = 0
        for parquet_path in sorted(subreddit_dir.glob("*.parquet")):
            if max_month_files_per_subreddit > 0 and yielded_for_subreddit >= max_month_files_per_subreddit:
                break
            if max_total_month_files > 0 and yielded_total >= max_total_month_files:
                return
            yield subreddit, parquet_path
            yielded_for_subreddit += 1
            yielded_total += 1


# ----------------------------- ISO week -----------------------------


def iso_week_start_from_unix(created_utc: int) -> date:
    """Function summary: return the UTC Monday date that begins the ISO week containing the unix timestamp."""
    dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc).date()
    return dt - timedelta(days=dt.weekday())


# ----------------------------- Per-shard aggregation -----------------------------


def select_shard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep only columns we need from a comment_features shard for memory efficiency."""
    needed = {
        "id",
        "subreddit",
        "author",
        "created_utc",
        "n_words_comment",
        "total_word_chars_comment",
        "sentence_count_comment",
        "list_structure_flag",
        "strict_ai_word_hits_total",
        "extended_ai_word_hits_total",
        "assistant_tone_phrase_count",
        "contraction_count",
        "full_form_count",
        "passive_count",
        "toxic_lexicon_hits",
        "semicolon_count",
        "em_dash_count",
        "en_dash_count",
        "ascii_double_hyphen_count",
        "colon_count",
        "open_paren_count",
        "curly_quote_count",
        "markdown_bold_pair_count",
        "markdown_heading_line_count",
        "hedging_phrase_hits",
        "polite_closer_hits",
        "signposting_phrase_hits",
        "avg_words_per_sentence_comment",
    }
    needed.update(MEAN_FEATURES_REQUIRED)
    needed.update(MEAN_FEATURES_OPTIONAL)
    keep = [c for c in needed if c in frame.columns]
    return frame[keep].copy()


def filter_valid_authors(frame: pd.DataFrame) -> pd.DataFrame:
    """Function summary: drop rows with empty / deleted / AutoModerator / bot-name authors and rows with no usable timing/length."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "author" not in out.columns:
        return out.iloc[0:0]
    out["author"] = out["author"].astype("string").fillna("")
    mask_author = out["author"].str.len() > 0
    mask_author &= out["author"] != "[deleted]"
    mask_author &= out["author"] != "AutoModerator"
    mask_author &= ~out["author"].apply(is_bot_name_heuristic)
    out = out[mask_author].copy()
    if "created_utc" in out.columns:
        out["created_utc"] = pd.to_numeric(out["created_utc"], errors="coerce")
        out = out[out["created_utc"].notna()].copy()
    else:
        return out.iloc[0:0]
    if "n_words_comment" in out.columns:
        out["n_words_comment"] = pd.to_numeric(out["n_words_comment"], errors="coerce").fillna(0.0)
        out = out[out["n_words_comment"] > 0].copy()
    return out


def aggregate_shard_to_user_week_subreddit(frame: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    """Function summary: collapse a filtered per-comment shard into per (author, iso_week_start, subreddit) sums for later merge across shards."""
    if frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df["iso_week_start"] = df["created_utc"].astype("int64").map(iso_week_start_from_unix).map(lambda d: d.isoformat())
    df["subreddit"] = subreddit
    df["__one"] = 1

    integer_sum_cols: List[str] = [
        "n_words_comment",
        "total_word_chars_comment",
        "sentence_count_comment",
        "strict_ai_word_hits_total",
        "extended_ai_word_hits_total",
        "assistant_tone_phrase_count",
        "contraction_count",
        "full_form_count",
        "passive_count",
        "toxic_lexicon_hits",
        "semicolon_count",
        "em_dash_count",
        "en_dash_count",
        "ascii_double_hyphen_count",
        "colon_count",
        "open_paren_count",
        "curly_quote_count",
        "markdown_bold_pair_count",
        "markdown_heading_line_count",
        "hedging_phrase_hits",
        "polite_closer_hits",
        "signposting_phrase_hits",
        "list_structure_flag",
    ]
    integer_sum_cols = [c for c in integer_sum_cols if c in df.columns]
    for col in integer_sum_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    sum_specs: Dict[str, str] = {col: "sum" for col in integer_sum_cols}
    sum_specs["__one"] = "sum"

    for feat in MEAN_FEATURES_REQUIRED + MEAN_FEATURES_OPTIONAL:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        val_col = f"__{feat}_val"
        sq_col = f"__{feat}_sq"
        mask_col = f"__{feat}_mask"
        df[val_col] = s.where(s.notna(), 0.0)
        df[sq_col] = df[val_col] ** 2
        df[mask_col] = s.notna().astype(int)
        sum_specs[val_col] = "sum"
        sum_specs[sq_col] = "sum"
        sum_specs[mask_col] = "sum"

    out = (
        df.groupby(["author", "iso_week_start", "subreddit"], sort=False)
        .agg(sum_specs)
        .reset_index()
    )
    rename_map = {"__one": "n_comments"}
    if "n_words_comment" in out.columns:
        rename_map["n_words_comment"] = "n_words"
    if "list_structure_flag" in out.columns:
        rename_map["list_structure_flag"] = "list_structure_flag_sum"
    for feat in MEAN_FEATURES_REQUIRED + MEAN_FEATURES_OPTIONAL:
        if f"__{feat}_val" in out.columns:
            rename_map[f"__{feat}_val"] = f"{feat}_sum"
            rename_map[f"__{feat}_sq"] = f"{feat}_sumsq"
            rename_map[f"__{feat}_mask"] = f"{feat}_n"
    out = out.rename(columns=rename_map)
    return out


# ----------------------------- Cross-shard merge into final panel -----------------------------


def merge_user_week_subreddit_rows(intermediate: pd.DataFrame) -> pd.DataFrame:
    """Function summary: collapse (author, iso_week_start, subreddit) intermediate rows into one row per (author, iso_week_start) with subreddit-mix metadata."""
    if intermediate.empty:
        return pd.DataFrame()

    sum_columns_int = [
        "n_words",
        "total_word_chars_comment",
        "sentence_count_comment",
        "strict_ai_word_hits_total",
        "extended_ai_word_hits_total",
        "assistant_tone_phrase_count",
        "contraction_count",
        "full_form_count",
        "passive_count",
        "toxic_lexicon_hits",
        "semicolon_count",
        "em_dash_count",
        "en_dash_count",
        "ascii_double_hyphen_count",
        "colon_count",
        "open_paren_count",
        "curly_quote_count",
        "markdown_bold_pair_count",
        "markdown_heading_line_count",
        "hedging_phrase_hits",
        "polite_closer_hits",
        "signposting_phrase_hits",
        "list_structure_flag_sum",
        "n_comments",
    ]
    sum_columns_int = [c for c in sum_columns_int if c in intermediate.columns]

    sum_columns_mean: List[str] = []
    for feat in MEAN_FEATURES_REQUIRED + MEAN_FEATURES_OPTIONAL:
        for suffix in ("_sum", "_sumsq", "_n"):
            col = f"{feat}{suffix}"
            if col in intermediate.columns:
                sum_columns_mean.append(col)

    all_sum_cols = sum_columns_int + sum_columns_mean

    # Subreddit mix per (author, iso_week_start) from the intermediate row set.
    mix_rows: List[Dict[str, Any]] = []
    for (author, iso_week_start), grp in intermediate.groupby(["author", "iso_week_start"], sort=False):
        if grp.empty:
            continue
        words = pd.to_numeric(grp["n_words"], errors="coerce").fillna(0.0).astype(float).values
        subs = grp["subreddit"].astype(str).values
        total_words = float(words.sum())
        if total_words <= 0:
            top_sub = subs[0] if len(subs) else ""
            concentration = 1.0 if len(subs) >= 1 else 0.0
        else:
            shares = words / total_words
            concentration = float((shares ** 2).sum())
            top_idx = int(words.argmax())
            top_sub = str(subs[top_idx])
        top_topic = TOPIC_MAP.get(top_sub, "other")
        mix_rows.append(
            {
                "author": str(author),
                "iso_week_start": str(iso_week_start),
                "top_subreddit": top_sub,
                "subreddit_concentration": float(concentration),
                "top_topic": top_topic,
                "n_subreddits": int(len(subs)),
            }
        )
    mix_df = pd.DataFrame(mix_rows)

    aggregated = (
        intermediate.groupby(["author", "iso_week_start"], sort=False)[all_sum_cols].sum().reset_index()
    )

    panel = aggregated.merge(mix_df, on=["author", "iso_week_start"], how="left")

    # Derived display rates per 100 words.
    n_words = panel["n_words"].astype(float).where(panel["n_words"].astype(float) > 0)
    for rate_col, (raw_hits_src, raw_hits_panel) in RATE_FEATURES.items():
        if raw_hits_panel in panel.columns:
            panel[rate_col] = (panel[raw_hits_panel].astype(float) / n_words) * 100.0
            panel[rate_col] = panel[rate_col].fillna(0.0)
    if {"full_form_count", "contraction_count"}.issubset(panel.columns):
        panel["formality_balance_100w"] = (
            (panel["full_form_count"].astype(float) - panel["contraction_count"].astype(float)) / n_words
        ) * 100.0
        panel["formality_balance_100w"] = panel["formality_balance_100w"].fillna(0.0)
    if "list_structure_flag_sum" in panel.columns:
        denom = panel["n_comments"].astype(float).where(panel["n_comments"].astype(float) > 0)
        panel["list_structure_intensity"] = (panel["list_structure_flag_sum"].astype(float) / denom).fillna(0.0)

    # Complexity index from weekly totals (matches the daily aggregate's ratio-of-sums formula).
    if {"sentence_count_comment", "total_word_chars_comment", "n_words"}.issubset(panel.columns):
        n_words_safe = panel["n_words"].astype(float).where(panel["n_words"].astype(float) > 0, 1.0)
        sent_safe = panel["sentence_count_comment"].astype(float).where(panel["sentence_count_comment"].astype(float) > 0, 1.0)
        mean_sentence_length = panel["n_words"].astype(float) / sent_safe
        mean_word_length = panel["total_word_chars_comment"].astype(float) / n_words_safe
        complexity = 0.5 * mean_sentence_length + 0.5 * mean_word_length
        complexity = complexity.where(panel["n_words"].astype(float) > 0, 0.0)
        panel["complexity_index"] = complexity.astype(float)

    # Mean-feature display means from sum / n (only counts non-NaN comments).
    for feat in MEAN_FEATURES_REQUIRED + MEAN_FEATURES_OPTIONAL:
        sum_col = f"{feat}_sum"
        n_col = f"{feat}_n"
        if sum_col in panel.columns and n_col in panel.columns:
            n_safe = panel[n_col].astype(float).where(panel[n_col].astype(float) > 0)
            panel[f"{feat}_mean"] = (panel[sum_col].astype(float) / n_safe).where(n_safe.notna(), float("nan"))

    return panel.sort_values(["author", "iso_week_start"]).reset_index(drop=True)


# ----------------------------- Per-shard processing pipeline -----------------------------


def process_shard(
    file_path: Path,
    subreddit: str,
    max_days_per_month: int,
    stats: ProfilingStats,
) -> pd.DataFrame:
    """Function summary: read one comment_features shard and emit (author, iso_week, subreddit) intermediate rows."""
    print(
        f"[prepare_user_week_style_panel] shard_start subreddit={subreddit} month={file_path.stem}",
        flush=True,
    )
    t0 = time.perf_counter()
    frame = pd.read_parquet(file_path)
    stats.phase_read_s += time.perf_counter() - t0
    stats.rows_read += int(len(frame))
    if frame.empty:
        print(
            f"[prepare_user_week_style_panel] shard_empty subreddit={subreddit} month={file_path.stem}",
            flush=True,
        )
        return pd.DataFrame()
    if "subreddit" in frame.columns:
        frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
    if frame.empty:
        return pd.DataFrame()
    if max_days_per_month > 0 and "date_utc" in frame.columns:
        keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
        frame = frame[frame["date_utc"].isin(keep_days)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = select_shard_columns(frame)
    frame = filter_valid_authors(frame)
    if frame.empty:
        return pd.DataFrame()
    stats.rows_kept += int(len(frame))
    t1 = time.perf_counter()
    intermediate = aggregate_shard_to_user_week_subreddit(frame, subreddit=subreddit)
    stats.phase_aggregate_s += time.perf_counter() - t1
    stats.files_processed += 1
    print(
        f"[prepare_user_week_style_panel] shard_done subreddit={subreddit} month={file_path.stem} "
        f"rows_in={int(len(frame))} user_week_rows={int(len(intermediate))}",
        flush=True,
    )
    return intermediate


def assign_month_key(iso_week_start: str) -> str:
    """Function summary: derive a YYYY-MM partitioning key from an ISO week start date string."""
    return iso_week_start[:7]


# ----------------------------- Output -----------------------------


def write_panel_outputs(panel: pd.DataFrame, paths: RuntimePaths, stats: ProfilingStats) -> None:
    """Function summary: write the merged user-week panel and per-month interim shards."""
    if panel.empty:
        print("[prepare_user_week_style_panel] panel_empty: nothing to write", flush=True)
        return
    t0 = time.perf_counter()
    paths.user_week_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.interim_panel_dir.mkdir(parents=True, exist_ok=True)
    merged_path = paths.user_week_tables_dir / "user_week_panel.parquet"
    panel.to_parquet(merged_path, index=False, compression="zstd")
    print(
        f"[prepare_user_week_style_panel] wrote merged_panel rows={len(panel)} path={merged_path}",
        flush=True,
    )
    panel_with_month = panel.copy()
    panel_with_month["__month_key"] = panel_with_month["iso_week_start"].astype(str).map(assign_month_key)
    months_emitted = 0
    for month_key, group in panel_with_month.groupby("__month_key", sort=True):
        month_path = paths.interim_panel_dir / f"{month_key}.parquet"
        group.drop(columns="__month_key").to_parquet(month_path, index=False, compression="zstd")
        months_emitted += 1
    stats.months_emitted = months_emitted
    stats.phase_write_s += time.perf_counter() - t0


def emit_profiling(stats: ProfilingStats, profile: bool, profile_output: str) -> None:
    """Function summary: print and optionally persist profiling counters for performance benchmarking."""
    if not profile and not profile_output:
        return
    payload = stats.as_dict()
    print(f"[prepare_user_week_style_panel] profile={json.dumps(payload, sort_keys=True)}", flush=True)
    if profile_output:
        Path(profile_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_notes(path: Path) -> None:
    """Function summary: write a short notes file documenting panel schema and design choices for reproducibility."""
    lines = [
        "User-Week Style Panel Notes",
        "============================",
        "",
        "Unit of observation: one row per (author, iso_week_start). ISO weeks are anchored to UTC",
        "Monday and stored as YYYY-MM-DD date strings (the Monday's date).",
        "",
        "Author hygiene:",
        "- empty author, [deleted], AutoModerator, and bot-name heuristic accounts are dropped.",
        "",
        "Row inclusion:",
        "- Per comment: needs created_utc and n_words_comment > 0.",
        "- Per user-week: kept when total n_words >= --min_words_per_week_for_keep (default 0).",
        "",
        "Schema buckets:",
        "- counts: n_comments, n_words (precision weights downstream).",
        "- rate features: <feat>_rate_100w (display) AND raw integer hit counts (for pooled rates), including em/en dash,",
        "  ascii double-hyphen, colon/paren/curly-quote, markdown bold/heading, hedging/polite/signposting phrase rates.",
        "- formality_balance_100w computed from full_form_count and contraction_count.",
        "- list_structure_intensity = list_structure_flag_sum / n_comments (mean of binary flag).",
        "- complexity_index from weekly totals (n_words, total_word_chars_comment, sentence_count_comment),",
        "  matching the ratio-of-sums formula used in prepare_event_time_metrics.py.",
        "- mean features: <feat>_mean (display), <feat>_sum, <feat>_sumsq, <feat>_n; n counts only",
        "  comments with non-NaN values so coverage stays honest.",
        "- subreddit mix: top_subreddit, subreddit_concentration (Herfindahl), n_subreddits, top_topic.",
        "",
        "Topic mapping (top_topic):",
        "- coding: AskProgramming, CodingHelp, learnprogramming",
        "- politics: Ask_Politics, NeutralPolitics, PoliticalDiscussion, politics, moderatepolitics",
        "- career: cscareerquestions, ITCareerQuestions, csMajors",
        "- general_questions: answers, OutOfTheLoop, TooAfraidToAsk",
        "- other: any subreddit not in the map (should not occur with the configured primary list).",
        "",
        "Outputs:",
        "- data/interim/political_forums/user_week_style_panel/<YYYY-MM>.parquet (one shard per month).",
        "- results/tables/user_week/user_week_panel.parquet (merged panel for downstream).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- main -----------------------------


def main() -> None:
    """Function summary: run the full per-author per-ISO-week panel build and write all output artifacts."""
    args = parse_args()
    config = load_config(args.config)
    subreddits = list(config["subreddits"]["primary"])
    paths = build_paths(config)

    if not paths.comment_features_dir.is_dir():
        raise FileNotFoundError(
            f"comment_features directory not found: {paths.comment_features_dir}. "
            "Run scripts/compute_comment_features.py or scripts/merge_ml_shards_into_comment_features.py first."
        )

    month_jobs = list(
        iter_monthly_files(
            paths.comment_features_dir,
            subreddits,
            max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
            max_total_month_files=int(args.max_total_month_files),
        )
    )
    if not month_jobs:
        raise FileNotFoundError(
            f"No comment_features parquet files found under: {paths.comment_features_dir}. "
            "Run scripts/compute_comment_features.py or scripts/merge_ml_shards_into_comment_features.py first."
        )

    print(
        f"[prepare_user_week_style_panel] start subreddits={len(set(s for s, _ in month_jobs))} files={len(month_jobs)}",
        flush=True,
    )

    stats = ProfilingStats()
    intermediate_parts: List[pd.DataFrame] = []
    for subreddit, file_path in month_jobs:
        part = process_shard(
            file_path=file_path,
            subreddit=subreddit,
            max_days_per_month=int(args.max_days_per_month),
            stats=stats,
        )
        if not part.empty:
            intermediate_parts.append(part)

    if not intermediate_parts:
        print("[prepare_user_week_style_panel] no_intermediate_rows: nothing to merge", flush=True)
        emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))
        return

    intermediate = pd.concat(intermediate_parts, ignore_index=True)
    print(
        f"[prepare_user_week_style_panel] intermediate_rows={len(intermediate)} (pre-collapse across subreddits)",
        flush=True,
    )

    panel = merge_user_week_subreddit_rows(intermediate)
    if int(args.min_words_per_week_for_keep) > 0:
        before = int(len(panel))
        panel = panel[panel["n_words"].astype(float) >= float(args.min_words_per_week_for_keep)].copy()
        print(
            f"[prepare_user_week_style_panel] applied min_words_per_week_for_keep={int(args.min_words_per_week_for_keep)} "
            f"rows_before={before} rows_after={len(panel)}",
            flush=True,
        )
    stats.user_week_rows_emitted = int(len(panel))

    write_panel_outputs(panel=panel, paths=paths, stats=stats)
    write_notes(paths.user_week_tables_dir / "user_week_panel_notes.txt")
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))

    print(
        f"[prepare_user_week_style_panel] done user_week_rows={stats.user_week_rows_emitted} "
        f"unique_authors={panel['author'].nunique() if not panel.empty else 0} "
        f"unique_weeks={panel['iso_week_start'].nunique() if not panel.empty else 0}",
        flush=True,
    )


if __name__ == "__main__":
    main()
