"""
Script summary:
This script builds a per-author, per-ISO-week panel from enriched monthly Parquet
shards on `cleaned_monthly_chunks/` (Italy polarization study).
Each row represents one user's writing in one calendar week (UTC ISO week,
Monday start) across all configured forums they posted in.

Foundation for within-user pre/post analysis around the Italy ChatGPT ban
(`event_window.launch_day_utc`): aggregate to (author, iso_week) and compare
each user's post-ban writing to their own pre-ban baseline.

The panel keeps display-friendly weekly aggregates (rates per 100 words, weighted
means) and precision-preserving raw fields (hit counts, sums, sums-of-squares,
comment counts) for pooled standard errors downstream.

Functionality:
- Reads `paths.interim_dir/cleaned_monthly_chunks/`.
- Skips screening-excluded subreddits by default (same rule as feature passes); use
  `--include-excluded` only for audit runs.
- Filters out empty/deleted authors, AutoModerator, and bot-name heuristic accounts.
- Skips rows with missing `created_utc` or `n_words_comment <= 0`.
- Computes per-row `iso_week_start` (Monday in UTC, ISO 8601 date string).
- Aggregates per (author, iso_week_start, subreddit) within each shard, then
  pools intermediate rows across shards and collapses across subreddit to one
  row per (author, iso_week_start). Subreddit mix survives as `top_subreddit`,
  `subreddit_concentration`, and `top_topic`.
- Writes monthly shards under `paths.interim_dir/user_week_panel/` and a merged panel at
  `paths.tables_dir/user_week/user_week_panel.parquet` (Italy: under `italy_polarization/`).
- Optional bounded controls (`--max_total_month_files`, `--max_days_per_month`)
  and a soft post-aggregation filter `--min_words_per_week_for_keep` (default 0
  so the same panel can serve strict and loose downstream cohorts).

How to apply/run:
- Full run:
  `.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml`
- Bounded benchmark:
  `.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml --max_total_month_files 2 --max_days_per_month 10 --profile`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import importlib.util
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

POLE_SHARE_EPS = 1.0e-6


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

from src.config_utils import (
    load_config,
    load_screening_pooled,
    resolve_primary_subreddits,
    screening_by_subreddit,
    should_skip_screened_subreddit,
    subreddit_screening_action,
    subreddit_topic_map,
    topic_groups,
)


# ----------------------------- Screening / subreddit selection -----------------------------


def subreddits_for_panel(config: Dict[str, Any], include_excluded: bool = False) -> List[str]:
    """Function summary: primary subreddits that pass screening (analysis sample by default).

    Parameters:
    - config: loaded study YAML.
    - include_excluded: when True, retain screening-excluded forums for audit.

    Returns:
    - Sorted subreddit names to scan for user-week aggregation.
    """
    tables_dir = Path(config["paths"]["tables_dir"])
    screening_by_sub = screening_by_subreddit(load_screening_pooled(tables_dir))
    out: List[str] = []
    skipped = 0
    for subreddit in resolve_primary_subreddits(config):
        action = subreddit_screening_action(screening_by_sub, subreddit)
        if should_skip_screened_subreddit(action, include_excluded=include_excluded):
            skipped += 1
            continue
        out.append(subreddit)
    if skipped:
        print(
            f"[prepare_user_week_style_panel] skip_excluded_subreddits={skipped} "
            f"include_excluded={include_excluded}",
            flush=True,
        )
    return out


# ----------------------------- Topic / hygiene helpers -----------------------------

# Rate features: panel_rate_column -> (shard_hits_col, panel_hits_col).
ENRICHED_RATE_FEATURES: Dict[str, tuple[str, str]] = {
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
    "ai_style_rate_100w": ("ai_style_hits", "ai_style_hits"),
    "other_side_salience_rate_100w": ("other_side_salience_hits", "other_side_salience_hits"),
    "aggression_rate_100w": ("aggression_hits", "aggression_hits"),
    "left_rate_100w": ("left_hits", "left_hits"),
    "right_rate_100w": ("right_hits", "right_hits"),
    "center_rate_100w": ("center_hits", "center_hits"),
}

# Mean features stored as `<feat>_mean`, with `<feat>_sum`, `<feat>_sumsq`,
# `<feat>_n` so pooled mean and pooled SE are recoverable downstream. NaN values
# are skipped per comment so coverage stays honest (n counts only non-NaN).
MEAN_FEATURES_REQUIRED: List[str] = [
    "comment_length_words",
    "avg_words_per_sentence_comment",
]

ENRICHED_MEAN_FEATURES: List[str] = [
    "ttr_50w",
    "net_ideology",
    "extremity",
    "ambivalence",
    "negative_rate_100w",
    "anger_rate_100w",
    "issue_eu_rate_100w",
    "issue_migration_rate_100w",
    "issue_economy_rate_100w",
    "issue_culture_rate_100w",
]

# Semantic-axis weekly means (word-weighted; NaN comments excluded from n).
SEM_AXIS_MEAN_FEATURES: List[str] = [
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_economic",
    "sem_axis_cultural",
    "sem_axis_nationalism",
    "sem_axis_anti_establishment",
    "sem_axis_coverage",
]

# Complexity is a ratio-of-sums metric (matches `compute_complexity_index` in
# ratio-of-sums complexity); we keep the three raw totals per week so
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

    input_shards_dir: Path
    input_mode: str
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
    parser = argparse.ArgumentParser(
        description="Build per-author per-ISO-week panel from enriched cleaned_monthly_chunks."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
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
        "--include-excluded",
        action="store_true",
        help="Include screening-excluded subreddits (default: analysis sample only).",
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
    interim_panel_dir = interim_dir / "user_week_panel"
    user_week_logs_dir = logs_dir / "user_week"
    user_week_tables_dir.mkdir(parents=True, exist_ok=True)
    interim_panel_dir.mkdir(parents=True, exist_ok=True)
    user_week_logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        input_shards_dir=interim_dir / "cleaned_monthly_chunks",
        input_mode="enriched_shards",
        interim_panel_dir=interim_panel_dir,
        tables_dir=tables_dir,
        user_week_tables_dir=user_week_tables_dir,
        logs_dir=user_week_logs_dir,
    )


def rate_features_for_config(config: Dict[str, Any]) -> Dict[str, tuple[str, str]]:
    """Function summary: return rate-feature map for enriched-shard input.

    Parameters:
    - config: study YAML (unused; kept for API stability).

    Returns:
    - Mapping panel_rate_column -> (shard_hits_col, panel_hits_col).
    """
    _ = config
    return dict(ENRICHED_RATE_FEATURES)


def mean_features_for_config(config: Dict[str, Any]) -> List[str]:
    """Function summary: return mean features to aggregate for enriched shards.

    Parameters:
    - config: study YAML (unused; kept for API stability).

    Returns:
    - List of mean feature column names on shards.
    """
    _ = config
    return list(MEAN_FEATURES_REQUIRED) + list(ENRICHED_MEAN_FEATURES) + list(SEM_AXIS_MEAN_FEATURES)


def normalize_input_shard(frame: pd.DataFrame) -> pd.DataFrame:
    """Function summary: align Italy enriched shard column names with panel expectations.

    Parameters:
    - frame: raw shard rows.

    Returns:
    - Frame with n_words_comment and derived hit columns when needed.
    """
    out = frame.copy()
    if "n_words" in out.columns and "n_words_comment" not in out.columns:
        out["n_words_comment"] = pd.to_numeric(out["n_words"], errors="coerce")
    elif "n_words_comment" not in out.columns and "n_words" not in out.columns:
        out["n_words_comment"] = 0.0
    if "n_words_comment" in out.columns:
        nw = pd.to_numeric(out["n_words_comment"], errors="coerce").fillna(0.0)
    else:
        nw = pd.Series(0.0, index=out.index)
    derived: List[tuple[str, str]] = [
        ("ai_style_hits", "ai_style_rate_100w"),
        ("other_side_salience_hits", "other_side_salience_rate_100w"),
        ("aggression_hits", "aggression_rate_100w"),
        ("left_hits", "left_rate_100w"),
        ("right_hits", "right_rate_100w"),
        ("center_hits", "center_rate_100w"),
    ]
    for hits_col, rate_col in derived:
        if hits_col in out.columns:
            continue
        if rate_col in out.columns:
            rate = pd.to_numeric(out[rate_col], errors="coerce").fillna(0.0)
            out[hits_col] = (rate * nw / 100.0).round().astype(float)
    return out


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


def select_shard_columns(frame: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: keep only columns we need from an input shard for memory efficiency."""
    frame = normalize_input_shard(frame)
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
        "ai_style_hits",
        "other_side_salience_hits",
        "aggression_hits",
        "left_hits",
        "right_hits",
        "center_hits",
    }
    needed.update(MEAN_FEATURES_REQUIRED)
    needed.update(mean_features_for_config(config))
    needed.update({"has_sem_axis"})
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


def aggregate_shard_to_user_week_subreddit(
    frame: pd.DataFrame, subreddit: str, config: Dict[str, Any]
) -> pd.DataFrame:
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
        "ai_style_hits",
        "other_side_salience_hits",
        "aggression_hits",
        "left_hits",
        "right_hits",
        "center_hits",
    ]
    integer_sum_cols = [c for c in integer_sum_cols if c in df.columns]
    for col in integer_sum_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "has_sem_axis" in df.columns:
        df["has_sem_axis"] = pd.to_numeric(df["has_sem_axis"], errors="coerce").fillna(0.0)

    sum_specs: Dict[str, str] = {col: "sum" for col in integer_sum_cols}
    sum_specs["__one"] = "sum"
    if "has_sem_axis" in df.columns:
        sum_specs["has_sem_axis"] = "sum"

    for feat in mean_features_for_config(config):
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
    if "has_sem_axis" in out.columns:
        rename_map["has_sem_axis"] = "has_sem_axis_sum"
    if "list_structure_flag" in out.columns:
        rename_map["list_structure_flag"] = "list_structure_flag_sum"
    for feat in mean_features_for_config(config):
        if f"__{feat}_val" in out.columns:
            rename_map[f"__{feat}_val"] = f"{feat}_sum"
            rename_map[f"__{feat}_sq"] = f"{feat}_sumsq"
            rename_map[f"__{feat}_mask"] = f"{feat}_n"
    out = out.rename(columns=rename_map)
    return out


# ----------------------------- Cross-shard merge into final panel -----------------------------


def merge_user_week_subreddit_rows(
    intermediate: pd.DataFrame, subreddit_to_topic: Dict[str, str], config: Dict[str, Any]
) -> pd.DataFrame:
    """Function summary: collapse (author, iso_week_start, subreddit) rows to one row per user-week with topic labels."""
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
        "ai_style_hits",
        "other_side_salience_hits",
        "aggression_hits",
        "left_hits",
        "right_hits",
        "center_hits",
        "has_sem_axis_sum",
    ]
    sum_columns_int = [c for c in sum_columns_int if c in intermediate.columns]

    sum_columns_mean: List[str] = []
    for feat in mean_features_for_config(config):
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
        top_topic = subreddit_to_topic.get(top_sub, "other")
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
    for rate_col, (raw_hits_src, raw_hits_panel) in rate_features_for_config(config).items():
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
    for feat in mean_features_for_config(config):
        sum_col = f"{feat}_sum"
        n_col = f"{feat}_n"
        if sum_col in panel.columns and n_col in panel.columns:
            n_safe = panel[n_col].astype(float).where(panel[n_col].astype(float) > 0)
            panel[f"{feat}_mean"] = (panel[sum_col].astype(float) / n_safe).where(n_safe.notna(), float("nan"))

    panel = add_pole_share_column(panel)
    if "has_sem_axis_sum" in panel.columns and "n_comments" in panel.columns:
        denom = panel["n_comments"].astype(float).where(panel["n_comments"].astype(float) > 0)
        panel["share_scored"] = (
            panel["has_sem_axis_sum"].astype(float) / denom
        ).where(denom.notna(), float("nan"))
    return panel.sort_values(["author", "iso_week_start"]).reset_index(drop=True)


def add_pole_share_column(panel: pd.DataFrame, eps: float = POLE_SHARE_EPS) -> pd.DataFrame:
    """Function summary: derive weekly pole_share from pooled left/right/center ideology hits.

    Parameters:
    - panel: user-week panel with left_hits, right_hits, center_hits.
    - eps: stabilizer when ideology hit total is zero.

    Returns:
    - Copy with pole_share column (NaN when no ideology hits in the week).
    """
    if panel.empty or not {"left_hits", "right_hits", "center_hits"}.issubset(panel.columns):
        return panel
    out = panel.copy()
    left = out["left_hits"].astype(float)
    right = out["right_hits"].astype(float)
    center = out["center_hits"].astype(float)
    ideology_total = left + right + center
    out["pole_share"] = np.where(
        ideology_total > 0,
        (left + right) / (ideology_total + float(eps)),
        np.nan,
    )
    return out


# ----------------------------- Per-shard processing pipeline -----------------------------


def process_shard(
    file_path: Path,
    subreddit: str,
    max_days_per_month: int,
    stats: ProfilingStats,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: read one enriched shard and emit (author, iso_week, subreddit) intermediate rows."""
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
    frame = select_shard_columns(frame, config)
    frame = filter_valid_authors(frame)
    if frame.empty:
        return pd.DataFrame()
    stats.rows_kept += int(len(frame))
    t1 = time.perf_counter()
    intermediate = aggregate_shard_to_user_week_subreddit(frame, subreddit=subreddit, config=config)
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


def write_notes(path: Path, groups: Dict[str, List[str]]) -> None:
    """Function summary: write notes documenting panel schema and config-driven topic mapping for reproducibility."""
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
        "- complexity_index from weekly totals (n_words, total_word_chars_comment, sentence_count_comment).",
        "- pole_share: (left_hits + right_hits) / (left + right + center + eps) from weekly hit totals.",
        "- mean features: <feat>_mean (display), <feat>_sum, <feat>_sumsq, <feat>_n; n counts only",
        "  comments with non-NaN values so coverage stays honest.",
        "- semantic axes (sem_axis_*): word-weighted weekly means from enriched shards; do not interpret",
        "  raw levels across languages (same rule as forum semantic DiD). share_scored = has_sem_axis_sum / n_comments.",
        "- subreddit mix: top_subreddit, subreddit_concentration (Herfindahl), n_subreddits, top_topic.",
        "",
        "Topic mapping (top_topic, loaded from config/topics):",
        "- other: any subreddit not in the map (should not occur with the configured primary list).",
        "",
        "Outputs:",
        "- paths.interim_dir/user_week_panel/<YYYY-MM>.parquet (one shard per month).",
        "- paths.tables_dir/user_week/user_week_panel.parquet (merged panel for downstream).",
    ]
    for topic_name in sorted(groups.keys()):
        sub_list = ", ".join(groups[topic_name]) if groups[topic_name] else "(none)"
        lines.insert(-5, f"- {topic_name}: {sub_list}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- main -----------------------------


def main() -> None:
    """Function summary: run the full per-author per-ISO-week panel build and write all output artifacts."""
    args = parse_args()
    config = load_config(args.config)
    config_topic_map = subreddit_topic_map(config, include_topic_aliases=False)
    config_topic_groups = topic_groups(config)
    subreddits = subreddits_for_panel(config, include_excluded=bool(args.include_excluded))
    paths = build_paths(config)

    if not paths.input_shards_dir.is_dir():
        hint = (
            "Run enrich + compute_enriched_shard_features --pass all "
            "(or compute_polarization/ai_use/comment_style) first."
        )
        raise FileNotFoundError(f"Input shards directory not found: {paths.input_shards_dir}. {hint}")

    month_jobs = list(
        iter_monthly_files(
            paths.input_shards_dir,
            subreddits,
            max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
            max_total_month_files=int(args.max_total_month_files),
        )
    )
    if not month_jobs:
        raise FileNotFoundError(
            f"No parquet files found under: {paths.input_shards_dir} (input_mode={paths.input_mode})."
        )

    print(
        f"[prepare_user_week_style_panel] start subreddits={len(set(s for s, _ in month_jobs))} files={len(month_jobs)}",
        flush=True,
    )

    stats = ProfilingStats()
    intermediate_parts: List[pd.DataFrame] = []
    shards_missing_sem_axis = 0
    for subreddit, file_path in month_jobs:
        import pyarrow.parquet as pq

        if "sem_axis_ideology" not in pq.ParquetFile(file_path).schema.names:
            shards_missing_sem_axis += 1
        part = process_shard(
            file_path=file_path,
            subreddit=subreddit,
            max_days_per_month=int(args.max_days_per_month),
            stats=stats,
            config=config,
        )
        if not part.empty:
            intermediate_parts.append(part)
    if shards_missing_sem_axis:
        print(
            f"[prepare_user_week_style_panel] shards_missing_sem_axis={shards_missing_sem_axis} "
            f"(of {len(month_jobs)} kept subreddits); run compute_enriched_shard_features.py --pass semaxis",
            flush=True,
        )

    if not intermediate_parts:
        print("[prepare_user_week_style_panel] no_intermediate_rows: nothing to merge", flush=True)
        emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))
        sys.exit(1)

    intermediate = pd.concat(intermediate_parts, ignore_index=True)
    print(
        f"[prepare_user_week_style_panel] intermediate_rows={len(intermediate)} (pre-collapse across subreddits)",
        flush=True,
    )

    panel = merge_user_week_subreddit_rows(intermediate, config_topic_map, config)
    if panel.empty:
        print("[prepare_user_week_style_panel] panel_empty after merge: nothing to write", flush=True)
        emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))
        sys.exit(1)
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
    write_notes(paths.user_week_tables_dir / "user_week_panel_notes.txt", config_topic_groups)
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))

    print(
        f"[prepare_user_week_style_panel] done user_week_rows={stats.user_week_rows_emitted} "
        f"unique_authors={panel['author'].nunique() if not panel.empty else 0} "
        f"unique_weeks={panel['iso_week_start'].nunique() if not panel.empty else 0}",
        flush=True,
    )


if __name__ == "__main__":
    main()
