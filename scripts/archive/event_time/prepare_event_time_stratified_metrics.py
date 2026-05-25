"""
Script summary:
Builds pooled daily event-time tables stratified by (1) user series — old users,
new users (all comments), and observed debut comments in each subreddit (all cohorts);
(2) comment length_bucket (short / medium / long from comment_features). Phase 1
scans all comment_features shards to find each author's first observed post per
subreddit (min created_utc, tie-break min id); cohorts compare that timestamp to
config launch_day_utc. Phase 2 re-aggregates the same lexical/ML daily metrics as
prepare_event_time_metrics.py without repetition/Jaccard. Writes CSVs under
results/tables/event_time/ plus a short notes file documenting definitions and
left-censoring.

How to apply/run:
- Requires merged comment_features with author and created_utc (run
  merge_ml_shards_into_comment_features.py or compute_comment_features with those
  columns in cleaned Parquet).
- `.venv/bin/python scripts/archive/event_time/prepare_event_time_stratified_metrics.py --config config/archive/ai_adoption_political_forums_setup.yaml`
- Bounded smoke test:
  `.venv/bin/python scripts/archive/event_time/prepare_event_time_stratified_metrics.py --config config/archive/ai_adoption_political_forums_setup.yaml --max_month_files_per_subreddit 1 --max_days_per_month 5`
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys
import time
from typing import Any, Dict, Iterable

import numpy as np
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

import prepare_event_time_metrics as pem

from src.config_utils import load_config, utc_ts

USER_SERIES_ORDER = ("old", "new", "debut_observed")
LENGTH_BUCKET_ORDER = ("short", "medium", "long")


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for config path and the same bounded scan flags as prepare_event_time_metrics."""
    parser = argparse.ArgumentParser(description="Prepare stratified pooled event-time metrics from comment_features.")
    parser.add_argument("--config", type=str, default="config/archive/ai_adoption_political_forums_setup.yaml")
    parser.add_argument("--max_month_files_per_subreddit", type=int, default=0)
    parser.add_argument("--max_total_month_files", type=int, default=0)
    parser.add_argument("--max_days_per_month", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_output", type=str, default="")
    return parser.parse_args()


def iter_monthly_jobs(
    comment_features_dir: Path,
    subreddits: Iterable[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
) -> list[tuple[str, Path]]:
    """Function summary: list (subreddit, parquet_path) jobs in deterministic scan order."""
    return list(
        pem.iter_monthly_files(
            comment_features_dir,
            subreddits,
            max_month_files_per_subreddit=max_month_files_per_subreddit,
            max_total_month_files=max_total_month_files,
        )
    )


def build_first_seen_debut_table(
    jobs: list[tuple[str, Path]],
) -> pd.DataFrame:
    """Function summary: scan minimal columns across shards and return one row per (author, subreddit) with first_ts and debut_id_str.

    Parameters:
    - jobs: ordered list of (subreddit, monthly_parquet_path) from iter_monthly_jobs.

    Returns:
    - DataFrame columns author, subreddit, first_seen_ts (int64), debut_id (str).
    """
    parts: list[pd.DataFrame] = []
    for subreddit, file_path in jobs:
        need = {"author", "created_utc", "id"}
        frame = pd.read_parquet(file_path, columns=sorted(need))
        if "subreddit" in frame.columns:
            frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
        frame["author"] = frame["author"].astype("string").fillna("").str.strip()
        frame = frame[frame["author"] != ""].copy()
        frame["created_utc"] = pd.to_numeric(frame["created_utc"], errors="coerce")
        frame = frame.dropna(subset=["created_utc"]).copy()
        frame["created_utc"] = frame["created_utc"].astype("int64")
        frame["id_str"] = frame["id"].astype("string").fillna("").astype(str)
        frame["subreddit"] = subreddit
        parts.append(frame[["author", "subreddit", "created_utc", "id_str"]])
    if not parts:
        return pd.DataFrame(columns=["author", "subreddit", "first_seen_ts", "debut_id"])
    big = pd.concat(parts, ignore_index=True)
    big = big.sort_values(["author", "subreddit", "created_utc", "id_str"])
    first = big.drop_duplicates(["author", "subreddit"], keep="first")
    out = first.rename(columns={"created_utc": "first_seen_ts", "id_str": "debut_id"})[
        ["author", "subreddit", "first_seen_ts", "debut_id"]
    ]
    return out.reset_index(drop=True)


def daily_metric_row_from_group(
    group: pd.DataFrame,
    subreddit: str,
    date_utc: str,
    user_series: str | None,
    length_bucket: str | None,
) -> Dict[str, Any] | None:
    """Function summary: compute one daily aggregate dict matching prepare_event_time_metrics row schema (no repetition).

    Parameters:
    - group: comment-level rows for one calendar day (and stratum filter already applied).
    - subreddit: forum name.
    - date_utc: ISO day key string.
    - user_series: old | new | debut_observed, or None when aggregating by length only.
    - length_bucket: short | medium | long, or None when aggregating by user_series only.

    Returns:
    - Metric dict or None when group is empty.
    """

    def numeric_series(col: str, default: float = 0.0) -> pd.Series:
        """Function summary: coerce column to numeric or return constant series."""
        if col in group.columns:
            return pd.to_numeric(group[col], errors="coerce")
        return pd.Series([default] * len(group), index=group.index, dtype="float64")

    n_comments = int(len(group))
    if n_comments <= 0:
        return None
    n_words = int(numeric_series("n_words_comment", default=0.0).fillna(0.0).sum())
    strict_total_hits = int(pd.to_numeric(group["strict_ai_word_hits_total"], errors="coerce").fillna(0.0).sum())
    extended_total_hits = int(pd.to_numeric(group["extended_ai_word_hits_total"], errors="coerce").fillna(0.0).sum())
    semicolon_total = float(numeric_series("semicolon_count", default=0.0).fillna(0.0).sum())
    em_dash_total = float(numeric_series("em_dash_count", default=0.0).fillna(0.0).sum())
    em_dash_extended_total = float(numeric_series("em_dash_extended_count", default=0.0).fillna(0.0).sum())
    en_dash_total = float(numeric_series("en_dash_count", default=0.0).fillna(0.0).sum())
    ascii_ddh_total = float(numeric_series("ascii_double_hyphen_count", default=0.0).fillna(0.0).sum())
    colon_total = float(numeric_series("colon_count", default=0.0).fillna(0.0).sum())
    colon_extended_total = float(numeric_series("colon_extended_count", default=0.0).fillna(0.0).sum())
    open_paren_total = float(numeric_series("open_paren_count", default=0.0).fillna(0.0).sum())
    curly_quote_total = float(numeric_series("curly_quote_count", default=0.0).fillna(0.0).sum())
    quote_all_total = float(numeric_series("quote_all_count", default=0.0).fillna(0.0).sum())
    quote_curly_share_num = float(numeric_series("quote_curly_share_num", default=0.0).fillna(0.0).sum())
    quote_curly_share_den = float(numeric_series("quote_curly_share_den", default=0.0).fillna(0.0).sum())
    url_total = float(numeric_series("url_count", default=0.0).fillna(0.0).sum())
    time_expression_total = float(numeric_series("time_expression_count", default=0.0).fillna(0.0).sum())
    md_bold_total = float(numeric_series("markdown_bold_pair_count", default=0.0).fillna(0.0).sum())
    md_head_total = float(numeric_series("markdown_heading_line_count", default=0.0).fillna(0.0).sum())
    hedging_total = float(numeric_series("hedging_phrase_hits", default=0.0).fillna(0.0).sum())
    polite_total = float(numeric_series("polite_closer_hits", default=0.0).fillna(0.0).sum())
    signpost_total = float(numeric_series("signposting_phrase_hits", default=0.0).fillna(0.0).sum())
    if "avg_words_per_sentence_comment" in group.columns:
        wps_series = pd.to_numeric(group["avg_words_per_sentence_comment"], errors="coerce")
    else:
        wps_series = pd.Series([float("nan")] * len(group), index=group.index, dtype="float64")
    avg_wps_mean = float(wps_series.dropna().mean()) if bool(wps_series.notna().any()) else float("nan")
    contraction_total = float(numeric_series("contraction_count", default=0.0).fillna(0.0).sum())
    full_form_total = float(numeric_series("full_form_count", default=0.0).fillna(0.0).sum())
    assistant_tone_total = float(numeric_series("assistant_tone_phrase_count", default=0.0).fillna(0.0).sum())
    toxic_lexicon_total = float(numeric_series("toxic_lexicon_hits", default=0.0).fillna(0.0).sum())
    sum_words = float(numeric_series("n_words_comment", default=0.0).fillna(0.0).sum())
    sum_word_chars = float(numeric_series("total_word_chars_comment", default=0.0).fillna(0.0).sum())
    sum_sentences = float(numeric_series("sentence_count_comment", default=0.0).fillna(0.0).sum())
    complexity_index = pem.compute_complexity_index(
        total_sentences=int(sum_sentences),
        total_words=int(sum_words),
        total_word_chars=int(sum_word_chars),
        n_comments=n_comments,
    )
    perplexity_series = numeric_series("perplexity")
    row: Dict[str, Any] = {
        "subreddit": subreddit,
        "date_utc": str(date_utc),
        "n_comments": n_comments,
        "n_words": n_words,
        "semicolon_rate_100w": pem.safe_rate_100w(semicolon_total, n_words),
        "em_dash_rate_100w": pem.safe_rate_100w(em_dash_total, n_words),
        "em_dash_extended_rate_100w": pem.safe_rate_100w(em_dash_extended_total, n_words),
        "en_dash_rate_100w": pem.safe_rate_100w(en_dash_total, n_words),
        "ascii_double_hyphen_rate_100w": pem.safe_rate_100w(ascii_ddh_total, n_words),
        "colon_rate_100w": pem.safe_rate_100w(colon_total, n_words),
        "colon_extended_rate_100w": pem.safe_rate_100w(colon_extended_total, n_words),
        "open_paren_rate_100w": pem.safe_rate_100w(open_paren_total, n_words),
        "curly_quote_rate_100w": pem.safe_rate_100w(curly_quote_total, n_words),
        "quote_all_rate_100w": pem.safe_rate_100w(quote_all_total, n_words),
        "quote_curly_share": float(quote_curly_share_num / quote_curly_share_den) if quote_curly_share_den > 0 else 0.0,
        "quote_curly_share_num": float(quote_curly_share_num),
        "quote_curly_share_den": float(quote_curly_share_den),
        "url_rate_100w": pem.safe_rate_100w(url_total, n_words),
        "time_expression_rate_100w": pem.safe_rate_100w(time_expression_total, n_words),
        "markdown_bold_pair_rate_100w": pem.safe_rate_100w(md_bold_total, n_words),
        "markdown_heading_line_rate_100w": pem.safe_rate_100w(md_head_total, n_words),
        "hedging_phrase_rate_100w": pem.safe_rate_100w(hedging_total, n_words),
        "polite_closer_rate_100w": pem.safe_rate_100w(polite_total, n_words),
        "signposting_phrase_rate_100w": pem.safe_rate_100w(signpost_total, n_words),
        "avg_words_per_sentence_mean": float(avg_wps_mean),
        "comment_length_words": float(pd.to_numeric(group["comment_length_words"], errors="coerce").fillna(0.0).mean()),
        "complexity_index": float(complexity_index),
        "ai_word_rate_100w": pem.safe_rate_100w(strict_total_hits, n_words),
        "ai_word_extended_rate_100w": pem.safe_rate_100w(extended_total_hits, n_words),
        "vader_compound_mean": float(numeric_series("vader_compound", default=0.0).fillna(0.0).mean()),
        "vader_negativity_mean": float(numeric_series("vader_negativity", default=0.0).fillna(0.0).mean()),
        "toxicity_score": float(numeric_series("toxicity_score_comment", default=0.0).fillna(0.0).mean()),
        "toxic_lexicon_rate_100w": pem.safe_rate_100w(toxic_lexicon_total, n_words),
        "contraction_rate_100w": pem.safe_rate_100w(contraction_total, n_words),
        "full_form_rate_100w": pem.safe_rate_100w(full_form_total, n_words),
        "formality_balance_100w": pem.safe_rate_100w(full_form_total - contraction_total, n_words),
        "list_structure_intensity": float(numeric_series("list_structure_flag", default=0.0).fillna(0.0).mean()),
        "assistant_tone_rate_100w": pem.safe_rate_100w(assistant_tone_total, n_words),
        "strict_ai_word_hits_total": strict_total_hits,
        "extended_ai_word_hits_total": extended_total_hits,
        "detector_primary_human_score": float(numeric_series("detector_primary_human_score").dropna().mean()),
        "detector_secondary_human_score": float(numeric_series("detector_secondary_human_score").dropna().mean()),
        "hostility_score": float(numeric_series("hostility_score").dropna().mean()),
        "emotion_anger": float(numeric_series("emotion_anger").dropna().mean()),
        "emotion_fear": float(numeric_series("emotion_fear").dropna().mean()),
        "emotion_sadness": float(numeric_series("emotion_sadness").dropna().mean()),
        "emotion_surprise": float(numeric_series("emotion_surprise").dropna().mean()),
        "passive_rate_100w": pem.safe_rate_100w(numeric_series("passive_count", default=0.0).fillna(0.0).sum(), n_words),
        "perplexity_mean": float(perplexity_series.dropna().mean()),
        "coverage_detector_primary": float(numeric_series("detector_primary_human_score").notna().mean()),
        "coverage_detector_secondary": float(numeric_series("detector_secondary_human_score").notna().mean()),
        "coverage_perplexity": float(perplexity_series.notna().mean()),
        "coverage_hostility": float(numeric_series("hostility_score").notna().mean()),
        "coverage_emotion": float(numeric_series("emotion_anger").notna().mean()),
        "detector_low_confidence_share": float((group["detector_confidence_flag"] == "low").mean())
        if "detector_confidence_flag" in group.columns
        else 0.0,
    }
    if user_series is not None:
        row["user_series"] = user_series
    if length_bucket is not None:
        row["length_bucket"] = length_bucket
    return row


def tag_frame_with_cohort(
    frame: pd.DataFrame,
    debut: pd.DataFrame,
    launch_ts: int,
) -> pd.DataFrame:
    """Function summary: left-join first_seen/debut and add user_cohort, is_debut_row, id_str columns.

    Parameters:
    - frame: one monthly comment_features shard.
    - debut: output of build_first_seen_debut_table.
    - launch_ts: unix seconds for launch_day_utc.

    Returns:
    - Copy of frame with tags; rows with missing author or created_utc get user_cohort NaN.
    """
    out = frame.copy()
    out["author"] = out["author"].astype("string").fillna("").str.strip() if "author" in out.columns else ""
    out["id_str"] = out["id"].astype("string").fillna("").astype(str) if "id" in out.columns else ""
    out["created_utc"] = pd.to_numeric(out.get("created_utc", float("nan")), errors="coerce")
    merged = out.merge(debut, on=["author", "subreddit"], how="left", suffixes=("", "_debut"))
    merged["user_cohort"] = pd.NA
    ok = merged["author"].astype(str).str.len() > 0
    ok &= merged["created_utc"].notna()
    ok &= merged["first_seen_ts"].notna()
    first_ts = pd.to_numeric(merged.loc[ok, "first_seen_ts"], errors="coerce")
    merged.loc[ok, "user_cohort"] = np.where(first_ts < int(launch_ts), "old", "new")
    merged["is_debut_row"] = False
    deb_ok = ok & (
        pd.to_numeric(merged["created_utc"], errors="coerce").astype("Int64")
        == pd.to_numeric(merged["first_seen_ts"], errors="coerce").astype("Int64")
    ) & (merged["id_str"].astype(str) == merged["debut_id"].astype(str))
    merged.loc[deb_ok, "is_debut_row"] = True
    return merged


def aggregate_stratified_month(
    frame: pd.DataFrame,
    subreddit: str,
    debut: pd.DataFrame,
    launch_ts: int,
    max_days_per_month: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Function summary: emit daily rows for user_series and length_bucket stratifications for one shard.

    Returns:
    - (user_series_rows, length_bucket_rows) lists of metric dicts.
    """
    user_rows: list[Dict[str, Any]] = []
    len_rows: list[Dict[str, Any]] = []
    required = {"date_utc", "n_words_comment", "strict_ai_word_hits_total", "extended_ai_word_hits_total"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing} for subreddit={subreddit}")
    if "subreddit" in frame.columns:
        frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
    if frame.empty:
        return user_rows, len_rows
    if max_days_per_month > 0:
        keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
        frame = frame[frame["date_utc"].isin(keep_days)].copy()
    if frame.empty:
        return user_rows, len_rows
    if "length_bucket" not in frame.columns:
        raise ValueError(f"Missing length_bucket in comment_features for subreddit={subreddit}")
    tagged = tag_frame_with_cohort(frame, debut, launch_ts)
    for date_utc, day in tagged.groupby("date_utc", sort=True):
        for series in USER_SERIES_ORDER:
            if series == "old":
                g = day[day["user_cohort"] == "old"]
            elif series == "new":
                g = day[day["user_cohort"] == "new"]
            else:
                g = day[day["is_debut_row"]]
            row = daily_metric_row_from_group(g, subreddit, str(date_utc), user_series=series, length_bucket=None)
            if row is not None:
                user_rows.append(row)
        for bucket in LENGTH_BUCKET_ORDER:
            g = day[day["length_bucket"].astype(str) == bucket]
            row = daily_metric_row_from_group(g, subreddit, str(date_utc), user_series=None, length_bucket=bucket)
            if row is not None:
                len_rows.append(row)
    return user_rows, len_rows


def pool_stratified_subreddit_days(
    metrics_df: pd.DataFrame,
    stratum_col: str,
    launch_ts: int,
) -> pd.DataFrame:
    """Function summary: pool across subreddits weighted by n_comments/n_words per (date_utc, stratum)."""
    if metrics_df.empty:
        return pd.DataFrame()
    pooled_rows: list[Dict[str, Any]] = []
    for (date_utc, strat), group in metrics_df.groupby(["date_utc", stratum_col], sort=True):
        n_comments = int(group["n_comments"].sum())
        n_words = int(group["n_words"].sum())
        if n_comments <= 0:
            continue
        pooled_rows.append(
            {
                "subreddit": "ALL",
                "date_utc": str(date_utc),
                stratum_col: str(strat),
                "n_comments": n_comments,
                "n_words": n_words,
                "semicolon_rate_100w": pem.safe_rate_100w(
                    (group["semicolon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                ),
                "em_dash_rate_100w": pem.safe_rate_100w((group["em_dash_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "em_dash_rate_100w" in group.columns
                else 0.0,
                "em_dash_extended_rate_100w": pem.safe_rate_100w(
                    (group["em_dash_extended_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "em_dash_extended_rate_100w" in group.columns
                else 0.0,
                "en_dash_rate_100w": pem.safe_rate_100w((group["en_dash_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "en_dash_rate_100w" in group.columns
                else 0.0,
                "ascii_double_hyphen_rate_100w": pem.safe_rate_100w(
                    (group["ascii_double_hyphen_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "ascii_double_hyphen_rate_100w" in group.columns
                else 0.0,
                "colon_rate_100w": pem.safe_rate_100w((group["colon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "colon_rate_100w" in group.columns
                else 0.0,
                "colon_extended_rate_100w": pem.safe_rate_100w(
                    (group["colon_extended_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "colon_extended_rate_100w" in group.columns
                else 0.0,
                "open_paren_rate_100w": pem.safe_rate_100w(
                    (group["open_paren_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "open_paren_rate_100w" in group.columns
                else 0.0,
                "curly_quote_rate_100w": pem.safe_rate_100w(
                    (group["curly_quote_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "curly_quote_rate_100w" in group.columns
                else 0.0,
                "quote_all_rate_100w": pem.safe_rate_100w(
                    (group["quote_all_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "quote_all_rate_100w" in group.columns
                else 0.0,
                "quote_curly_share_num": float(group["quote_curly_share_num"].sum()) if "quote_curly_share_num" in group.columns else 0.0,
                "quote_curly_share_den": float(group["quote_curly_share_den"].sum()) if "quote_curly_share_den" in group.columns else 0.0,
                "quote_curly_share": (
                    float(group["quote_curly_share_num"].sum()) / float(group["quote_curly_share_den"].sum())
                    if "quote_curly_share_num" in group.columns
                    and "quote_curly_share_den" in group.columns
                    and float(group["quote_curly_share_den"].sum()) > 0.0
                    else 0.0
                ),
                "url_rate_100w": pem.safe_rate_100w((group["url_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "url_rate_100w" in group.columns
                else 0.0,
                "time_expression_rate_100w": pem.safe_rate_100w(
                    (group["time_expression_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "time_expression_rate_100w" in group.columns
                else 0.0,
                "markdown_bold_pair_rate_100w": pem.safe_rate_100w(
                    (group["markdown_bold_pair_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "markdown_bold_pair_rate_100w" in group.columns
                else 0.0,
                "markdown_heading_line_rate_100w": pem.safe_rate_100w(
                    (group["markdown_heading_line_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "markdown_heading_line_rate_100w" in group.columns
                else 0.0,
                "hedging_phrase_rate_100w": pem.safe_rate_100w(
                    (group["hedging_phrase_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "hedging_phrase_rate_100w" in group.columns
                else 0.0,
                "polite_closer_rate_100w": pem.safe_rate_100w(
                    (group["polite_closer_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "polite_closer_rate_100w" in group.columns
                else 0.0,
                "signposting_phrase_rate_100w": pem.safe_rate_100w(
                    (group["signposting_phrase_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "signposting_phrase_rate_100w" in group.columns
                else 0.0,
                "avg_words_per_sentence_mean": float(
                    group["avg_words_per_sentence_mean"].mul(group["n_comments"]).sum() / float(n_comments)
                )
                if "avg_words_per_sentence_mean" in group.columns
                else float("nan"),
                "comment_length_words": float(group["comment_length_words"].mul(group["n_comments"]).sum())
                / float(n_comments),
                "complexity_index": float(group["complexity_index"].mul(group["n_comments"]).sum()) / float(n_comments),
                "ai_word_rate_100w": pem.safe_rate_100w(group["strict_ai_word_hits_total"].sum(), n_words),
                "ai_word_extended_rate_100w": pem.safe_rate_100w(group["extended_ai_word_hits_total"].sum(), n_words),
                "vader_compound_mean": float(group["vader_compound_mean"].mul(group["n_comments"]).sum()) / float(n_comments),
                "vader_negativity_mean": float(group["vader_negativity_mean"].mul(group["n_comments"]).sum())
                / float(n_comments),
                "toxicity_score": float(group["toxicity_score"].mul(group["n_comments"]).sum()) / float(n_comments),
                "toxic_lexicon_rate_100w": pem.safe_rate_100w(
                    (group["toxic_lexicon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                ),
                "contraction_rate_100w": pem.safe_rate_100w(
                    (group["contraction_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                ),
                "full_form_rate_100w": pem.safe_rate_100w(
                    (group["full_form_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                ),
                "formality_balance_100w": pem.safe_rate_100w(
                    ((group["full_form_rate_100w"] - group["contraction_rate_100w"]) * group["n_words"] / 100.0).sum(),
                    n_words,
                ),
                "list_structure_intensity": float(group["list_structure_intensity"].mul(group["n_comments"]).sum())
                / float(n_comments),
                "assistant_tone_rate_100w": pem.safe_rate_100w(
                    (group["assistant_tone_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                ),
                "strict_ai_word_hits_total": int(group["strict_ai_word_hits_total"].sum()),
                "extended_ai_word_hits_total": int(group["extended_ai_word_hits_total"].sum()),
                "detector_primary_human_score": float(
                    group["detector_primary_human_score"].mul(group["n_comments"]).sum() / float(n_comments)
                ),
                "detector_secondary_human_score": float(
                    group["detector_secondary_human_score"].mul(group["n_comments"]).sum() / float(n_comments)
                ),
                "passive_rate_100w": pem.safe_rate_100w(
                    (group["passive_rate_100w"] * group["n_words"] / 100.0).sum(),
                    n_words,
                ),
                "perplexity_mean": float(group["perplexity_mean"].mul(group["n_comments"]).sum() / float(n_comments)),
                "hostility_score": float(group["hostility_score"].mul(group["n_comments"]).sum() / float(n_comments)),
                "emotion_anger": float(group["emotion_anger"].mul(group["n_comments"]).sum() / float(n_comments)),
                "emotion_fear": float(group["emotion_fear"].mul(group["n_comments"]).sum() / float(n_comments)),
                "emotion_sadness": float(group["emotion_sadness"].mul(group["n_comments"]).sum() / float(n_comments)),
                "emotion_surprise": float(group["emotion_surprise"].mul(group["n_comments"]).sum() / float(n_comments)),
                "coverage_detector_primary": float(
                    group["coverage_detector_primary"].mul(group["n_comments"]).sum() / float(n_comments)
                ),
                "coverage_detector_secondary": float(
                    group["coverage_detector_secondary"].mul(group["n_comments"]).sum() / float(n_comments)
                ),
                "coverage_perplexity": float(group["coverage_perplexity"].mul(group["n_comments"]).sum() / float(n_comments)),
                "coverage_hostility": float(group["coverage_hostility"].mul(group["n_comments"]).sum() / float(n_comments)),
                "coverage_emotion": float(group["coverage_emotion"].mul(group["n_comments"]).sum() / float(n_comments)),
                "detector_low_confidence_share": float(
                    group["detector_low_confidence_share"].mul(group["n_comments"]).sum() / float(n_comments)
                ),
            }
        )
    out = pd.DataFrame(pooled_rows).sort_values(["date_utc", stratum_col]).reset_index(drop=True)
    out = pem.add_event_time_columns(out, launch_ts)
    out = add_ai_likeness_index_by_stratum(out, stratum_col)
    return out


def add_ai_likeness_index_by_stratum(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """Function summary: apply pem.add_ai_likeness_index within each stratum slice then concatenate."""
    if df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby(stratum_col, sort=False):
        parts.append(pem.add_ai_likeness_index(grp.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True).sort_values(["date_utc", stratum_col]).reset_index(drop=True)


def build_length_bucket_shares(pooled_length: pd.DataFrame) -> pd.DataFrame:
    """Function summary: compute pooled daily share of comments in each length_bucket (sums to 1 per day)."""
    if pooled_length.empty:
        return pd.DataFrame()
    rows: list[Dict[str, Any]] = []
    for date_utc, g in pooled_length.groupby("date_utc", sort=True):
        total = float(g["n_comments"].sum())
        if total <= 0:
            continue
        by_bucket = g.set_index("length_bucket")["n_comments"].reindex(LENGTH_BUCKET_ORDER).fillna(0.0).astype(float)
        row: Dict[str, Any] = {
            "date_utc": str(date_utc),
            "n_comments_total": int(total),
            "share_short": float(by_bucket.get("short", 0.0) / total),
            "share_medium": float(by_bucket.get("medium", 0.0) / total),
            "share_long": float(by_bucket.get("long", 0.0) / total),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date_utc").reset_index(drop=True)


def write_stratified_notes(path: Path, launch_day: str, start_day: str) -> None:
    """Function summary: write definitions, left-censoring, and explicit exclusion of repetition/Jaccard for stratified outputs."""
    lines = [
        "Stratified Event-Time Metrics Notes",
        "===================================",
        "",
        f"Launch anchor (UTC): {launch_day}",
        f"Earliest observed history in default pipeline (event_window.start_utc): {start_day}",
        "",
        "User cohort (per author, per subreddit):",
        "- old: first observed created_utc in that subreddit is strictly before launch.",
        "- new: first observed created_utc is on or after launch.",
        "- debut_observed: subset of rows that are the author's observed debut in that subreddit",
        "  (min created_utc, tie-break min id), regardless of old/new cohort.",
        "",
        "Left-censoring:",
        "- Cohorts use earliest appearance in comment_features only. True first posts before the corpus start can be missing,",
        "  so some users may be labeled new when they are not.",
        "",
        "Length buckets (from comment_features.length_bucket):",
        "- short: <20 words; medium: 20-49; long: >=50.",
        "",
        "Stratified tables intentionally omit repetition_template_similarity (Jaccard stream metric).",
        "",
        "Lexical extensions (em/en dash, ASCII double-hyphen, punctuation/markdown proxies, hedging/polite/signposting phrase rates,",
        "avg_words_per_sentence_mean) mirror prepare_event_time_metrics.py column names where present in comment_features shards.",
        "Strict AI basket is the stem-aware top-10 from prepare_event_time_metrics.py (entries with `*` match Porter stems).",
        "",
        "Figures (plot_event_time_stratified_metrics.py):",
        "- Saved under results/figures/event_time/stratified_pooled/user_series/ and .../length_bucket/, each with daily/weekly/rolling_daily/.",
        "- Length-bucket plots omit detector, perplexity, hostility, emotion, and coverage metrics (not meaningful when stratifier is length).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: orchestrate debut scan, stratified aggregation, pooled exports, and notes."""
    args = parse_args()
    config = load_config(args.config)
    launch_ts = utc_ts(str(config["event_window"]["launch_day_utc"]))
    start_ts = utc_ts(str(config["event_window"]["start_utc"]))
    launch_day = datetime.fromtimestamp(launch_ts, tz=timezone.utc).date().isoformat()
    start_day = datetime.fromtimestamp(start_ts, tz=timezone.utc).date().isoformat()
    subreddits = list(config["subreddits"]["primary"])
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    event_time_dir = tables_dir / "event_time"
    event_time_dir.mkdir(parents=True, exist_ok=True)
    comment_features_dir = interim_dir / "comment_features"
    if not comment_features_dir.is_dir():
        raise FileNotFoundError(f"comment_features not found: {comment_features_dir}")

    jobs = iter_monthly_jobs(
        comment_features_dir,
        subreddits,
        int(args.max_month_files_per_subreddit),
        int(args.max_total_month_files),
    )
    if not jobs:
        raise FileNotFoundError(f"No parquet under {comment_features_dir}")

    t0 = time.perf_counter()
    print("[prepare_event_time_stratified_metrics] phase=first_seen_scan", flush=True)
    debut = build_first_seen_debut_table(jobs)
    print(f"[prepare_event_time_stratified_metrics] debut_keys={len(debut)} elapsed_s={time.perf_counter() - t0:.1f}", flush=True)

    user_rows: list[Dict[str, Any]] = []
    len_rows: list[Dict[str, Any]] = []
    for subreddit, file_path in jobs:
        print(f"[prepare_event_time_stratified_metrics] aggregate subreddit={subreddit} month={file_path.stem}", flush=True)
        frame = pd.read_parquet(file_path)
        if frame.empty:
            continue
        u_part, l_part = aggregate_stratified_month(
            frame,
            subreddit,
            debut,
            launch_ts,
            int(args.max_days_per_month),
        )
        user_rows.extend(u_part)
        len_rows.extend(l_part)

    user_sub = pd.DataFrame(user_rows).sort_values(["subreddit", "user_series", "date_utc"]).reset_index(drop=True)
    len_sub = pd.DataFrame(len_rows).sort_values(["subreddit", "length_bucket", "date_utc"]).reset_index(drop=True)

    pooled_user = pool_stratified_subreddit_days(user_sub, "user_series", launch_ts)
    pooled_len = pool_stratified_subreddit_days(len_sub, "length_bucket", launch_ts)
    shares = build_length_bucket_shares(pooled_len)
    if not shares.empty:
        shares = pem.add_event_time_columns(shares, launch_ts)

    user_out = event_time_dir / "event_time_daily_metrics_pooled_by_user_cohort.csv"
    len_out = event_time_dir / "event_time_daily_metrics_pooled_by_length_bucket.csv"
    share_out = event_time_dir / "event_time_length_bucket_daily_shares_pooled.csv"
    pooled_user.to_csv(user_out, index=False)
    pooled_len.to_csv(len_out, index=False)
    shares.to_csv(share_out, index=False)
    write_stratified_notes(event_time_dir / "event_time_stratified_metrics_notes.txt", launch_day, start_day)

    if bool(args.profile) or str(args.profile_output or "").strip():
        payload = {
            "debut_keys": int(len(debut)),
            "user_subreddit_day_rows": int(len(user_sub)),
            "length_subreddit_day_rows": int(len(len_sub)),
            "pooled_user_rows": int(len(pooled_user)),
            "pooled_length_rows": int(len(pooled_len)),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[prepare_event_time_stratified_metrics] profile={json.dumps(payload, sort_keys=True)}", flush=True)
        if str(args.profile_output or "").strip():
            Path(str(args.profile_output)).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[prepare_event_time_stratified_metrics] wrote {user_out}", flush=True)
    print(f"[prepare_event_time_stratified_metrics] wrote {len_out}", flush=True)
    print(f"[prepare_event_time_stratified_metrics] wrote {share_out}", flush=True)


if __name__ == "__main__":
    main()
