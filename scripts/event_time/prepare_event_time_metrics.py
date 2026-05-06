"""
Script summary:
This script prepares metric-ready daily event-time aggregates from reusable
`comment_features/` Parquet shards (per comment, precomputed lexical and ML fields).
It aggregates to subreddit-day and pooled series, merges optional daily
repetition/template similarity from `compute_daily_repetition_similarity.py`, and
writes CSV tables used by event-time plots.

Functionality:
- Requires `data/interim/.../comment_features/<subreddit>/<YYYY-MM>.parquet` (run
  `compute_comment_features.py` or `merge_ml_shards_into_comment_features.py` first).
- Left-merges `results/tables/event_time/repetition_daily_by_subreddit.csv` when
  present (otherwise `repetition_template_similarity` is all NaN); generate that file
  with `compute_daily_repetition_similarity.py`.
- Computes pooled metrics, AI-likeness index, and exports under `results/tables/event_time/`.
- Writes compatibility export `results/tables/event_time_daily_metrics.csv`.

How to apply/run:
- `.venv/bin/python scripts/event_time/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`
- Bounded benchmark example:
  `.venv/bin/python scripts/event_time/prepare_event_time_metrics.py --config config/political_forums_setup.yaml --max_month_files_per_subreddit 1 --max_days_per_month 10 --profile`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys
import time
from typing import Any, Dict, Iterable

import pandas as pd

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

from src.config_utils import load_config, utc_ts

STRICT_AI_WORDS = [
    "delv*",
    "realm*",
    "landscape",
    "testament",
    "leverage",
    "meticul*",
    "intric*",
    "crucial*",
    "underscore*",
    "showcas*",
]


@dataclass
class RuntimePaths:
    """Function summary: store resolved runtime input and output paths for this script."""

    comment_features_dir: Path
    event_time_tables_dir: Path
    tables_dir: Path


@dataclass
class ProfilingStats:
    """Function summary: store cumulative runtime counters and workload volume for phase-level profiling."""

    phase_read_s: float = 0.0
    phase_validate_s: float = 0.0
    phase_aggregate_s: float = 0.0
    phase_postprocess_s: float = 0.0
    phase_write_s: float = 0.0
    files_processed: int = 0
    days_processed: int = 0
    comments_processed: int = 0
    rows_read: int = 0

    def merge(self, other: "ProfilingStats") -> None:
        """Function summary: merge another profiling payload into this instance."""
        self.phase_read_s += float(other.phase_read_s)
        self.phase_validate_s += float(other.phase_validate_s)
        self.phase_aggregate_s += float(other.phase_aggregate_s)
        self.phase_postprocess_s += float(other.phase_postprocess_s)
        self.phase_write_s += float(other.phase_write_s)
        self.files_processed += int(other.files_processed)
        self.days_processed += int(other.days_processed)
        self.comments_processed += int(other.comments_processed)
        self.rows_read += int(other.rows_read)

    def as_dict(self) -> Dict[str, Any]:
        """Function summary: return a stable dictionary representation for logs and optional export."""
        seconds_per_100k_comments = 0.0
        if self.comments_processed > 0:
            core_total = self.phase_read_s + self.phase_validate_s + self.phase_aggregate_s + self.phase_postprocess_s + self.phase_write_s
            seconds_per_100k_comments = (core_total / float(self.comments_processed)) * 100000.0
        return {
            "phase_read_s": round(self.phase_read_s, 4),
            "phase_validate_s": round(self.phase_validate_s, 4),
            "phase_aggregate_s": round(self.phase_aggregate_s, 4),
            "phase_postprocess_s": round(self.phase_postprocess_s, 4),
            "phase_write_s": round(self.phase_write_s, 4),
            "files_processed": int(self.files_processed),
            "days_processed": int(self.days_processed),
            "comments_processed": int(self.comments_processed),
            "rows_read": int(self.rows_read),
            "seconds_per_100k_comments": round(seconds_per_100k_comments, 4),
        }


def parse_args() -> argparse.Namespace:
    """Function summary: parse command line options for config path and bounded benchmark controls."""
    parser = argparse.ArgumentParser(description="Prepare event-time metric tables from comment_features shards.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--max_month_files_per_subreddit",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process at most this many monthly files per subreddit.",
    )
    parser.add_argument(
        "--max_total_month_files",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process at most this many monthly files total.",
    )
    parser.add_argument(
        "--max_days_per_month",
        type=int,
        default=0,
        help="Bounded benchmark control: if >0, process only the first N days per monthly file.",
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
    """Function summary: resolve configured locations and ensure event-time output folders exist."""
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    event_time_tables_dir = tables_dir / "event_time"
    event_time_tables_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        comment_features_dir=interim_dir / "comment_features",
        event_time_tables_dir=event_time_tables_dir,
        tables_dir=tables_dir,
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


def safe_rate_100w(count: float, n_words: int) -> float:
    """Function summary: compute per-100-word rate with explicit zero-denominator handling."""
    if n_words <= 0:
        return 0.0
    return (float(count) / float(n_words)) * 100.0


def weighted_mean_nullable(values: pd.Series, weights: pd.Series) -> float:
    """Function summary: compute weighted mean over non-null values, returning NaN when no valid weight remains."""
    value_num = pd.to_numeric(values, errors="coerce")
    weight_num = pd.to_numeric(weights, errors="coerce")
    valid = value_num.notna() & weight_num.notna() & (weight_num > 0)
    if not bool(valid.any()):
        return float("nan")
    denom = float(weight_num.loc[valid].sum())
    if denom <= 0:
        return float("nan")
    numer = float((value_num.loc[valid] * weight_num.loc[valid]).sum())
    return numer / denom


def compute_complexity_index(total_sentences: int, total_words: int, total_word_chars: int, n_comments: int) -> float:
    """Function summary: compute a stable lexical/syntactic complexity proxy from daily aggregates."""
    if n_comments <= 0 or total_words <= 0:
        return 0.0
    mean_sentence_length = float(total_words) / float(max(total_sentences, 1))
    mean_word_length = float(total_word_chars) / float(total_words)
    return 0.5 * mean_sentence_length + 0.5 * mean_word_length


def aggregate_daily_metrics_from_comment_features(
    comment_features_dir: Path,
    subreddits: list[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
    max_days_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame, ProfilingStats, pd.DataFrame]:
    """Function summary: aggregate daily subreddit metrics directly from reusable comment-level feature parquet shards."""
    rows: list[Dict[str, Any]] = []
    ai_word_long_rows: list[Dict[str, Any]] = []
    validation_rows: list[Dict[str, Any]] = []
    stats = ProfilingStats()
    month_jobs = list(
        iter_monthly_files(
            comment_features_dir,
            subreddits,
            max_month_files_per_subreddit=max_month_files_per_subreddit,
            max_total_month_files=max_total_month_files,
        )
    )
    for subreddit, file_path in month_jobs:
        print(
            f"[prepare_event_time_metrics] feature_source_start subreddit={subreddit} month={file_path.stem}",
            flush=True,
        )
        t0 = time.perf_counter()
        frame = pd.read_parquet(file_path)
        stats.phase_read_s += time.perf_counter() - t0
        stats.rows_read += int(len(frame))
        if frame.empty:
            print(
                f"[prepare_event_time_metrics] feature_source_empty subreddit={subreddit} month={file_path.stem}",
                flush=True,
            )
            continue
        stats.files_processed += 1
        required = {"date_utc", "n_words_comment", "strict_ai_word_hits_total", "extended_ai_word_hits_total"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns {missing} in comment feature file: {file_path}")
        if "subreddit" in frame.columns:
            frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
        if frame.empty:
            continue
        if max_days_per_month > 0:
            keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
            frame = frame[frame["date_utc"].isin(keep_days)].copy()
        if frame.empty:
            continue

        for date_utc, group in frame.groupby("date_utc", sort=True):
            def numeric_series(col: str, default: float = 0.0) -> pd.Series:
                """Function summary: return one numeric pandas series for a column or a default-valued fallback series."""
                if col in group.columns:
                    return pd.to_numeric(group[col], errors="coerce")
                return pd.Series([default] * len(group), index=group.index, dtype="float64")

            n_comments = int(len(group))
            n_words = int(numeric_series("n_words_comment", default=0.0).fillna(0.0).sum())
            if n_comments <= 0:
                continue
            strict_total_hits = int(pd.to_numeric(group["strict_ai_word_hits_total"], errors="coerce").fillna(0.0).sum())
            extended_total_hits = int(pd.to_numeric(group["extended_ai_word_hits_total"], errors="coerce").fillna(0.0).sum())
            semicolon_total = float(numeric_series("semicolon_count", default=0.0).fillna(0.0).sum())
            semicolon_extended_total = float(numeric_series("semicolon_extended_count", default=0.0).fillna(0.0).sum())
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
            complexity_index = compute_complexity_index(
                total_sentences=int(sum_sentences),
                total_words=int(sum_words),
                total_word_chars=int(sum_word_chars),
                n_comments=n_comments,
            )
            perplexity_series = numeric_series("perplexity")
            row = {
                "subreddit": subreddit,
                "date_utc": str(date_utc),
                "n_comments": n_comments,
                "n_words": n_words,
                "semicolon_rate_100w": safe_rate_100w(semicolon_total, n_words),
                "semicolon_extended_rate_100w": safe_rate_100w(semicolon_extended_total, n_words),
                "em_dash_rate_100w": safe_rate_100w(em_dash_total, n_words),
                "em_dash_extended_rate_100w": safe_rate_100w(em_dash_extended_total, n_words),
                "en_dash_rate_100w": safe_rate_100w(en_dash_total, n_words),
                "ascii_double_hyphen_rate_100w": safe_rate_100w(ascii_ddh_total, n_words),
                "colon_rate_100w": safe_rate_100w(colon_total, n_words),
                "colon_extended_rate_100w": safe_rate_100w(colon_extended_total, n_words),
                "open_paren_rate_100w": safe_rate_100w(open_paren_total, n_words),
                "curly_quote_rate_100w": safe_rate_100w(curly_quote_total, n_words),
                "quote_all_rate_100w": safe_rate_100w(quote_all_total, n_words),
                "quote_curly_share": float(quote_curly_share_num / quote_curly_share_den) if quote_curly_share_den > 0 else 0.0,
                "quote_curly_share_num": float(quote_curly_share_num),
                "quote_curly_share_den": float(quote_curly_share_den),
                "url_rate_100w": safe_rate_100w(url_total, n_words),
                "time_expression_rate_100w": safe_rate_100w(time_expression_total, n_words),
                "markdown_bold_pair_rate_100w": safe_rate_100w(md_bold_total, n_words),
                "markdown_heading_line_rate_100w": safe_rate_100w(md_head_total, n_words),
                "hedging_phrase_rate_100w": safe_rate_100w(hedging_total, n_words),
                "polite_closer_rate_100w": safe_rate_100w(polite_total, n_words),
                "signposting_phrase_rate_100w": safe_rate_100w(signpost_total, n_words),
                "avg_words_per_sentence_mean": float(avg_wps_mean),
                "comment_length_words": float(pd.to_numeric(group["comment_length_words"], errors="coerce").fillna(0.0).mean()),
                "complexity_index": float(complexity_index),
                "ai_word_rate_100w": safe_rate_100w(strict_total_hits, n_words),
                "ai_word_extended_rate_100w": safe_rate_100w(extended_total_hits, n_words),
                "vader_compound_mean": float(numeric_series("vader_compound", default=0.0).fillna(0.0).mean()),
                "vader_negativity_mean": float(numeric_series("vader_negativity", default=0.0).fillna(0.0).mean()),
                "toxicity_score": float(numeric_series("toxicity_score_comment", default=0.0).fillna(0.0).mean()),
                "toxic_lexicon_rate_100w": safe_rate_100w(toxic_lexicon_total, n_words),
                "contraction_rate_100w": safe_rate_100w(contraction_total, n_words),
                "full_form_rate_100w": safe_rate_100w(full_form_total, n_words),
                "formality_balance_100w": safe_rate_100w(full_form_total - contraction_total, n_words),
                "list_structure_intensity": float(numeric_series("list_structure_flag", default=0.0).fillna(0.0).mean()),
                "assistant_tone_rate_100w": safe_rate_100w(assistant_tone_total, n_words),
                "strict_ai_word_hits_total": strict_total_hits,
                "extended_ai_word_hits_total": extended_total_hits,
                "detector_primary_human_score": float(
                    numeric_series("detector_primary_human_score").dropna().mean()
                ),
                "detector_secondary_human_score": float(
                    numeric_series("detector_secondary_human_score").dropna().mean()
                ),
                "hostility_score": float(numeric_series("hostility_score").dropna().mean()),
                "emotion_anger": float(numeric_series("emotion_anger").dropna().mean()),
                "emotion_fear": float(numeric_series("emotion_fear").dropna().mean()),
                "emotion_sadness": float(numeric_series("emotion_sadness").dropna().mean()),
                "emotion_surprise": float(numeric_series("emotion_surprise").dropna().mean()),
                "passive_rate_100w": safe_rate_100w(
                    numeric_series("passive_count", default=0.0).fillna(0.0).sum(), n_words
                ),
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
            rows.append(row)

            for word in STRICT_AI_WORDS:
                col_name = f"strict_word_count__{word}"
                hits = int(numeric_series(col_name, default=0.0).fillna(0.0).sum())
                ai_word_long_rows.append(
                    {
                        "subreddit": subreddit,
                        "date_utc": str(date_utc),
                        "word": word,
                        "word_group": "strict_individual",
                        "hits": hits,
                        "rate_100w": safe_rate_100w(hits, n_words),
                        "n_words": n_words,
                    }
                )
            ai_word_long_rows.append(
                {
                    "subreddit": subreddit,
                    "date_utc": str(date_utc),
                    "word": "strict_10_combined",
                    "word_group": "strict_combined",
                    "hits": strict_total_hits,
                    "rate_100w": safe_rate_100w(strict_total_hits, n_words),
                    "n_words": n_words,
                }
            )
            ai_word_long_rows.append(
                {
                    "subreddit": subreddit,
                    "date_utc": str(date_utc),
                    "word": "extended_combined",
                    "word_group": "extended_combined",
                    "hits": extended_total_hits,
                    "rate_100w": safe_rate_100w(extended_total_hits, n_words),
                    "n_words": n_words,
                }
            )

        stats.days_processed += int(frame["date_utc"].nunique())
        stats.comments_processed += int(len(frame))
        assoc_subset = frame[["comment_length_words", "passive_rate_100w", "perplexity", "detector_primary_human_score"]].copy()
        if not assoc_subset.empty:
            assoc_corr = assoc_subset.corr(numeric_only=True)
            validation_rows.append(
                {
                    "subreddit": subreddit,
                    "month": file_path.stem,
                    "corr_human_vs_length": float(assoc_corr.loc["detector_primary_human_score", "comment_length_words"])
                    if "detector_primary_human_score" in assoc_corr.index and "comment_length_words" in assoc_corr.columns
                    else float("nan"),
                    "corr_human_vs_passive": float(assoc_corr.loc["detector_primary_human_score", "passive_rate_100w"])
                    if "detector_primary_human_score" in assoc_corr.index and "passive_rate_100w" in assoc_corr.columns
                    else float("nan"),
                    "corr_human_vs_perplexity": float(assoc_corr.loc["detector_primary_human_score", "perplexity"])
                    if "detector_primary_human_score" in assoc_corr.index and "perplexity" in assoc_corr.columns
                    else float("nan"),
                    "n_comments": int(len(frame)),
                }
            )
        print(
            f"[prepare_event_time_metrics] feature_source_done subreddit={subreddit} month={file_path.stem} "
            f"days={int(frame['date_utc'].nunique())} comments={int(len(frame))}",
            flush=True,
        )
    metrics_df = pd.DataFrame(rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True) if rows else pd.DataFrame()
    ai_word_long_df = (
        pd.DataFrame(ai_word_long_rows).sort_values(["subreddit", "date_utc", "word_group", "word"]).reset_index(drop=True)
        if ai_word_long_rows
        else pd.DataFrame()
    )
    validation_df = pd.DataFrame(validation_rows).sort_values(["subreddit", "month"]).reset_index(drop=True) if validation_rows else pd.DataFrame()
    return metrics_df, ai_word_long_df, stats, validation_df


def add_event_time_columns(df: pd.DataFrame, launch_ts: int) -> pd.DataFrame:
    """Function summary: attach date and event-time offset columns relative to launch day."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    launch_date = datetime.fromtimestamp(launch_ts, tz=timezone.utc).replace(tzinfo=None)
    out["event_time_t"] = (out["date"] - launch_date).dt.days.astype(int)
    return out


def zscore(series: pd.Series) -> pd.Series:
    """Function summary: compute z-score with zero-variance guard to keep deterministic outputs."""
    std = float(series.std(ddof=0))
    if std == 0.0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - float(series.mean())) / std


def add_ai_likeness_index(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: compute transparent AI-likeness composite from normalized component metrics."""
    out = df.copy()
    out["z_ai_word_rate_100w"] = zscore(out["ai_word_rate_100w"])
    out["z_formality_balance_100w"] = zscore(out["formality_balance_100w"])
    out["z_assistant_tone_rate_100w"] = zscore(out["assistant_tone_rate_100w"])
    out["z_list_structure_intensity"] = zscore(out["list_structure_intensity"])
    out["z_contraction_rate_100w"] = zscore(out["contraction_rate_100w"])
    out["ai_likeness_index"] = (
        out["z_ai_word_rate_100w"]
        + out["z_formality_balance_100w"]
        + out["z_assistant_tone_rate_100w"]
        + out["z_list_structure_intensity"]
        - out["z_contraction_rate_100w"]
    )
    return out


def build_pooled_daily(metrics_df: pd.DataFrame, ai_word_long_df: pd.DataFrame, launch_ts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function summary: aggregate subreddit-day metrics into pooled day series and pooled word-rate long table."""
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pooled_rows: list[Dict[str, Any]] = []
    for date_utc, group in metrics_df.groupby("date_utc", sort=True):
        n_comments = int(group["n_comments"].sum())
        n_words = int(group["n_words"].sum())
        if n_comments <= 0:
            continue
        rep_pooled = float("nan")
        if "repetition_template_similarity" in group.columns:
            rep_pooled = weighted_mean_nullable(group["repetition_template_similarity"], group["n_comments"])
        pooled_rows.append(
            {
                "subreddit": "ALL",
                "date_utc": str(date_utc),
                "n_comments": n_comments,
                "n_words": n_words,
                "semicolon_rate_100w": safe_rate_100w((group["semicolon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "semicolon_extended_rate_100w": safe_rate_100w(
                    (group["semicolon_extended_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "semicolon_extended_rate_100w" in group.columns
                else 0.0,
                "em_dash_rate_100w": safe_rate_100w((group["em_dash_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "em_dash_rate_100w" in group.columns
                else 0.0,
                "em_dash_extended_rate_100w": safe_rate_100w(
                    (group["em_dash_extended_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "em_dash_extended_rate_100w" in group.columns
                else 0.0,
                "en_dash_rate_100w": safe_rate_100w((group["en_dash_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "en_dash_rate_100w" in group.columns
                else 0.0,
                "ascii_double_hyphen_rate_100w": safe_rate_100w(
                    (group["ascii_double_hyphen_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "ascii_double_hyphen_rate_100w" in group.columns
                else 0.0,
                "colon_rate_100w": safe_rate_100w((group["colon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "colon_rate_100w" in group.columns
                else 0.0,
                "colon_extended_rate_100w": safe_rate_100w(
                    (group["colon_extended_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "colon_extended_rate_100w" in group.columns
                else 0.0,
                "open_paren_rate_100w": safe_rate_100w((group["open_paren_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "open_paren_rate_100w" in group.columns
                else 0.0,
                "curly_quote_rate_100w": safe_rate_100w((group["curly_quote_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "curly_quote_rate_100w" in group.columns
                else 0.0,
                "quote_all_rate_100w": safe_rate_100w((group["quote_all_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
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
                "url_rate_100w": safe_rate_100w((group["url_rate_100w"] * group["n_words"] / 100.0).sum(), n_words)
                if "url_rate_100w" in group.columns
                else 0.0,
                "time_expression_rate_100w": safe_rate_100w(
                    (group["time_expression_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "time_expression_rate_100w" in group.columns
                else 0.0,
                "markdown_bold_pair_rate_100w": safe_rate_100w(
                    (group["markdown_bold_pair_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "markdown_bold_pair_rate_100w" in group.columns
                else 0.0,
                "markdown_heading_line_rate_100w": safe_rate_100w(
                    (group["markdown_heading_line_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "markdown_heading_line_rate_100w" in group.columns
                else 0.0,
                "hedging_phrase_rate_100w": safe_rate_100w(
                    (group["hedging_phrase_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "hedging_phrase_rate_100w" in group.columns
                else 0.0,
                "polite_closer_rate_100w": safe_rate_100w(
                    (group["polite_closer_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "polite_closer_rate_100w" in group.columns
                else 0.0,
                "signposting_phrase_rate_100w": safe_rate_100w(
                    (group["signposting_phrase_rate_100w"] * group["n_words"] / 100.0).sum(), n_words
                )
                if "signposting_phrase_rate_100w" in group.columns
                else 0.0,
                "avg_words_per_sentence_mean": weighted_mean_nullable(
                    group["avg_words_per_sentence_mean"], group["n_comments"]
                )
                if "avg_words_per_sentence_mean" in group.columns
                else float("nan"),
                "comment_length_words": float(group["comment_length_words"].mul(group["n_comments"]).sum()) / float(n_comments),
                "complexity_index": float(group["complexity_index"].mul(group["n_comments"]).sum()) / float(n_comments),
                "ai_word_rate_100w": safe_rate_100w(group["strict_ai_word_hits_total"].sum(), n_words),
                "ai_word_extended_rate_100w": safe_rate_100w(group["extended_ai_word_hits_total"].sum(), n_words),
                "vader_compound_mean": float(group["vader_compound_mean"].mul(group["n_comments"]).sum()) / float(n_comments),
                "vader_negativity_mean": float(group["vader_negativity_mean"].mul(group["n_comments"]).sum()) / float(n_comments),
                "toxicity_score": float(group["toxicity_score"].mul(group["n_comments"]).sum()) / float(n_comments),
                "toxic_lexicon_rate_100w": safe_rate_100w((group["toxic_lexicon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "contraction_rate_100w": safe_rate_100w((group["contraction_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "full_form_rate_100w": safe_rate_100w((group["full_form_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "formality_balance_100w": safe_rate_100w(
                    ((group["full_form_rate_100w"] - group["contraction_rate_100w"]) * group["n_words"] / 100.0).sum(),
                    n_words,
                ),
                "list_structure_intensity": float(group["list_structure_intensity"].mul(group["n_comments"]).sum()) / float(n_comments),
                "repetition_template_similarity": rep_pooled,
                "assistant_tone_rate_100w": safe_rate_100w((group["assistant_tone_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "strict_ai_word_hits_total": int(group["strict_ai_word_hits_total"].sum()),
                "extended_ai_word_hits_total": int(group["extended_ai_word_hits_total"].sum()),
                "detector_primary_human_score": weighted_mean_nullable(
                    group["detector_primary_human_score"], group["n_comments"]
                )
                if "detector_primary_human_score" in group.columns
                else float("nan"),
                "detector_secondary_human_score": weighted_mean_nullable(
                    group["detector_secondary_human_score"], group["n_comments"]
                )
                if "detector_secondary_human_score" in group.columns
                else float("nan"),
                "passive_rate_100w": safe_rate_100w(
                    (group["passive_rate_100w"] * group["n_words"] / 100.0).sum(),
                    n_words,
                )
                if "passive_rate_100w" in group.columns
                else float("nan"),
                "perplexity_mean": weighted_mean_nullable(group["perplexity_mean"], group["n_comments"])
                if "perplexity_mean" in group.columns
                else float("nan"),
                "hostility_score": weighted_mean_nullable(group["hostility_score"], group["n_comments"])
                if "hostility_score" in group.columns
                else float("nan"),
                "emotion_anger": weighted_mean_nullable(group["emotion_anger"], group["n_comments"])
                if "emotion_anger" in group.columns
                else float("nan"),
                "emotion_fear": weighted_mean_nullable(group["emotion_fear"], group["n_comments"])
                if "emotion_fear" in group.columns
                else float("nan"),
                "emotion_sadness": weighted_mean_nullable(group["emotion_sadness"], group["n_comments"])
                if "emotion_sadness" in group.columns
                else float("nan"),
                "emotion_surprise": weighted_mean_nullable(group["emotion_surprise"], group["n_comments"])
                if "emotion_surprise" in group.columns
                else float("nan"),
                "coverage_detector_primary": float(
                    group["coverage_detector_primary"].mul(group["n_comments"]).sum() / float(n_comments)
                )
                if "coverage_detector_primary" in group.columns
                else float("nan"),
                "coverage_detector_secondary": float(
                    group["coverage_detector_secondary"].mul(group["n_comments"]).sum() / float(n_comments)
                )
                if "coverage_detector_secondary" in group.columns
                else float("nan"),
                "coverage_perplexity": float(group["coverage_perplexity"].mul(group["n_comments"]).sum() / float(n_comments))
                if "coverage_perplexity" in group.columns
                else float("nan"),
                "coverage_hostility": float(group["coverage_hostility"].mul(group["n_comments"]).sum() / float(n_comments))
                if "coverage_hostility" in group.columns
                else float("nan"),
                "coverage_emotion": float(group["coverage_emotion"].mul(group["n_comments"]).sum() / float(n_comments))
                if "coverage_emotion" in group.columns
                else float("nan"),
                "detector_low_confidence_share": float(
                    group["detector_low_confidence_share"].mul(group["n_comments"]).sum() / float(n_comments)
                )
                if "detector_low_confidence_share" in group.columns
                else float("nan"),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows).sort_values("date_utc").reset_index(drop=True)
    pooled_df = add_event_time_columns(pooled_df, launch_ts)
    pooled_df = add_ai_likeness_index(pooled_df)

    pooled_word_long = (
        ai_word_long_df.groupby(["date_utc", "word", "word_group"], as_index=False)[["hits", "n_words"]]
        .sum()
        .assign(subreddit="ALL")
    )
    pooled_word_long["rate_100w"] = 0.0
    nonzero_mask = pooled_word_long["n_words"] > 0
    pooled_word_long.loc[nonzero_mask, "rate_100w"] = (
        pooled_word_long.loc[nonzero_mask, "hits"] / pooled_word_long.loc[nonzero_mask, "n_words"] * 100.0
    )
    pooled_word_long = add_event_time_columns(pooled_word_long, launch_ts)
    pooled_word_long = pooled_word_long.sort_values(["date_utc", "word_group", "word"]).reset_index(drop=True)
    return pooled_df, pooled_word_long


def emit_profiling(stats: ProfilingStats, profile: bool, profile_output: str) -> None:
    """Function summary: print and optionally write profiling metrics for bounded baseline comparisons."""
    if not profile and not profile_output:
        return
    payload = stats.as_dict()
    print(f"[prepare_event_time_metrics] profile={json.dumps(payload, sort_keys=True)}", flush=True)
    if profile_output:
        Path(profile_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_notes(path: Path) -> None:
    """Function summary: write concise metric definitions and interpretation caveats for reproducibility."""
    lines = [
        "Event-Time Metrics Notes",
        "========================",
        "",
        "Primary AI word list (strict_10; stem-aware where * is present):",
        "- delv*, realm*, landscape, testament, leverage, meticul*, intric*, crucial*, underscore*, showcas*",
        "- Stem rule: entries ending with `*` are matched as Porter stems (e.g., `delv*` -> delve/delved/delving).",
        "",
        "Toxicity channels:",
        "- toxicity_score: VADER negativity mean = mean(max(0, -compound)).",
        "- toxic_lexicon_rate_100w: lexical incidence per 100 words from a lightweight toxic lexicon.",
        "- hostility_score: classifier-derived hostility probability mean (when comment_features are available).",
        "",
        "Additional comment-feature metrics (when available):",
        "- detector_primary_human_score / detector_secondary_human_score: detector-based human-likelihood means.",
        "- passive_rate_100w: passive-construction proxy rate per 100 words.",
        "- em_dash_rate_100w / en_dash_rate_100w: Unicode U+2014 / U+2013 counts per 100 words.",
        "- ascii_double_hyphen_rate_100w: count of spaced ` -- ` tokens per 100 words.",
        "- colon_rate_100w / open_paren_rate_100w: punctuation density (colon inflated by URLs/timestamps).",
        "- curly_quote_rate_100w: curly quote characters per 100 words.",
        "- markdown_bold_pair_rate_100w / markdown_heading_line_rate_100w: **...** spans and ATX heading lines per 100 words.",
        "- hedging_phrase_rate_100w / polite_closer_rate_100w / signposting_phrase_rate_100w: disjoint phrase-list hits per 100 words.",
        "- avg_words_per_sentence_mean: mean of per-comment words/sentence (NaN-skipped) per day.",
        "- perplexity_mean: average language-model perplexity.",
        "- emotion_anger/fear/sadness/surprise: mean emotion classifier scores.",
        "- coverage_* columns: per-day non-null coverage by metric.",
        "",
        "AI-likeness index (z-score composite):",
        "- z(ai_word_rate_100w) + z(formality_balance_100w) + z(assistant_tone_rate_100w)",
        "- + z(list_structure_intensity) - z(contraction_rate_100w)",
        "",
        "Repetition / template similarity:",
        "- Merged from results/tables/event_time/repetition_daily_by_subreddit.csv when present;",
        "  generate with scripts/features/compute_daily_repetition_similarity.py. If missing, column is NaN.",
        "",
        "No minimum comment length filter is applied.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_repetition_daily_metrics(metrics_df: pd.DataFrame, repetition_csv: Path) -> pd.DataFrame:
    """Function summary: left-join `repetition_template_similarity` from standalone repetition CSV.

    Parameters:
    - metrics_df: subreddit-day rows from comment_features aggregation (no repetition column yet).
    - repetition_csv: path to `repetition_daily_by_subreddit.csv`.

    Returns:
    - metrics_df with `repetition_template_similarity` (NaN when file missing or invalid).
    """
    out = metrics_df.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    out["subreddit"] = out["subreddit"].astype(str)
    if not repetition_csv.exists():
        print(
            f"[prepare_event_time_metrics] repetition_csv_missing path={repetition_csv}; "
            "setting repetition_template_similarity to NaN (run scripts/features/compute_daily_repetition_similarity.py).",
            flush=True,
        )
        out["repetition_template_similarity"] = float("nan")
        return out
    rep = pd.read_csv(repetition_csv)
    need = {"subreddit", "date_utc", "repetition_template_similarity"}
    missing = sorted(need.difference(rep.columns))
    if missing:
        print(
            f"[prepare_event_time_metrics] repetition_skip invalid_csv missing_columns={missing} path={repetition_csv}",
            flush=True,
        )
        out["repetition_template_similarity"] = float("nan")
        return out
    rep = rep[list(need)].copy()
    rep["date_utc"] = rep["date_utc"].astype(str)
    rep["subreddit"] = rep["subreddit"].astype(str)
    out = out.merge(rep, on=["subreddit", "date_utc"], how="left")
    return out


def main() -> None:
    """Function summary: run full event-time metric preparation and write all required output artifacts."""
    args = parse_args()
    config = load_config(args.config)
    launch_ts = utc_ts(config["event_window"]["launch_day_utc"])
    subreddits = list(config["subreddits"]["primary"])
    paths = build_paths(config)

    if not paths.comment_features_dir.is_dir():
        raise FileNotFoundError(
            f"comment_features directory not found: {paths.comment_features_dir}. "
            "Run scripts/features/compute_comment_features.py or scripts/features/merge_ml_shards_into_comment_features.py first."
        )
    preview_jobs = list(
        iter_monthly_files(
            paths.comment_features_dir,
            subreddits,
            max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
            max_total_month_files=int(args.max_total_month_files),
        )
    )
    if not preview_jobs:
        raise FileNotFoundError(
            f"No comment_features parquet files found under: {paths.comment_features_dir}. "
            "Run scripts/features/compute_comment_features.py or scripts/features/merge_ml_shards_into_comment_features.py first."
        )

    print("[prepare_event_time_metrics] mode=comment_features_only", flush=True)
    metrics_df, ai_word_long_df, stats, validation_df = aggregate_daily_metrics_from_comment_features(
        comment_features_dir=paths.comment_features_dir,
        subreddits=subreddits,
        max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
        max_total_month_files=int(args.max_total_month_files),
        max_days_per_month=int(args.max_days_per_month),
    )
    if metrics_df.empty:
        raise FileNotFoundError(
            f"No daily metric rows produced from comment_features under: {paths.comment_features_dir}"
        )

    metrics_df = merge_repetition_daily_metrics(
        metrics_df, paths.event_time_tables_dir / "repetition_daily_by_subreddit.csv"
    )

    post_started_at = time.perf_counter()
    metrics_df = add_event_time_columns(metrics_df, launch_ts)
    metrics_df = add_ai_likeness_index(metrics_df)
    pooled_df, pooled_word_long = build_pooled_daily(metrics_df, ai_word_long_df, launch_ts)

    ai_word_long_df = add_event_time_columns(ai_word_long_df, launch_ts)
    ai_word_long_df = pd.concat([ai_word_long_df, pooled_word_long], ignore_index=True).sort_values(
        ["subreddit", "date_utc", "word_group", "word"]
    )
    stats.phase_postprocess_s += time.perf_counter() - post_started_at

    write_started_at = time.perf_counter()
    metrics_df.to_csv(paths.event_time_tables_dir / "event_time_daily_metrics_by_subreddit.csv", index=False)
    pooled_df.to_csv(paths.event_time_tables_dir / "event_time_daily_metrics_pooled.csv", index=False)
    ai_word_long_df.to_csv(paths.event_time_tables_dir / "ai_word_rates_daily_long.csv", index=False)
    if not validation_df.empty:
        validation_df.to_csv(paths.event_time_tables_dir / "comment_feature_validation_associations.csv", index=False)
    write_notes(paths.event_time_tables_dir / "event_time_metrics_notes.txt")

    # Backward-compatible export for existing figure script convention.
    pooled_df.to_csv(paths.tables_dir / "event_time_daily_metrics.csv", index=False)
    stats.phase_write_s += time.perf_counter() - write_started_at
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))


if __name__ == "__main__":
    main()
