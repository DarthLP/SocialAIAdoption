"""
Script summary:
This script prepares metric-ready daily event-time aggregates from cleaned Reddit
monthly chunks for subreddit-level and pooled analysis. It computes linguistic,
AI-style, and affect/safety proxy metrics, then writes reproducible tables used
by event-time plots and robustness checks.

Functionality:
- Reads cleaned monthly Parquet files from:
  `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
- Computes daily subreddit and pooled aggregates for:
  - semicolon rate
  - comment length
  - complexity index
  - AI-likeness index
  - AI-typical word rates (strict 10-word basket and extended basket)
  - toxicity proxy (VADER negativity and lexical toxicity rate)
- Adds additional AI-use candidates:
  - formality markers (contraction vs full-form rates)
  - list-structure intensity
  - repetition/template similarity proxy
  - assistant-tone phrase frequency
- Writes grouped outputs under `results/tables/event_time/`.
- Writes compatibility export to `results/tables/event_time_daily_metrics.csv`.
- Supports bounded benchmark runs with month/day sampling limits.
- Supports optional month-level parallel processing and phase timing logs.

How to apply/run:
- `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`
- Bounded benchmark example:
  `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml --max_month_files_per_subreddit 1 --max_days_per_month 10 --profile`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
LIST_STRUCTURE_RE = re.compile(r"(^\s*[-*]\s+)|(^\s*\d+[.)]\s+)|(\bfirst\b|\bsecond\b|\bthird\b)", re.IGNORECASE)
TOXIC_LEXICON = {
    "idiot",
    "moron",
    "stupid",
    "dumb",
    "trash",
    "garbage",
    "hate",
    "loser",
    "pathetic",
    "shut up",
    "kill",
}
CONTRACTIONS = {
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "i'm",
    "you're",
    "we're",
    "they're",
    "it's",
    "that's",
    "there's",
    "what's",
    "who's",
}
FULL_FORMS = {
    "cannot",
    "do not",
    "does not",
    "did not",
    "is not",
    "are not",
    "was not",
    "were not",
    "have not",
    "has not",
    "had not",
    "i am",
    "you are",
    "we are",
    "they are",
    "it is",
    "that is",
    "there is",
    "what is",
    "who is",
}
STRICT_AI_WORDS = [
    "explore",
    "captivate",
    "tapestry",
    "leverage",
    "embrace",
    "resonate",
    "dynamic",
    "testament",
    "delve",
    "elevate",
]
EXTENDED_AI_WORDS = [
    "additionally",
    "furthermore",
    "moreover",
    "underscore",
    "pivotal",
    "crucial",
    "intricate",
    "vibrant",
    "comprehensive",
    "robust",
    "enhance",
    "fostering",
    "showcase",
    "landscape",
    "valuable",
    "meticulous",
    "garner",
    "bolstered",
]
ASSISTANT_TONE_PHRASES = [
    "it is important to note",
    "it is worth noting",
    "in conclusion",
    "in summary",
    "moreover",
    "furthermore",
    "additionally",
    "this underscores",
    "a testament to",
]
FULL_FORM_PATTERNS = tuple(FULL_FORMS)
ASSISTANT_TONE_PATTERNS = tuple(ASSISTANT_TONE_PHRASES)
TOXIC_PATTERNS = tuple(TOXIC_LEXICON)
EXTENDED_AI_WORD_SET = frozenset(EXTENDED_AI_WORDS)
CONTRACTIONS_SET = frozenset(CONTRACTIONS)


@dataclass
class RuntimePaths:
    """Function summary: store resolved runtime input and output paths for this script."""

    cleaned_monthly_chunks_dir: Path
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
    """Function summary: parse command line options for config path and repetition window."""
    parser = argparse.ArgumentParser(description="Prepare event-time metric tables from cleaned daily chunks.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--similarity_window",
        type=int,
        default=20,
        help="How many previous comments to compare for template similarity within each day/subreddit.",
    )
    parser.add_argument(
        "--min_words_for_similarity",
        type=int,
        default=0,
        help="If >0, skip repetition similarity updates for comments shorter than this token count.",
    )
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
        "--workers",
        type=int,
        default=1,
        help="Number of month-level worker processes. Use 1 for sequential mode.",
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
        cleaned_monthly_chunks_dir=interim_dir / "cleaned_monthly_chunks",
        event_time_tables_dir=event_time_tables_dir,
        tables_dir=tables_dir,
    )


def iter_monthly_files(
    cleaned_monthly_chunks_dir: Path,
    subreddits: Iterable[str],
    max_month_files_per_subreddit: int = 0,
    max_total_month_files: int = 0,
) -> Iterable[tuple[str, Path]]:
    """Function summary: yield existing cleaned monthly Parquet paths for each configured subreddit."""
    yielded_total = 0
    for subreddit in sorted(subreddits):
        subreddit_dir = cleaned_monthly_chunks_dir / subreddit
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


def validate_month_schema(df: pd.DataFrame, file_path: Path, subreddit: str) -> pd.DataFrame:
    """Function summary: validate required columns for monthly interim data and return cleaned frame."""
    required_cols = {"body", "date_utc", "subreddit"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing} in monthly interim file: {file_path}")
    out = df.copy()
    out = out[out["subreddit"].astype("string") == subreddit]
    out["body"] = out["body"].astype("string")
    out["date_utc"] = out["date_utc"].astype("string")
    out = out.dropna(subset=["body", "date_utc"])
    return out


def tokenize_words(text: str) -> list[str]:
    """Function summary: tokenize text into lowercase word-like tokens for lexical rate calculations."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def count_phrase_occurrences(text_lc: str, phrase: str) -> int:
    """Function summary: count non-overlapping case-normalized phrase matches in one comment text."""
    if not phrase:
        return 0
    return text_lc.count(phrase)


def count_full_form_occurrences(text_lc: str) -> int:
    """Function summary: count occurrences of configured full-form patterns in one comment text."""
    total = 0
    for phrase in FULL_FORM_PATTERNS:
        total += text_lc.count(phrase)
    return total


def is_list_structured(text: str) -> bool:
    """Function summary: detect list-like structure markers in one comment body."""
    return bool(LIST_STRUCTURE_RE.search(text or ""))


def comment_similarity_to_recent(tokens: set[str], recent_tokens: deque[set[str]]) -> float:
    """Function summary: compute max Jaccard similarity against a bounded history of prior comments."""
    if not tokens or not recent_tokens:
        return 0.0
    best = 0.0
    for prev in recent_tokens:
        union = tokens | prev
        if not union:
            continue
        score = len(tokens & prev) / len(union)
        if score > best:
            best = score
    return best


def safe_rate_100w(count: float, n_words: int) -> float:
    """Function summary: compute per-100-word rate with explicit zero-denominator handling."""
    if n_words <= 0:
        return 0.0
    return (float(count) / float(n_words)) * 100.0


def compute_complexity_index(total_sentences: int, total_words: int, total_word_chars: int, n_comments: int) -> float:
    """Function summary: compute a stable lexical/syntactic complexity proxy from daily aggregates."""
    if n_comments <= 0 or total_words <= 0:
        return 0.0
    mean_sentence_length = float(total_words) / float(max(total_sentences, 1))
    mean_word_length = float(total_word_chars) / float(total_words)
    return 0.5 * mean_sentence_length + 0.5 * mean_word_length


def build_empty_counter() -> Dict[str, Any]:
    """Function summary: create initialized mutable counters for one day/subreddit aggregation unit."""
    strict_word_counts = {word: 0 for word in STRICT_AI_WORDS}
    return {
        "n_comments": 0,
        "n_words": 0,
        "total_semicolons": 0,
        "total_word_chars": 0,
        "total_sentences": 0,
        "sum_comment_length_words": 0,
        "vader_compound_sum": 0.0,
        "vader_negativity_sum": 0.0,
        "list_structured_count": 0,
        "similarity_sum": 0.0,
        "contraction_count": 0,
        "full_form_count": 0,
        "assistant_tone_phrase_count": 0,
        "toxic_lexicon_hits": 0,
        "strict_word_counts": strict_word_counts,
        "extended_ai_word_hits": 0,
    }


def update_counter_for_comment(
    counter: Dict[str, Any],
    body: str,
    analyzer: SentimentIntensityAnalyzer,
    similarity_window: int,
    min_words_for_similarity: int,
    recent_tokens: deque[set[str]],
) -> None:
    """Function summary: update one aggregation counter from a single comment body text."""
    text = body or ""
    text_lc = text.lower()
    words = tokenize_words(text)
    n_words_comment = len(words)
    word_set = set(words)
    token_counter = Counter(words)

    counter["n_comments"] += 1
    counter["n_words"] += n_words_comment
    counter["sum_comment_length_words"] += n_words_comment
    counter["total_semicolons"] += text.count(";")
    counter["total_word_chars"] += sum(len(w) for w in words)
    sentence_count = sum(1 for _ in SENTENCE_SPLIT_RE.finditer(text))
    counter["total_sentences"] += max(sentence_count, 1 if n_words_comment > 0 else 0)
    counter["list_structured_count"] += int(is_list_structured(text))

    # Repetition/template proxy: max Jaccard to recent comments in same day/subreddit.
    if n_words_comment >= min_words_for_similarity:
        similarity = comment_similarity_to_recent(word_set, recent_tokens)
        counter["similarity_sum"] += similarity
        recent_tokens.append(word_set)
        while len(recent_tokens) > similarity_window:
            recent_tokens.popleft()

    if text.strip():
        sentiment = analyzer.polarity_scores(text)
        compound = float(sentiment.get("compound", 0.0))
    else:
        compound = 0.0
    counter["vader_compound_sum"] += compound
    counter["vader_negativity_sum"] += max(0.0, -compound)

    contraction_hits = int(sum(token_counter.get(w, 0) for w in CONTRACTIONS_SET))
    full_form_hits = count_full_form_occurrences(text_lc)
    counter["contraction_count"] += contraction_hits
    counter["full_form_count"] += full_form_hits

    phrase_hits = sum(count_phrase_occurrences(text_lc, phrase) for phrase in ASSISTANT_TONE_PATTERNS)
    counter["assistant_tone_phrase_count"] += phrase_hits

    toxic_hits = sum(count_phrase_occurrences(text_lc, toxic_item) for toxic_item in TOXIC_PATTERNS)
    counter["toxic_lexicon_hits"] += toxic_hits

    for word in STRICT_AI_WORDS:
        counter["strict_word_counts"][word] += int(token_counter.get(word, 0))
    counter["extended_ai_word_hits"] += int(sum(count for token, count in token_counter.items() if token in EXTENDED_AI_WORD_SET))


def process_month_file(
    subreddit: str,
    file_path: Path,
    similarity_window: int,
    min_words_for_similarity: int,
    max_days_per_month: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], ProfilingStats]:
    """Function summary: process one monthly file into daily metric rows plus profiling counters."""
    analyzer = SentimentIntensityAnalyzer()
    rows: list[Dict[str, Any]] = []
    ai_word_long_rows: list[Dict[str, Any]] = []
    stats = ProfilingStats()

    t0 = time.perf_counter()
    month_df = pd.read_parquet(file_path, columns=["body", "date_utc", "subreddit"])
    stats.phase_read_s += time.perf_counter() - t0
    stats.rows_read += int(len(month_df))

    t1 = time.perf_counter()
    month_df = validate_month_schema(month_df, file_path=file_path, subreddit=subreddit)
    stats.phase_validate_s += time.perf_counter() - t1

    if month_df.empty:
        stats.files_processed += 1
        return rows, ai_word_long_rows, stats

    t2 = time.perf_counter()
    for day_idx, (date_utc, day_df) in enumerate(month_df.groupby("date_utc", sort=True)):
        if max_days_per_month > 0 and day_idx >= max_days_per_month:
            break
        counter = build_empty_counter()
        recent_tokens: deque[set[str]] = deque()
        for body in day_df["body"]:
            update_counter_for_comment(
                counter=counter,
                body=str(body or ""),
                analyzer=analyzer,
                similarity_window=similarity_window,
                min_words_for_similarity=min_words_for_similarity,
                recent_tokens=recent_tokens,
            )

        n_comments = int(counter["n_comments"])
        n_words = int(counter["n_words"])
        strict_total_hits = int(sum(counter["strict_word_counts"].values()))
        row = {
            "subreddit": subreddit,
            "date_utc": date_utc,
            "n_comments": n_comments,
            "n_words": n_words,
            "semicolon_rate_100w": safe_rate_100w(counter["total_semicolons"], n_words),
            "comment_length_words": (float(counter["sum_comment_length_words"]) / float(n_comments)) if n_comments else 0.0,
            "complexity_index": compute_complexity_index(
                total_sentences=int(counter["total_sentences"]),
                total_words=n_words,
                total_word_chars=int(counter["total_word_chars"]),
                n_comments=n_comments,
            ),
            "ai_word_rate_100w": safe_rate_100w(strict_total_hits, n_words),
            "ai_word_extended_rate_100w": safe_rate_100w(counter["extended_ai_word_hits"], n_words),
            "vader_compound_mean": (float(counter["vader_compound_sum"]) / float(n_comments)) if n_comments else 0.0,
            "vader_negativity_mean": (float(counter["vader_negativity_sum"]) / float(n_comments)) if n_comments else 0.0,
            "toxicity_score": (float(counter["vader_negativity_sum"]) / float(n_comments)) if n_comments else 0.0,
            "toxic_lexicon_rate_100w": safe_rate_100w(counter["toxic_lexicon_hits"], n_words),
            "contraction_rate_100w": safe_rate_100w(counter["contraction_count"], n_words),
            "full_form_rate_100w": safe_rate_100w(counter["full_form_count"], n_words),
            "formality_balance_100w": safe_rate_100w(counter["full_form_count"], n_words)
            - safe_rate_100w(counter["contraction_count"], n_words),
            "list_structure_intensity": (float(counter["list_structured_count"]) / float(n_comments)) if n_comments else 0.0,
            "repetition_template_similarity": (float(counter["similarity_sum"]) / float(n_comments)) if n_comments else 0.0,
            "assistant_tone_rate_100w": safe_rate_100w(counter["assistant_tone_phrase_count"], n_words),
            "strict_ai_word_hits_total": strict_total_hits,
            "extended_ai_word_hits_total": int(counter["extended_ai_word_hits"]),
        }
        rows.append(row)

        for word in STRICT_AI_WORDS:
            ai_word_long_rows.append(
                {
                    "subreddit": subreddit,
                    "date_utc": date_utc,
                    "word": word,
                    "word_group": "strict_individual",
                    "hits": int(counter["strict_word_counts"][word]),
                    "rate_100w": safe_rate_100w(counter["strict_word_counts"][word], n_words),
                    "n_words": n_words,
                }
            )
        ai_word_long_rows.append(
            {
                "subreddit": subreddit,
                "date_utc": date_utc,
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
                "date_utc": date_utc,
                "word": "extended_combined",
                "word_group": "extended_combined",
                "hits": int(counter["extended_ai_word_hits"]),
                "rate_100w": safe_rate_100w(counter["extended_ai_word_hits"], n_words),
                "n_words": n_words,
            }
        )
        stats.days_processed += 1
        stats.comments_processed += n_comments
    stats.phase_aggregate_s += time.perf_counter() - t2
    stats.files_processed += 1
    return rows, ai_word_long_rows, stats


def aggregate_daily_metrics(
    cleaned_monthly_chunks_dir: Path,
    subreddits: list[str],
    similarity_window: int,
    min_words_for_similarity: int,
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
    max_days_per_month: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, ProfilingStats]:
    """Function summary: aggregate subreddit-day metrics and strict-word long-format rates from cleaned monthly files."""
    rows: list[Dict[str, Any]] = []
    ai_word_long_rows: list[Dict[str, Any]] = []
    stats = ProfilingStats()
    started_at = time.perf_counter()
    months_processed = 0
    month_jobs = list(
        iter_monthly_files(
            cleaned_monthly_chunks_dir,
            subreddits,
            max_month_files_per_subreddit=max_month_files_per_subreddit,
            max_total_month_files=max_total_month_files,
        )
    )

    if workers > 1 and month_jobs:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [
                pool.submit(
                    process_month_file,
                    subreddit,
                    file_path,
                    int(similarity_window),
                    int(min_words_for_similarity),
                    int(max_days_per_month),
                )
                for subreddit, file_path in month_jobs
            ]
            for future in as_completed(futures):
                file_rows, file_word_rows, file_stats = future.result()
                rows.extend(file_rows)
                ai_word_long_rows.extend(file_word_rows)
                stats.merge(file_stats)
                months_processed += 1
                elapsed = time.perf_counter() - started_at
                print(
                    f"[prepare_event_time_metrics] finished month_job={months_processed}/{len(month_jobs)} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    else:
        for subreddit, file_path in month_jobs:
            file_rows, file_word_rows, file_stats = process_month_file(
                subreddit=subreddit,
                file_path=file_path,
                similarity_window=similarity_window,
                min_words_for_similarity=min_words_for_similarity,
                max_days_per_month=max_days_per_month,
            )
            rows.extend(file_rows)
            ai_word_long_rows.extend(file_word_rows)
            stats.merge(file_stats)
            months_processed += 1
            elapsed = time.perf_counter() - started_at
            print(
                f"[prepare_event_time_metrics] finished subreddit={subreddit} month={file_path.stem} months_done={months_processed} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True) if rows else pd.DataFrame()
    ai_word_long_df = (
        pd.DataFrame(ai_word_long_rows).sort_values(["subreddit", "date_utc", "word_group", "word"]).reset_index(drop=True)
        if ai_word_long_rows
        else pd.DataFrame()
    )
    return metrics_df, ai_word_long_df, stats


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
        pooled_rows.append(
            {
                "subreddit": "ALL",
                "date_utc": str(date_utc),
                "n_comments": n_comments,
                "n_words": n_words,
                "semicolon_rate_100w": safe_rate_100w((group["semicolon_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
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
                "repetition_template_similarity": float(group["repetition_template_similarity"].mul(group["n_comments"]).sum())
                / float(n_comments),
                "assistant_tone_rate_100w": safe_rate_100w((group["assistant_tone_rate_100w"] * group["n_words"] / 100.0).sum(), n_words),
                "strict_ai_word_hits_total": int(group["strict_ai_word_hits_total"].sum()),
                "extended_ai_word_hits_total": int(group["extended_ai_word_hits_total"].sum()),
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
        "Primary AI word list (strict_10):",
        "- explore, captivate, tapestry, leverage, embrace, resonate, dynamic, testament, delve, elevate",
        "",
        "Toxicity channels:",
        "- toxicity_score: VADER negativity mean = mean(max(0, -compound)).",
        "- toxic_lexicon_rate_100w: lexical incidence per 100 words from a lightweight toxic lexicon.",
        "",
        "AI-likeness index (z-score composite):",
        "- z(ai_word_rate_100w) + z(formality_balance_100w) + z(assistant_tone_rate_100w)",
        "- + z(list_structure_intensity) - z(contraction_rate_100w)",
        "",
        "No minimum comment length filter is applied.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run full event-time metric preparation and write all required output artifacts."""
    args = parse_args()
    config = load_config(args.config)
    launch_ts = utc_ts(config["event_window"]["launch_day_utc"])
    subreddits = list(config["subreddits"]["primary"])
    paths = build_paths(config)

    metrics_df, ai_word_long_df, stats = aggregate_daily_metrics(
        cleaned_monthly_chunks_dir=paths.cleaned_monthly_chunks_dir,
        subreddits=subreddits,
        similarity_window=int(args.similarity_window),
        min_words_for_similarity=int(args.min_words_for_similarity),
        max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
        max_total_month_files=int(args.max_total_month_files),
        max_days_per_month=int(args.max_days_per_month),
        workers=max(1, int(args.workers)),
    )
    if metrics_df.empty:
        raise FileNotFoundError(f"No cleaned monthly chunk files found under: {paths.cleaned_monthly_chunks_dir}")

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
    write_notes(paths.event_time_tables_dir / "event_time_metrics_notes.txt")

    # Backward-compatible export for existing figure script convention.
    pooled_df.to_csv(paths.tables_dir / "event_time_daily_metrics.csv", index=False)
    stats.phase_write_s += time.perf_counter() - write_started_at
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))


if __name__ == "__main__":
    main()
