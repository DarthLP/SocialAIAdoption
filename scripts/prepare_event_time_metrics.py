"""
Script summary:
This script prepares metric-ready daily event-time aggregates from cleaned Reddit
daily chunks for subreddit-level and pooled analysis. It computes linguistic,
AI-style, and affect/safety proxy metrics, then writes reproducible tables used
by event-time plots and robustness checks.

Functionality:
- Reads cleaned daily files from:
  `data/interim/political_forums/cleaned_daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`.
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

How to apply/run:
- `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
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


@dataclass
class RuntimePaths:
    """Function summary: store resolved runtime input and output paths for this script."""

    cleaned_daily_chunks_dir: Path
    event_time_tables_dir: Path
    tables_dir: Path


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
    return parser.parse_args()


def build_paths(config: Dict[str, Any]) -> RuntimePaths:
    """Function summary: resolve configured locations and ensure event-time output folders exist."""
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    event_time_tables_dir = tables_dir / "event_time"
    event_time_tables_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        cleaned_daily_chunks_dir=interim_dir / "cleaned_daily_chunks",
        event_time_tables_dir=event_time_tables_dir,
        tables_dir=tables_dir,
    )


def iter_daily_files(cleaned_daily_chunks_dir: Path, subreddits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Function summary: yield existing daily cleaned NDJSON paths for each configured subreddit."""
    for subreddit in sorted(subreddits):
        subreddit_dir = cleaned_daily_chunks_dir / subreddit
        if not subreddit_dir.exists():
            continue
        for ndjson_path in sorted(subreddit_dir.glob("*.ndjson")):
            yield subreddit, ndjson_path


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
    for phrase in FULL_FORMS:
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
    recent_tokens: deque[set[str]],
) -> None:
    """Function summary: update one aggregation counter from a single comment body text."""
    text = body or ""
    text_lc = text.lower()
    words = tokenize_words(text)
    n_words_comment = len(words)
    word_set = set(words)

    counter["n_comments"] += 1
    counter["n_words"] += n_words_comment
    counter["sum_comment_length_words"] += n_words_comment
    counter["total_semicolons"] += text.count(";")
    counter["total_word_chars"] += sum(len(w) for w in words)
    sentence_parts = [p for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    counter["total_sentences"] += max(len(sentence_parts), 1 if n_words_comment > 0 else 0)
    counter["list_structured_count"] += int(is_list_structured(text))

    # Repetition/template proxy: max Jaccard to recent comments in same day/subreddit.
    similarity = comment_similarity_to_recent(word_set, recent_tokens)
    counter["similarity_sum"] += similarity
    recent_tokens.append(word_set)
    while len(recent_tokens) > similarity_window:
        recent_tokens.popleft()

    sentiment = analyzer.polarity_scores(text)
    compound = float(sentiment.get("compound", 0.0))
    counter["vader_compound_sum"] += compound
    counter["vader_negativity_sum"] += max(0.0, -compound)

    contraction_hits = sum(1 for w in words if w in CONTRACTIONS)
    full_form_hits = count_full_form_occurrences(text_lc)
    counter["contraction_count"] += contraction_hits
    counter["full_form_count"] += full_form_hits

    phrase_hits = 0
    for phrase in ASSISTANT_TONE_PHRASES:
        phrase_hits += count_phrase_occurrences(text_lc, phrase)
    counter["assistant_tone_phrase_count"] += phrase_hits

    toxic_hits = 0
    for toxic_item in TOXIC_LEXICON:
        toxic_hits += count_phrase_occurrences(text_lc, toxic_item)
    counter["toxic_lexicon_hits"] += toxic_hits

    token_counter = Counter(words)
    for word in STRICT_AI_WORDS:
        counter["strict_word_counts"][word] += int(token_counter.get(word, 0))
    counter["extended_ai_word_hits"] += int(sum(token_counter.get(w, 0) for w in EXTENDED_AI_WORDS))


def aggregate_daily_metrics(
    cleaned_daily_chunks_dir: Path,
    subreddits: list[str],
    similarity_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function summary: aggregate subreddit-day metrics and strict-word long-format rates from cleaned files."""
    analyzer = SentimentIntensityAnalyzer()
    rows: list[Dict[str, Any]] = []
    ai_word_long_rows: list[Dict[str, Any]] = []

    for subreddit, file_path in iter_daily_files(cleaned_daily_chunks_dir, subreddits):
        date_utc = file_path.stem
        counter = build_empty_counter()
        recent_tokens: deque[set[str]] = deque()

        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                body = str(record.get("body") or "")
                update_counter_for_comment(
                    counter=counter,
                    body=body,
                    analyzer=analyzer,
                    similarity_window=similarity_window,
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

    metrics_df = pd.DataFrame(rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True) if rows else pd.DataFrame()
    ai_word_long_df = (
        pd.DataFrame(ai_word_long_rows).sort_values(["subreddit", "date_utc", "word_group", "word"]).reset_index(drop=True)
        if ai_word_long_rows
        else pd.DataFrame()
    )
    return metrics_df, ai_word_long_df


def add_event_time_columns(df: pd.DataFrame, launch_ts: int) -> pd.DataFrame:
    """Function summary: attach date and event-time offset columns relative to launch day."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date_utc"], utc=True).dt.tz_convert(None)
    launch_date = datetime.fromtimestamp(launch_ts, tz=timezone.utc).date()
    out["event_time_t"] = out["date"].dt.date.apply(lambda d: (d - launch_date).days).astype(int)
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
    pooled_word_long["rate_100w"] = pooled_word_long.apply(lambda r: safe_rate_100w(r["hits"], int(r["n_words"])), axis=1)
    pooled_word_long = add_event_time_columns(pooled_word_long, launch_ts)
    pooled_word_long = pooled_word_long.sort_values(["date_utc", "word_group", "word"]).reset_index(drop=True)
    return pooled_df, pooled_word_long


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

    metrics_df, ai_word_long_df = aggregate_daily_metrics(
        cleaned_daily_chunks_dir=paths.cleaned_daily_chunks_dir,
        subreddits=subreddits,
        similarity_window=int(args.similarity_window),
    )
    if metrics_df.empty:
        raise FileNotFoundError(f"No cleaned daily chunk files found under: {paths.cleaned_daily_chunks_dir}")

    metrics_df = add_event_time_columns(metrics_df, launch_ts)
    metrics_df = add_ai_likeness_index(metrics_df)
    pooled_df, pooled_word_long = build_pooled_daily(metrics_df, ai_word_long_df, launch_ts)

    ai_word_long_df = add_event_time_columns(ai_word_long_df, launch_ts)
    ai_word_long_df = pd.concat([ai_word_long_df, pooled_word_long], ignore_index=True).sort_values(
        ["subreddit", "date_utc", "word_group", "word"]
    )

    metrics_df.to_csv(paths.event_time_tables_dir / "event_time_daily_metrics_by_subreddit.csv", index=False)
    pooled_df.to_csv(paths.event_time_tables_dir / "event_time_daily_metrics_pooled.csv", index=False)
    ai_word_long_df.to_csv(paths.event_time_tables_dir / "ai_word_rates_daily_long.csv", index=False)
    write_notes(paths.event_time_tables_dir / "event_time_metrics_notes.txt")

    # Backward-compatible export for existing figure script convention.
    pooled_df.to_csv(paths.tables_dir / "event_time_daily_metrics.csv", index=False)


if __name__ == "__main__":
    main()
