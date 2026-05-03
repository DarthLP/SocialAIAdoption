"""
Script summary:
This script computes reusable per-comment feature rows from cleaned monthly Reddit
Parquet files and writes feature parquet shards per subreddit-month. It keeps all
comments (including short comments) and stores confidence/coverage metadata for
metrics that are less reliable on short texts.

Functionality:
- Reads cleaned monthly input from:
  `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
- Passes through `author` and `created_utc` when present in the cleaned schema (empty
  author and null `created_utc` otherwise) for downstream user- and time-ordered analyses.
- Computes lexical/style/toxicity features for event-time aggregation, including
  dash/typography counts, markdown proxies, hedging/polite/signposting phrase hits,
  and average words per sentence.
- Computes additional per-comment outputs:
  - detector primary and secondary AI/human scores
  - passive voice proxy counts/rates
  - perplexity (optional, model-based)
  - hostility score (optional, model-based)
  - emotion scores (anger, fear, sadness, surprise; optional, model-based)
- Adds length buckets and detector confidence flags without excluding short comments.
- Supports bounded runs, skip-existing behavior, device auto-selection, and profiling.

How to apply/run:
- Full run (public model defaults):
  `.venv/bin/python scripts/compute_comment_features.py --config config/political_forums_setup.yaml`
- Fast bounded benchmark:
  `.venv/bin/python scripts/compute_comment_features.py --config config/political_forums_setup.yaml --max_total_month_files 2 --max_days_per_month 10 --profile`
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
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

from src.config_utils import load_config

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
LIST_STRUCTURE_RE = re.compile(r"(^\s*[-*]\s+)|(^\s*\d+[.)]\s+)|(\bfirst\b|\bsecond\b|\bthird\b)", re.IGNORECASE)
PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|get|gets|got)\s+\w+(?:ed|en)\b",
    flags=re.IGNORECASE,
)
MARKDOWN_BOLD_PAIR_RE = re.compile(r"\*\*.+?\*\*", re.DOTALL)
MARKDOWN_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)
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
# Disjoint from ASSISTANT_TONE_PHRASES: hedging / epistemic softeners (substring counts on lowercased text).
HEDGING_PHRASES = [
    "may not be",
    "might not be",
    "could be",
    "possibly",
    "perhaps",
    "likely that",
    "it seems",
    "appears to",
    "generally speaking",
    "in many cases",
    "to some extent",
    "not necessarily",
    "tends to",
    "often the case",
]
# Chat-assistant style closers (disjoint from assistant_tone list above).
POLITE_CLOSER_PHRASES = [
    "hope this helps",
    "let me know if",
    "happy to help",
    "feel free to",
    "happy to clarify",
    "if you have any questions",
    "let me know if you need",
    "glad to help",
]
# Signposting without overlapping assistant_tone entries (no moreover/furthermore/additionally/in conclusion/in summary).
SIGNPOSTING_PHRASES = [
    "firstly",
    "secondly",
    "thirdly",
    "on the other hand",
    "in other words",
    "to recap",
    "to summarize",
    "short answer",
    "long answer",
    "my take",
    "overall",
    "in short",
]
FULL_FORM_PATTERNS = tuple(FULL_FORMS)
ASSISTANT_TONE_PATTERNS = tuple(ASSISTANT_TONE_PHRASES)
HEDGING_PATTERNS = tuple(HEDGING_PHRASES)
POLITE_CLOSER_PATTERNS = tuple(POLITE_CLOSER_PHRASES)
SIGNPOSTING_PATTERNS = tuple(SIGNPOSTING_PHRASES)
TOXIC_PATTERNS = tuple(TOXIC_LEXICON)
EXTENDED_AI_WORD_SET = frozenset(EXTENDED_AI_WORDS)
CONTRACTIONS_SET = frozenset(CONTRACTIONS)

REQUIRED_CLEANED_COLUMNS = ("id", "subreddit", "date_utc", "body")
OPTIONAL_CLEANED_META_COLUMNS = ("author", "created_utc")


def read_cleaned_month_for_features(file_path: Path) -> pd.DataFrame:
    """Function summary: load one cleaned monthly Parquet with required columns plus optional metadata.

    Parameters:
    - file_path: path to cleaned `<subreddit>/<YYYY-MM>.parquet`.

    Returns:
    - DataFrame with id, subreddit, date_utc, body, author (string), created_utc (nullable Int64).
    """
    required = list(REQUIRED_CLEANED_COLUMNS)
    optional = list(OPTIONAL_CLEANED_META_COLUMNS)
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(file_path).names)
    except Exception:
        full = pd.read_parquet(file_path)
        schema_names = set(full.columns)
        missing_req = set(required) - schema_names
        if missing_req:
            raise ValueError(f"Missing required columns {sorted(missing_req)} in {file_path}")
        keep = [c for c in required + optional if c in full.columns]
        frame = full[keep].copy()
    else:
        missing_req = set(required) - schema_names
        if missing_req:
            raise ValueError(f"Missing required columns {sorted(missing_req)} in {file_path}")
        cols = required + [c for c in optional if c in schema_names]
        frame = pd.read_parquet(file_path, columns=cols)
    if "author" not in frame.columns:
        frame["author"] = ""
    frame["author"] = frame["author"].astype("string").fillna("")
    if "created_utc" not in frame.columns:
        frame["created_utc"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    else:
        frame["created_utc"] = pd.to_numeric(frame["created_utc"], errors="coerce").astype("Int64")
    return frame


@dataclass
class ProfilingStats:
    """Function summary: store cumulative runtime counters for bounded-run profiling output."""

    phase_read_s: float = 0.0
    phase_feature_s: float = 0.0
    phase_model_s: float = 0.0
    phase_write_s: float = 0.0
    files_processed: int = 0
    comments_processed: int = 0

    def merge(self, other: "ProfilingStats") -> None:
        """Function summary: merge another profiling payload into this instance."""
        self.phase_read_s += float(other.phase_read_s)
        self.phase_feature_s += float(other.phase_feature_s)
        self.phase_model_s += float(other.phase_model_s)
        self.phase_write_s += float(other.phase_write_s)
        self.files_processed += int(other.files_processed)
        self.comments_processed += int(other.comments_processed)

    def as_dict(self) -> Dict[str, Any]:
        """Function summary: return a stable dictionary representation for logs and optional export."""
        seconds_per_100k_comments = 0.0
        if self.comments_processed > 0:
            total = self.phase_read_s + self.phase_feature_s + self.phase_model_s + self.phase_write_s
            seconds_per_100k_comments = (total / float(self.comments_processed)) * 100000.0
        return {
            "phase_read_s": round(self.phase_read_s, 4),
            "phase_feature_s": round(self.phase_feature_s, 4),
            "phase_model_s": round(self.phase_model_s, 4),
            "phase_write_s": round(self.phase_write_s, 4),
            "files_processed": int(self.files_processed),
            "comments_processed": int(self.comments_processed),
            "seconds_per_100k_comments": round(seconds_per_100k_comments, 4),
        }


def parse_args() -> argparse.Namespace:
    """Function summary: parse command line options controlling feature extraction scope and runtime."""
    parser = argparse.ArgumentParser(description="Compute reusable per-comment features from cleaned monthly chunks.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing output parquet shards.")
    parser.add_argument("--workers", type=int, default=1, help="Month-level workers. Keep at 1 on slower disks.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model-based feature inference.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Inference device for model-based features.",
    )
    parser.add_argument("--max_month_files_per_subreddit", type=int, default=0)
    parser.add_argument("--max_total_month_files", type=int, default=0)
    parser.add_argument("--max_days_per_month", type=int, default=0)
    parser.add_argument("--subreddits", type=str, default="", help="Comma-separated subreddit allow-list.")
    parser.add_argument("--months", type=str, default="", help="Comma-separated YYYY-MM allow-list.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_output", type=str, default="")
    return parser.parse_args()


def tokenize_words(text: str) -> list[str]:
    """Function summary: tokenize one text into lowercase word-like units for lexical feature calculations."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def count_phrase_occurrences(text_lc: str, phrase: str) -> int:
    """Function summary: count non-overlapping phrase matches in lowercased text."""
    if not phrase:
        return 0
    return text_lc.count(phrase)


def count_full_form_occurrences(text_lc: str) -> int:
    """Function summary: count occurrences of configured full-form patterns in one lowercased text."""
    total = 0
    for phrase in FULL_FORM_PATTERNS:
        total += text_lc.count(phrase)
    return total


def count_ascii_double_hyphen(text: str) -> int:
    """Function summary: count occurrences of spaced double hyphen token ` -- ` in raw text (ASCII dash proxy)."""
    return (text or "").count(" -- ")


def count_curly_quotes(text: str) -> int:
    """Function summary: count Unicode curly single/double quote characters in one text."""
    s = text or ""
    return int(sum(s.count(ch) for ch in "\u201c\u201d\u2018\u2019"))


def count_markdown_bold_pairs(text: str) -> int:
    """Function summary: count non-greedy **...** markdown bold spans in one text."""
    return len(MARKDOWN_BOLD_PAIR_RE.findall(text or ""))


def count_markdown_heading_lines(text: str) -> int:
    """Function summary: count Markdown ATX-style heading line starts (# through ######)."""
    return len(MARKDOWN_HEADING_LINE_RE.findall(text or ""))


def sum_phrase_hits(text_lc: str, phrases: tuple[str, ...]) -> int:
    """Function summary: sum substring hit counts for each phrase in lowercased text."""
    return int(sum(count_phrase_occurrences(text_lc, p) for p in phrases))


def avg_words_per_sentence(n_words: int, sentence_count: int) -> float:
    """Function summary: return words divided by sentence_count when both positive; else NaN."""
    if n_words <= 0 or sentence_count <= 0:
        return float("nan")
    return float(n_words) / float(sentence_count)


def length_bucket(n_words: int) -> str:
    """Function summary: map per-comment word count to stable length bucket labels."""
    if n_words < 20:
        return "short"
    if n_words < 50:
        return "medium"
    return "long"


def detector_confidence_flag(n_words: int) -> str:
    """Function summary: assign detector confidence tier by length without excluding any rows."""
    if n_words < 20:
        return "low"
    if n_words < 50:
        return "medium"
    return "high"


def choose_device(device_arg: str) -> str:
    """Function summary: resolve runtime inference device with Apple Silicon MPS auto fallback to CPU."""
    if device_arg in {"cpu", "mps"}:
        return device_arg
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def iter_monthly_files(
    cleaned_monthly_chunks_dir: Path,
    subreddits: Iterable[str],
    allowed_months: set[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
) -> Iterable[tuple[str, Path]]:
    """Function summary: yield existing cleaned monthly parquet files with optional subreddit/month and count filters."""
    total = 0
    for subreddit in sorted(subreddits):
        sub_dir = cleaned_monthly_chunks_dir / subreddit
        if not sub_dir.exists():
            continue
        per_sub_count = 0
        for file_path in sorted(sub_dir.glob("*.parquet")):
            month = file_path.stem
            if allowed_months and month not in allowed_months:
                continue
            if max_month_files_per_subreddit > 0 and per_sub_count >= max_month_files_per_subreddit:
                break
            if max_total_month_files > 0 and total >= max_total_month_files:
                return
            yield subreddit, file_path
            per_sub_count += 1
            total += 1


def maybe_load_transformers_pipeline(task: str, model_id: str, device: str) -> Any:
    """Function summary: lazily initialize a transformers pipeline and return None when dependency/model init fails."""
    if not model_id:
        return None
    try:
        from transformers import pipeline

        if device == "cpu":
            device_index = -1
        elif device == "mps":
            device_index = "mps"
        else:
            device_index = -1
        return pipeline(task, model=model_id, truncation=True, device=device_index)
    except Exception:
        return None


def label_score_as_ai_probability(result: Dict[str, Any]) -> float:
    """Function summary: convert one classifier result dict into normalized AI probability."""
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))
    if "human" in label or label.endswith("0"):
        return float(max(0.0, min(1.0, 1.0 - score)))
    return float(max(0.0, min(1.0, score)))


def batch_text_classification_scores(pipe: Any, texts: list[str], batch_size: int) -> list[float]:
    """Function summary: run batched text-classification inference and return one probability score per text."""
    if pipe is None:
        return [float("nan")] * len(texts)
    out: list[float] = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start : start + max(1, int(batch_size))]
        try:
            results = pipe(batch, truncation=True, batch_size=max(1, int(batch_size)))
            if isinstance(results, dict):
                results = [results]
            out.extend([label_score_as_ai_probability(r) for r in results])
        except Exception:
            out.extend([float("nan")] * len(batch))
    return out


def batch_emotion_scores(pipe: Any, texts: list[str], batch_size: int) -> Dict[str, list[float]]:
    """Function summary: run batched emotion classification and map outputs to anger/fear/sadness/surprise columns."""
    keys = ["anger", "fear", "sadness", "surprise"]
    scores: Dict[str, list[float]] = {k: [] for k in keys}
    if pipe is None:
        for k in keys:
            scores[k] = [float("nan")] * len(texts)
        return scores
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start : start + max(1, int(batch_size))]
        try:
            results = pipe(batch, truncation=True, batch_size=max(1, int(batch_size)), top_k=None)
            if results and isinstance(results[0], dict):
                results = [[r] for r in results]
            for row_result in results:
                row_map = {k: 0.0 for k in keys}
                for item in row_result:
                    label = str(item.get("label", "")).lower()
                    value = float(item.get("score", 0.0))
                    if label in row_map:
                        row_map[label] = value
                for k in keys:
                    scores[k].append(row_map[k])
        except Exception:
            for k in keys:
                scores[k].extend([float("nan")] * len(batch))
    return scores


def init_perplexity_components(model_id: str, device: str) -> tuple[Any, Any]:
    """Function summary: initialize tokenizer and causal LM for perplexity scoring or return (None, None) on failure."""
    if not model_id:
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if device == "mps":
            model = model.to("mps")
        else:
            model = model.to("cpu")
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def batch_perplexity(texts: list[str], tokenizer: Any, model: Any, batch_size: int, device: str) -> list[float]:
    """Function summary: compute batched per-comment perplexity with truncation and robust error fallbacks."""
    if tokenizer is None or model is None:
        return [float("nan")] * len(texts)
    try:
        import torch
    except Exception:
        return [float("nan")] * len(texts)

    out: list[float] = []
    target_device = "mps" if device == "mps" else "cpu"
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start : start + max(1, int(batch_size))]
        try:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            enc = {k: v.to(target_device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc, labels=enc["input_ids"])
                logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = enc["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size())
            if "attention_mask" in enc:
                mask = enc["attention_mask"][:, 1:].contiguous()
                seq_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                seq_loss = token_losses.mean(dim=1)
            batch_ppl = torch.exp(seq_loss).detach().cpu().tolist()
            out.extend([float(v) for v in batch_ppl])
        except Exception:
            out.extend([float("nan")] * len(batch))
    return out


def process_month_file(
    subreddit: str,
    file_path: Path,
    output_path: Path,
    overwrite: bool,
    max_days_per_month: int,
    device_arg: str,
    batch_size: int,
    model_config: Dict[str, str],
) -> ProfilingStats:
    """Function summary: compute per-comment features for one subreddit-month parquet and write one output shard."""
    stats = ProfilingStats()
    if output_path.exists() and not overwrite:
        print(
            f"[compute_comment_features] skip_existing subreddit={subreddit} month={file_path.stem} out={output_path}",
            flush=True,
        )
        return stats

    print(f"[compute_comment_features] start subreddit={subreddit} month={file_path.stem}", flush=True)
    t_read = time.perf_counter()
    frame = pd.read_parquet(file_path, columns=["id", "subreddit", "date_utc", "body"])
    stats.phase_read_s += time.perf_counter() - t_read
    frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
    frame["body"] = frame["body"].astype("string").fillna("")
    frame["date_utc"] = frame["date_utc"].astype("string")
    if max_days_per_month > 0 and not frame.empty:
        keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
        frame = frame[frame["date_utc"].isin(keep_days)].copy()
    if frame.empty:
        print(f"[compute_comment_features] empty_after_filter subreddit={subreddit} month={file_path.stem}", flush=True)
        return stats
    stats.comments_processed += int(len(frame))

    analyzer = SentimentIntensityAnalyzer()
    t_feat = time.perf_counter()
    records: list[Dict[str, Any]] = []
    texts: list[str] = []
    for row in frame.itertuples(index=False):
        text = str(getattr(row, "body", "") or "")
        text_lc = text.lower()
        words = tokenize_words(text)
        n_words = len(words)
        word_counter = Counter(words)
        sentence_count = sum(1 for _ in SENTENCE_SPLIT_RE.finditer(text))
        sentence_count = max(sentence_count, 1 if n_words > 0 else 0)
        total_word_chars = int(sum(len(w) for w in words))
        sentiment = analyzer.polarity_scores(text) if text.strip() else {"compound": 0.0}
        compound = float(sentiment.get("compound", 0.0))
        strict_word_counts = {word: int(word_counter.get(word, 0)) for word in STRICT_AI_WORDS}
        strict_hits_total = int(sum(strict_word_counts.values()))
        extended_hits_total = int(sum(count for token, count in word_counter.items() if token in EXTENDED_AI_WORD_SET))
        assistant_tone_hits = int(sum(count_phrase_occurrences(text_lc, p) for p in ASSISTANT_TONE_PATTERNS))
        toxic_hits = int(sum(count_phrase_occurrences(text_lc, p) for p in TOXIC_PATTERNS))
        contraction_hits = int(sum(word_counter.get(w, 0) for w in CONTRACTIONS_SET))
        full_form_hits = int(count_full_form_occurrences(text_lc))
        passive_hits = int(len(PASSIVE_RE.findall(text)))
        passive_rate = (float(passive_hits) / float(n_words) * 100.0) if n_words > 0 else 0.0
        em_dash_count = int(text.count("\u2014"))
        en_dash_count = int(text.count("\u2013"))
        ascii_double_hyphen_count = int(count_ascii_double_hyphen(text))
        colon_count = int(text.count(":"))
        open_paren_count = int(text.count("("))
        curly_quote_count = int(count_curly_quotes(text))
        markdown_bold_pair_count = int(count_markdown_bold_pairs(text))
        markdown_heading_line_count = int(count_markdown_heading_lines(text))
        hedging_phrase_hits = int(sum_phrase_hits(text_lc, HEDGING_PATTERNS))
        polite_closer_hits = int(sum_phrase_hits(text_lc, POLITE_CLOSER_PATTERNS))
        signposting_phrase_hits = int(sum_phrase_hits(text_lc, SIGNPOSTING_PATTERNS))
        avg_wps = avg_words_per_sentence(n_words, sentence_count)
        author_val = str(getattr(row, "author", "") or "")
        created_raw = getattr(row, "created_utc", pd.NA)
        if pd.isna(created_raw):
            created_out: Any = pd.NA
        else:
            try:
                created_out = int(created_raw)
            except (TypeError, ValueError):
                created_out = pd.NA
        record: Dict[str, Any] = {
            "id": str(getattr(row, "id", "") or ""),
            "subreddit": str(getattr(row, "subreddit", "") or ""),
            "date_utc": str(getattr(row, "date_utc", "") or ""),
            "author": author_val,
            "created_utc": created_out,
            "body": text,
            "n_words_comment": int(n_words),
            "comment_length_words": float(n_words),
            "length_bucket": length_bucket(n_words),
            "detector_confidence_flag": detector_confidence_flag(n_words),
            "semicolon_count": int(text.count(";")),
            "em_dash_count": em_dash_count,
            "en_dash_count": en_dash_count,
            "ascii_double_hyphen_count": ascii_double_hyphen_count,
            "colon_count": colon_count,
            "open_paren_count": open_paren_count,
            "curly_quote_count": curly_quote_count,
            "markdown_bold_pair_count": markdown_bold_pair_count,
            "markdown_heading_line_count": markdown_heading_line_count,
            "hedging_phrase_hits": hedging_phrase_hits,
            "polite_closer_hits": polite_closer_hits,
            "signposting_phrase_hits": signposting_phrase_hits,
            "avg_words_per_sentence_comment": float(avg_wps),
            "total_word_chars_comment": int(total_word_chars),
            "sentence_count_comment": int(sentence_count),
            "vader_compound": float(compound),
            "vader_negativity": float(max(0.0, -compound)),
            "toxicity_score_comment": float(max(0.0, -compound)),
            "toxic_lexicon_hits": int(toxic_hits),
            "contraction_count": int(contraction_hits),
            "full_form_count": int(full_form_hits),
            "formality_balance_count": int(full_form_hits - contraction_hits),
            "assistant_tone_phrase_count": int(assistant_tone_hits),
            "list_structure_flag": int(bool(LIST_STRUCTURE_RE.search(text))),
            "passive_count": int(passive_hits),
            "passive_rate_100w": float(passive_rate),
            "strict_ai_word_hits_total": int(strict_hits_total),
            "extended_ai_word_hits_total": int(extended_hits_total),
            "features_computed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for word in STRICT_AI_WORDS:
            record[f"strict_word_count__{word}"] = int(strict_word_counts[word])
        records.append(record)
        texts.append(text[:2000])
    stats.phase_feature_s += time.perf_counter() - t_feat

    t_model = time.perf_counter()
    device = choose_device(device_arg)
    detector_primary_pipe = maybe_load_transformers_pipeline("text-classification", model_config["detector_primary"], device)
    detector_secondary_pipe = maybe_load_transformers_pipeline("text-classification", model_config["detector_secondary"], device)
    hostility_pipe = maybe_load_transformers_pipeline("text-classification", model_config["hostility"], device)
    emotion_pipe = maybe_load_transformers_pipeline("text-classification", model_config["emotion"], device)
    ppl_tokenizer, ppl_model = init_perplexity_components(model_config["perplexity"], device)
    print(
        "[compute_comment_features] models "
        f"subreddit={subreddit} month={file_path.stem} device={device} "
        f"primary={'ok' if detector_primary_pipe is not None else 'na'} "
        f"secondary={'ok' if detector_secondary_pipe is not None else 'na'} "
        f"hostility={'ok' if hostility_pipe is not None else 'na'} "
        f"emotion={'ok' if emotion_pipe is not None else 'na'} "
        f"perplexity={'ok' if ppl_model is not None else 'na'}",
        flush=True,
    )

    primary_ai = batch_text_classification_scores(detector_primary_pipe, texts, batch_size=batch_size)
    secondary_ai = batch_text_classification_scores(detector_secondary_pipe, texts, batch_size=batch_size)
    hostility_scores = batch_text_classification_scores(hostility_pipe, texts, batch_size=batch_size)
    emotion_scores = batch_emotion_scores(emotion_pipe, texts, batch_size=batch_size)
    ppl_scores = batch_perplexity(texts, ppl_tokenizer, ppl_model, batch_size=batch_size, device=device)
    stats.phase_model_s += time.perf_counter() - t_model

    for idx, record in enumerate(records):
        record["detector_primary_ai_prob"] = float(primary_ai[idx]) if idx < len(primary_ai) else float("nan")
        record["detector_primary_human_score"] = (
            float(1.0 - primary_ai[idx]) if idx < len(primary_ai) and pd.notna(primary_ai[idx]) else float("nan")
        )
        record["detector_secondary_ai_prob"] = float(secondary_ai[idx]) if idx < len(secondary_ai) else float("nan")
        record["detector_secondary_human_score"] = (
            float(1.0 - secondary_ai[idx]) if idx < len(secondary_ai) and pd.notna(secondary_ai[idx]) else float("nan")
        )
        record["hostility_score"] = float(hostility_scores[idx]) if idx < len(hostility_scores) else float("nan")
        record["emotion_anger"] = float(emotion_scores["anger"][idx])
        record["emotion_fear"] = float(emotion_scores["fear"][idx])
        record["emotion_sadness"] = float(emotion_scores["sadness"][idx])
        record["emotion_surprise"] = float(emotion_scores["surprise"][idx])
        record["perplexity"] = float(ppl_scores[idx]) if idx < len(ppl_scores) else float("nan")
        record["log_perplexity"] = (
            float(math.log(record["perplexity"])) if pd.notna(record["perplexity"]) and record["perplexity"] > 0 else float("nan")
        )
        record["detector_primary_model_id"] = model_config["detector_primary"]
        record["detector_secondary_model_id"] = model_config["detector_secondary"]
        record["hostility_model_id"] = model_config["hostility"]
        record["emotion_model_id"] = model_config["emotion"]
        record["perplexity_model_id"] = model_config["perplexity"]
        record["device_used"] = device

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_write = time.perf_counter()
    out_df.to_parquet(output_path, index=False, compression="zstd")
    stats.phase_write_s += time.perf_counter() - t_write
    stats.files_processed += 1
    print(
        f"[compute_comment_features] done subreddit={subreddit} month={file_path.stem} "
        f"comments={len(out_df)} read_s={stats.phase_read_s:.2f} feat_s={stats.phase_feature_s:.2f} "
        f"model_s={stats.phase_model_s:.2f} write_s={stats.phase_write_s:.2f}",
        flush=True,
    )
    return stats


def emit_profiling(stats: ProfilingStats, profile: bool, profile_output: str) -> None:
    """Function summary: print and optionally persist profiling counters for performance benchmarking."""
    if not profile and not profile_output:
        return
    payload = stats.as_dict()
    print(f"[compute_comment_features] profile={json.dumps(payload, sort_keys=True)}", flush=True)
    if profile_output:
        Path(profile_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: orchestrate per-month comment feature extraction and write reusable parquet shards."""
    args = parse_args()
    config = load_config(args.config)
    interim_dir = Path(config["paths"]["interim_dir"])
    cleaned_dir = interim_dir / "cleaned_monthly_chunks"
    out_dir = interim_dir / "comment_features"
    configured_subreddits = list(config["subreddits"]["primary"])
    if args.subreddits.strip():
        allow_subs = {s.strip() for s in args.subreddits.split(",") if s.strip()}
        configured_subreddits = [s for s in configured_subreddits if s in allow_subs]
    allow_months = {m.strip() for m in args.months.split(",") if m.strip()}

    feature_cfg = config.get("comment_features", {})
    model_config = {
        "detector_primary": str(feature_cfg.get("detector_primary_model", "desklib/ai-text-detector-v1.01")),
        "detector_secondary": str(
            feature_cfg.get("detector_secondary_model", "fakespot-ai/roberta-base-ai-text-detection-v1")
        ),
        "hostility": str(feature_cfg.get("hostility_model", "unitary/unbiased-toxic-roberta")),
        "emotion": str(feature_cfg.get("emotion_model", "j-hartmann/emotion-english-distilroberta-base")),
        "perplexity": str(feature_cfg.get("perplexity_model", "gpt2")),
    }

    jobs = list(
        iter_monthly_files(
            cleaned_monthly_chunks_dir=cleaned_dir,
            subreddits=configured_subreddits,
            allowed_months=allow_months,
            max_month_files_per_subreddit=int(args.max_month_files_per_subreddit),
            max_total_month_files=int(args.max_total_month_files),
        )
    )
    if not jobs:
        raise FileNotFoundError(f"No cleaned monthly files found under: {cleaned_dir}")

    stats = ProfilingStats()
    started_at = time.perf_counter()
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = []
            for subreddit, file_path in jobs:
                futures.append(
                    pool.submit(
                        process_month_file,
                        subreddit,
                        file_path,
                        out_dir / subreddit / f"{file_path.stem}.parquet",
                        bool(args.overwrite),
                        int(args.max_days_per_month),
                        str(args.device),
                        int(args.batch_size),
                        model_config,
                    )
                )
            for idx, future in enumerate(as_completed(futures), start=1):
                stats.merge(future.result())
                elapsed = time.perf_counter() - started_at
                print(f"[compute_comment_features] completed={idx}/{len(futures)} elapsed_s={elapsed:.1f}", flush=True)
    else:
        for idx, (subreddit, file_path) in enumerate(jobs, start=1):
            file_stats = process_month_file(
                subreddit=subreddit,
                file_path=file_path,
                output_path=out_dir / subreddit / f"{file_path.stem}.parquet",
                overwrite=bool(args.overwrite),
                max_days_per_month=int(args.max_days_per_month),
                device_arg=str(args.device),
                batch_size=int(args.batch_size),
                model_config=model_config,
            )
            stats.merge(file_stats)
            elapsed = time.perf_counter() - started_at
            print(
                f"[compute_comment_features] completed={idx}/{len(jobs)} subreddit={subreddit} month={file_path.stem} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))


if __name__ == "__main__":
    main()
