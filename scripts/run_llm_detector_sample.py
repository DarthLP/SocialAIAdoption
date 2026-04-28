"""
Script summary:
This script runs optional CPU-only LLM-likeness detector checks on a stratified
sample of cleaned comments. It is designed as a robustness layer that complements
full-corpus lexical/structure metrics with sampled detector-based signals.

Functionality:
- Reads cleaned monthly chunks from
  `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
- Builds deterministic subreddit x day stratified samples using a fixed seed.
- Computes two detector-style outputs:
  - heuristic_llm_style_score (free, local, no additional dependencies)
  - optional Hugging Face classifier probability (if transformers is installed)
- Aggregates sampled scores to daily subreddit and pooled tables.
- Logs pinned model metadata for reproducibility.

How to apply/run:
- Heuristic-only sampled robustness:
  `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml`
- Heuristic + optional HF classifier:
  `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml --use_hf_model`
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts

HF_MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"
HF_MODEL_REVISION = "main"
WORD_RE = re.compile(r"[A-Za-z']+")
ASSISTANT_PHRASES = [
    "in conclusion",
    "in summary",
    "it is important to note",
    "it is worth noting",
    "moreover",
    "furthermore",
    "additionally",
    "a testament to",
]
AI_WORDS = {
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
}


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI options for sampling behavior and optional model usage."""
    parser = argparse.ArgumentParser(description="Run optional sampled LLM detector checks.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument("--sample_per_day_subreddit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument(
        "--use_hf_model",
        action="store_true",
        help="Run optional Hugging Face classifier scoring if transformers is available.",
    )
    return parser.parse_args()


def iter_monthly_files(cleaned_monthly_chunks_dir: Path, subreddits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Function summary: iterate cleaned monthly Parquet files for configured subreddits."""
    for subreddit in sorted(subreddits):
        sub_dir = cleaned_monthly_chunks_dir / subreddit
        if not sub_dir.exists():
            continue
        for parquet_path in sorted(sub_dir.glob("*.parquet")):
            yield subreddit, parquet_path


def stable_hash_as_float(text: str) -> float:
    """Function summary: map text deterministically to [0, 1) for stable sampling."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def tokenize(text: str) -> list[str]:
    """Function summary: tokenize text into lowercase words for lightweight style features."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def heuristic_llm_style_score(text: str) -> float:
    """Function summary: compute a bounded heuristic LLM-style score using lexical and phrase cues."""
    text_lc = (text or "").lower()
    tokens = tokenize(text_lc)
    if not tokens:
        return 0.0
    ai_word_hits = sum(1 for t in tokens if t in AI_WORDS)
    phrase_hits = sum(text_lc.count(phrase) for phrase in ASSISTANT_PHRASES)
    long_word_ratio = sum(1 for t in tokens if len(t) >= 8) / float(len(tokens))
    score = (
        min(1.0, ai_word_hits / 4.0) * 0.45
        + min(1.0, phrase_hits / 2.0) * 0.35
        + min(1.0, long_word_ratio / 0.35) * 0.20
    )
    return float(max(0.0, min(1.0, score)))


def maybe_build_hf_pipeline(use_hf_model: bool) -> Any:
    """Function summary: initialize optional HF text-classification pipeline when requested."""
    if not use_hf_model:
        return None
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - runtime environment dependent.
        raise ImportError(
            "transformers is required for --use_hf_model. Install it in .venv first."
        ) from exc
    return pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        revision=HF_MODEL_REVISION,
        truncation=True,
    )


def score_with_hf(pipe: Any, text: str) -> float:
    """Function summary: score one text with optional HF detector and normalize to AI probability-like value."""
    if pipe is None:
        return float("nan")
    result = pipe(text[:2000])[0]
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))
    if "human" in label:
        return 1.0 - score
    return score


def main() -> None:
    """Function summary: run sampled detector workflow and write daily subreddit plus pooled outputs."""
    args = parse_args()
    random.seed(int(args.seed))
    config = load_config(args.config)
    launch_ts = utc_ts(config["event_window"]["launch_day_utc"])
    launch_date = datetime.fromtimestamp(launch_ts, tz=timezone.utc).date()

    cleaned_monthly_chunks_dir = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    tables_dir = Path(config["paths"]["tables_dir"]) / "event_time"
    tables_dir.mkdir(parents=True, exist_ok=True)
    subreddits = list(config["subreddits"]["primary"])

    hf_pipe = maybe_build_hf_pipeline(bool(args.use_hf_model))

    rows: list[Dict[str, Any]] = []
    sample_target = int(args.sample_per_day_subreddit)
    for subreddit, file_path in iter_monthly_files(cleaned_monthly_chunks_dir, subreddits):
        month_df = pd.read_parquet(file_path)
        required = {"id", "body", "date_utc", "subreddit"}
        missing = sorted(required.difference(month_df.columns))
        if missing:
            raise ValueError(f"Missing required columns {missing} in {file_path}")
        month_df = month_df[month_df["subreddit"].astype("string") == subreddit]
        if month_df.empty:
            continue
        for date_utc, day_df in month_df.groupby("date_utc", sort=True):
            sampled_records: list[Dict[str, Any]] = []
            for row in day_df.itertuples(index=False):
                text = str(getattr(row, "body", "") or "")
                comment_id = str(getattr(row, "id", "") or "")
                # Stable Bernoulli gate then cap by target with deterministic ranking.
                h = stable_hash_as_float(f"{subreddit}|{date_utc}|{comment_id}")
                if h <= min(1.0, sample_target / 5000.0):
                    sampled_records.append({"id": comment_id, "text": text, "h": h})
            sampled_records = sorted(sampled_records, key=lambda r: (r["h"], str(r["id"])))[:sample_target]

            if not sampled_records:
                continue
            heuristic_scores = [heuristic_llm_style_score(rec["text"]) for rec in sampled_records]
            hf_scores = [score_with_hf(hf_pipe, rec["text"]) for rec in sampled_records] if hf_pipe is not None else []
            event_time_t = (datetime.fromisoformat(str(date_utc)).date() - launch_date).days
            out_row = {
                "subreddit": subreddit,
                "date_utc": str(date_utc),
                "event_time_t": int(event_time_t),
                "sample_size": len(sampled_records),
                "heuristic_llm_style_score_mean": float(sum(heuristic_scores) / len(heuristic_scores)),
                "hf_llm_probability_mean": float(sum(hf_scores) / len(hf_scores)) if hf_scores else float("nan"),
                "hf_model_id": HF_MODEL_ID if hf_pipe is not None else "",
                "hf_model_revision": HF_MODEL_REVISION if hf_pipe is not None else "",
                "seed": int(args.seed),
            }
            rows.append(out_row)

    if not rows:
        raise FileNotFoundError(f"No sampled records found under: {cleaned_monthly_chunks_dir}")

    by_subreddit_df = pd.DataFrame(rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True)
    pooled_rows = []
    for date_utc, group in by_subreddit_df.groupby("date_utc", sort=True):
        total_n = int(group["sample_size"].sum())
        if total_n <= 0:
            continue
        pooled_rows.append(
            {
                "subreddit": "ALL",
                "date_utc": str(date_utc),
                "event_time_t": int(group["event_time_t"].iloc[0]),
                "sample_size": total_n,
                "heuristic_llm_style_score_mean": float(
                    (group["heuristic_llm_style_score_mean"] * group["sample_size"]).sum() / total_n
                ),
                "hf_llm_probability_mean": float((group["hf_llm_probability_mean"] * group["sample_size"]).sum() / total_n)
                if group["hf_llm_probability_mean"].notna().any()
                else float("nan"),
                "hf_model_id": HF_MODEL_ID if bool(args.use_hf_model) else "",
                "hf_model_revision": HF_MODEL_REVISION if bool(args.use_hf_model) else "",
                "seed": int(args.seed),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows).sort_values("date_utc").reset_index(drop=True)
    out_df = pd.concat([by_subreddit_df, pooled_df], ignore_index=True)
    out_df.to_csv(tables_dir / "llm_detector_sample_scores_daily.csv", index=False)


if __name__ == "__main__":
    main()
