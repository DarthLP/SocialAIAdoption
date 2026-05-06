"""
Script summary:
Computes the daily subreddit-level repetition / template-similarity metric used in
event-time analysis: mean over comments of the max Jaccard overlap between each
comment's word set and any of the last K prior comments within the same calendar
day and subreddit. Reads cleaned monthly Parquet (same layout as feature extraction),
sorts rows by `created_utc` then `id` within each day for a time-ordered stream, and
writes a narrow CSV for `prepare_event_time_metrics.py` to left-merge.

Functionality:
- Loads `config/political_forums_setup.yaml` and scans `cleaned_monthly_chunks/`.
- For each (subreddit, date_utc), sorts comments by `created_utc` ascending (stable
  tie-break on `id`); if `created_utc` is entirely missing for a shard, falls back
  to file order and logs once per monthly file.
- Emits `results/tables/event_time/repetition_daily_by_subreddit.csv` with columns
  `subreddit`, `date_utc`, `repetition_template_similarity`, `n_comments`.

How to apply/run:
- Full run:
  `.venv/bin/python scripts/features/compute_daily_repetition_similarity.py --config config/political_forums_setup.yaml`
- Bounded sample:
  `.venv/bin/python scripts/features/compute_daily_repetition_similarity.py --config config/political_forums_setup.yaml --max_total_month_files 2 --max_days_per_month 10`
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

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

from src.config_utils import load_config


def _load_compute_comment_features_module():
    """Function summary: import `compute_comment_features` as a module for shared Parquet readers."""
    path = Path(__file__).resolve().parent / "compute_comment_features.py"
    name = "_ccf_repetition_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ccf = _load_compute_comment_features_module()


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI flags for config path, deque window, and bounded runs."""
    parser = argparse.ArgumentParser(
        description="Compute daily repetition_template_similarity from cleaned monthly chunks."
    )
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--similarity_window",
        type=int,
        default=20,
        help="How many previous comments to retain in the deque for max-Jaccard comparisons.",
    )
    parser.add_argument(
        "--min_words_for_similarity",
        type=int,
        default=0,
        help="If >0, only comments with at least this many tokens update the deque and similarity sum.",
    )
    parser.add_argument("--max_month_files_per_subreddit", type=int, default=0)
    parser.add_argument("--max_total_month_files", type=int, default=0)
    parser.add_argument("--max_days_per_month", type=int, default=0)
    parser.add_argument("--subreddits", type=str, default="", help="Comma-separated subreddit allow-list.")
    parser.add_argument("--months", type=str, default="", help="Comma-separated YYYY-MM allow-list.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Override output CSV path (default: <tables_dir>/event_time/repetition_daily_by_subreddit.csv).",
    )
    return parser.parse_args()


def iter_monthly_files(
    cleaned_monthly_chunks_dir: Path,
    subreddits: Iterable[str],
    allowed_months: set[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
) -> Iterable[tuple[str, Path]]:
    """Function summary: yield cleaned monthly parquet paths under each configured subreddit."""
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


def comment_similarity_to_recent(tokens: set[str], recent_tokens: deque[set[str]]) -> float:
    """Function summary: return max Jaccard similarity between `tokens` and each prior set in `recent_tokens`."""
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


def aggregate_similarity_for_ordered_texts(
    texts: list[str],
    similarity_window: int,
    min_words_for_similarity: int,
) -> tuple[float, int]:
    """Function summary: run the legacy deque logic over an ordered list of comment bodies.

    Parameters:
    - texts: comment bodies in processing order (typically time-sorted within a day).
    - similarity_window: maximum prior comments kept in the deque.
    - min_words_for_similarity: minimum token count to contribute similarity and deque updates.

    Returns:
    - Tuple of (repetition_template_similarity = similarity_sum / n_comments, n_comments).
    """
    recent_tokens: deque[set[str]] = deque()
    similarity_sum = 0.0
    n_comments = 0
    for text in texts:
        words = _ccf.tokenize_words(text or "")
        n_words = len(words)
        word_set = set(words)
        n_comments += 1
        if n_words >= min_words_for_similarity:
            similarity = comment_similarity_to_recent(word_set, recent_tokens)
            similarity_sum += float(similarity)
            recent_tokens.append(word_set)
            while len(recent_tokens) > similarity_window:
                recent_tokens.popleft()
    repetition = (float(similarity_sum) / float(n_comments)) if n_comments else 0.0
    return repetition, n_comments


def process_month_file(
    subreddit: str,
    file_path: Path,
    similarity_window: int,
    min_words_for_similarity: int,
    max_days_per_month: int,
    logged_fallback: set[Path],
) -> list[dict[str, Any]]:
    """Function summary: emit one dict per (subreddit, date_utc) with repetition and n_comments."""
    frame = _ccf.read_cleaned_month_for_features(file_path)
    frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
    frame["body"] = frame["body"].astype("string").fillna("")
    frame["date_utc"] = frame["date_utc"].astype("string")
    if max_days_per_month > 0 and not frame.empty:
        keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
        frame = frame[frame["date_utc"].isin(keep_days)].copy()
    if frame.empty:
        return []

    if file_path not in logged_fallback and "created_utc" in frame.columns:
        if bool(frame["created_utc"].isna().all()):
            print(
                f"[compute_daily_repetition_similarity] created_utc_all_null subreddit={subreddit} "
                f"month={file_path.stem} using_file_order",
                flush=True,
            )
            logged_fallback.add(file_path)

    rows: list[dict[str, Any]] = []
    for date_utc, group in frame.groupby("date_utc", sort=True):
        g = group.copy()
        if "created_utc" in g.columns and not bool(g["created_utc"].isna().all()):
            g = g.sort_values(by=["created_utc", "id"], ascending=[True, True], na_position="last")
        texts = g["body"].astype(str).tolist()
        rep, n = aggregate_similarity_for_ordered_texts(
            texts,
            similarity_window=int(similarity_window),
            min_words_for_similarity=int(min_words_for_similarity),
        )
        rows.append(
            {
                "subreddit": subreddit,
                "date_utc": str(date_utc),
                "repetition_template_similarity": float(rep),
                "n_comments": int(n),
            }
        )
    return rows


def main() -> None:
    """Function summary: scan cleaned monthly shards, compute daily repetition, write one CSV."""
    args = parse_args()
    config = load_config(args.config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    cleaned_dir = interim_dir / "cleaned_monthly_chunks"
    out_path = (
        Path(args.output).expanduser().resolve()
        if str(args.output).strip()
        else (tables_dir / "event_time" / "repetition_daily_by_subreddit.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configured_subreddits = list(config["subreddits"]["primary"])
    if args.subreddits.strip():
        allow_subs = {s.strip() for s in args.subreddits.split(",") if s.strip()}
        configured_subreddits = [s for s in configured_subreddits if s in allow_subs]
    allow_months = {m.strip() for m in args.months.split(",") if m.strip()}

    jobs = list(
        iter_monthly_files(
            cleaned_dir,
            configured_subreddits,
            allow_months,
            int(args.max_month_files_per_subreddit),
            int(args.max_total_month_files),
        )
    )
    if not jobs:
        raise FileNotFoundError(f"No cleaned monthly files found under: {cleaned_dir}")

    logged_fallback: set[Path] = set()
    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, (subreddit, file_path) in enumerate(jobs, start=1):
        part = process_month_file(
            subreddit=subreddit,
            file_path=file_path,
            similarity_window=int(args.similarity_window),
            min_words_for_similarity=int(args.min_words_for_similarity),
            max_days_per_month=int(args.max_days_per_month),
            logged_fallback=logged_fallback,
        )
        all_rows.extend(part)
        print(
            f"[compute_daily_repetition_similarity] done {idx}/{len(jobs)} subreddit={subreddit} "
            f"month={file_path.stem} day_rows={len(part)} elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )

    out_df = pd.DataFrame(all_rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"[compute_daily_repetition_similarity] wrote rows={len(out_df)} path={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
