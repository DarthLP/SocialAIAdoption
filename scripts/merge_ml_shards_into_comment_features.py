"""
Script summary:
Merges optional Colab-produced `comment_features_ml/` Parquet shards by comment `id` into
feature rows computed from cleaned monthly chunks. Lexical/rule-based fields reuse the same
implementations as `compute_comment_features.py` (loaded via importlib); ML columns come from
the ML shards when present. Writes the combined schema under `comment_features/` so
`prepare_event_time_metrics.py --prefer_comment_features` works unchanged.

Typical workflow:
1. Colab: standalone `colab_compute_comment_features_gpu.ipynb` → `interim_dir/comment_features_ml/`
2. Local: this script → `interim_dir/comment_features/` (with or without ML shards; missing
   months leave ML columns NaN/empty).

How to apply/run:
- With ML shards (default `--ml_subdir comment_features_ml`):
  `.venv/bin/python scripts/merge_ml_shards_into_comment_features.py --config config/political_forums_setup.yaml`
- ML shard missing for a month: ML columns are left NaN/empty for that shard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_PROJECT_LOG = "[merge_ml_shards_into_comment_features]"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


def _load_monolithic_script():
    """Function summary: load `compute_comment_features` as a module for shared lexical helpers."""
    path = PROJECT_ROOT / "scripts" / "compute_comment_features.py"
    name = "_ccf_merge_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ccf = _load_monolithic_script()
ProfilingStats = _ccf.ProfilingStats


def iter_monthly_files(
    cleaned_monthly_chunks_dir: Path,
    subreddits: Iterable[str],
    allowed_months: set[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
) -> Iterable[tuple[str, Path]]:
    """Function summary: yield cleaned monthly parquet paths (same behaviour as compute_comment_features)."""
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


ML_MERGE_COLUMNS = [
    "detector_primary_ai_prob",
    "detector_primary_human_score",
    "detector_secondary_ai_prob",
    "detector_secondary_human_score",
    "hostility_score",
    "emotion_anger",
    "emotion_fear",
    "emotion_sadness",
    "emotion_surprise",
    "perplexity",
    "log_perplexity",
    "detector_primary_model_id",
    "detector_secondary_model_id",
    "hostility_model_id",
    "emotion_model_id",
    "perplexity_model_id",
    "device_used",
    "ml_features_computed_at_utc",
]


def _add_empty_ml_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add ML columns with nulls when no `comment_features_ml` shard exists."""
    out = frame.copy()
    for col in ML_MERGE_COLUMNS:
        if col in out.columns:
            continue
        if col.endswith("_model_id") or col == "device_used":
            out[col] = ""
        elif col == "ml_features_computed_at_utc":
            out[col] = pd.NA
        else:
            out[col] = float("nan")
    return out


def _merge_ml_shard(lex_df: pd.DataFrame, ml_path: Path) -> pd.DataFrame:
    """Function summary: left-join ML columns from one Parquet shard on comment `id`."""
    if not ml_path.exists():
        return _add_empty_ml_columns(lex_df)
    ml = pd.read_parquet(ml_path)
    drop_overlap = [c for c in ("subreddit", "date_utc") if c in ml.columns]
    ml_only = ml.drop(columns=drop_overlap, errors="ignore")
    merged = lex_df.merge(ml_only, on="id", how="left", suffixes=("", "_ml_dup"))
    dup_cols = [c for c in merged.columns if c.endswith("_ml_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    return _add_empty_ml_columns(merged)


def _final_column_order() -> list[str]:
    """Function summary: return column order matching monolithic `compute_comment_features` output."""
    strict_words = list(_ccf.STRICT_AI_WORDS)
    base = [
        "id",
        "subreddit",
        "date_utc",
        "body",
        "n_words_comment",
        "comment_length_words",
        "length_bucket",
        "detector_confidence_flag",
        "semicolon_count",
        "total_word_chars_comment",
        "sentence_count_comment",
        "vader_compound",
        "vader_negativity",
        "toxicity_score_comment",
        "toxic_lexicon_hits",
        "contraction_count",
        "full_form_count",
        "formality_balance_count",
        "assistant_tone_phrase_count",
        "list_structure_flag",
        "passive_count",
        "passive_rate_100w",
        "strict_ai_word_hits_total",
        "extended_ai_word_hits_total",
        "features_computed_at_utc",
    ]
    for w in strict_words:
        base.append(f"strict_word_count__{w}")
    base.extend(ML_MERGE_COLUMNS)
    return base


def build_lexical_records(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    """Function summary: build one dict per row with lexical fields only (mirrors monolithic script)."""
    analyzer = SentimentIntensityAnalyzer()
    records: list[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        text = str(getattr(row, "body", "") or "")
        text_lc = text.lower()
        words = _ccf.tokenize_words(text)
        n_words = len(words)
        word_counter = Counter(words)
        sentence_count = sum(1 for _ in _ccf.SENTENCE_SPLIT_RE.finditer(text))
        sentence_count = max(sentence_count, 1 if n_words > 0 else 0)
        total_word_chars = int(sum(len(w) for w in words))
        sentiment = analyzer.polarity_scores(text) if text.strip() else {"compound": 0.0}
        compound = float(sentiment.get("compound", 0.0))
        strict_word_counts = {word: int(word_counter.get(word, 0)) for word in _ccf.STRICT_AI_WORDS}
        strict_hits_total = int(sum(strict_word_counts.values()))
        extended_hits_total = int(
            sum(count for token, count in word_counter.items() if token in _ccf.EXTENDED_AI_WORD_SET)
        )
        assistant_tone_hits = int(sum(_ccf.count_phrase_occurrences(text_lc, p) for p in _ccf.ASSISTANT_TONE_PATTERNS))
        toxic_hits = int(sum(_ccf.count_phrase_occurrences(text_lc, p) for p in _ccf.TOXIC_PATTERNS))
        contraction_hits = int(sum(word_counter.get(w, 0) for w in _ccf.CONTRACTIONS_SET))
        full_form_hits = int(_ccf.count_full_form_occurrences(text_lc))
        passive_hits = int(len(_ccf.PASSIVE_RE.findall(text)))
        passive_rate = (float(passive_hits) / float(n_words) * 100.0) if n_words > 0 else 0.0
        record: Dict[str, Any] = {
            "id": str(getattr(row, "id", "") or ""),
            "subreddit": str(getattr(row, "subreddit", "") or ""),
            "date_utc": str(getattr(row, "date_utc", "") or ""),
            "body": text,
            "n_words_comment": int(n_words),
            "comment_length_words": float(n_words),
            "length_bucket": _ccf.length_bucket(n_words),
            "detector_confidence_flag": _ccf.detector_confidence_flag(n_words),
            "semicolon_count": int(text.count(";")),
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
            "list_structure_flag": int(bool(_ccf.LIST_STRUCTURE_RE.search(text))),
            "passive_count": int(passive_hits),
            "passive_rate_100w": float(passive_rate),
            "strict_ai_word_hits_total": int(strict_hits_total),
            "extended_ai_word_hits_total": int(extended_hits_total),
            "features_computed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for word in _ccf.STRICT_AI_WORDS:
            record[f"strict_word_count__{word}"] = int(strict_word_counts[word])
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for merge + lexical extraction via monolithic helpers."""
    parser = argparse.ArgumentParser(
        description="Merge optional comment_features_ml Parquet shards and write final comment_features/ tree."
    )
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ml_subdir",
        type=str,
        default="comment_features_ml",
        help="Directory under interim_dir holding ML-only shards (mirror of cleaned_monthly_chunks layout).",
    )
    parser.add_argument("--max_month_files_per_subreddit", type=int, default=0)
    parser.add_argument("--max_total_month_files", type=int, default=0)
    parser.add_argument("--max_days_per_month", type=int, default=0)
    parser.add_argument("--subreddits", type=str, default="")
    parser.add_argument("--months", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_output", type=str, default="")
    return parser.parse_args()


def process_month_merge(
    subreddit: str,
    file_path: Path,
    output_path: Path,
    ml_root: Path,
    overwrite: bool,
    max_days_per_month: int,
    column_order: list[str],
) -> ProfilingStats:
    """Function summary: compute lexical rows for one shard, merge ML, write final comment_features parquet."""
    stats = ProfilingStats()
    if output_path.exists() and not overwrite:
        print(
            f"{_PROJECT_LOG} skip_existing subreddit={subreddit} month={file_path.stem} out={output_path}",
            flush=True,
        )
        return stats

    print(f"{_PROJECT_LOG} start subreddit={subreddit} month={file_path.stem}", flush=True)
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
        print(
            f"{_PROJECT_LOG} empty_after_filter subreddit={subreddit} month={file_path.stem}",
            flush=True,
        )
        return stats
    stats.comments_processed += int(len(frame))

    t_feat = time.perf_counter()
    records = build_lexical_records(frame)
    stats.phase_feature_s += time.perf_counter() - t_feat

    lex_df = pd.DataFrame.from_records(records)
    ml_path = ml_root / subreddit / f"{file_path.stem}.parquet"
    t_merge = time.perf_counter()
    merged = _merge_ml_shard(lex_df, ml_path)
    stats.phase_model_s += time.perf_counter() - t_merge

    ordered = [c for c in column_order if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    out_df = merged[ordered + extra]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_write = time.perf_counter()
    out_df.to_parquet(output_path, index=False, compression="zstd")
    stats.phase_write_s += time.perf_counter() - t_write
    stats.files_processed += 1
    print(
        f"{_PROJECT_LOG} done subreddit={subreddit} month={file_path.stem} "
        f"comments={len(out_df)} ml_shard={'ok' if ml_path.exists() else 'missing'}",
        flush=True,
    )
    return stats


def emit_profiling(stats: ProfilingStats, profile: bool, profile_output: str) -> None:
    """Function summary: print or save profiling stats."""
    if not profile and not profile_output:
        return
    payload = stats.as_dict()
    print(f"{_PROJECT_LOG} profile={json.dumps(payload, sort_keys=True)}", flush=True)
    if profile_output:
        Path(profile_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run merge + lexical extraction for all configured monthly shards."""
    args = parse_args()
    config = load_config(args.config)
    interim_dir = Path(config["paths"]["interim_dir"])
    cleaned_dir = interim_dir / "cleaned_monthly_chunks"
    out_dir = interim_dir / "comment_features"
    ml_root = interim_dir / args.ml_subdir

    configured_subreddits = list(config["subreddits"]["primary"])
    if args.subreddits.strip():
        allow_subs = {s.strip() for s in args.subreddits.split(",") if s.strip()}
        configured_subreddits = [s for s in configured_subreddits if s in allow_subs]
    allow_months = {m.strip() for m in args.months.split(",") if m.strip()}

    column_order = _final_column_order()

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
                        process_month_merge,
                        subreddit,
                        file_path,
                        out_dir / subreddit / f"{file_path.stem}.parquet",
                        ml_root,
                        bool(args.overwrite),
                        int(args.max_days_per_month),
                        column_order,
                    )
                )
            for idx, future in enumerate(as_completed(futures), start=1):
                stats.merge(future.result())
                print(
                    f"{_PROJECT_LOG} completed={idx}/{len(futures)} elapsed_s={time.perf_counter() - started_at:.1f}",
                    flush=True,
                )
    else:
        for idx, (subreddit, file_path) in enumerate(jobs, start=1):
            file_stats = process_month_merge(
                subreddit=subreddit,
                file_path=file_path,
                output_path=out_dir / subreddit / f"{file_path.stem}.parquet",
                ml_root=ml_root,
                overwrite=bool(args.overwrite),
                max_days_per_month=int(args.max_days_per_month),
                column_order=column_order,
            )
            stats.merge(file_stats)
            print(
                f"{_PROJECT_LOG} completed={idx}/{len(jobs)} subreddit={subreddit} month={file_path.stem} elapsed_s={time.perf_counter() - started_at:.1f}",
                flush=True,
            )
    emit_profiling(stats=stats, profile=bool(args.profile), profile_output=str(args.profile_output or ""))


if __name__ == "__main__":
    main()
