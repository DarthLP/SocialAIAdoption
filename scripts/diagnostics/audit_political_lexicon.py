"""
Script summary:
Sanity-check political lexicon scoring on benchmark Italy polarization subreddits.

Functionality:
- Reports lexicon term counts and per-subreddit word-weighted political rates.
- Samples example comments with lexicon hits from politicaITA, litigi, and Italia.

How to apply/run:
  .venv/bin/python scripts/diagnostics/audit_political_lexicon.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BENCHMARK_SUBREDDITS = [
    "politicaITA",
    "litigi",
    "Italia",
    "oknotizie",
    "BancaDelMeme",
    "news_and_talk",
    "europe",
    "ukpolitics",
]


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

from src.config_utils import (  # noqa: E402
    build_subreddit_metadata_table,
    load_config,
    parallel_political_lexicon_path,
    resolve_primary_subreddits,
)
from src.political_lexicon import (  # noqa: E402
    get_lexicon,
    political_rate_100w,
    score_comment_political_salience,
)


def read_parquet_shard_safe(shard: Path, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Function summary: read a Parquet shard, skipping corrupt or empty files.

    Parameters:
    - shard: path to shard file.
    - columns: optional column subset.

    Returns:
    - DataFrame or None if unreadable.
    """
    if not shard.is_file() or shard.stat().st_size < 8:
        return None
    try:
        if columns is None:
            return pd.read_parquet(shard)
        return pd.read_parquet(shard, columns=columns)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit political lexicon coverage.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--sample-per-forum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def subreddit_stats(
    interim_dir: Path, subreddit: str, lex_lang: str, parallel_csv: Path
) -> Dict[str, Any]:
    """Function summary: word-weighted rate and hit share for one subreddit.

    Parameters:
    - interim_dir: interim data root.
    - subreddit: forum name.
    - lex_lang: lexicon language code.
    - parallel_csv: graded parallel lexicon CSV path.

    Returns:
    - Summary statistics dict.
    """
    shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    total_points = 0
    total_g1 = total_g2 = total_g3 = 0
    total_words = 0
    n_with_hits = 0
    n_comments = 0
    n_shards_read = 0
    n_shards_skipped = 0
    if not shard_dir.exists():
        return {"subreddit": subreddit, "lexicon": lex_lang, "n_comments": 0}
    for shard in sorted(shard_dir.glob("*.parquet")):
        df = read_parquet_shard_safe(shard, columns=["body"])
        if df is None or df.empty:
            n_shards_skipped += 1
            continue
        n_shards_read += 1
        for body in df["body"].astype(str).tolist():
            scored = score_comment_political_salience(
                body, lex_lang, PROJECT_ROOT, csv_path=parallel_csv
            )
            points = int(scored["political_weighted_points"])
            total_points += points
            total_g1 += int(scored["political_g1_hits"])
            total_g2 += int(scored["political_g2_hits"])
            total_g3 += int(scored["political_g3_hits"])
            nw = int(scored["n_words"])
            total_words += nw
            n_comments += 1
            if points > 0:
                n_with_hits += 1
    return {
        "subreddit": subreddit,
        "lexicon": lex_lang,
        "n_comments": n_comments,
        "n_shards_read": n_shards_read,
        "n_shards_skipped": n_shards_skipped,
        "word_weighted_political_rate_100w": political_rate_100w(total_points, total_words),
        "comment_hit_share": (n_with_hits / n_comments) if n_comments else 0.0,
        "total_weighted_points": total_points,
        "total_g1_hits": total_g1,
        "total_g2_hits": total_g2,
        "total_g3_hits": total_g3,
        "total_words": total_words,
    }


def sample_hit_comments(
    interim_dir: Path,
    subreddit: str,
    lex_lang: str,
    parallel_csv: Path,
    n_sample: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Function summary: random sample of comments with at least one lexicon hit.

    Parameters:
    - interim_dir: interim root.
    - subreddit: forum name.
    - lex_lang: lexicon code.
    - n_sample: max samples to return.
    - rng: random generator.

    Returns:
    - List of {subreddit, body_snippet, hits} dicts.
    """
    hits_pool: List[Dict[str, str]] = []
    shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    if not shard_dir.exists():
        return hits_pool
    for shard in sorted(shard_dir.glob("*.parquet")):
        df = read_parquet_shard_safe(shard, columns=["body"])
        if df is None or df.empty:
            continue
        for body in df["body"].astype(str).tolist():
            scored = score_comment_political_salience(
                body, lex_lang, PROJECT_ROOT, csv_path=parallel_csv
            )
            points = int(scored["political_weighted_points"])
            if points > 0:
                snippet = body[:200].replace("\n", " ")
                hits_pool.append(
                    {
                        "subreddit": subreddit,
                        "weighted_points": str(points),
                        "g1": str(scored["political_g1_hits"]),
                        "g2": str(scored["political_g2_hits"]),
                        "g3": str(scored["political_g3_hits"]),
                        "body_snippet": snippet,
                    }
                )
    if len(hits_pool) <= n_sample:
        return hits_pool
    return rng.sample(hits_pool, n_sample)


def main() -> None:
    """Function summary: run lexicon audit and write CSV samples."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    out_dir = tables_dir / "cleaning_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    meta_table = build_subreddit_metadata_table(config, project_root=PROJECT_ROOT)
    parallel_csv = parallel_political_lexicon_path(config, project_root=PROJECT_ROOT)

    term_rows: List[Dict[str, Any]] = []
    for lang in ("it", "en", "de"):
        singles, phrases = get_lexicon(PROJECT_ROOT, lang)
        term_rows.append(
            {
                "lang": lang,
                "path": str(parallel_csv),
                "n_terms": len(singles) + len(phrases),
            }
        )
    pd.DataFrame(term_rows).to_csv(out_dir / "lexicon_term_counts.csv", index=False)
    print(f"[audit_political_lexicon] wrote lexicon_term_counts languages={len(term_rows)}", flush=True)

    subreddits = resolve_primary_subreddits(config)
    print(f"[audit_political_lexicon] subreddits={len(subreddits)}", flush=True)
    stat_rows: List[Dict[str, Any]] = []
    for idx, subreddit in enumerate(subreddits, start=1):
        lex_lang = meta_table[subreddit]["primary_lexicon"]
        print(
            f"[audit_political_lexicon] subreddit_start {idx}/{len(subreddits)} subreddit={subreddit}",
            flush=True,
        )
        stat_rows.append(subreddit_stats(interim_dir, subreddit, lex_lang, parallel_csv))
    stats_df = pd.DataFrame(stat_rows)
    stats_df = stats_df.sort_values("word_weighted_political_rate_100w", ascending=False)
    stats_df.to_csv(out_dir / "lexicon_audit_subreddit_rates.csv", index=False)
    print("[audit_political_lexicon] wrote lexicon_audit_subreddit_rates", flush=True)

    print(f"[audit_political_lexicon] sampling_hits forums={len(BENCHMARK_SUBREDDITS)}", flush=True)
    sample_rows: List[Dict[str, str]] = []
    for subreddit in BENCHMARK_SUBREDDITS:
        if subreddit not in meta_table:
            continue
        lex_lang = meta_table[subreddit]["primary_lexicon"]
        sample_rows.extend(
            sample_hit_comments(
                interim_dir, subreddit, lex_lang, parallel_csv, args.sample_per_forum, rng
            )
        )
    pd.DataFrame(sample_rows).to_csv(out_dir / "lexicon_audit_sample_hits.csv", index=False)

    print(f"[audit_political_lexicon] wrote tables under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
