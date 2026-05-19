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
from typing import Any, Dict, List

import pandas as pd

BENCHMARK_SUBREDDITS = ["politicaITA", "litigi", "Italia", "oknotizie", "BancaDelMeme"]


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import (  # noqa: E402
    build_subreddit_metadata_table,
    load_config,
    resolve_primary_subreddits,
)
from src.political_lexicon import (  # noqa: E402
    count_political_hits,
    get_lexicon,
    lexicon_path,
    political_rate_100w,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit political lexicon coverage.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--sample-per-forum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def subreddit_stats(interim_dir: Path, subreddit: str, lex_lang: str) -> Dict[str, Any]:
    """Function summary: word-weighted rate and hit share for one subreddit.

    Parameters:
    - interim_dir: interim data root.
    - subreddit: forum name.
    - lex_lang: lexicon language code.

    Returns:
    - Summary statistics dict.
    """
    shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    total_hits = 0
    total_words = 0
    n_with_hits = 0
    n_comments = 0
    if not shard_dir.exists():
        return {"subreddit": subreddit, "n_comments": 0}
    for shard in sorted(shard_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["body"])
        for body in df["body"].astype(str).tolist():
            hits, nw = count_political_hits(body, lex_lang, PROJECT_ROOT)
            total_hits += hits
            total_words += nw
            n_comments += 1
            if hits > 0:
                n_with_hits += 1
    return {
        "subreddit": subreddit,
        "lexicon": lex_lang,
        "n_comments": n_comments,
        "word_weighted_political_rate_100w": political_rate_100w(total_hits, total_words),
        "comment_hit_share": (n_with_hits / n_comments) if n_comments else 0.0,
        "total_hits": total_hits,
        "total_words": total_words,
    }


def sample_hit_comments(
    interim_dir: Path,
    subreddit: str,
    lex_lang: str,
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
        df = pd.read_parquet(shard, columns=["body"])
        for body in df["body"].astype(str).tolist():
            h, _ = count_political_hits(body, lex_lang, PROJECT_ROOT)
            if h > 0:
                snippet = body[:200].replace("\n", " ")
                hits_pool.append({"subreddit": subreddit, "hits": str(h), "body_snippet": snippet})
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

    term_rows: List[Dict[str, Any]] = []
    for lang in ("it", "en", "de", "es"):
        path = lexicon_path(PROJECT_ROOT, lang)
        terms, _ = get_lexicon(PROJECT_ROOT, lang)
        term_rows.append({"lang": lang, "path": str(path), "n_terms": len(terms)})
    pd.DataFrame(term_rows).to_csv(out_dir / "lexicon_term_counts.csv", index=False)

    stat_rows: List[Dict[str, Any]] = []
    for subreddit in resolve_primary_subreddits(config):
        lex_lang = meta_table[subreddit]["primary_lexicon"]
        stat_rows.append(subreddit_stats(interim_dir, subreddit, lex_lang))
    stats_df = pd.DataFrame(stat_rows)
    stats_df = stats_df.sort_values("word_weighted_political_rate_100w", ascending=False)
    stats_df.to_csv(out_dir / "lexicon_audit_subreddit_rates.csv", index=False)

    sample_rows: List[Dict[str, str]] = []
    for subreddit in BENCHMARK_SUBREDDITS:
        if subreddit not in meta_table:
            continue
        lex_lang = meta_table[subreddit]["primary_lexicon"]
        sample_rows.extend(
            sample_hit_comments(interim_dir, subreddit, lex_lang, args.sample_per_forum, rng)
        )
    pd.DataFrame(sample_rows).to_csv(out_dir / "lexicon_audit_sample_hits.csv", index=False)

    print(f"[audit_political_lexicon] wrote tables under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
