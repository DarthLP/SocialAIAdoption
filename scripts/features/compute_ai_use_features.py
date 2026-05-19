"""
Script summary:
Add language-matched AI-style lexicon features to enriched Italy polarization Parquet shards.

Functionality:
- Scores ai_style_{lang}.txt per comment (primary_lexicon).
- Adds lightweight style proxies (length, em dash, sentence-length variance).
- Idempotent re-run drops prior AI-use columns before rewrite.

How to apply/run:
  .venv/bin/python scripts/features/compute_ai_use_features.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/compute_ai_use_features.py --config config/italy_polarization_setup.yaml --subreddit politicaITA
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

AI_USE_COLUMNS = [
    "ai_style_hits",
    "ai_style_rate_100w",
    "ai_n_chars",
    "ai_avg_words_per_sentence",
    "ai_exclamation_rate_100w",
    "ai_caps_word_share",
    "ai_em_dash_count",
    "ai_sentence_length_variance",
]


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
    load_ai_use_config,
    load_config,
    resolve_primary_subreddits,
)
from src.political_lexicon import score_comment_ai_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute AI-use lexicon features on enriched Parquet.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None, help="Process one subreddit only.")
    parser.add_argument("--max-shards", type=int, default=None, help="Max parquet files per subreddit.")
    parser.add_argument("--include-excluded", action="store_true")
    return parser.parse_args()


def enrich_ai_use_row(body: str, lex_lang: str, lang_comment: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Function summary: score one comment for AI-use features.

    Parameters:
    - body: comment text.
    - lex_lang: primary_lexicon language.
    - lang_comment: langid label.
    - cfg: ai_use config dict.

    Returns:
    - Column value dict for AI_USE_COLUMNS.
    """
    if cfg.get("lang_match_filter") and lang_comment != lex_lang:
        return {col: 0.0 for col in AI_USE_COLUMNS}
    scored = score_comment_ai_style(body, lex_lang, PROJECT_ROOT)
    return {
        "ai_style_hits": scored["ai_style_hits"],
        "ai_style_rate_100w": scored["ai_style_rate_100w"],
        "ai_n_chars": scored["n_chars"],
        "ai_avg_words_per_sentence": scored["avg_words_per_sentence"],
        "ai_exclamation_rate_100w": scored["exclamation_rate_100w"],
        "ai_caps_word_share": scored["caps_word_share"],
        "ai_em_dash_count": scored["em_dash_count"],
        "ai_sentence_length_variance": scored["sentence_length_variance"],
    }


def process_shard(path: Path, lex_lang: str, cfg: Dict[str, Any]) -> int:
    """Function summary: add AI-use columns to one Parquet shard.

    Parameters:
    - path: parquet file path.
    - lex_lang: lexicon language.
    - cfg: ai_use settings.

    Returns:
    - Number of rows written.
    """
    df = pd.read_parquet(path)
    if "body" not in df.columns or "primary_lexicon" not in df.columns:
        raise ValueError(f"Run enrich_cleaned_chunks.py first: missing columns in {path}")
    out = df.drop(columns=[c for c in AI_USE_COLUMNS if c in df.columns], errors="ignore").copy()
    lang_col = out["lang_comment"].astype(str) if "lang_comment" in out.columns else pd.Series([lex_lang] * len(out))
    rows: List[Dict[str, float]] = []
    for body, lang_c in zip(out["body"].astype(str).tolist(), lang_col.tolist(), strict=True):
        rows.append(enrich_ai_use_row(body, lex_lang, str(lang_c), cfg))
    for col in AI_USE_COLUMNS:
        out[col] = [r[col] for r in rows]
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return len(out)


def main() -> None:
    """Function summary: run AI-use feature pass on all eligible shards."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    cfg = load_ai_use_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    total = 0
    for subreddit in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if args.max_shards:
            shards = shards[: args.max_shards]
        sample = pd.read_parquet(shards[0], columns=["primary_lexicon"]) if shards else None
        lex_lang = str(sample["primary_lexicon"].iloc[0]) if sample is not None and len(sample) else "it"
        for shard in shards:
            n = process_shard(shard, lex_lang, cfg)
            total += n
            print(f"[compute_ai_use_features] {subreddit}/{shard.name} rows={n}", flush=True)
    print(f"[compute_ai_use_features] done total_rows={total}", flush=True)


if __name__ == "__main__":
    main()
