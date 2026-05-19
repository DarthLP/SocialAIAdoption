"""
Script summary:
Add polarization lexicon features and thread roll-ups to enriched Italy polarization Parquet.

Functionality:
- Ideology (L/C/R), affect, other-side salience, aggression, issue rates; derived indices.
- Thread-level ideology aggregates (comment- and word-weighted).
- No AI-style columns (see compute_ai_use_features.py).

How to apply/run:
  .venv/bin/python scripts/features/compute_polarization_features.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

POLARIZATION_COLUMNS = [
    "left_hits",
    "center_hits",
    "right_hits",
    "left_rate_100w",
    "center_rate_100w",
    "right_rate_100w",
    "net_ideology",
    "extremity",
    "ambivalence",
    "other_side_salience_hits",
    "other_side_salience_rate_100w",
    "aggression_hits",
    "aggression_rate_100w",
    "negative_rate_100w",
    "anger_rate_100w",
    "issue_eu_rate_100w",
    "issue_migration_rate_100w",
    "issue_economy_rate_100w",
    "issue_culture_rate_100w",
    "has_left_hit",
    "has_right_hit",
    "has_other_side_hit",
    "pol_n_chars",
    "pol_avg_words_per_sentence",
    "pol_exclamation_rate_100w",
    "pol_caps_word_share",
    "pol_em_dash_count",
    "thread_net_ideology_comment_wt",
    "thread_net_ideology_word_wt",
    "thread_has_both_ideology_sides",
    "thread_other_side_salience_share",
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

from src.config_utils import load_config, load_polarization_config, resolve_primary_subreddits  # noqa: E402
from src.political_lexicon import score_comment_polarization, tokenize  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute polarization lexicon features.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def _style_proxies(body: str, n_words: int) -> Dict[str, float]:
    """Function summary: language-agnostic style counts for polarization shard.

    Parameters:
    - body: text.
    - n_words: token count.

    Returns:
    - Style proxy fields.
    """
    text = body or ""
    sentences = [s for s in text.replace("!", ".").split(".") if s.strip()]
    n_sentences = max(1, len(sentences))
    tokens = tokenize(text)
    caps = sum(1 for t in tokens if len(t) > 1 and t.isupper())
    return {
        "pol_n_chars": float(len(text)),
        "pol_avg_words_per_sentence": float(n_words) / float(n_sentences) if n_words > 0 else 0.0,
        "pol_exclamation_rate_100w": 100.0 * float(text.count("!")) / float(n_words) if n_words > 0 else 0.0,
        "pol_caps_word_share": float(caps) / float(n_words) if n_words > 0 else 0.0,
        "pol_em_dash_count": float(text.count("\u2014")),
    }


def score_row(body: str, lex_lang: str, lang_comment: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Function summary: polarization scores for one comment.

    Parameters:
    - body: comment text.
    - lex_lang: primary_lexicon.
    - lang_comment: langid code.
    - cfg: polarization config.

    Returns:
    - Flat score dict.
    """
    if cfg.get("lang_match_filter") and lang_comment != lex_lang:
        return {col: 0.0 for col in POLARIZATION_COLUMNS}
    scored = score_comment_polarization(
        body,
        lex_lang,
        PROJECT_ROOT,
        negation_window=int(cfg.get("negation_window_tokens", 3)),
        eps=float(cfg.get("eps", 1.0e-6)),
    )
    if scored.get("n_words", 0) <= 0:
        base = {col: 0.0 for col in POLARIZATION_COLUMNS}
        base.update(_style_proxies(body, 0))
        return base
    n_words = int(scored["n_words"])
    row = {
        "left_hits": scored.get("left_hits", 0.0),
        "center_hits": scored.get("center_hits", 0.0),
        "right_hits": scored.get("right_hits", 0.0),
        "left_rate_100w": scored.get("left_rate_100w", 0.0),
        "center_rate_100w": scored.get("center_rate_100w", 0.0),
        "right_rate_100w": scored.get("right_rate_100w", 0.0),
        "net_ideology": scored.get("net_ideology", 0.0),
        "extremity": scored.get("extremity", 0.0),
        "ambivalence": scored.get("ambivalence", 0.0),
        "other_side_salience_hits": scored.get("other_side_salience_hits", 0.0),
        "other_side_salience_rate_100w": scored.get("other_side_salience_rate_100w", 0.0),
        "aggression_hits": scored.get("aggression_hits", 0.0),
        "aggression_rate_100w": scored.get("aggression_rate_100w", 0.0),
        "negative_rate_100w": scored.get("negative_rate_100w", 0.0),
        "anger_rate_100w": scored.get("anger_rate_100w", 0.0),
        "issue_eu_rate_100w": scored.get("issue_eu_rate_100w", 0.0),
        "issue_migration_rate_100w": scored.get("issue_migration_rate_100w", 0.0),
        "issue_economy_rate_100w": scored.get("issue_economy_rate_100w", 0.0),
        "issue_culture_rate_100w": scored.get("issue_culture_rate_100w", 0.0),
        "has_left_hit": scored.get("has_left_hit", 0.0),
        "has_right_hit": scored.get("has_right_hit", 0.0),
        "has_other_side_hit": scored.get("has_other_side_hit", 0.0),
    }
    row.update(_style_proxies(body, n_words))
    return row


def add_thread_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: merge thread-level ideology and salience aggregates.

    Parameters:
    - df: comment-level frame with thread_id and polarization columns.

    Returns:
    - Frame with thread_* columns merged.
    """
    if "thread_id" not in df.columns:
        df["thread_id"] = df.get("link_id", pd.Series(dtype="string")).astype(str)
    grp = df.groupby("thread_id", as_index=False)
    stats = grp.agg(
        thread_net_ideology_comment_wt=("net_ideology", "mean"),
        thread_left_any=("has_left_hit", "max"),
        thread_right_any=("has_right_hit", "max"),
        thread_other_side_share=("has_other_side_hit", "mean"),
        thread_nw=("n_words", "sum"),
    )
    weighted = (
        df.assign(_wx=df["net_ideology"] * df["n_words"].astype(float))
        .groupby("thread_id", as_index=False)
        .agg(_num=("_wx", "sum"), thread_nw2=("n_words", "sum"))
    )
    weighted["thread_net_ideology_word_wt"] = weighted.apply(
        lambda r: float(r["_num"]) / float(r["thread_nw2"]) if r["thread_nw2"] > 0 else 0.0,
        axis=1,
    )
    stats["thread_has_both_ideology_sides"] = (
        (stats["thread_left_any"] > 0) & (stats["thread_right_any"] > 0)
    ).astype(float)
    stats = stats.merge(
        weighted[["thread_id", "thread_net_ideology_word_wt"]],
        on="thread_id",
        how="left",
    )
    stats = stats.rename(columns={"thread_other_side_share": "thread_other_side_salience_share"})
    merge_cols = [
        "thread_id",
        "thread_net_ideology_comment_wt",
        "thread_net_ideology_word_wt",
        "thread_has_both_ideology_sides",
        "thread_other_side_salience_share",
    ]
    return df.merge(stats[merge_cols], on="thread_id", how="left")


def process_shard(path: Path, lex_lang: str, cfg: Dict[str, Any]) -> int:
    """Function summary: add polarization columns to one Parquet file.

    Parameters:
    - path: shard path.
    - lex_lang: lexicon language.
    - cfg: polarization settings.

    Returns:
    - Row count.
    """
    df = pd.read_parquet(path)
    required = {"body", "primary_lexicon", "n_words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}; run enrich first.")
    out = df.drop(columns=[c for c in POLARIZATION_COLUMNS if c in df.columns], errors="ignore").copy()
    lang_series = out["lang_comment"].astype(str) if "lang_comment" in out.columns else pd.Series([lex_lang] * len(out))
    rows = [
        score_row(body, lex_lang, lang_c, cfg)
        for body, lang_c in zip(out["body"].astype(str).tolist(), lang_series.tolist(), strict=True)
    ]
    for col in POLARIZATION_COLUMNS:
        if col.startswith("thread_"):
            continue
        out[col] = [r.get(col, 0.0) for r in rows]
    out = add_thread_rollups(out)
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return len(out)


def main() -> None:
    """Function summary: run polarization feature pass."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    cfg = load_polarization_config(config)
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
        if not shards:
            continue
        sample = pd.read_parquet(shards[0], columns=["primary_lexicon"])
        lex_lang = str(sample["primary_lexicon"].iloc[0])
        for shard in shards:
            n = process_shard(shard, lex_lang, cfg)
            total += n
            print(f"[compute_polarization_features] {subreddit}/{shard.name} rows={n}", flush=True)
    print(f"[compute_polarization_features] done total_rows={total}", flush=True)


if __name__ == "__main__":
    main()
