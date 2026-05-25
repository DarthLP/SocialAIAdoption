"""
Script summary:
Shared runner for in-place feature passes on enriched `cleaned_monthly_chunks/` Parquet.

Functionality:
- Dispatches polarization, AI-use, and comment-style passes with common screening/subreddit iteration.
- Used by `compute_enriched_shard_features.py` and thin wrapper scripts.

How to apply/run:
- Not executed standalone; import via `main_with_pass` from sibling entry scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import pandas as pd


def read_parquet_shard_safe(shard: Path) -> pd.DataFrame | None:
    """Function summary: read an enriched monthly Parquet shard, skipping corrupt files.

    Parameters:
    - shard: path to Parquet file.

    Returns:
    - DataFrame or None if unreadable or empty.
    """
    if not shard.is_file() or shard.stat().st_size < 8:
        return None
    try:
        return pd.read_parquet(shard)
    except Exception:
        return None


def _setup_project_root(caller_file: Path) -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    for parent in caller_file.resolve().parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller_file)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root(Path(__file__))

from src.comment_style import STYLE_COUNT_COLUMNS, score_comment_style  # noqa: E402
from src.config_utils import (  # noqa: E402
    load_ai_use_config,
    load_comment_style_config,
    load_config,
    load_polarization_config,
    load_screening_pooled,
    resolve_primary_subreddits,
    screening_by_subreddit,
    shard_dir_is_enriched,
    should_skip_screened_subreddit,
    subreddit_primary_lexicon,
    subreddit_screening_action,
)
from src.political_lexicon import score_comment_ai_style, score_comment_polarization  # noqa: E402
from src.v4_lexicon import all_v4_column_names  # noqa: E402

PassName = Literal["polarization", "ai", "style"]
PASS_ORDER: tuple[PassName, ...] = ("polarization", "ai", "style")

POLARIZATION_COMMENT_COLUMNS: tuple[str, ...] = (
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
    *tuple(all_v4_column_names()),
)

POLARIZATION_THREAD_COLUMNS: tuple[str, ...] = (
    "thread_net_ideology_comment_wt",
    "thread_net_ideology_word_wt",
    "thread_has_both_ideology_sides",
    "thread_other_side_salience_share",
)

POLARIZATION_COLUMNS: tuple[str, ...] = POLARIZATION_COMMENT_COLUMNS + POLARIZATION_THREAD_COLUMNS

AI_USE_COLUMNS = [
    "ai_style_hits",
    "ai_style_rate_100w",
    "ai_sentence_length_variance",
]


def parse_args(
    fixed_pass: PassName | None = None,
    prog: str | None = None,
) -> argparse.Namespace:
    """Function summary: parse CLI for enriched shard feature passes.

    Parameters:
    - fixed_pass: when set, ``--pass`` is omitted and this pass is used.
    - prog: optional program name for argparse help.

    Returns:
    - Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compute features on enriched cleaned_monthly_chunks Parquet (in-place).",
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None, help="Process one subreddit only.")
    parser.add_argument("--max-shards", type=int, default=None, help="Max parquet files per subreddit.")
    parser.add_argument("--include-excluded", action="store_true")
    if fixed_pass is None:
        parser.add_argument(
            "--pass",
            dest="pass_name",
            choices=list(PASS_ORDER) + ["all"],
            default="all",
            help="Feature pass: polarization, ai, style, or all (default all).",
        )
    return parser.parse_args()


def _polarization_score_row(
    body: str,
    lex_lang: str,
    lang_comment: str,
    cfg: Dict[str, Any],
    project_root: Path,
) -> Dict[str, float]:
    """Function summary: polarization scores for one comment."""
    row = {col: 0.0 for col in POLARIZATION_COMMENT_COLUMNS}
    if cfg.get("lang_match_filter") and lang_comment != lex_lang:
        return row
    scored = score_comment_polarization(
        body,
        lex_lang,
        project_root,
        negation_window=int(cfg.get("negation_window_tokens", 3)),
        eps=float(cfg.get("eps", 1.0e-6)),
    )
    if scored.get("n_words", 0) <= 0:
        return row
    for col in POLARIZATION_COMMENT_COLUMNS:
        if col not in scored:
            raise KeyError(
                f"score_comment_polarization missing required column {col!r} "
                f"(lex_lang={lex_lang!r}); update scorer or POLARIZATION_COMMENT_COLUMNS"
            )
        row[col] = float(scored[col])
    return row


def _add_thread_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: merge thread-level ideology and salience aggregates."""
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


def _process_polarization_shard(
    path: Path, lex_lang: str, cfg: Dict[str, Any], project_root: Path
) -> int:
    """Function summary: add polarization columns to one Parquet shard."""
    df = read_parquet_shard_safe(path)
    if df is None or df.empty:
        return 0
    required = {"body", "primary_lexicon", "n_words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}; run enrich first.")
    out = df.drop(columns=[c for c in POLARIZATION_COLUMNS if c in df.columns], errors="ignore").copy()
    lang_series = (
        out["lang_comment"].astype(str) if "lang_comment" in out.columns else pd.Series([lex_lang] * len(out))
    )
    rows = [
        _polarization_score_row(body, lex_lang, lang_c, cfg, project_root)
        for body, lang_c in zip(out["body"].astype(str).tolist(), lang_series.tolist(), strict=True)
    ]
    for col in POLARIZATION_COLUMNS:
        if col.startswith("thread_"):
            continue
        out[col] = [r.get(col, 0.0) for r in rows]
    out = _add_thread_rollups(out)
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return len(out)


def _process_ai_shard(path: Path, lex_lang: str, cfg: Dict[str, Any], project_root: Path) -> int:
    """Function summary: add AI-use columns to one Parquet shard."""
    df = read_parquet_shard_safe(path)
    if df is None or df.empty:
        return 0
    if "body" not in df.columns or "primary_lexicon" not in df.columns:
        raise ValueError(f"Run enrich_cleaned_chunks.py first: missing columns in {path}")
    out = df.drop(columns=[c for c in AI_USE_COLUMNS if c in df.columns], errors="ignore").copy()
    lang_col = out["lang_comment"].astype(str) if "lang_comment" in out.columns else pd.Series([lex_lang] * len(out))
    rows: List[Dict[str, float]] = []
    for body, lang_c in zip(out["body"].astype(str).tolist(), lang_col.tolist(), strict=True):
        if cfg.get("lang_match_filter") and lang_c != lex_lang:
            rows.append({col: 0.0 for col in AI_USE_COLUMNS})
        else:
            scored = score_comment_ai_style(body, lex_lang, project_root)
            rows.append(
                {
                    "ai_style_hits": scored["ai_style_hits"],
                    "ai_style_rate_100w": scored["ai_style_rate_100w"],
                    "ai_sentence_length_variance": scored["sentence_length_variance"],
                }
            )
    for col in AI_USE_COLUMNS:
        out[col] = [r[col] for r in rows]
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return len(out)


def _process_style_shard(path: Path, lex_lang: str, cfg: Dict[str, Any], project_root: Path) -> int:
    """Function summary: add style columns to one Parquet shard."""
    df = read_parquet_shard_safe(path)
    if df is None or df.empty:
        return 0
    if "body" not in df.columns or "primary_lexicon" not in df.columns:
        raise ValueError(f"Run enrich_cleaned_chunks.py first: missing columns in {path}")
    out = df.drop(columns=[c for c in STYLE_COUNT_COLUMNS if c in df.columns], errors="ignore").copy()
    lang_col = out["lang_comment"].astype(str) if "lang_comment" in out.columns else pd.Series([lex_lang] * len(out))
    rows: List[Dict[str, int | float]] = []
    for body, lang_c in zip(out["body"].astype(str).tolist(), lang_col.tolist(), strict=True):
        if cfg.get("lang_match_filter") and lang_c != lex_lang:
            rows.append({col: 0 for col in STYLE_COUNT_COLUMNS})
        else:
            rows.append(
                score_comment_style(
                    body,
                    lex_lang,
                    project_root,
                    enable_phrase_lexicons=bool(cfg.get("enable_phrase_lexicons", True)),
                    lang_match_filter=bool(cfg.get("lang_match_filter", False)),
                    lang_comment=lang_c,
                )
            )
    for col in STYLE_COUNT_COLUMNS:
        out[col] = [r[col] for r in rows]
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return len(out)


def _passes_to_run(pass_name: str) -> Sequence[PassName]:
    """Function summary: resolve which passes to execute."""
    if pass_name == "all":
        return PASS_ORDER
    return (pass_name,)  # type: ignore[return-value]


def run_passes(
    args: argparse.Namespace,
    project_root: Path,
    log_prefix: str,
    passes: Sequence[PassName],
) -> None:
    """Function summary: iterate subreddits/shards and run the requested feature passes.

    Parameters:
    - args: CLI namespace with config, subreddit, max_shards, include_excluded.
    - project_root: repository root.
    - log_prefix: log tag prefix.
    - passes: ordered pass names to run per shard.
    """
    config = load_config(project_root / args.config)
    pol_cfg = load_polarization_config(config)
    ai_cfg = load_ai_use_config(config)
    style_cfg = load_comment_style_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    screening_by_sub = screening_by_subreddit(load_screening_pooled(tables_dir))
    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    processors = {
        "polarization": (_process_polarization_shard, pol_cfg),
        "ai": (_process_ai_shard, ai_cfg),
        "style": (_process_style_shard, style_cfg),
    }
    total = 0
    for subreddit in subs:
        action = subreddit_screening_action(screening_by_sub, subreddit)
        if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
            print(f"[{log_prefix}] skip excluded subreddit={subreddit}", flush=True)
            continue
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        if not shard_dir_is_enriched(shard_dir):
            print(
                f"[{log_prefix}] skip subreddit={subreddit}: "
                "shards not enriched; run enrich_cleaned_chunks.py first",
                flush=True,
            )
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if args.max_shards:
            shards = shards[: args.max_shards]
        if not shards:
            continue
        lex_lang = subreddit_primary_lexicon(config, subreddit, project_root=project_root)
        for shard in shards:
            for pass_name in passes:
                fn, cfg = processors[pass_name]
                n = fn(shard, lex_lang, cfg, project_root)
                total += n
                print(f"[{log_prefix}] pass={pass_name} {subreddit}/{shard.name} rows={n}", flush=True)
    print(f"[{log_prefix}] done total_rows={total}", flush=True)


def main_with_pass(
    fixed_pass: PassName | None = None,
    prog: str | None = None,
    caller_file: str | Path | None = None,
) -> None:
    """Function summary: CLI entry for one or more enriched-shard feature passes.

    Parameters:
    - fixed_pass: optional single pass (wrapper scripts).
    - prog: argparse program name.
    - caller_file: ``__file__`` of the invoking script for bootstrap.
    """
    _ = caller_file  # wrappers pass __file__ for future use; PROJECT_ROOT set at import
    project_root = PROJECT_ROOT
    args = parse_args(fixed_pass=fixed_pass, prog=prog)
    pass_name = fixed_pass if fixed_pass is not None else getattr(args, "pass_name", "all")
    log_prefix = prog or "compute_enriched_shard_features"
    run_passes(args, project_root, log_prefix, _passes_to_run(pass_name))
