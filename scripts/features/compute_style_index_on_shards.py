"""
Script summary:
Add style_index_llm columns and em_dash feature fields to enriched comment Parquet shards.

Functionality:
- Primary style_index_llm + leave-one-out ablation columns (frozen weights from style_index_stats.json).
- Drops legacy v1/v2/v3 column names on each pass.
- Persists extended em_dash_* columns.

How to apply/run:
  .venv/bin/python scripts/diagnostics/validate_style_index_weights.py --config config/italy_polarization_setup.yaml --max-shards 30
  .venv/bin/python scripts/features/compute_style_index_on_shards.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.style_index_llm import (
    ABLATION_DROP_FEATURES,
    PRIMARY_COL,
    ablation_column_name,
)


def _setup_project_root() -> Path:
    """Function summary: resolve repo root."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config, resolve_primary_subreddits, tables_subdir  # noqa: E402
from src.style_index import (  # noqa: E402
    comment_feature_dict,
    compute_style_index_llm,
    load_style_index_stats,
    style_index_stats_filename,
)

INDEX_COLS = (PRIMARY_COL,) + tuple(ablation_column_name(d) for d in ABLATION_DROP_FEATURES)
EM_DASH_COLS = (
    "em_dash_count",
    "em_dash_extended_count",
    "em_dash_rate_100w",
    "em_dash_any",
)
LEGACY_DROP_COLS = (
    "style_index_full",
    "style_index_reduced",
    "style_index_lexical_v2",
    "style_index_formality_v2",
    "style_index_formality_punct_v2",
    "style_index_formality_reweighted_v3",
    "style_index_punct_reweighted_v3",
    "style_index_llm_v3",
    "style_index_llm_v3_no_ai_style",
    "style_index_llm_v3_no_em_dash",
    "style_index_llm_v3_no_semicolon_colon",
    "style_index_llm_v3_no_hedging_phrase",
    "style_index_llm_v3_no_exclamation",
    "style_index_llm_v3_no_caps_word",
) + tuple(f"style_index_llm_v3_only_{f}" for f in (
    "ai_style",
    "em_dash",
    "semicolon_colon",
    "hedging_phrase",
    "exclamation",
    "caps_word",
))


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Write style_index_llm onto enriched shards.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=None)
    return p.parse_args()


def _process_shard(
    path: Path,
    stats: Dict[str, Any],
    project_root: Path,
) -> int:
    """Function summary: update one Parquet shard with style_index_llm columns."""
    df = pd.read_parquet(path)
    if df.empty:
        return 0
    index_cols, index_lists = list(INDEX_COLS), {c: [] for c in INDEX_COLS}
    ttr_list: List[float] = []
    read_list: List[float] = []
    log_len_list: List[float] = []
    em_dash_count_list: List[float] = []
    em_dash_ext_list: List[float] = []
    em_dash_rate_list: List[float] = []
    em_dash_any_list: List[float] = []

    df = df.drop(columns=[c for c in LEGACY_DROP_COLS if c in df.columns], errors="ignore")

    for _, row in df.iterrows():
        lang = str(row.get("primary_lexicon", row.get("lang", "it")))
        feats = comment_feature_dict(str(row.get("body", "")), lang, project_root)
        vals = compute_style_index_llm(feats, stats, lang)
        for col in index_cols:
            index_lists[col].append(float(vals.get(col, float("nan"))))
        ttr_list.append(feats.get("ttr_50w"))
        read_list.append(feats.get("readability"))
        log_len_list.append(feats.get("log_len"))
        em_n = float(feats.get("em_dash_count", 0.0) or 0.0)
        em_dash_count_list.append(em_n)
        em_dash_ext_list.append(em_n)
        em_dash_rate_list.append(feats.get("em_dash_rate_100w"))
        em_dash_any_list.append(feats.get("em_dash_any"))

    for col in index_cols:
        if index_lists[col]:
            df[col] = index_lists[col]
    df["ttr_50w"] = ttr_list
    df["readability"] = read_list
    df["log_len"] = log_len_list
    for col, vals in zip(
        EM_DASH_COLS,
        (em_dash_count_list, em_dash_ext_list, em_dash_rate_list, em_dash_any_list),
    ):
        df[col] = vals
    df.to_parquet(path, index=False)
    return len(df)


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    did_dir = PROJECT_ROOT / tables_subdir(config, "did")
    stats_path = did_dir / style_index_stats_filename()
    stats = load_style_index_stats(stats_path)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    n_shards = 0
    for sub in resolve_primary_subreddits(config):
        for shard in sorted((shard_root / sub).glob("*.parquet")):
            if args.max_shards is not None and n_shards >= args.max_shards:
                break
            _process_shard(shard, stats, PROJECT_ROOT)
            n_shards += 1
            print(f"[compute_style_index_on_shards] {shard.name} ok", flush=True)
        if args.max_shards is not None and n_shards >= args.max_shards:
            break


if __name__ == "__main__":
    main()
