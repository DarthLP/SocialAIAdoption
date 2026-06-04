"""
Script summary:
Add style_index_full, style_index_reduced, and feature columns to enriched comment Parquet shards.

Functionality:
- Uses did/style_index_stats.json from fit_style_index_stats.py.
- In-place shard updates following the enriched-shard iteration pattern.

How to apply/run:
  .venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/compute_style_index_on_shards.py --config config/italy_polarization_setup.yaml --max-shards 5
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict

import pandas as pd


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
from src.style_index import comment_feature_dict, compute_index, load_style_index_stats  # noqa: E402

INDEX_COLS = ("style_index_full", "style_index_reduced", "ttr_50w", "readability", "log_len")


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Write style indices onto enriched shards.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=None)
    return p.parse_args()


def _process_shard(path: Path, stats: Dict[str, Any], project_root: Path) -> int:
    """Function summary: update one Parquet shard with index columns."""
    df = pd.read_parquet(path)
    if df.empty:
        return 0
    full_list = []
    red_list = []
    ttr_list = []
    read_list = []
    log_len_list = []
    for _, row in df.iterrows():
        lang = str(row.get("primary_lexicon", row.get("lang", "it")))
        feats = comment_feature_dict(str(row.get("body", "")), lang, project_root)
        full, red = compute_index(feats, stats, lang)
        full_list.append(full)
        red_list.append(red)
        ttr_list.append(feats.get("ttr_50w"))
        read_list.append(feats.get("readability"))
        log_len_list.append(feats.get("log_len"))
    df["style_index_full"] = full_list
    df["style_index_reduced"] = red_list
    df["ttr_50w"] = ttr_list
    df["readability"] = read_list
    df["log_len"] = log_len_list
    df.to_parquet(path, index=False)
    return len(df)


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    stats_path = PROJECT_ROOT / tables_subdir(config, "did") / "style_index_stats.json"
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
