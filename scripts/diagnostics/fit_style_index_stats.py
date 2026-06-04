"""
Script summary:
Fit pre-period style-index calibration (clip bounds, mu, sigma) from March 2023 comments.

Functionality:
- Samples enriched comment shards, computes per-comment features, filters pre-period.
- Writes results/tables/italy_polarization/did/style_index_stats.json.

How to apply/run:
  .venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

FIT_SHARD_COLUMNS: Sequence[str] = ("body", "date_utc", "primary_lexicon")


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
from src.style_index import comment_feature_dict, fit_preperiod_stats, save_style_index_stats  # noqa: E402


def _read_shard(path: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Function summary: read projected columns from one Parquet shard.

    Parameters:
    - path: shard path.
    - columns: desired column names.

    Returns:
    - DataFrame subset or None if unreadable / no requested columns.
    """
    try:
        import pyarrow.parquet as pq  # noqa: WPS433

        avail: List[str] = [c for c in columns if c in pq.read_schema(path).names]
        if not avail:
            return None
        return pd.read_parquet(path, columns=avail)
    except Exception:
        try:
            df = pd.read_parquet(path)
            keep = [c for c in columns if c in df.columns]
            return df[keep] if keep else None
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    """Function summary: CLI args."""
    p = argparse.ArgumentParser(description="Fit style index pre-period stats.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=50)
    return p.parse_args()


def main() -> None:
    """Function summary: fit and save style_index_stats.json."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    subs = resolve_primary_subreddits(config)
    rows = []
    n = 0
    for sub in subs:
        for shard in sorted((shard_root / sub).glob("*.parquet")):
            if n >= args.max_shards:
                break
            df = _read_shard(shard, FIT_SHARD_COLUMNS)
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                feats = comment_feature_dict(str(r.get("body", "")), str(r.get("primary_lexicon", "it")), PROJECT_ROOT)
                feats["date_utc"] = str(r.get("date_utc", ""))[:10]
                feats["lang"] = str(r.get("primary_lexicon", "it"))
                rows.append(feats)
            n += 1
        if n >= args.max_shards:
            break
    if not rows:
        raise FileNotFoundError(
            f"No shards under {shard_root}; run feature enrichment on cleaned_monthly_chunks first."
        )
    stats = fit_preperiod_stats(pd.DataFrame(rows))
    out = tables_subdir(config, "did") / "style_index_stats.json"
    save_style_index_stats(stats, PROJECT_ROOT / out)
    print(f"[fit_style_index_stats] wrote {out} langs={list(stats.get('languages', {}).keys())}", flush=True)


if __name__ == "__main__":
    main()
