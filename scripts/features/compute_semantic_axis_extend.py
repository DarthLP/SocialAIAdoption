"""
Script summary:
Append extended fastText semantic-axis columns to enriched Parquet without overwriting legacy scores.

Functionality:
- Thin wrapper around `_enriched_shard_runner.py` (--pass semaxis_extend).
- Requires existing sem_axis_ideology/emotion/aggression columns and reuses NPZ vector caches.

How to apply/run:
  .venv/bin/python scripts/features/compute_semantic_axis_extend.py --config config/italy_polarization_setup.yaml --workers 1
  .venv/bin/python scripts/features/compute_semantic_axis_extend.py --config config/italy_polarization_setup.yaml --subreddit Italia --max-shards 1
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    """Function summary: load sibling _enriched_shard_runner module."""
    path = Path(__file__).resolve().parent / "_enriched_shard_runner.py"
    spec = importlib.util.spec_from_file_location("_enriched_shard_runner_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    _load_runner().main_with_pass(
        fixed_pass="semaxis_extend",
        prog="compute_semantic_axis_extend",
        caller_file=__file__,
    )
