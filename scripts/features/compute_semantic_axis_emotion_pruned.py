"""
Script summary:
Append leakage-pruned emotion semantic-axis column to enriched Parquet shards.

Functionality:
- Thin wrapper around `_enriched_shard_runner.py` (--pass semaxis_emotion_pruned).
- Scores sem_axis_emotion_pruned from pruned cognition pole seeds (emotion_neg_*_pruned.txt).
- Reuses NPZ vector caches from the legacy semaxis pass; does not overwrite sem_axis_emotion.

How to apply/run:
  .venv/bin/python scripts/features/compute_semantic_axis_emotion_pruned.py --config config/italy_polarization_setup.yaml --workers 1
  .venv/bin/python scripts/features/compute_semantic_axis_emotion_pruned.py --config config/italy_polarization_setup.yaml --subreddit Italia --max-shards 1
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    """Function summary: load sibling _enriched_shard_runner module.

    Returns:
    - Loaded runner module with main_with_pass().
    """
    path = Path(__file__).resolve().parent / "_enriched_shard_runner.py"
    spec = importlib.util.spec_from_file_location("_enriched_shard_runner_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    _load_runner().main_with_pass(
        fixed_pass="semaxis_emotion_pruned",
        prog="compute_semantic_axis_emotion_pruned",
        caller_file=__file__,
    )
