"""
Script summary:
Add fastText semantic-axis features to enriched Italy polarization Parquet shards.

Functionality:
- Thin wrapper around `_enriched_shard_runner.py` (--pass semaxis).
- Requires fastText models under data/external/embeddings/ (see download_fasttext_models.py).

How to apply/run:
  .venv/bin/python scripts/features/compute_semantic_axis_features.py --config config/italy_polarization_setup.yaml --workers 1
  .venv/bin/python scripts/features/compute_semantic_axis_features.py --config config/italy_polarization_setup.yaml --lex-lang it --workers 1
  .venv/bin/python scripts/features/compute_semantic_axis_features.py --config config/italy_polarization_setup.yaml --subreddit Italia --max-shards 1

Language waves (default): all IT shards, then EN, then DE; ProcessPool restarts between waves
so fastText models unload. Use --workers 1 on machines with ~8GB RAM.
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
        fixed_pass="semaxis",
        prog="compute_semantic_axis_features",
        caller_file=__file__,
    )
