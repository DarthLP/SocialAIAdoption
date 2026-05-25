"""
Script summary:
Run one or more in-place feature passes on enriched Italy polarization Parquet shards.

Functionality:
- `--pass all` runs polarization, AI-use, and comment-style (in that order) per shard.
- Delegates to `_enriched_shard_runner.py`.

How to apply/run:
  .venv/bin/python scripts/features/compute_enriched_shard_features.py --config config/italy_polarization_setup.yaml --pass all
  .venv/bin/python scripts/features/compute_enriched_shard_features.py --config config/italy_polarization_setup.yaml --pass style --subreddit Italia --max-shards 1
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
    _load_runner().main_with_pass(prog="compute_enriched_shard_features", caller_file=__file__)
