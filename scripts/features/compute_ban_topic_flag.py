"""
Script summary:
Append is_ban_topic boolean to enriched monthly Parquet shards (in-place, additive).

Functionality:
- Thin wrapper around `_enriched_shard_runner.py` (--pass bantopic).
- Case-insensitive multilingual regex on comment body; idempotent unless --force.

How to apply/run:
  .venv/bin/python scripts/features/compute_ban_topic_flag.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/compute_ban_topic_flag.py --config config/italy_polarization_setup.yaml --subreddit Italia --max-shards 1
  .venv/bin/python scripts/features/compute_ban_topic_flag.py --config config/italy_polarization_setup.yaml --force
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
        fixed_pass="bantopic",
        prog="compute_ban_topic_flag",
        caller_file=__file__,
    )
