"""
Module summary:
Picklable process-pool entry point for enriched-shard feature passes.

Functionality:
- Spawned workers import this module by name; it bootstraps sys.path and delegates
  to ``scripts/features/_enriched_shard_runner.py``.

How to apply/run:
- Not invoked directly; used via ``run_passes`` in ``_enriched_shard_runner.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple


def _load_runner_module(project_root: Path):
    """Function summary: load the enriched shard runner module from disk.

    Parameters:
    - project_root: repository root.

    Returns:
    - Loaded runner module object.
    """
    runner_path = project_root / "scripts" / "features" / "_enriched_shard_runner.py"
    name = "_enriched_shard_runner"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(runner_path):
        return cached
    spec = importlib.util.spec_from_file_location(name, runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {runner_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def feature_shard_worker(
    shard_str: str,
    subreddit: str,
    lex_lang: str,
    pass_name: str,
    config_path_str: str,
    project_root_str: str,
    force_flag: str = "0",
) -> Tuple[str, str, str, int, float]:
    """Function summary: process-pool worker for one shard and feature pass.

    Parameters:
    - shard_str: absolute parquet path.
    - subreddit: subreddit name for logging.
    - lex_lang: primary lexicon language.
    - pass_name: polarization, semaxis, ai, style, bantopic, or all.
    - config_path_str: study YAML path.
    - project_root_str: repository root path.
    - force_flag: "1" to recompute is_ban_topic when present.

    Returns:
    - Tuple (subreddit, shard_name, pass_name, rows, elapsed_sec).
    """
    root = Path(project_root_str)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    runner = _load_runner_module(root)
    return runner._feature_shard_worker(
        shard_str,
        subreddit,
        lex_lang,
        pass_name,
        config_path_str,
        project_root_str,
        force_flag,
    )
