"""
Script summary:
Apply Italian subreddit discovery results to italy_polarization_setup.yaml by writing
discovered_italian and rebuilding primary from control lists plus discovered names.

Functionality:
- Reads candidate_italian_subreddits.csv (or a custom --candidates-csv).
- Optionally merges discovery_seeds_italian from config.
- Writes sorted discovered_italian and primary lists back to the YAML config.

How to run:
  .venv/bin/python scripts/discovery/apply_discovery_to_config.py \
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/discovery/apply_discovery_to_config.py \
    --config config/italy_polarization_setup.yaml --include-all-candidates
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml



def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config, resolve_primary_subreddits, subreddit_control_lists  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for applying discovery to config."""
    parser = argparse.ArgumentParser(description="Write discovered Italian subs into study config.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--candidates-csv",
        type=str,
        default="",
        help="Override path to candidate_italian_subreddits.csv",
    )
    parser.add_argument(
        "--include-all-candidates",
        action="store_true",
        help="Include every row in candidates CSV (default: same as all rows).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print primary list without writing YAML.",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> List[str]:
    """Function summary: load subreddit names from discovery candidates CSV."""
    df = pd.read_csv(path)
    if "subreddit" not in df.columns:
        raise ValueError(f"Missing subreddit column in {path}")
    return sorted({str(s).strip() for s in df["subreddit"].tolist() if str(s).strip()})


def write_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Function summary: persist updated config dict to YAML."""
    header = (
        "# Updated by scripts/discovery/apply_discovery_to_config.py\n"
        "# primary is explicit union of controls, seeds, and discovered_italian.\n"
    )
    text = header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    """Function summary: merge discovery candidates into config primary/discovered lists."""
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    tables_dir = Path(config["paths"]["tables_dir"])
    candidates_path = (
        Path(args.candidates_csv)
        if args.candidates_csv
        else tables_dir / "discovery" / "candidate_italian_subreddits.csv"
    )
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    discovered = load_candidates(candidates_path)
    lists = subreddit_control_lists(config)
    seeds = lists.get("discovery_seeds_italian", [])
    for seed in seeds:
        if seed not in discovered:
            discovered.append(seed)
    discovered = sorted(set(discovered))

    if "subreddits" not in config or not isinstance(config["subreddits"], dict):
        config["subreddits"] = {}
    config["subreddits"]["discovered_italian"] = discovered
    config["subreddits"]["primary"] = []
    config["subreddits"]["primary"] = resolve_primary_subreddits(config)

    print(f"discovered_italian ({len(discovered)}): {', '.join(discovered[:20])}{'...' if len(discovered) > 20 else ''}")
    print(f"primary ({len(config['subreddits']['primary'])} subs)")

    if args.dry_run:
        return
    write_config(config_path, config)
    print(f"Updated {config_path}")


if __name__ == "__main__":
    main()
