"""
Script summary:
Regenerate DiD overview figures from saved did_summary.csv without re-estimating.

Functionality:
- Loads estimates/summary/did_summary.csv and calls generate_overview_figures.
- Use after estimation or when tuning forest/heatmap/DDD plot styling.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_did_overview.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


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

from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.outputs import add_strategy_labels, generate_overview_figures  # noqa: E402
from src.did.paths import did_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for overview figure regeneration."""
    parser = argparse.ArgumentParser(description="Regenerate DiD overview figures.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def main() -> None:
    """Function summary: load did_summary and write overview/*.png."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    summary_path, _ = did_summary_paths(config)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; run did_event_study.py first.")
    summary = pd.read_csv(summary_path)
    fig_dir = figures_subdir(config, "did")
    fig_dir.mkdir(parents=True, exist_ok=True)
    generate_overview_figures(add_strategy_labels(summary), fig_dir)
    print(f"[plot_did_overview] wrote overview figures under {fig_dir / 'overview'}", flush=True)


if __name__ == "__main__":
    main()
