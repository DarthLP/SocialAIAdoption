"""
Script summary:
Build aggregated DiD panels for event studies (topic_family, language, language_universe).

Functionality:
- Rolls lexical descriptives and semantic-axis panels to 1d and 3d launch-aligned bins.
- Writes did/panels/aggregated/did_{level}_{1,3}d.csv for downstream event-study estimation.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only --bin-days 1
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only --bin-days 3
  .venv/bin/python scripts/diagnostics/prepare_did_merged_panels.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_did_aggregated_panels.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


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

from src.config_utils import load_config  # noqa: E402
from src.did.aggregated import (  # noqa: E402
    AGGREGATED_PANEL_LEVELS,
    AggregatedPanelKey,
    build_aggregated_panels,
)
from src.did.panels import wordfish_forum_v2_available  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build aggregated DiD panels (1d and 3d).")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild panels even if cached CSVs exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Function summary: write aggregated panel CSVs under did/panels/aggregated/."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.refresh:
        from src.did.paths import did_panels_dir

        agg_dir = did_panels_dir(config, "aggregated")
        for name in AGGREGATED_PANEL_LEVELS:
            for bd in (1, 3):
                p = agg_dir / f"did_{name}_{bd}d.csv"
                if p.is_file():
                    p.unlink()
    panels = build_aggregated_panels(config)
    for level in AGGREGATED_PANEL_LEVELS:
        for bd in (1, 3):
            df = panels.get(AggregatedPanelKey(level, bd))
            print(
                f"[prepare_did_aggregated_panels] {level} {bd}d rows={len(df)}",
                flush=True,
            )
    if wordfish_forum_v2_available(config):
        print("[prepare_did_aggregated_panels] forum v2 available", flush=True)
    print("[prepare_did_aggregated_panels] done", flush=True)


if __name__ == "__main__":
    main()
