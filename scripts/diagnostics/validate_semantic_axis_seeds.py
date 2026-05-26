"""
Script summary:
Validate semantic-axis seed in-vocab coverage and held-out word axis directions.

Functionality:
- Loads fastText models and seed poles per language (no enriched shards required).
- Writes semantic_axis_seed_coverage.csv and semantic_axis_axis_sanity.csv.

How to apply/run:
  .venv/bin/python scripts/diagnostics/validate_semantic_axis_seeds.py --config config/italy_polarization_setup.yaml
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

from src.config_utils import load_config, load_semantic_axis_config, tables_subdir  # noqa: E402
from src.embeddings import (  # noqa: E402
    ensure_exclusive_vector_lang,
    held_out_axis_sanity_report,
    seed_coverage_report,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate semantic-axis seeds and held-out checks.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def main() -> None:
    """Function summary: write seed coverage and axis sanity tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    sem_cfg = load_semantic_axis_config(config)
    out_dir = tables_subdir(config, "semantic_axis")
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = []
    sanity_rows = []
    for lang in ("it", "en", "de"):
        ensure_exclusive_vector_lang(lang, sem_cfg)
        coverage_rows.extend(
            seed_coverage_report(lang, PROJECT_ROOT, sem_cfg, config=config)
        )
        sanity_rows.extend(
            held_out_axis_sanity_report(lang, PROJECT_ROOT, sem_cfg, config=config)
        )

    pd.DataFrame(coverage_rows).to_csv(out_dir / "semantic_axis_seed_coverage.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(out_dir / "semantic_axis_axis_sanity.csv", index=False)
    n_pass = sum(int(r["pass"]) for r in sanity_rows)
    print(
        f"[validate_semantic_axis_seeds] wrote {out_dir} "
        f"coverage_rows={len(coverage_rows)} sanity_pass={n_pass}/{len(sanity_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
