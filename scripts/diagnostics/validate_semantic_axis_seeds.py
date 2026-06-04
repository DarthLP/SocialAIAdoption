"""
Script summary:
Validate semantic-axis seed in-vocab coverage and held-out word axis directions.

Functionality:
- Loads fastText models and seed poles per language (no enriched shards required).
- Writes semantic_axis_seed_coverage.csv and semantic_axis_axis_sanity.csv (seven axes including extended issue dimensions).

How to apply/run:
  .venv/bin/python scripts/diagnostics/validate_semantic_axis_seeds.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

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
    get_axes_for_language,
    held_out_axis_sanity_report,
    run_language_vector_wave,
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

    coverage_rows: list = []
    sanity_rows: list = []

    def _validate_lang(lang: str, kv: Any) -> None:
        """Function summary: coverage + held-out sanity for one loaded language."""
        if kv is None:
            print(f"[validate_semantic_axis_seeds] skip lang={lang}: fastText model missing", flush=True)
            return
        print(f"[validate_semantic_axis_seeds] lang={lang} coverage + sanity ...", flush=True)
        coverage_rows.extend(
            seed_coverage_report(lang, PROJECT_ROOT, sem_cfg, config=config, kv=kv)
        )
        axis_vecs = get_axes_for_language(lang, PROJECT_ROOT, sem_cfg, config=config)
        sanity_rows.extend(
            held_out_axis_sanity_report(
                lang, PROJECT_ROOT, sem_cfg, config=config, kv=kv, axis_vecs=axis_vecs
            )
        )

    print("[validate_semantic_axis_seeds] 3 language waves (one model at a time)", flush=True)
    run_language_vector_wave(PROJECT_ROOT, sem_cfg, _validate_lang)

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
