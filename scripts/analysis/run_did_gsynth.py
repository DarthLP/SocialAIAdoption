"""
Script summary:
Run generalized synthetic control (gsynth) on country-day language aggregated panels.

Functionality:
- Loads did/panels/aggregated/did_language_{1,3}d.csv.
- Estimates synthetic Italy ATT with R gsynth (rpy2) or augmented SC fallback.
- Writes ATT paths and placebo-in-space/time p-values to did/estimates/gsynth/.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_aggregated_panels.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/run_did_gsynth.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/run_did_gsynth.py --bin-days 3 --outcomes sem_axis_ideology
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
from src.did.gsynth import outcomes_for_gsynth, run_gsynth_att, write_gsynth_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for gsynth estimation."""
    parser = argparse.ArgumentParser(description="Generalized synthetic control on language panel.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=3, choices=(1, 3))
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome_ids (default: headline outcomes).",
    )
    return parser.parse_args()


def main() -> None:
    """Function summary: run gsynth for each requested outcome."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    oids = args.outcomes.split(",") if args.outcomes else None
    specs = outcomes_for_gsynth(config, oids)
    for oc in specs:
        try:
            res = run_gsynth_att(config, oc, bin_days=args.bin_days)
            write_gsynth_outputs(config, res, oc.outcome_id, args.bin_days)
            print(
                f"[gsynth] {oc.outcome_id} backend={res.backend} "
                f"p_space={res.inference.get('p_placebo_space')} "
                f"p_time={res.inference.get('p_placebo_time')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[gsynth] skip {oc.outcome_id}: {exc}", flush=True)


if __name__ == "__main__":
    main()
