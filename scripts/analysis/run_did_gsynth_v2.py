"""
Script summary:
Run gsynth v2 (demeaned SC + pre-fit gate) on country-day language aggregated panels.

Functionality:
- Loads did/panels/aggregated/did_language_{3}d.csv.
- Demeans each unit by pre-ban mean; estimates synthetic Italy ATT (R gsynth or augmented SC).
- Applies hard pre-fit gate; writes att/inference CSVs to did/estimates/gsynth_v2/.
- Exports pre-ban diagnostic PNGs; aborts only if pole_share passes pre-fit but sign mismatches did_summary.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_aggregated_panels.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/run_did_gsynth_v2.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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
from src.did.gsynth import (  # noqa: E402
    GSYNTH_V2_OUTCOMES,
    GsynthV2Result,
    POLE_SHARE_SIGN_REFERENCE,
    load_did_summary_beta,
    outcomes_for_gsynth_v2,
    plot_gsynth_v2_prefit,
    pole_share_sign_gate_should_abort,
    run_gsynth_v2_att,
    write_gsynth_v2_outputs,
    write_gsynth_v2_readme,
)
from src.did.paths import gsynth_v2_figure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for gsynth v2 estimation."""
    parser = argparse.ArgumentParser(
        description="Generalized synthetic control v2 on language aggregated panel."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=3, choices=(1, 3))
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome_ids (default: GSYNTH_V2_OUTCOMES).",
    )
    return parser.parse_args()


def _fit_all(
    config: Dict[str, Any],
    bin_days: int,
    outcome_ids: Optional[List[str]],
) -> List[GsynthV2Result]:
    """Function summary: estimate gsynth v2 for each outcome in the fixed set.

    Parameters:
    - config: study YAML.
    - bin_days: panel bin width.
    - outcome_ids: optional outcome override list.

    Returns:
    - List of GsynthV2Result (skips outcomes that fail to load).
    """
    specs = outcomes_for_gsynth_v2(config, outcome_ids)
    results: List[GsynthV2Result] = []
    for oc in specs:
        try:
            res = run_gsynth_v2_att(config, oc, bin_days=bin_days)
            results.append(res)
            p_floor = res.inference.get("placebo_p_floor", float("nan"))
            print(
                f"[gsynth_v2] {oc.outcome_id} backend={res.backend} "
                f"pre_fit_ok={res.gate['pre_fit_ok']} "
                f"p_floor={p_floor:.4g} (1/(n_placebos+1)) "
                f"p_space={res.inference.get('p_placebo_space')} "
                f"p_time={res.inference.get('p_placebo_time')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[gsynth_v2] skip {oc.outcome_id}: {exc}", flush=True)
    return results


def _write_diagnostic_pngs(
    config: Dict[str, Any],
    results: List[GsynthV2Result],
    launch: str,
) -> None:
    """Function summary: write pre-ban overlay PNG for each estimated outcome.

    Parameters:
    - config: study YAML.
    - results: gsynth v2 results.
    - launch: ban launch date string.
    """
    fig_dir = gsynth_v2_figure_dir(config)
    for res in results:
        oid = res.inference["outcome_id"]
        plot_gsynth_v2_prefit(
            oid,
            res.att,
            launch,
            fig_dir / f"{oid}_prefit_overlay.png",
            pre_fit_ok=bool(res.gate["pre_fit_ok"]),
        )


def _pole_share_sign_check(
    config: Dict[str, Any],
    results: List[GsynthV2Result],
) -> Optional[Dict[str, Any]]:
    """Function summary: evaluate pole_share sign gate; abort if pre-fit ok but sign mismatch.

    Parameters:
    - config: study YAML.
    - results: gsynth v2 results.

    Returns:
    - Sign metadata dict for inference CSV, or None if pole_share not in results.

    Raises:
    - SystemExit: when pre_fit_ok and sign disagrees with did_summary TWFE.
    """
    ref = POLE_SHARE_SIGN_REFERENCE
    pole = next((r for r in results if r.inference["outcome_id"] == "pole_share"), None)
    if pole is None:
        return None

    did_beta = load_did_summary_beta(
        config,
        ref["outcome_id"],
        ref["strategy_id"],
        ref["spec"],
    )
    pre_fit_ok = bool(pole.gate["pre_fit_ok"])
    mean_post = float(pole.gate["mean_post_att"])
    sign_checked = pre_fit_ok
    sign_pass = True
    if sign_checked:
        sign_pass = not pole_share_sign_gate_should_abort(pre_fit_ok, mean_post, did_beta)

    sign_meta = {
        "did_summary_beta": did_beta,
        "did_summary_strategy": ref["strategy_id"],
        "did_summary_spec": ref["spec"],
        "sign_gate_checked": sign_checked,
        "sign_gate_pass": sign_pass if sign_checked else True,
    }

    if pole_share_sign_gate_should_abort(pre_fit_ok, mean_post, did_beta):
        print(
            f"[gsynth_v2] ABORT: pole_share pre_fit_ok=True but sign mismatch: "
            f"gsynth mean_post_att={mean_post:+.6f} vs did_summary beta={did_beta:+.6f} "
            f"({ref['strategy_id']}/{ref['spec']})",
            flush=True,
        )
        sys.exit(1)

    if not pre_fit_ok:
        print(
            "[gsynth_v2] pole_share failed pre-fit gate; skipping sign check, continuing",
            flush=True,
        )

    return sign_meta


def main() -> None:
    """Function summary: run gsynth v2 for each outcome; write outputs unless sign gate aborts."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    oids = args.outcomes.split(",") if args.outcomes else None

    from scripts.diagnostics.descriptives_util import event_dates_from_config

    _, _, launch, _ = event_dates_from_config(config)

    results = _fit_all(config, args.bin_days, oids)
    if not results:
        print("[gsynth_v2] no outcomes estimated", flush=True)
        sys.exit(1)

    _write_diagnostic_pngs(config, results, launch)

    pole_sign_meta = _pole_share_sign_check(config, results)

    for res in results:
        sign_meta = pole_sign_meta if res.inference["outcome_id"] == "pole_share" else None
        write_gsynth_v2_outputs(
            config,
            res,
            res.inference["outcome_id"],
            args.bin_days,
            sign_meta=sign_meta,
        )

    write_gsynth_v2_readme(config)

    print("[gsynth_v2] summary:", flush=True)
    for res in results:
        g = res.gate
        print(
            f"  {res.inference['outcome_id']:24s} backend={res.backend:12s} "
            f"pre_fit_ok={str(g['pre_fit_ok']):5s} "
            f"mean_pre={g['mean_pre_att']:+.4f} mean_post={g['mean_post_att']:+.4f} "
            f"p_floor={res.inference.get('placebo_p_floor', float('nan')):.4g}",
            flush=True,
        )
    print("[gsynth_v2] done", flush=True)


if __name__ == "__main__":
    main()
