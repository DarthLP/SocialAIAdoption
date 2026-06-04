"""
Script summary:
Patch did_summary.csv inference columns (wild_p, p_placebo_space) without re-estimating betas.

Functionality:
- Recomputes placebo-in-space and restricted WCB p-values from saved panels.
- Updates rows in did_summary.csv in place.

How to apply/run:
  .venv/bin/python scripts/diagnostics/patch_did_inference.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/patch_did_inference.py --bootstrap-draws 499
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
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

from src.config_utils import load_config  # noqa: E402
from src.did.inference import placebo_in_space_p, wild_cluster_bootstrap_p  # noqa: E402
from src.did.outcomes import OUTCOME_REGISTRY  # noqa: E402
from src.did.panels import AnalysisPanels, build_analysis_panels, slice_panel_for_ddd  # noqa: E402
from src.did.paths import did_summary_paths  # noqa: E402
from src.did.specs import (  # noqa: E402
    StrategySpec,
    is_cross_country_strategy,
    is_placebo_in_space_eligible_strategy,
    is_wcb_eligible_strategy,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for inference patch."""
    parser = argparse.ArgumentParser(description="Patch DiD summary inference p-values.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bootstrap-draws", type=int, default=999)
    parser.add_argument(
        "--wcb-only",
        action="store_true",
        help="Only recompute wild_p (skip placebo-in-space).",
    )
    parser.add_argument(
        "--placebo-only",
        action="store_true",
        help="Only recompute placebo-in-space (skip WCB).",
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome_ids to patch (default: all).",
    )
    return parser.parse_args()


def _panel_for_family(panels: AnalysisPanels, family: str) -> pd.DataFrame:
    """Function summary: map outcome family to analysis panel."""
    if family in ("wordfish_author_v2",):
        return panels.auth_v2
    if family in ("wordfish_author",):
        return panels.auth_v1
    if family == "semantic_axis_author_week":
        return panels.auth_semantic
    return panels.sub_v1


def _resolve_col(panel: pd.DataFrame, column: str) -> str | None:
    """Function summary: return column name if present."""
    if column in panel.columns:
        return column
    alt = column.replace("_mean", "")
    return alt if alt in panel.columns else None


def main() -> None:
    """Function summary: patch inference columns in did_summary.csv."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    summary_path, _ = did_summary_paths(config)
    df = pd.read_csv(summary_path)
    if "placebo_note" not in df.columns:
        df["placebo_note"] = pd.Series([np.nan] * len(df), dtype=object)
    else:
        df["placebo_note"] = df["placebo_note"].astype(object)
    panels = build_analysis_panels(config)
    oc_map = {o.outcome_id: o for o in OUTCOME_REGISTRY}
    outcome_filter = set(args.outcomes.split(",")) if args.outcomes else None

    for idx, row in df.iterrows():
        if outcome_filter and str(row["outcome_id"]) not in outcome_filter:
            continue
        if str(row.get("spec", "full_ban")) != "full_ban":
            continue
        sid = str(row["strategy_id"])
        oid = str(row["outcome_id"])
        oc = oc_map.get(oid)
        if oc is None:
            continue
        if is_cross_country_strategy(sid) and not is_placebo_in_space_eligible_strategy(sid):
            df.at[idx, "p_placebo_space"] = np.nan
            df.at[idx, "perm_p"] = np.nan
            df.at[idx, "placebo_p_floor"] = np.nan
            df.at[idx, "placebo_note"] = "not_applicable_single_country_contrast"
            continue
        if sid == "within_italy_ddd":
            panel = slice_panel_for_ddd(panels.slice_panel)
            entity_col = "subreddit"
        elif is_cross_country_strategy(sid) or is_wcb_eligible_strategy(sid):
            panel = _panel_for_family(panels, oc.family)
            entity_col = "author" if "author" in panel.columns else "entity_id"
        else:
            continue
        y_col = _resolve_col(panel, str(row.get("column", oc.column)))
        if y_col is None:
            y_col = _resolve_col(panel, oc.column)
        if y_col is None:
            continue
        strat = StrategySpec(sid)
        try:
            if is_placebo_in_space_eligible_strategy(sid) and not args.wcb_only:
                pis = placebo_in_space_p(panel, strat, y_col, entity_col=entity_col)
                df.at[idx, "p_placebo_space"] = pis.p
                df.at[idx, "perm_p"] = pis.p
                df.at[idx, "perm_p_beta"] = pis.perm_p_beta
                df.at[idx, "perm_p_t"] = pis.perm_p_t
                df.at[idx, "placebo_p_floor"] = pis.p_floor
                df.at[idx, "placebo_note"] = np.nan
            if is_wcb_eligible_strategy(sid) and not args.placebo_only:
                wp = wild_cluster_bootstrap_p(
                    panel,
                    strat,
                    y_col,
                    n_draws=args.bootstrap_draws,
                    entity_col=entity_col,
                    time_col="time_id",
                )
                if np.isfinite(wp):
                    df.at[idx, "wild_p"] = wp
        except Exception:
            continue

    df.to_csv(summary_path, index=False)
    n_wild = int(df["wild_p"].notna().sum())
    n_space = int(df["p_placebo_space"].notna().sum()) if "p_placebo_space" in df.columns else 0
    print(f"[patch_did_inference] wild_p finite: {n_wild}, p_placebo_space finite: {n_space}", flush=True)


if __name__ == "__main__":
    main()
