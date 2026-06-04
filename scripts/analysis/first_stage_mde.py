"""
Script summary:
Minimum detectable effect (MDE) table for first-stage DiD outcomes from saved TWFE summaries.

Functionality:
- Reads cross_country_all × full_ban SE from did/estimates/summary/by_outcome/*.csv.
- Computes MDE = 2.8 × SE and compares to comment-weighted IT pre-period baselines.
- Plausibility columns at 1%, 2%, 5% adoption shares (gap = 2× baseline).

How to apply/run:
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --families lexical
  .venv/bin/python scripts/analysis/first_stage_mde.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/first_stage_mde.py --weighted  # after estimates_weighted exist
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _setup_project_root() -> Path:
    """Function summary: resolve repo root."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config  # noqa: E402
from src.did.outcomes import FIRST_STAGE_OUTCOMES  # noqa: E402
from src.did.paths import did_panels_dir, did_summary_dir  # noqa: E402

MDE_Z = 2.8
STRATEGY = "cross_country_all"
SPEC = "full_ban"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for MDE table."""
    parser = argparse.ArgumentParser(description="First-stage MDE from saved DiD outputs.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Read SE from estimates_weighted/ (requires weighted DiD run).",
    )
    return parser.parse_args()


def _weighted_baseline(panel: pd.DataFrame, col: str) -> float:
    """Function summary: IT pre-period comment-weighted mean of outcome column."""
    pre = panel[(panel["post"].astype(int) == 0) & (panel["IT"].astype(int) == 1)].copy()
    if pre.empty or col not in pre.columns:
        return float("nan")
    y = pd.to_numeric(pre[col], errors="coerce")
    w = pd.to_numeric(pre["n_comments"], errors="coerce").astype(float)
    mask = y.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(y[mask], weights=w[mask]))


def build_mde_table(config: Dict[str, Any], *, weighted: bool = False) -> pd.DataFrame:
    """Function summary: assemble first_stage_mde rows for FIRST_STAGE_OUTCOMES."""
    from src.did.paths import did_estimates_dir

    summary_dir = did_summary_dir(config, weighted=weighted)
    panel_path = did_panels_dir(config, "subreddit") / "did_subreddit_panel_1d.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    panel = pd.read_csv(panel_path)
    if "n_comments" not in panel.columns and "n_comments_x" in panel.columns:
        panel["n_comments"] = panel["n_comments_x"]

    rows: List[Dict[str, Any]] = []
    for oid in FIRST_STAGE_OUTCOMES:
        by_path = summary_dir / "by_outcome" / f"{oid}.csv"
        if not by_path.is_file():
            print(f"[first_stage_mde] skip {oid}: missing {by_path.name}", flush=True)
            continue
        est = pd.read_csv(by_path)
        sub = est[
            (est["strategy_id"].astype(str) == STRATEGY)
            & (est["spec"].astype(str) == SPEC)
        ]
        if sub.empty:
            continue
        row = sub.iloc[0]
        se = float(row.get("se", np.nan))
        mde = MDE_Z * se if np.isfinite(se) else float("nan")
        col = str(row.get("column", oid))
        baseline = _weighted_baseline(panel, col)
        mde_ratio = mde / baseline if baseline and np.isfinite(baseline) and baseline != 0 else float("nan")
        plausible_gap = 2.0 * baseline if np.isfinite(baseline) else float("nan")
        row_out: Dict[str, Any] = {
            "outcome_id": oid,
            "strategy_id": STRATEGY,
            "spec": SPEC,
            "se": se,
            "mde": mde,
            "baseline_it_pre_weighted": baseline,
            "mde_over_baseline": mde_ratio,
            "plausible_gap_2x_baseline": plausible_gap,
            "weighted_run": int(weighted),
        }
        for share in (0.01, 0.02, 0.05):
            effect = share * plausible_gap if np.isfinite(plausible_gap) else float("nan")
            row_out[f"plausible_effect_{int(share * 100)}pct"] = effect
            row_out[f"mde_over_plausible_{int(share * 100)}pct"] = (
                mde / effect if effect and np.isfinite(effect) and effect != 0 else float("nan")
            )
        rows.append(row_out)
    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: write first_stage_mde.csv."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    df = build_mde_table(config, weighted=bool(args.weighted))
    out_path = did_summary_dir(config) / "first_stage_mde.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[first_stage_mde] wrote {out_path} ({len(df)} rows)", flush=True)
    if "ai_style_rate" in df["outcome_id"].astype(str).values:
        r = df[df["outcome_id"] == "ai_style_rate"].iloc[0]
        print(
            f"  ai_style_rate mde/plausible_2pct={r.get('mde_over_plausible_2pct', float('nan')):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
