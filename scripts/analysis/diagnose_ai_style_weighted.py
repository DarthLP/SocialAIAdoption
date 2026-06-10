"""
Script summary:
Diagnostic CSV for comment-weighted ai_style_rate full_ban puzzle (Italy ChatGPT-ban DiD).

Functionality:
- Weighted TWFE (n_comments) on cross_country_all for ai_style_rate.
- Post-phase decomposition: post_short_3d, post_medium_7d, post_long_tail, post_first_2bd.
- Same spec on ban-topic-excluded panel (exbantopic variant; in-memory only).
- Leave-one-out: drop each of the top-3 Italian forums by total n_comments, re-estimate.
- Writes ai_style_weighted_diagnosis.csv; does not touch did_summary.csv.

How to apply/run:
  .venv/bin/python scripts/analysis/diagnose_ai_style_weighted.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

OUTCOME_ID = "ai_style_rate"
Y_COL = "ai_style_rate_100w_mean"
STRATEGY_ID = "cross_country_all"
FULL_BAN = "full_ban"
WEIGHTS_COL = "n_comments"
POST_PHASE_SPECS = ("post_short_3d", "post_medium_7d", "post_long_tail", "post_first_2bd")
LOO_TOP_N = 3
ATTENUATION_THRESHOLD_PCT = 30.0


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
from src.did.estimate import run_strategy_twfe  # noqa: E402
from src.did.panels import build_analysis_panels  # noqa: E402
from src.did.paths import did_estimates_dir  # noqa: E402
from src.did.specs import StrategySpec, activate_post_phases_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Weighted ai_style_rate diagnosis CSV.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: estimates_weighted/ai_style_weighted_diagnosis.csv).",
    )
    return parser.parse_args()


def _entity_col(panel: pd.DataFrame) -> str:
    """Function summary: entity id column for subreddit-day panel."""
    return "entity_id" if "entity_id" in panel.columns else "subreddit"


def _estimate(
    panel: pd.DataFrame,
    spec: str,
    *,
    entity_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Function summary: weighted TWFE for ai_style_rate on one post spec.

    Parameters:
    - panel: subreddit-day panel with outcome and n_comments.
    - spec: post_mode (full_ban or post-phase id).
    - entity_col: optional entity column override.

    Returns:
    - Result dict from run_strategy_twfe (beta, pvalue, n_obs, ...).
    """
    if panel.empty or Y_COL not in panel.columns:
        return {"beta": float("nan"), "pvalue": float("nan"), "n_obs": 0, "estimation_note": "empty_panel"}
    ent = entity_col or _entity_col(panel)
    strat = StrategySpec(STRATEGY_ID, post_mode=spec)
    return run_strategy_twfe(
        panel,
        strat,
        Y_COL,
        entity_col=ent,
        time_col="time_id",
        cluster_col=ent,
        panel_kind="subreddit_day",
        weights=WEIGHTS_COL,
    )


def _top_italian_forums(panel: pd.DataFrame, n: int = LOO_TOP_N) -> List[str]:
    """Function summary: top-n Italian subreddits by total n_comments in panel.

    Parameters:
    - panel: subreddit-day panel with topic_family and n_comments.
    - n: number of forums to return.

    Returns:
    - List of subreddit names descending by weight.
    """
    if "topic_family" not in panel.columns or "n_comments" not in panel.columns:
        return []
    it = panel[panel["topic_family"].astype(str).str.startswith("it_")].copy()
    if it.empty:
        return []
    w = it.groupby("subreddit")["n_comments"].sum().sort_values(ascending=False)
    return [str(s) for s in w.head(n).index.tolist()]


def _post_phase_verdict(phase_rows: List[Dict[str, Any]], full_ban_beta: float) -> str:
    """Function summary: one-line verdict on whether positive coef is in post_long_tail.

    Parameters:
    - phase_rows: list of dicts with spec and beta for post-phase checks.
    - full_ban_beta: baseline weighted full_ban beta.

    Returns:
    - Verdict string (may be empty).
    """
    if not np.isfinite(full_ban_beta) or full_ban_beta <= 0:
        return "Full_ban weighted coefficient not positive"
    by_spec = {str(r["spec"]): float(r.get("beta", np.nan)) for r in phase_rows}
    long_beta = by_spec.get("post_long_tail", float("nan"))
    if not np.isfinite(long_beta):
        return "post_long_tail estimate unavailable"
    others = [by_spec.get(s, float("nan")) for s in POST_PHASE_SPECS if s != "post_long_tail"]
    finite_others = [b for b in others if np.isfinite(b)]
    if long_beta > 0 and (not finite_others or long_beta >= max(finite_others)):
        return "Positive weighted coefficient concentrated in post_long_tail (late April window)"
    return "Positive weighted full_ban not concentrated in post_long_tail"


def _exbantopic_verdict(attenuation_pct: float) -> str:
    """Function summary: verdict when exbantopic attenuates baseline weighted beta.

    Parameters:
    - attenuation_pct: percent attenuation (base−exo)/base×100.

    Returns:
    - Verdict string or empty.
    """
    if np.isfinite(attenuation_pct) and attenuation_pct > ATTENUATION_THRESHOLD_PCT:
        return "Exbantopic panel attenuates weighted full_ban by >30%"
    return ""


def _attenuation_pct(baseline_beta: float, variant_beta: float) -> float:
    """Function summary: percent attenuation of baseline toward zero."""
    if not np.isfinite(baseline_beta) or baseline_beta == 0:
        return float("nan")
    return float(100.0 * (baseline_beta - variant_beta) / baseline_beta)


def build_diagnosis_table(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: assemble all diagnosis rows for ai_style_rate weighted TWFE.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - DataFrame ready to write as ai_style_weighted_diagnosis.csv.
    """
    activate_post_phases_from_config(config)
    panels = build_analysis_panels(config, families=["lexical"], variant=None)
    panel = panels.sub_v1
    if Y_COL not in panel.columns:
        alt = Y_COL.replace("_mean", "")
        if alt in panel.columns:
            panel = panel.copy()
            panel[Y_COL] = panel[alt]
        else:
            raise ValueError(f"Missing {Y_COL} on subreddit panel")

    exo_panels = build_analysis_panels(config, families=["lexical"], variant="exbantopic")
    exo_panel = exo_panels.sub_v1
    if Y_COL not in exo_panel.columns and Y_COL.replace("_mean", "") in exo_panel.columns:
        exo_panel = exo_panel.copy()
        exo_panel[Y_COL] = exo_panel[Y_COL.replace("_mean", "")]

    baseline = _estimate(panel, FULL_BAN)
    baseline_beta = float(baseline.get("beta", np.nan))

    rows: List[Dict[str, Any]] = []
    phase_betas: List[Dict[str, Any]] = []

    for spec in POST_PHASE_SPECS:
        res = _estimate(panel, spec)
        row = {
            "check_type": "post_phase",
            "spec": spec,
            "dropped_forum": "",
            "beta": res.get("beta"),
            "pvalue": res.get("pvalue"),
            "n_obs": res.get("n_obs"),
            "baseline_beta": baseline_beta,
            "attenuation_pct": float("nan"),
            "verdict": "",
        }
        rows.append(row)
        phase_betas.append(row)

    rows.append(
        {
            "check_type": "post_phase_summary",
            "spec": "",
            "dropped_forum": "",
            "beta": float("nan"),
            "pvalue": float("nan"),
            "n_obs": float("nan"),
            "baseline_beta": baseline_beta,
            "attenuation_pct": float("nan"),
            "verdict": _post_phase_verdict(phase_betas, baseline_beta),
        }
    )

    exo_res = _estimate(exo_panel, FULL_BAN)
    exo_beta = float(exo_res.get("beta", np.nan))
    att = _attenuation_pct(baseline_beta, exo_beta)
    rows.append(
        {
            "check_type": "exbantopic",
            "spec": FULL_BAN,
            "dropped_forum": "",
            "beta": exo_beta,
            "pvalue": exo_res.get("pvalue"),
            "n_obs": exo_res.get("n_obs"),
            "baseline_beta": baseline_beta,
            "attenuation_pct": att,
            "verdict": _exbantopic_verdict(att),
        }
    )

    for forum in _top_italian_forums(panel, LOO_TOP_N):
        filtered = panel[panel["subreddit"].astype(str) != forum].copy()
        loo_res = _estimate(filtered, FULL_BAN)
        rows.append(
            {
                "check_type": "loo",
                "spec": FULL_BAN,
                "dropped_forum": forum,
                "beta": loo_res.get("beta"),
                "pvalue": loo_res.get("pvalue"),
                "n_obs": loo_res.get("n_obs"),
                "baseline_beta": baseline_beta,
                "attenuation_pct": _attenuation_pct(baseline_beta, float(loo_res.get("beta", np.nan))),
                "verdict": "",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: write diagnosis CSV under estimates_weighted/."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    df = build_diagnosis_table(config)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = did_estimates_dir(config, weighted=True) / "ai_style_weighted_diagnosis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[diagnose_ai_style_weighted] wrote {out_path} ({len(df)} rows)", flush=True)
    summary = df[df["check_type"] == "post_phase_summary"]
    if not summary.empty and str(summary.iloc[0].get("verdict", "")):
        print(f"  post_phase verdict: {summary.iloc[0]['verdict']}", flush=True)
    exo = df[df["check_type"] == "exbantopic"]
    if not exo.empty and str(exo.iloc[0].get("verdict", "")):
        print(f"  exbantopic verdict: {exo.iloc[0]['verdict']}", flush=True)


if __name__ == "__main__":
    main()
