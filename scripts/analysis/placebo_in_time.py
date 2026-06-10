"""
Script summary:
Placebo-in-time inference with fixed-length post windows for Italy ChatGPT-ban TWFE.

Functionality:
- Re-estimates cross_country_all static TWFE at placebo ban dates 2023-03-08..2023-03-24 using
  a fixed 7-day post window on data strictly before 2023-03-31 (real ban excluded).
- True-date 7d statistic uses the full event-window panel (post days may include April).
- Reports rank-based permutation p for beta and t; optional 3-day post window.
- Combined space×time p ranks the true 7d beta among placebo-date × placebo-country draws.
- Exchangeability assumes calendar placebo dates are interchangeable under the sharp null.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/placebo_in_time.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/placebo_in_time.py --outcomes ai_style_rate,pole_share
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRUE_BAN_DATE = "2023-03-31"
TRUNCATE_BEFORE = "2023-03-31"
PLACEBO_START = "2023-03-08"
PLACEBO_END = "2023-03-24"


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
from src.did.inference import (  # noqa: E402
    _apply_placebo_treat,
    _control_countries_in_sample,
    _is_italy_country,
    assign_entity_country_series,
)
from src.did.outcomes import OUTCOME_REGISTRY  # noqa: E402
from src.did.panels import build_analysis_panels  # noqa: E402
from src.did.paths import did_summary_dir  # noqa: E402
from src.did.robustness import placebo_panel_fixed_post_window  # noqa: E402
from src.did.specs import StrategySpec, filter_strategy_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for placebo-in-time."""
    parser = argparse.ArgumentParser(description="Placebo-in-time with fixed post windows.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--outcomes",
        type=str,
        default="ai_style_rate,pole_share,sem_axis_emotion,net_ideology,aggression_rate,sem_axis_ideology_var",
        help="Comma-separated outcome_ids.",
    )
    parser.add_argument("--post-days", type=int, default=7, help="Fixed post-window length (primary).")
    parser.add_argument(
        "--also-3d",
        action="store_true",
        help="Also compute 3-day fixed-window statistics.",
    )
    return parser.parse_args()


def _placebo_dates() -> List[str]:
    """Function summary: daily placebo ban dates from PLACEBO_START through PLACEBO_END."""
    start = datetime.strptime(PLACEBO_START, "%Y-%m-%d")
    end = datetime.strptime(PLACEBO_END, "%Y-%m-%d")
    out: List[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _ri_two_sided(true_stat: float, placebo_stats: List[float]) -> Tuple[float, int]:
    """Function summary: rank-based two-sided permutation p with +1 denominator.

    Parameters:
    - true_stat: Italy estimate.
    - placebo_stats: placebo draw statistics.

    Returns:
    - Tuple (p_value, n_placebo_draws_used).
    """
    vals = [float(x) for x in placebo_stats if np.isfinite(x)]
    n = len(vals)
    if not np.isfinite(true_stat) or n == 0:
        return float("nan"), n
    n_ge = sum(1 for x in vals if abs(x) >= abs(true_stat))
    p = float((n_ge + 1) / (n + 1))
    p_floor = 1.0 / (n + 1)
    return min(1.0, max(p_floor, p)), n


def _estimate_7d(
    panel: pd.DataFrame,
    outcome_col: str,
    placebo_date: str,
    *,
    post_days: int,
    truncate: bool,
) -> Dict[str, float]:
    """Function summary: TWFE beta/se/t for fixed-window placebo date."""
    work = placebo_panel_fixed_post_window(
        panel,
        placebo_date,
        post_days=post_days,
        truncate_before=TRUNCATE_BEFORE if truncate else None,
    )
    strat = StrategySpec("cross_country_all")
    work = filter_strategy_sample(work, strat)
    if work.empty or outcome_col not in work.columns:
        return {"beta": float("nan"), "se": float("nan"), "t": float("nan")}
    res = run_strategy_twfe(work, strat, outcome_col)
    beta = float(res.get("beta", np.nan))
    se = float(res.get("se", np.nan))
    t = float(beta / se) if se and se > 0 and np.isfinite(beta) else float("nan")
    return {"beta": beta, "se": se, "t": t}


def _space_placebo_betas(
    panel: pd.DataFrame,
    outcome_col: str,
    placebo_date: str,
    post_days: int,
    entity_col: str = "entity_id",
) -> List[float]:
    """Function summary: placebo-country betas for one placebo date (fixed 7d window)."""
    from src.did.estimate import estimate_twfe

    work = placebo_panel_fixed_post_window(
        panel, placebo_date, post_days=post_days, truncate_before=TRUNCATE_BEFORE
    )
    strat = StrategySpec("cross_country_all")
    work = filter_strategy_sample(work, strat)
    ent_country = assign_entity_country_series(work, entity_col)
    controls = work[~ent_country.map(_is_italy_country)].copy()
    if controls.empty:
        return []
    countries = _control_countries_in_sample(controls, entity_col, strat)
    betas: List[float] = []
    for fake_c in countries:
        try:
            pl = _apply_placebo_treat(controls, fake_c, entity_col)
        except ValueError:
            continue
        r = estimate_twfe(pl, outcome_col, entity_col, "time_id")
        b = r.get("beta", np.nan)
        if np.isfinite(b):
            betas.append(float(b))
    return betas


def _resolve_outcome_column(panel: pd.DataFrame, outcome_id: str) -> str:
    """Function summary: map outcome_id to panel column name."""
    spec = next((o for o in OUTCOME_REGISTRY if o.outcome_id == outcome_id), None)
    if spec is None:
        raise ValueError(f"Unknown outcome_id: {outcome_id}")
    col = spec.column
    if col not in panel.columns:
        raise ValueError(f"Outcome {outcome_id} missing column {col!r} on subreddit panel")
    return col


def run_placebo_in_time(
    config: Dict[str, Any],
    outcome_ids: List[str],
    post_days: int = 7,
    also_3d: bool = False,
) -> pd.DataFrame:
    """Function summary: build placebo_in_time summary table for requested outcomes."""
    panels = build_analysis_panels(config, families=["lexical", "semantic_axis"])
    panel = panels.sub_v1
    if panel.empty:
        raise RuntimeError("Empty subreddit panel; run prepare_did_subreddit_panel.py")

    rows: List[Dict[str, Any]] = []
    p_dates = _placebo_dates()
    rank_dates = p_dates + [TRUE_BAN_DATE]

    for oid in outcome_ids:
        y_col = _resolve_outcome_column(panel, oid)
        strat_full = StrategySpec("cross_country_all", post_mode="full_ban")
        full_panel = filter_strategy_sample(panel, strat_full)
        ref = run_strategy_twfe(full_panel, strat_full, y_col) if not full_panel.empty else {}
        beta_ref = float(ref.get("beta", np.nan))
        t_ref = (
            float(beta_ref / ref["se"])
            if ref.get("se") and ref["se"] > 0 and np.isfinite(beta_ref)
            else float("nan")
        )

        for pdate in rank_dates:
            truncate = pdate != TRUE_BAN_DATE
            est7 = _estimate_7d(panel, y_col, pdate, post_days=post_days, truncate=truncate)
            row: Dict[str, Any] = {
                "outcome_id": oid,
                "placebo_date": pdate,
                "post_days": post_days,
                "truncated_pre_ban": int(truncate),
                "beta_7d": est7["beta"],
                "se_7d": est7["se"],
                "t_7d": est7["t"],
                "beta_full_ban_ref": beta_ref,
                "t_full_ban_ref": t_ref,
            }
            if also_3d:
                est3 = _estimate_7d(panel, y_col, pdate, post_days=3, truncate=truncate)
                row.update(
                    {
                        "beta_3d": est3["beta"],
                        "se_3d": est3["se"],
                        "t_3d": est3["t"],
                    }
                )
            rows.append(row)

        sub = pd.DataFrame([r for r in rows if r["outcome_id"] == oid])
        placebo_betas = [
            float(r["beta_7d"])
            for r in sub.to_dict("records")
            if r["placebo_date"] in p_dates and np.isfinite(r["beta_7d"])
        ]
        placebo_ts = [
            float(r["t_7d"]) for r in sub.to_dict("records") if r["placebo_date"] in p_dates and np.isfinite(r["t_7d"])
        ]
        true_row = sub[sub["placebo_date"] == TRUE_BAN_DATE].iloc[0]
        p_b, n_b = _ri_two_sided(float(true_row["beta_7d"]), placebo_betas)
        p_t, n_t = _ri_two_sided(float(true_row["t_7d"]), placebo_ts)

        space_pool: List[float] = []
        for pdate in p_dates:
            space_pool.extend(_space_placebo_betas(panel, y_col, pdate, post_days))

        p_st, _ = _ri_two_sided(float(true_row["beta_7d"]), space_pool)

        for r in rows:
            if r["outcome_id"] != oid:
                continue
            if r["placebo_date"] == TRUE_BAN_DATE:
                r["perm_p_beta_7d"] = p_b
                r["perm_p_t_7d"] = p_t
                r["perm_p_combined_7d"] = p_st
                r["n_placebo_dates"] = len(p_dates)
                r["n_space_placebos"] = len(space_pool)

    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: CLI entry for placebo-in-time table."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    outcome_ids = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    df = run_placebo_in_time(
        config,
        outcome_ids,
        post_days=int(args.post_days),
        also_3d=bool(args.also_3d),
    )
    out_dir = did_summary_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "placebo_in_time.csv"
    df.to_csv(out_path, index=False)
    print(f"[placebo_in_time] wrote {out_path} ({len(df)} rows)", flush=True)
    for oid in outcome_ids:
        tr = df[(df["outcome_id"] == oid) & (df["placebo_date"] == TRUE_BAN_DATE)]
        if not tr.empty:
            r = tr.iloc[0]
            print(
                f"  {oid}: perm_p_beta_7d={r.get('perm_p_beta_7d', float('nan')):.4g} "
                f"perm_p_t_7d={r.get('perm_p_t_7d', float('nan')):.4g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
