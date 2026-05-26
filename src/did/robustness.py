"""
Robustness re-estimates: placebo date, window trims.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.did.estimate import run_strategy_twfe
from src.did.specs import StrategySpec, rel_day_from_date


def placebo_panel(panel: pd.DataFrame, placebo_date: str = "2023-03-16") -> pd.DataFrame:
    """Function summary: redefine post/rel_day for placebo ban date."""
    out = panel.copy()
    out["rel_day"] = rel_day_from_date(out["date_utc"], "", placebo=True, placebo_date=placebo_date)
    out["post"] = (out["rel_day"] >= 0).astype(int)
    return out


def window_trim(panel: pd.DataFrame, days: int) -> pd.DataFrame:
    """Function summary: keep rel_day within symmetric window."""
    return panel[panel["rel_day"].between(-days, days)].copy()


def run_robustness_grid(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    launch: str,
    placebo_date: str = "2023-03-16",
) -> List[Dict[str, Any]]:
    """Function summary: placebo and ±window sensitivity rows.

    Returns:
    - List of dicts for robustness_<outcome>.csv.
    """
    rows: List[Dict[str, Any]] = []
    base = run_strategy_twfe(panel, strategy, y_col)
    rows.append({"check": "baseline", "beta": base["beta"], "se": base["se"], "pvalue": base["pvalue"]})
    pl = placebo_panel(panel, placebo_date)
    pl_strat = StrategySpec(strategy.strategy_id + "_placebo", description="placebo t*")
    r = run_strategy_twfe(pl, pl_strat, y_col)
    rows.append({"check": f"placebo_{placebo_date}", "beta": r["beta"], "se": r["se"], "pvalue": r["pvalue"]})
    for w in (14, 30):
        trimmed = window_trim(panel, w)
        r = run_strategy_twfe(trimmed, strategy, y_col, window_days=w)
        rows.append({"check": f"window_pm{w}", "beta": r["beta"], "se": r["se"], "pvalue": r["pvalue"]})
    return rows
