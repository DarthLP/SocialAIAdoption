"""
Robustness re-estimates: placebo dates, window trims, placebo-in-space.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from src.did.estimate import run_strategy_twfe
from src.did.inference import placebo_in_space_p
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


def _placebo_dates_from_launch(launch: str) -> List[str]:
    """Function summary: calendar placebo dates at launch-45, -30, -15 days and 2023-03-16."""
    launch_dt = datetime.strptime(launch[:10], "%Y-%m-%d")
    offsets = (-45, -30, -15)
    dates = [(launch_dt + timedelta(days=o)).strftime("%Y-%m-%d") for o in offsets]
    dates.append("2023-03-16")
    return dates


def run_robustness_grid(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    launch: str,
    placebo_date: str = "2023-03-16",
) -> List[Dict[str, Any]]:
    """Function summary: placebo dates, window trims, and placebo-in-space row.

    Parameters:
    - panel: analysis panel.
    - strategy: typically cross_country_all.
    - y_col: outcome column.
    - launch: ban launch date (UTC).
    - placebo_date: legacy single placebo (also included in date grid).

    Returns:
    - List of dicts for robustness_<outcome>.csv.
    """
    rows: List[Dict[str, Any]] = []
    base = run_strategy_twfe(panel, strategy, y_col)
    rows.append(
        {
            "check": "baseline",
            "beta": base["beta"],
            "se": base["se"],
            "pvalue": base["pvalue"],
            "p_placebo_space": float("nan"),
        }
    )

    pis = placebo_in_space_p(panel, strategy, y_col)
    rows.append(
        {
            "check": "placebo_in_space",
            "beta": pis.beta_italy,
            "se": float("nan"),
            "pvalue": float("nan"),
            "p_placebo_space": pis.p,
            "placebo_p_floor": pis.p_floor,
            "placebo_betas": ",".join(f"{b:.6g}" for b in pis.placebo_betas),
        }
    )

    seen = set()
    for pdate in _placebo_dates_from_launch(launch):
        if pdate in seen:
            continue
        seen.add(pdate)
        pl = placebo_panel(panel, pdate)
        pl_strat = StrategySpec(strategy.strategy_id + "_placebo", description="placebo t*")
        r = run_strategy_twfe(pl, pl_strat, y_col)
        rows.append(
            {
                "check": f"placebo_{pdate}",
                "beta": r["beta"],
                "se": r["se"],
                "pvalue": r["pvalue"],
                "p_placebo_space": float("nan"),
            }
        )

    for w in (14, 30):
        trimmed = window_trim(panel, w)
        r = run_strategy_twfe(trimmed, strategy, y_col, window_days=w)
        rows.append(
            {
                "check": f"window_pm{w}",
                "beta": r["beta"],
                "se": r["se"],
                "pvalue": r["pvalue"],
                "p_placebo_space": float("nan"),
            }
        )
    return rows
