"""
Author×week panel regressions for Italy within-person analysis (entity FE).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from src.user_week.panel_prep import EVENT_STUDY_REFERENCE_WEEK

MIN_OBS = 30
MIN_CLUSTERS = 12


def _rel_week_column(rel_week: int) -> str:
    """Function summary: safe column name for rel_week dummy (formula-safe, no minus signs)."""
    w = int(rel_week)
    if w < 0:
        return f"rw_m{abs(w)}"
    if w > 0:
        return f"rw_p{w}"
    return "rw_0"


def _empty_result(n_obs: int, n_clusters: int, note: str) -> Dict[str, Any]:
    """Function summary: standard failure dict for estimation helpers."""
    return {
        "beta": float("nan"),
        "se": float("nan"),
        "n_obs": int(n_obs),
        "n_clusters": int(n_clusters),
        "estimation_note": note,
    }


def _pack_result(
    beta: float,
    se: float,
    n_obs: int,
    n_clusters: int,
    estimation_note: str = "ok",
) -> Dict[str, Any]:
    """Function summary: standard success dict for estimation helpers."""
    return {
        "beta": float(beta),
        "se": float(se),
        "n_obs": int(n_obs),
        "n_clusters": int(n_clusters),
        "estimation_note": estimation_note,
    }


def estimate_user_week_entity_only(
    panel: pd.DataFrame,
    y_col: str,
    author_col: str = "author",
    time_col: str = "time_id",
) -> Dict[str, Any]:
    """Function summary: y ~ post + author FE; cluster SE at author (Italy national shock).

    Parameters:
    - panel: regression sample with post, author, time_id, and outcome column.
    - y_col: dependent variable column on panel.
    - author_col: entity identifier.
    - time_col: time index for PanelOLS.

    Returns:
    - Dict with beta (post), se, n_obs, n_clusters, estimation_note.
    """
    need = [author_col, time_col, y_col, "post"]
    work = panel[need].copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    work["time"] = pd.to_datetime(work[time_col].astype(str))
    work = work.set_index([author_col, "time"])
    n_clusters = work.index.get_level_values(0).nunique()
    if len(work) < MIN_OBS or n_clusters < MIN_CLUSTERS:
        return _empty_result(len(work), n_clusters, "insufficient_obs_or_clusters")
    try:
        mod = PanelOLS.from_formula(
            "y ~ post + EntityEffects",
            data=work,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except (ValueError, Exception):
        return _empty_result(len(work), n_clusters, "estimation_error")
    beta = float(res.params.get("post", np.nan))
    se = float(res.std_errors.get("post", np.nan))
    note = "ok_entity_fe_only" if np.isfinite(beta) else "post_absorbed"
    return _pack_result(beta, se, int(res.nobs), n_clusters, estimation_note=note)


def estimate_user_week_event_study(
    panel: pd.DataFrame,
    y_col: str,
    author_col: str = "author",
    time_col: str = "time_id",
    reference_week: int = EVENT_STUDY_REFERENCE_WEEK,
) -> List[Dict[str, Any]]:
    """Function summary: y ~ rel_week dummies + author FE; omit reference_week.

    Parameters:
    - panel: regression sample with rel_week, author, time_id, outcome.
    - y_col: dependent variable column.
    - author_col: entity id.
    - time_col: calendar week id.
    - reference_week: omitted rel_week category.

    Returns:
    - List of dicts per rel_week with beta, se, n_obs, n_clusters, rel_week.
    """
    need = [author_col, time_col, y_col, "rel_week"]
    work = panel[need].copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    work["rel_week"] = work["rel_week"].astype(int)
    rel_weeks = sorted(int(w) for w in work["rel_week"].unique() if int(w) != int(reference_week))
    if not rel_weeks:
        return []
    for w in rel_weeks:
        work[_rel_week_column(w)] = (work["rel_week"] == w).astype(float)
    rw_cols = [_rel_week_column(w) for w in rel_weeks]
    work["time"] = pd.to_datetime(work[time_col].astype(str))
    work = work.set_index([author_col, "time"])
    n_clusters = work.index.get_level_values(0).nunique()
    if len(work) < MIN_OBS or n_clusters < MIN_CLUSTERS:
        return []
    formula = "y ~ " + " + ".join(rw_cols) + " + EntityEffects"
    try:
        mod = PanelOLS.from_formula(formula, data=work, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except (ValueError, Exception):
        return []
    rows: List[Dict[str, Any]] = []
    for w in rel_weeks:
        name = _rel_week_column(w)
        if name not in res.params.index:
            continue
        beta = float(res.params[name])
        se = float(res.std_errors.get(name, np.nan))
        rows.append(
            {
                "rel_week": w,
                "beta": beta,
                "se": se,
                "n_obs": int(res.nobs),
                "n_clusters": n_clusters,
                "estimation_note": "ok" if np.isfinite(beta) else "absorbed",
            }
        )
    rows.sort(key=lambda r: r["rel_week"])
    return rows


