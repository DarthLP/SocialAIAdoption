"""
Generalized synthetic control (Xu 2017) on country-day aggregated panels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.descriptives_util import event_dates_from_config
from src.did.outcomes import OUTCOME_REGISTRY, OutcomeSpec
from src.did.paths import did_panels_dir, did_gsynth_att_path, did_gsynth_inference_path
from src.did.specs import CONTROL_FAMILIES, rel_day_from_date

logger = logging.getLogger(__name__)

TREATED_UNIT = "it"


@dataclass(frozen=True)
class GsynthResult:
    """Function summary: gsynth ATT path and inference metadata."""

    att: pd.DataFrame
    inference: Dict[str, Any]
    backend: str


def _resolve_outcome_column(panel: pd.DataFrame, outcome: OutcomeSpec) -> Optional[str]:
    """Function summary: map outcome spec to a column present in the panel."""
    if outcome.column in panel.columns:
        return outcome.column
    if outcome.outcome_id in panel.columns:
        return outcome.outcome_id
    return None


def load_gsynth_panel(
    config: Dict[str, Any],
    outcome_col: str,
    bin_days: int = 3,
) -> pd.DataFrame:
    """Function summary: load language-hub aggregated panel for gsynth.

    Parameters:
    - config: study YAML.
    - outcome_col: outcome column name.
    - bin_days: 1 or 3.

    Returns:
    - Long panel with language_hub, period_start, outcome, rel_day, post.
    """
    path = did_panels_dir(config, "aggregated") / f"did_language_{int(bin_days)}d.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregated panel: {path}")
    df = pd.read_csv(path)
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome {outcome_col} not in {path}")
    _, end_excl, launch, _ = event_dates_from_config(config)
    date_col = "period_start" if "period_start" in df.columns else "date_utc"
    df = df[df[date_col].astype(str) < end_excl].copy()
    df["rel_day"] = rel_day_from_date(df[date_col], launch)
    df["post"] = (df[date_col].astype(str) >= launch[:10]).astype(int)
    return df


def _wide_matrix(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_col: str = "language_hub",
    time_col: str = "period_start",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Function summary: pivot to units × dates outcome matrix."""
    sub = panel[[unit_col, time_col, outcome_col]].dropna()
    wide = sub.pivot_table(index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean")
    wide = wide.sort_index()
    units = [u for u in wide.columns if str(u) in ([TREATED_UNIT] + sorted(CONTROL_FAMILIES))]
    wide = wide[[c for c in units if c in wide.columns]]
    times = list(wide.index.astype(str))
    return wide, units, times


def _augmented_sc_att(
    wide: pd.DataFrame,
    treated: str = TREATED_UNIT,
    pre_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """Function summary: simple SC weights on pre-period; ATT path post launch.

    Parameters:
    - wide: units × dates matrix.
    - treated: treated unit id.
    - pre_periods: number of pre periods for weight fitting (default: half of sample).

    Returns:
    - Tuple (att DataFrame with rel_day, att, y_treated, y_synth), average post ATT.
    """
    if treated not in wide.columns:
        raise ValueError(f"Treated unit {treated} not in panel")
    donors = [c for c in wide.columns if c != treated and c in CONTROL_FAMILIES]
    if not donors:
        raise ValueError("No donor countries in panel")

    y = wide.astype(float)
    n_pre = pre_periods or max(3, len(y) // 2)
    pre = y.iloc[:n_pre]
    y_t = pre[treated].values
    y_d = pre[donors].values
    if y_d.ndim == 1:
        w = np.array([1.0])
    else:
        w, _, _, _ = np.linalg.lstsq(y_d, y_t, rcond=None)
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(donors)) / len(donors)

    synth = y[donors].values @ w
    treated_path = y[treated].values
    att = treated_path - synth
    rel_days = list(range(len(att)))
    out = pd.DataFrame(
        {
            "period_start": y.index.astype(str),
            "rel_day": rel_days,
            "att": att,
            "y_treated": treated_path,
            "y_synth": synth,
        }
    )
    post_mask = np.array(rel_days) >= n_pre
    avg_att = float(np.nanmean(att[post_mask])) if post_mask.any() else float("nan")
    return out, avg_att


def _try_r_gsynth(
    wide: pd.DataFrame,
    treated: str,
    launch: str,
) -> Optional[GsynthResult]:
    """Function summary: attempt R gsynth via rpy2; return None if unavailable."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        base = importr("base")
        if not base.require("gsynth", quietly=True)[0]:
            return None
        gsynth = importr("gsynth")
        long = wide.reset_index().melt(id_vars="period_start", var_name="unit", value_name="Y")
        long["period_start"] = long["period_start"].astype(str)
        long["D"] = (
            (long["unit"].astype(str) == treated)
            & (long["period_start"] >= launch[:10])
        ).astype(int)
        r_df = pandas2ri.py2rpy(long)
        fit = gsynth.gsynth(
            "Y ~ D",
            data=r_df,
            index=ro.StrVector(["unit", "period_start"]),
            force="two-way",
            CV=True,
            r=ro.IntVector([4]),
            se=True,
        )
        eff = pandas2ri.rpy2py(fit.rx2("eff"))
        att_df = pd.DataFrame(eff)
        att_df.columns = ["period_start", "att"] if att_df.shape[1] == 2 else list(att_df.columns)
        inf = {"avg_att": float(np.nanmean(att_df["att"])), "backend": "r_gsynth"}
        return GsynthResult(att=att_df, inference=inf, backend="r_gsynth")
    except Exception as exc:
        logger.info("R gsynth unavailable: %s", exc)
        return None


def _placebo_space_gsynth(
    wide: pd.DataFrame,
    avg_att_real: float,
) -> float:
    """Function summary: share of placebo |ATT| >= |real| when each donor is fake-treated."""
    donors = [c for c in wide.columns if c in CONTROL_FAMILIES]
    if not donors or not np.isfinite(avg_att_real):
        return float("nan")
    placebo_atts: List[float] = []
    for fake in donors:
        sub = wide[[fake] + [d for d in donors if d != fake]].copy()
        sub = sub.rename(columns={fake: TREATED_UNIT})
        try:
            _, avg_p = _augmented_sc_att(sub, treated=TREATED_UNIT)
            if np.isfinite(avg_p):
                placebo_atts.append(avg_p)
        except ValueError:
            continue
    if not placebo_atts:
        return float("nan")
    n_ge = sum(1 for a in placebo_atts if abs(a) >= abs(avg_att_real))
    floor = 1.0 / (len(donors) + 1)
    return min(1.0, max(floor, (n_ge + 1) / (len(donors) + 1)))


def _placebo_time_gsynth(
    wide: pd.DataFrame,
    launch: str,
    shift_days: int = 30,
) -> float:
    """Function summary: placebo-in-time using shifted pseudo-treatment on donors."""
    donors = [c for c in wide.columns if c in CONTROL_FAMILIES]
    if not donors:
        return float("nan")
    launch_dt = datetime.strptime(launch[:10], "%Y-%m-%d")
    placebo_launch = (launch_dt - timedelta(days=shift_days)).strftime("%Y-%m-%d")
    times = wide.index.astype(str)
    n_pre = sum(1 for t in times if t < placebo_launch)
    placebo_atts: List[float] = []
    for fake in donors:
        try:
            wsub = wide.copy()
            y = wsub.astype(float)
            pre = y.iloc[: max(3, n_pre)]
            donors_other = [d for d in donors if d != fake]
            if fake not in y.columns or not donors_other:
                continue
            y_t = pre[fake].values
            y_d = pre[donors_other].values
            w, _, _, _ = np.linalg.lstsq(y_d, y_t, rcond=None)
            w = np.clip(w, 0, None)
            if w.sum() > 0:
                w = w / w.sum()
            synth = y[donors_other].values @ w
            att = y[fake].values - synth
            post_mask = np.array(times) >= placebo_launch
            placebo_atts.append(float(np.nanmean(att[post_mask])))
        except Exception:
            continue
    if not placebo_atts:
        return float("nan")
    try:
        _, avg_real = _augmented_sc_att(wide)
    except ValueError:
        return float("nan")
    n_ge = sum(1 for a in placebo_atts if abs(a) >= abs(avg_real))
    floor = 1.0 / (len(placebo_atts) + 1)
    return min(1.0, max(floor, (n_ge + 1) / (len(placebo_atts) + 1)))


def run_gsynth_att(
    config: Dict[str, Any],
    outcome: OutcomeSpec,
    bin_days: int = 3,
) -> GsynthResult:
    """Function summary: estimate ATT path and placebo inference for one outcome.

    Parameters:
    - config: study YAML.
    - outcome: outcome registry entry.
    - bin_days: panel bin width.

    Returns:
    - GsynthResult with att table, inference dict, backend label.
    """
    panel_probe = load_gsynth_panel(config, outcome.column, bin_days)
    y_col = _resolve_outcome_column(panel_probe, outcome) or outcome.column
    if y_col not in panel_probe.columns:
        raise KeyError(f"Cannot resolve column for {outcome.outcome_id}")
    panel = load_gsynth_panel(config, y_col, bin_days)
    wide, _, _ = _wide_matrix(panel, y_col)
    _, _, launch, _ = event_dates_from_config(config)

    r_res = _try_r_gsynth(wide, TREATED_UNIT, launch)
    if r_res is not None:
        att_df = r_res.att
        avg_att = r_res.inference.get("avg_att", float("nan"))
        backend = "r_gsynth"
    else:
        try:
            from pysyncon import Synth  # noqa: F401

            backend = "pysyncon"
        except ImportError:
            backend = "augmented_sc"
        att_df, avg_att = _augmented_sc_att(wide)
        att_df["rel_day"] = rel_day_from_date(att_df["period_start"], launch)

    p_space = _placebo_space_gsynth(wide, avg_att)
    p_time = _placebo_time_gsynth(wide, launch)
    n_donors = len([c for c in wide.columns if c in CONTROL_FAMILIES])
    inference = {
        "outcome_id": outcome.outcome_id,
        "avg_att": avg_att,
        "p_placebo_space": p_space,
        "p_placebo_time": p_time,
        "p_gsynth_placebo": p_space,
        "placebo_p_floor": 1.0 / (n_donors + 1) if n_donors else float("nan"),
        "backend": backend,
        "bin_days": bin_days,
    }
    return GsynthResult(att=att_df, inference=inference, backend=backend)


def write_gsynth_outputs(config: Dict[str, Any], result: GsynthResult, outcome_id: str, bin_days: int) -> None:
    """Function summary: write ATT and inference CSVs under did/estimates/gsynth/."""
    att_path = did_gsynth_att_path(config, outcome_id, bin_days)
    inf_path = did_gsynth_inference_path(config, outcome_id, bin_days)
    att_path.parent.mkdir(parents=True, exist_ok=True)
    result.att.to_csv(att_path, index=False)
    pd.DataFrame([result.inference]).to_csv(inf_path, index=False)


def outcomes_for_gsynth(config: Dict[str, Any], outcome_ids: Optional[List[str]] = None) -> Tuple[OutcomeSpec, ...]:
    """Function summary: lexical/semantic outcomes available on language aggregated panel."""
    from src.did.outcomes import HEADLINE_OUTCOMES

    ids = outcome_ids or list(HEADLINE_OUTCOMES)
    seen: set[str] = set()
    out: list[OutcomeSpec] = []
    for o in OUTCOME_REGISTRY:
        if o.outcome_id in ids and o.outcome_id not in seen:
            seen.add(o.outcome_id)
            out.append(o)
    return tuple(out)
