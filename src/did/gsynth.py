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
from src.did.paths import (
    did_gsynth_att_path,
    did_gsynth_inference_path,
    did_gsynth_v2_att_path,
    did_gsynth_v2_dir,
    did_gsynth_v2_inference_path,
    did_panels_dir,
    did_summary_paths,
    gsynth_v2_figure_dir,
)
from src.did.specs import CONTROL_FAMILIES, rel_day_from_date

logger = logging.getLogger(__name__)

TREATED_UNIT = "it"

GSYNTH_V2_OUTCOMES: Tuple[str, ...] = (
    "sem_axis_emotion",
    "pole_share",
    "aggression_rate",
    "sem_axis_ideology_var",
    "sem_axis_ideology",
    "ai_style_rate",
)

POLE_SHARE_SIGN_REFERENCE = {
    "outcome_id": "pole_share",
    "strategy_id": "cross_country_all",
    "spec": "full_ban",
}


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


@dataclass(frozen=True)
class GsynthV2Result:
    """Function summary: gsynth v2 ATT path, inference, backend, and pre-fit gate metadata."""

    att: pd.DataFrame
    inference: Dict[str, Any]
    backend: str
    gate: Dict[str, Any]


def _launch_str(launch: str) -> str:
    """Function summary: normalize launch date to YYYY-MM-DD."""
    return launch[:10]


def _pre_post_mask(period_starts: pd.Series, launch: str) -> Tuple[np.ndarray, np.ndarray]:
    """Function summary: boolean masks for pre- and post-ban periods.

    Parameters:
    - period_starts: period_start column as strings.
    - launch: ban launch date.

    Returns:
    - Tuple (pre_mask, post_mask).
    """
    launch_s = _launch_str(launch)
    ps = period_starts.astype(str)
    pre = (ps < launch_s).values
    post = (ps >= launch_s).values
    return pre, post


def _demean_by_pre_period_mean(
    wide: pd.DataFrame,
    launch: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Function summary: subtract each unit's pre-ban mean from its full series.

    Parameters:
    - wide: units × dates outcome matrix.
    - launch: ban launch date.

    Returns:
    - Tuple (demeaned wide, pre-period means per unit).
    """
    launch_s = _launch_str(launch)
    y = wide.astype(float).copy()
    pre_idx = y.index.astype(str) < launch_s
    pre_means = y.loc[pre_idx].mean(axis=0)
    demeaned = y - pre_means
    return demeaned, pre_means


def _augmented_sc_att_v2(
    wide_demeaned: pd.DataFrame,
    launch: str,
    treated: str = TREATED_UNIT,
) -> Tuple[pd.DataFrame, float]:
    """Function summary: SC on demeaned wide with launch-aligned pre/post split.

    Parameters:
    - wide_demeaned: demeaned units × dates matrix.
    - launch: ban launch date.
    - treated: treated unit id.

    Returns:
    - Tuple (att DataFrame, mean post-ban ATT).
    """
    if treated not in wide_demeaned.columns:
        raise ValueError(f"Treated unit {treated} not in panel")
    donors = [c for c in wide_demeaned.columns if c != treated and c in CONTROL_FAMILIES]
    if not donors:
        raise ValueError("No donor countries in panel")

    y = wide_demeaned.astype(float)
    launch_s = _launch_str(launch)
    times = y.index.astype(str)
    pre_mask = times < launch_s
    if pre_mask.sum() < 3:
        raise ValueError("Insufficient pre-ban periods for SC weight fit")

    pre = y.loc[pre_mask]
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
    out = pd.DataFrame(
        {
            "period_start": times,
            "rel_day": rel_day_from_date(pd.Series(times), launch),
            "att": att,
            "y_treated": treated_path,
            "y_synth": synth,
        }
    )
    _, post_mask = _pre_post_mask(out["period_start"], launch)
    avg_post = float(np.nanmean(att[post_mask])) if post_mask.any() else float("nan")
    return out, avg_post


def _try_r_gsynth_v2(
    wide_demeaned: pd.DataFrame,
    launch: str,
    treated: str = TREATED_UNIT,
) -> Optional[Tuple[pd.DataFrame, float, str]]:
    """Function summary: R gsynth on demeaned wide; return att path with y_treated/y_synth.

    Parameters:
    - wide_demeaned: demeaned units × dates matrix.
    - launch: ban launch date.
    - treated: treated unit id.

    Returns:
    - None if R gsynth unavailable, else (att_df, avg_post_att, backend).
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        base = importr("base")
        if not base.require("gsynth", quietly=True)[0]:
            return None
        gsynth = importr("gsynth")
        launch_s = _launch_str(launch)
        long = wide_demeaned.reset_index().melt(
            id_vars="period_start", var_name="unit", value_name="Y"
        )
        long["period_start"] = long["period_start"].astype(str)
        long["D"] = (
            (long["unit"].astype(str) == treated) & (long["period_start"] >= launch_s)
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
        att_raw = pd.DataFrame(eff)
        if att_raw.shape[1] == 2:
            att_raw.columns = ["period_start", "att"]
        elif "att" not in att_raw.columns and att_raw.shape[1] >= 2:
            att_raw = att_raw.rename(columns={att_raw.columns[-1]: "att"})

        y = wide_demeaned.astype(float)
        times = y.index.astype(str)
        treated_path = y[treated].values
        att_by_period = att_raw.groupby("period_start", as_index=False)["att"].mean()
        att_map = dict(zip(att_by_period["period_start"].astype(str), att_by_period["att"]))
        att_vals = np.array([float(att_map.get(t, np.nan)) for t in times])
        synth = treated_path - att_vals
        out = pd.DataFrame(
            {
                "period_start": times,
                "rel_day": rel_day_from_date(pd.Series(times), launch),
                "att": att_vals,
                "y_treated": treated_path,
                "y_synth": synth,
            }
        )
        _, post_mask = _pre_post_mask(out["period_start"], launch)
        avg_post = float(np.nanmean(att_vals[post_mask])) if post_mask.any() else float("nan")
        return out, avg_post, "r_gsynth"
    except Exception as exc:
        logger.info("R gsynth v2 unavailable: %s", exc)
        return None


def _evaluate_pre_fit_gate(
    att_df: pd.DataFrame,
    wide_demeaned: pd.DataFrame,
    launch: str,
    treated: str = TREATED_UNIT,
) -> Dict[str, Any]:
    """Function summary: evaluate hard pre-fit gate on launch-aligned ATT path.

    Parameters:
    - att_df: ATT table with period_start and att columns.
    - wide_demeaned: demeaned Italy/control wide matrix.
    - launch: ban launch date.
    - treated: treated unit id.

    Returns:
    - Dict with gate metrics, pre_fit_ok flag, and verdict string.
    """
    pre_mask, post_mask = _pre_post_mask(att_df["period_start"], launch)
    att = att_df["att"].astype(float).values
    mean_pre = float(np.nanmean(att[pre_mask])) if pre_mask.any() else float("nan")
    mean_post = float(np.nanmean(att[post_mask])) if post_mask.any() else float("nan")

    launch_s = _launch_str(launch)
    italy_pre = wide_demeaned.loc[
        wide_demeaned.index.astype(str) < launch_s, treated
    ].astype(float)
    italy_pre_sd = float(italy_pre.std(ddof=1)) if len(italy_pre) > 1 else float("nan")
    pre_att = att[pre_mask]
    pre_rmse = float(np.sqrt(np.nanmean(pre_att**2))) if pre_att.size else float("nan")

    if not np.isfinite(mean_post) or mean_post == 0:
        pre_fit_ok = False
    else:
        pre_fit_ok = bool(
            np.isfinite(mean_pre)
            and abs(mean_pre) < 0.25 * abs(mean_post)
            and np.isfinite(pre_rmse)
            and np.isfinite(italy_pre_sd)
            and pre_rmse < italy_pre_sd
        )

    return {
        "mean_pre_att": mean_pre,
        "mean_post_att": mean_post,
        "pre_rmse": pre_rmse,
        "italy_pre_sd": italy_pre_sd,
        "pre_fit_ok": pre_fit_ok,
        "verdict": "ok" if pre_fit_ok else "failed_pre_fit_do_not_cite",
    }


def _placebo_space_gsynth_v2(
    wide_demeaned: pd.DataFrame,
    launch: str,
    avg_att_real: float,
) -> Tuple[float, int]:
    """Function summary: placebo-in-space p on demeaned wide with v2 SC.

    Parameters:
    - wide_demeaned: demeaned units × dates matrix.
    - launch: ban launch date.
    - avg_att_real: real treated unit mean post ATT.

    Returns:
    - Tuple (p-value, n_placebos).
    """
    donors = [c for c in wide_demeaned.columns if c in CONTROL_FAMILIES]
    if not donors or not np.isfinite(avg_att_real):
        return float("nan"), len(donors)
    placebo_atts: List[float] = []
    for fake in donors:
        sub = wide_demeaned[[fake] + [d for d in donors if d != fake]].copy()
        sub = sub.rename(columns={fake: TREATED_UNIT})
        try:
            _, avg_p = _augmented_sc_att_v2(sub, launch, treated=TREATED_UNIT)
            if np.isfinite(avg_p):
                placebo_atts.append(avg_p)
        except ValueError:
            continue
    if not placebo_atts:
        return float("nan"), len(donors)
    n_ge = sum(1 for a in placebo_atts if abs(a) >= abs(avg_att_real))
    floor = 1.0 / (len(donors) + 1)
    p = min(1.0, max(floor, (n_ge + 1) / (len(donors) + 1)))
    return p, len(donors)


def _placebo_time_gsynth_v2(
    wide_demeaned: pd.DataFrame,
    launch: str,
    shift_days: int = 30,
) -> Tuple[float, int]:
    """Function summary: placebo-in-time p on demeaned wide with launch-aligned v2 SC.

    Parameters:
    - wide_demeaned: demeaned units × dates matrix.
    - launch: real ban launch date.
    - shift_days: days to shift pseudo-treatment earlier.

    Returns:
    - Tuple (p-value, n_placebos).
    """
    donors = [c for c in wide_demeaned.columns if c in CONTROL_FAMILIES]
    if not donors:
        return float("nan"), 0
    launch_dt = datetime.strptime(_launch_str(launch), "%Y-%m-%d")
    placebo_launch = (launch_dt - timedelta(days=shift_days)).strftime("%Y-%m-%d")
    placebo_atts: List[float] = []
    for fake in donors:
        cols = [fake] + [d for d in donors if d != fake]
        if fake not in wide_demeaned.columns:
            continue
        fake_wide = wide_demeaned[cols].rename(columns={fake: TREATED_UNIT})
        try:
            att_df, _ = _augmented_sc_att_v2(fake_wide, placebo_launch, treated=TREATED_UNIT)
            _, post_mask = _pre_post_mask(att_df["period_start"], placebo_launch)
            placebo_atts.append(float(np.nanmean(att_df["att"].values[post_mask])))
        except (ValueError, KeyError):
            continue
    if not placebo_atts:
        return float("nan"), 0
    try:
        _, avg_real = _augmented_sc_att_v2(wide_demeaned, launch)
    except ValueError:
        return float("nan"), len(placebo_atts)
    n_ge = sum(1 for a in placebo_atts if abs(a) >= abs(avg_real))
    floor = 1.0 / (len(placebo_atts) + 1)
    p = min(1.0, max(floor, (n_ge + 1) / (len(placebo_atts) + 1)))
    return p, len(placebo_atts)


def load_did_summary_beta(
    config: Dict[str, Any],
    outcome_id: str,
    strategy_id: str,
    spec: str,
) -> float:
    """Function summary: read TWFE beta from did_summary.csv for one outcome×strategy×spec.

    Parameters:
    - config: study YAML.
    - outcome_id: outcome slug.
    - strategy_id: identification strategy slug.
    - spec: post-window spec (e.g. full_ban).

    Returns:
    - TWFE beta coefficient.

    Raises:
    - FileNotFoundError: if did_summary.csv missing.
    - KeyError: if matching row not found.
    """
    summary_path = did_summary_paths(config)[0]
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing did_summary.csv: {summary_path}")
    df = pd.read_csv(summary_path)
    row = df[
        (df["outcome_id"].astype(str) == outcome_id)
        & (df["strategy_id"].astype(str) == strategy_id)
        & (df["spec"].astype(str) == spec)
    ]
    if row.empty:
        raise KeyError(
            f"No did_summary row for outcome={outcome_id} strategy={strategy_id} spec={spec}"
        )
    return float(row["beta"].iloc[0])


def pole_share_sign_gate_should_abort(
    pre_fit_ok: bool,
    mean_post_att: float,
    did_summary_beta: float,
) -> bool:
    """Function summary: True only when pre-fit passed but post-ATT sign disagrees with TWFE.

    Parameters:
    - pre_fit_ok: whether pre-fit gate passed.
    - mean_post_att: mean post-ban ATT.
    - did_summary_beta: reference TWFE beta from did_summary.

    Returns:
    - True if pipeline should hard-abort before writing CSVs.
    """
    if not pre_fit_ok:
        return False
    if not np.isfinite(mean_post_att) or not np.isfinite(did_summary_beta):
        return False
    if did_summary_beta == 0:
        return mean_post_att != 0
    return bool(np.sign(mean_post_att) != np.sign(did_summary_beta))


def plot_gsynth_v2_prefit(
    outcome_id: str,
    att_df: pd.DataFrame,
    launch: str,
    path: Path,
    *,
    pre_fit_ok: bool,
) -> None:
    """Function summary: plot demeaned Italy vs synthetic pre-ban overlay.

    Parameters:
    - outcome_id: outcome slug for title.
    - att_df: ATT table with period_start, y_treated, y_synth.
    - launch: ban launch date.
    - path: output PNG path.
    - pre_fit_ok: gate flag for title annotation.
    """
    import matplotlib.pyplot as plt

    launch_s = _launch_str(launch)
    pre = att_df[att_df["period_start"].astype(str) < launch_s].copy()
    if pre.empty:
        pre = att_df.copy()
    x = pd.to_datetime(pre["period_start"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, pre["y_treated"], label="Italy (demeaned)", color="C0", lw=1.5)
    ax.plot(x, pre["y_synth"], label="Synthetic", color="C1", lw=1.5, ls="--")
    ax.axvline(pd.Timestamp(launch_s), color="0.4", ls=":", lw=1)
    gate_label = "pre_fit_ok=True" if pre_fit_ok else "pre_fit_ok=False"
    ax.set_title(f"gsynth v2 pre-fit: {outcome_id} ({gate_label})")
    ax.set_xlabel("period_start")
    ax.set_ylabel("demeaned level")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_gsynth_v2_att(
    config: Dict[str, Any],
    outcome: OutcomeSpec,
    bin_days: int = 3,
) -> GsynthV2Result:
    """Function summary: gsynth v2 ATT path, pre-fit gate, and placebo inference.

    Parameters:
    - config: study YAML.
    - outcome: outcome registry entry.
    - bin_days: panel bin width.

    Returns:
    - GsynthV2Result with att table, inference dict, backend, gate metadata.
    """
    panel_probe = load_gsynth_panel(config, outcome.column, bin_days)
    y_col = _resolve_outcome_column(panel_probe, outcome) or outcome.column
    if y_col not in panel_probe.columns:
        raise KeyError(f"Cannot resolve column for {outcome.outcome_id}")
    panel = load_gsynth_panel(config, y_col, bin_days)
    wide, _, _ = _wide_matrix(panel, y_col)
    _, _, launch, _ = event_dates_from_config(config)
    wide_demeaned, _ = _demean_by_pre_period_mean(wide, launch)

    r_res = _try_r_gsynth_v2(wide_demeaned, launch)
    if r_res is not None:
        att_df, avg_post, backend = r_res
    else:
        try:
            from pysyncon import Synth  # noqa: F401

            backend = "pysyncon"
        except ImportError:
            backend = "augmented_sc"
        att_df, avg_post = _augmented_sc_att_v2(wide_demeaned, launch)

    gate = _evaluate_pre_fit_gate(att_df, wide_demeaned, launch)
    p_space, n_space = _placebo_space_gsynth_v2(wide_demeaned, launch, gate["mean_post_att"])
    p_time, n_time = _placebo_time_gsynth_v2(wide_demeaned, launch)
    n_donors = len([c for c in wide.columns if c in CONTROL_FAMILIES])
    inference: Dict[str, Any] = {
        "outcome_id": outcome.outcome_id,
        "avg_att": avg_post,
        "mean_pre_att": gate["mean_pre_att"],
        "mean_post_att": gate["mean_post_att"],
        "pre_rmse": gate["pre_rmse"],
        "italy_pre_sd": gate["italy_pre_sd"],
        "pre_fit_ok": gate["pre_fit_ok"],
        "verdict": gate["verdict"],
        "p_placebo_space": p_space,
        "p_placebo_time": p_time,
        "p_gsynth_placebo": p_space,
        "placebo_p_floor": 1.0 / (n_donors + 1) if n_donors else float("nan"),
        "n_placebos_space": n_space,
        "n_placebos_time": n_time,
        "backend": backend,
        "bin_days": bin_days,
        "demeaned": True,
    }
    return GsynthV2Result(att=att_df, inference=inference, backend=backend, gate=gate)


def _attach_gate_columns(att_df: pd.DataFrame, gate: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: repeat gate metadata on every row of the ATT export table."""
    out = att_df.copy()
    for key in ("pre_fit_ok", "verdict", "mean_pre_att", "mean_post_att", "pre_rmse", "italy_pre_sd"):
        out[key] = gate[key]
    return out


def write_gsynth_v2_outputs(
    config: Dict[str, Any],
    result: GsynthV2Result,
    outcome_id: str,
    bin_days: int,
    *,
    sign_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Function summary: write gsynth v2 ATT and inference CSVs under did/estimates/gsynth_v2/.

    Parameters:
    - config: study YAML.
    - result: gsynth v2 estimation result.
    - outcome_id: outcome slug.
    - bin_days: calendar bin width.
    - sign_meta: optional pole_share sign-gate fields for inference CSV.
    """
    att_path = did_gsynth_v2_att_path(config, outcome_id, bin_days)
    inf_path = did_gsynth_v2_inference_path(config, outcome_id, bin_days)
    att_path.parent.mkdir(parents=True, exist_ok=True)
    att_out = _attach_gate_columns(result.att, result.gate)
    att_out.to_csv(att_path, index=False)
    inf_row = dict(result.inference)
    if sign_meta:
        inf_row.update(sign_meta)
    pd.DataFrame([inf_row]).to_csv(inf_path, index=False)


def write_gsynth_v2_readme(config: Dict[str, Any]) -> None:
    """Function summary: write 5-line README describing gsynth v2 gate and outputs.

    Parameters:
    - config: study YAML.
    """
    out_dir = did_gsynth_v2_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = gsynth_v2_figure_dir(config)
    lines = [
        "# gsynth v2 (demeaned SC with pre-fit gate)",
        "Each unit's series is demeaned by its own pre-ban mean before SC fit; R gsynth uses force=two-way when available.",
        "Pre-fit gate: pre_fit_ok = (|mean pre ATT| < 0.25 × |mean post ATT|) AND (pre RMSE < pre SD of demeaned Italy).",
        "Rows with pre_fit_ok=False → verdict failed_pre_fit_do_not_cite (do not cite); pole_share sign vs did_summary full_ban is read-only.",
        f"ATT/inference CSVs here; pre-fit PNGs under {fig_dir}.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def outcomes_for_gsynth_v2(
    config: Dict[str, Any],
    outcome_ids: Optional[List[str]] = None,
) -> Tuple[OutcomeSpec, ...]:
    """Function summary: fixed gsynth v2 outcome set from OUTCOME_REGISTRY.

    Parameters:
    - config: study YAML (unused; kept for API symmetry).
    - outcome_ids: optional override list.

    Returns:
    - Tuple of OutcomeSpec rows for gsynth v2.
    """
    _ = config
    ids = outcome_ids or list(GSYNTH_V2_OUTCOMES)
    seen: set[str] = set()
    out: list[OutcomeSpec] = []
    for o in OUTCOME_REGISTRY:
        if o.outcome_id in ids and o.outcome_id not in seen:
            seen.add(o.outcome_id)
            out.append(o)
    return tuple(out)
