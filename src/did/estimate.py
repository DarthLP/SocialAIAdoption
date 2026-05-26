"""
TWFE DiD, event-study, and triple-difference estimation via linearmodels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from linearmodels.panel import PanelOLS
except ImportError as exc:
    raise ImportError("linearmodels is required for DiD estimation") from exc

from scipy import stats

from src.did.specs import StrategySpec, filter_strategy_sample


def _prep_panel(df: pd.DataFrame, y_col: str, entity_col: str, time_col: str) -> pd.DataFrame:
    """Function summary: drop missing and set MultiIndex for PanelOLS."""
    work = df[[entity_col, time_col, y_col, "treat", "post"]].copy()
    work = work.rename(columns={y_col: "y", entity_col: "entity", time_col: "time"})
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["y"])
    work["treat_post"] = work["treat"].astype(float) * work["post"].astype(float)
    work["time"] = pd.to_datetime(work["time"].astype(str))
    work = work.set_index(["entity", "time"])
    return work


def estimate_twfe(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    cluster_col: str = "entity_id",
) -> Dict[str, Any]:
    """Function summary: two-way FE DiD on treat×post with clustered SEs.

    Returns:
    - Dict with beta, se, ci_low, ci_high, pvalue, n_obs, n_clusters, estimation_note.
    """
    work = _prep_panel(df, y_col, entity_col, time_col)
    if len(work) < 30 or work.index.get_level_values(0).nunique() < 3:
        return _empty_result(
            len(work),
            work.index.get_level_values(0).nunique() if len(work) else 0,
            "insufficient_obs_or_clusters",
        )
    try:
        mod = PanelOLS.from_formula(
            "y ~ treat_post + EntityEffects + TimeEffects",
            data=work,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except ValueError:
        return _empty_result(
            len(work),
            work.index.get_level_values(0).nunique(),
            "fully_absorbed",
        )
    except Exception:
        return _empty_result(len(work), work.index.get_level_values(0).nunique(), "estimation_error")
    beta = float(res.params.get("treat_post", np.nan))
    se = float(res.std_errors.get("treat_post", np.nan))
    if not np.isfinite(beta):
        return _pack_result(
            beta,
            se,
            int(res.nobs),
            int(df[entity_col].nunique()),
            estimation_note="treat_post_absorbed",
        )
    return _pack_result(beta, se, int(res.nobs), int(df[entity_col].nunique()), estimation_note="ok")


def estimate_event_study(
    df: pd.DataFrame,
    y_col: str,
    rel_col: str = "rel_day",
    ref_day: int = -1,
    window: int = 30,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: dynamic TWFE event study with lead/lag dummies × treat.

    Returns:
    - Summary dict (pretrend_F_p, etc.) and coefficient table.
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    work = work[work[rel_col].between(-window, window)]
    work["treat"] = work["treat"].astype(float)
    interact_cols: List[str] = []
    for k in sorted(work[rel_col].unique()):
        if int(k) == ref_day:
            continue
        col = f"es_{int(k)}"
        work[col] = ((work[rel_col] == k) * work["treat"]).astype(float)
        if work[col].sum() != 0:
            interact_cols.append(col)
    if not interact_cols:
        return _empty_result(len(work), work[entity_col].nunique(), "no_event_study_variation"), pd.DataFrame()
    formula_rhs = " + ".join(interact_cols) + " + EntityEffects + TimeEffects"
    work[time_col] = pd.to_datetime(work[time_col].astype(str))
    panel = work.set_index([entity_col, time_col])
    try:
        mod = PanelOLS.from_formula(f"y ~ {formula_rhs}", data=panel, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except Exception:
        return _empty_result(len(work), work[entity_col].nunique(), "event_study_failed"), pd.DataFrame()
    pretrend_params = [p for p in res.params.index if p.startswith("es_") and int(p.split("_")[1]) < ref_day]
    pretrend_f_p = float("nan")
    if len(pretrend_params) >= 1:
        try:
            restrictions = " + ".join(f"{p}" for p in pretrend_params) + " = 0"
            wt = res.wald_test(formula=restrictions)
            pretrend_f_p = float(wt.pval)
        except Exception:
            pretrend_f_p = float("nan")
    rows: List[Dict[str, Any]] = []
    for pname in res.params.index:
        if not pname.startswith("es_"):
            continue
        k = int(pname.split("_")[1])
        b = float(res.params[pname])
        se = float(res.std_errors[pname])
        rows.append(
            {
                "rel_day": k,
                "gamma": b,
                "se": se,
                "ci_low": b - 1.96 * se,
                "ci_high": b + 1.96 * se,
                "pvalue": float(2 * (1 - stats.norm.cdf(abs(b / se)))) if se > 0 else float("nan"),
            }
        )
    es_df = pd.DataFrame(rows).sort_values("rel_day") if rows else pd.DataFrame()
    summary = _pack_result(float("nan"), float("nan"), int(res.nobs), int(work[entity_col].nunique()), estimation_note="ok")
    summary["pretrend_F_p"] = pretrend_f_p
    return summary, es_df


def estimate_ddd(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> Dict[str, Any]:
    """Function summary: triple-diff IT×Post×Political with two-way FEs on slice panel."""
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    if work.empty:
        return _empty_result(0, 0, "empty_sample")
    if "political_universe" not in work.columns:
        work["political_universe"] = (
            work["universe_slice"].astype(str) == "in_political_tree"
        ).astype(int)
    pol_var = work.groupby(entity_col)["political_universe"].nunique()
    if pol_var.max() < 2:
        return _empty_result(
            len(work),
            int(work[entity_col].nunique()),
            "no_within_entity_political_variation",
        )
    work["post_pol"] = work["post"].astype(float) * work["political_universe"].astype(float)
    work[time_col] = pd.to_datetime(work[time_col].astype(str))
    panel = work.set_index([entity_col, time_col])
    it_const = work["IT"].nunique() <= 1 if "IT" in work.columns else True
    try:
        if it_const:
            # IT=1 throughout: it_post_pol ≡ post_pol; drop redundant IT interactions.
            mod = PanelOLS.from_formula(
                "y ~ post_pol + political_universe + EntityEffects + TimeEffects",
                data=panel,
                drop_absorbed=True,
            )
            coef_name = "post_pol"
        else:
            work["it_post_pol"] = (
                work["IT"].astype(float)
                * work["post"].astype(float)
                * work["political_universe"].astype(float)
            )
            work["it_post"] = work["IT"].astype(float) * work["post"].astype(float)
            work["it_pol"] = work["IT"].astype(float) * work["political_universe"].astype(float)
            panel = work.set_index([entity_col, time_col])
            mod = PanelOLS.from_formula(
                "y ~ it_post_pol + it_post + it_pol + post_pol + EntityEffects + TimeEffects",
                data=panel,
                drop_absorbed=True,
            )
            coef_name = "it_post_pol"
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        beta = float(res.params.get(coef_name, np.nan))
        se = float(res.std_errors.get(coef_name, np.nan))
        note = "ok" if np.isfinite(beta) else f"{coef_name}_absorbed"
        return _pack_result(beta, se, int(res.nobs), int(work[entity_col].nunique()), estimation_note=note)
    except ValueError:
        return _empty_result(len(work), int(work[entity_col].nunique()), "fully_absorbed")
    except Exception:
        return _empty_result(len(work), int(work[entity_col].nunique()), "estimation_error")


def estimate_twfe_entity_only(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> Dict[str, Any]:
    """Function summary: treat×post with entity FE only (author IT cohort; no time FE).

    Used when treat is constant so calendar time FE absorb the ban dummy.
    """
    work = df[[entity_col, time_col, y_col, "treat", "post"]].copy()
    work = work.rename(columns={y_col: "y", entity_col: "entity", time_col: "time"})
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["y"])
    work["treat_post"] = work["treat"].astype(float) * work["post"].astype(float)
    work["time"] = pd.to_datetime(work["time"].astype(str))
    work = work.set_index(["entity", "time"])
    if len(work) < 30 or work.index.get_level_values(0).nunique() < 3:
        return _empty_result(
            len(work),
            work.index.get_level_values(0).nunique() if len(work) else 0,
            "insufficient_obs_or_clusters",
        )
    try:
        mod = PanelOLS.from_formula(
            "y ~ treat_post + EntityEffects",
            data=work,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except (ValueError, Exception):
        return _empty_result(len(work), work.index.get_level_values(0).nunique(), "fully_absorbed")
    beta = float(res.params.get("treat_post", np.nan))
    se = float(res.std_errors.get("treat_post", np.nan))
    if not np.isfinite(beta):
        return _pack_result(
            beta,
            se,
            int(res.nobs),
            int(df[entity_col].nunique()),
            estimation_note="treat_post_absorbed",
        )
    return _pack_result(beta, se, int(res.nobs), int(df[entity_col].nunique()), estimation_note="ok_entity_fe_only")


def run_strategy_twfe(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    window_days: Optional[int] = None,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    cluster_col: str = "entity_id",
) -> Dict[str, Any]:
    """Function summary: TWFE for one strategy/outcome combination."""
    if strategy.strategy_id == "within_italy_ddd" or strategy.strategy_id.startswith("within_italy"):
        italy = panel[panel["IT"].astype(int) == 1].copy()
        return estimate_ddd(italy, y_col, entity_col, time_col)
    sample = filter_strategy_sample(panel, strategy, window_days=window_days)
    if sample.empty:
        return _empty_result(0, 0, "empty_sample")
    if strategy.strategy_id == "author_it_ban":
        return estimate_twfe_entity_only(sample, y_col, entity_col, time_col)
    if sample["treat"].nunique() < 2:
        return _empty_result(len(sample), int(sample[entity_col].nunique()), "no_treat_variation")
    return estimate_twfe(sample, y_col, entity_col, time_col, cluster_col)


def _pack_result(
    beta: float,
    se: float,
    n_obs: int,
    n_clusters: int,
    estimation_note: str = "ok",
) -> Dict[str, Any]:
    """Function summary: standard result dict with 95% CI and p-value."""
    ci_low = beta - 1.96 * se if np.isfinite(se) else float("nan")
    ci_high = beta + 1.96 * se if np.isfinite(se) else float("nan")
    p = float(2 * (1 - stats.norm.cdf(abs(beta / se)))) if se and se > 0 and np.isfinite(beta) else float("nan")
    return {
        "beta": beta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pvalue": p,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
        "pretrend_F_p": float("nan"),
        "estimation_note": estimation_note,
    }


def _empty_result(n_obs: int, n_clusters: int, estimation_note: str = "estimation_failed") -> Dict[str, Any]:
    """Function summary: NaN result when estimation fails."""
    return {
        "beta": float("nan"),
        "se": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "pvalue": float("nan"),
        "n_obs": n_obs,
        "n_clusters": n_clusters,
        "pretrend_F_p": float("nan"),
        "estimation_note": estimation_note,
    }
