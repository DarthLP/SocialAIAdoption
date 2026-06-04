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

from src.did.specs import (
    StrategySpec,
    filter_strategy_sample,
    is_entity_fe_only_strategy,
)


def _es_dummy_name(k: int) -> str:
    """Function summary: formula-safe lead/lag dummy name (negative k cannot be es_{k})."""
    if int(k) < 0:
        return f"es_neg{abs(int(k))}"
    return f"es_pos{int(k)}"


def _parse_es_dummy_name(name: str) -> Optional[int]:
    """Function summary: map es_neg14 / es_pos0 back to event-time integer."""
    if name.startswith("es_neg"):
        return -int(name[len("es_neg") :])
    if name.startswith("es_pos"):
        return int(name[len("es_pos") :])
    return None


def _insufficient_panel(
    sample: pd.DataFrame,
    entity_col: str,
    time_col: str,
    min_clusters: int = 12,
    min_obs_per_cluster: int = 3,
) -> bool:
    """Function summary: True when clusters are few and any cluster has < min_obs rows.

    Uses entity-level counts (not entity×time cells, which are 1 obs in daily panels).
    """
    del time_col  # reserved for future bin-level checks
    n_clusters = int(sample[entity_col].nunique()) if not sample.empty else 0
    if n_clusters >= min_clusters:
        return False
    ent_counts = sample.groupby(entity_col, observed=True).size()
    return bool((ent_counts < min_obs_per_cluster).any())


def design_matrix_condition_number(work: pd.DataFrame) -> float:
    """Function summary: condition number of TWFE design (treat_post + entity/time dummies).

    Parameters:
    - work: MultiIndex panel with treat_post column.

    Returns:
    - cond(X) or inf on failure.
    """
    try:
        entities = work.index.get_level_values(0)
        times = work.index.get_level_values(1)
        ent_d = pd.get_dummies(entities, drop_first=True)
        time_d = pd.get_dummies(times, drop_first=True)
        parts = [work["treat_post"].astype(float).values.reshape(-1, 1)]
        if ent_d.shape[1]:
            parts.append(ent_d.values)
        if time_d.shape[1]:
            parts.append(time_d.values)
        x = np.column_stack(parts)
        if x.shape[1] < 2:
            return float("inf")
        return float(np.linalg.cond(x))
    except Exception:
        return float("inf")


def apply_degeneracy_guard(
    result: Dict[str, Any],
    design_cond: float,
    median_abs_beta: float,
) -> Dict[str, Any]:
    """Function summary: NaN β and tag degenerate_collinear when fit is numerically unstable.

    Parameters:
    - result: TWFE result dict.
    - design_cond: design matrix condition number.
    - median_abs_beta: median |β| among ok estimates for this outcome.

    Returns:
    - Possibly updated result dict.
    """
    skip_notes = {"empty_sample", "no_treat_variation", "insufficient_panel", "fully_absorbed"}
    note = str(result.get("estimation_note", "ok"))
    beta = result.get("beta", float("nan"))
    if not np.isfinite(beta):
        return result
    if note in skip_notes and abs(beta) < 1e6:
        return result
    threshold = 100.0 * max(1.0, float(median_abs_beta))
    if abs(beta) > 1e6 or abs(beta) > threshold or design_cond > 1e10:
        out = dict(result)
        out.update(
            {
                "beta": float("nan"),
                "se": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "pvalue": float("nan"),
                "estimation_note": "degenerate_collinear",
            }
        )
        return out
    return result


UNRELIABLE_ESTIMATION_MARKERS: Tuple[str, ...] = (
    "degenerate_collinear",
    "insufficient_panel",
    "fully_absorbed",
    "empty_sample",
    "no_treat_variation",
    "treat_post_absorbed",
    "no_within_entity_political_variation",
    "estimation_failed",
    "estimation_error",
)


def _estimation_note_unreliable(note: str) -> bool:
    """Function summary: True when TWFE row should not be paired with a pretrend p-value."""
    n = str(note or "ok").strip()
    if n in ("ok", "ok_entity_fe_only"):
        return False
    return any(marker in n for marker in UNRELIABLE_ESTIMATION_MARKERS)


def annotate_pretrend_quality(row: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: tag pretrend interpretability and clear misleading pretrend_F_p values.

    Parameters:
    - row: one strategy×outcome TWFE summary dict (after degeneracy guard).

    Returns:
    - Copy with pretrend_quality and possibly NaN pretrend_F_p.
    """
    out = dict(row)
    strategy_id = str(out.get("strategy_id", ""))
    if strategy_id == "within_italy_ddd" or is_entity_fe_only_strategy(strategy_id):
        out["pretrend_quality"] = "not_estimated"
        out["pretrend_F_p"] = float("nan")
        return out

    beta = out.get("beta", float("nan"))
    note = str(out.get("estimation_note", "ok"))
    try:
        pretrend_p = float(out.get("pretrend_F_p", float("nan")))
    except (TypeError, ValueError):
        pretrend_p = float("nan")

    if not np.isfinite(beta) or _estimation_note_unreliable(note):
        out["pretrend_quality"] = "unreliable_estimate"
        out["pretrend_F_p"] = float("nan")
        return out

    if not np.isfinite(pretrend_p):
        out["pretrend_quality"] = "not_estimated"
        return out

    if pretrend_p < 0.05:
        out["pretrend_quality"] = "pretrend_reject"
    else:
        out["pretrend_quality"] = "ok"
    return out


def _panelols_weights(panel: pd.DataFrame, weights_col: str | None) -> pd.Series | None:
    """Function summary: aligned positive weights for PanelOLS constructor, or None.

    Parameters:
    - panel: MultiIndex panel used in from_formula.
    - weights_col: column name on panel (e.g. n_comments).

    Returns:
    - Float Series aligned to panel index, or None when unweighted.
    """
    if not weights_col or weights_col not in panel.columns:
        return None
    w = pd.to_numeric(panel[weights_col], errors="coerce").astype(float)
    w = w.fillna(1.0).clip(lower=1e-9)
    return w


def estimate_pretrend_f(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    rel_col: str = "rel_day",
    leads: Tuple[int, ...] = (-3, -2, -1),
    weights_col: str | None = None,
) -> Tuple[float, Optional[str]]:
    """Function summary: joint F-test that pre-ban treat×event-time leads are zero.

    Parameters:
    - df: filtered strategy sample with treat and rel_day.
    - y_col: outcome column.
    - entity_col, time_col: panel index columns.
    - rel_col: relative event time (days).
    - leads: pre-reference event times to interact with treat.

    Returns:
    - Tuple (pretrend_F_p, optional note suffix like insufficient_preperiods).
    """
    work = df.copy()
    if rel_col not in work.columns:
        return float("nan"), "insufficient_preperiods"
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    drop_cols = ["y"]
    if weights_col and weights_col in work.columns:
        drop_cols.append(weights_col)
    work = work.dropna(subset=drop_cols)
    work["treat"] = work["treat"].astype(float)
    rel_vals = set(int(v) for v in work[rel_col].dropna().unique())
    if not all(int(k) in rel_vals for k in leads):
        return float("nan"), "insufficient_preperiods"
    interact_cols: List[str] = []
    for k in leads:
        col = _es_dummy_name(int(k))
        work[col] = ((work[rel_col] == k) * work["treat"]).astype(float)
        if work[col].sum() == 0:
            return float("nan"), "insufficient_preperiods"
        interact_cols.append(col)
    work[time_col] = pd.to_datetime(work[time_col].astype(str))
    panel = work.set_index([entity_col, time_col])
    if len(panel) < 30 or panel.index.get_level_values(0).nunique() < 3:
        return float("nan"), "insufficient_preperiods"
    formula_rhs = " + ".join(interact_cols) + " + EntityEffects + TimeEffects"
    w = _panelols_weights(panel, weights_col)
    try:
        kwargs: Dict[str, Any] = {"drop_absorbed": True}
        if w is not None:
            kwargs["weights"] = w
        mod = PanelOLS.from_formula(f"y ~ {formula_rhs}", data=panel, **kwargs)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        restrictions = " + ".join(f"{p}" for p in interact_cols if p in res.params.index) + " = 0"
        if not restrictions.strip().startswith("="):
            wt = res.wald_test(formula=restrictions)
            return float(wt.pval), None
    except Exception:
        pass
    return float("nan"), None


def _prep_comment_regression(
    df: pd.DataFrame,
    y_col: str,
    author_col: str = "author",
    time_col: str = "time_id",
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Function summary: comment-level frame with y, treat_post, and FE keys."""
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    drop = ["y", author_col, time_col]
    if weights_col and weights_col in work.columns:
        work[weights_col] = pd.to_numeric(work[weights_col], errors="coerce")
        drop.append(weights_col)
    work = work.dropna(subset=drop)
    work[author_col] = work[author_col].astype(str)
    work[time_col] = work[time_col].astype(str)
    work["treat"] = work["treat"].astype(float)
    work["post"] = work["post"].astype(float)
    work["treat_post"] = work["treat"] * work["post"]
    return work


def _feols_result_to_dict(
    fit: Any,
    coef_name: str,
    n_obs: int,
    n_clusters: int,
    estimation_note: str = "ok",
) -> Dict[str, Any]:
    """Function summary: map pyfixest fit to standard TWFE result dict."""
    try:
        coefs = fit.coef()
        beta = float(coefs.loc[coef_name]) if coef_name in coefs.index else float("nan")
        se_frame = fit.se()
        se = float(se_frame.loc[coef_name]) if coef_name in se_frame.index else float("nan")
    except Exception:
        return _empty_result(n_obs, n_clusters, "estimation_error")
    if not np.isfinite(beta):
        return _pack_result(beta, se, n_obs, n_clusters, estimation_note=f"{coef_name}_absorbed")
    note = estimation_note if np.isfinite(se) else "estimation_error"
    return _pack_result(beta, se, n_obs, n_clusters, estimation_note=note)


def estimate_comment_feols(
    df: pd.DataFrame,
    y_col: str,
    author_col: str = "author",
    time_col: str = "time_id",
    cluster_col: str = "author",
    entity_only: bool = False,
    weights_col: str | None = None,
) -> Dict[str, Any]:
    """Function summary: comment-level DiD with author (+ calendar) absorbed FEs via pyfixest.

    Parameters:
    - df: comment rows with treat, post, author, time_id.
    - y_col: outcome column.
    - author_col: author identifier for FE and clustering.
    - time_col: calendar bin for time FE (omit when entity_only).
    - cluster_col: cluster variable for CRV1 SEs.
    - entity_only: if True, y ~ post | author (Italy-only national shock).

    Returns:
    - Standard result dict (beta, se, n_obs, n_clusters, estimation_note).
    """
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_result(0, 0, "pyfixest_missing")
    work = _prep_comment_regression(df, y_col, author_col, time_col, weights_col=weights_col)
    w_series = None
    if weights_col and weights_col in work.columns:
        w_series = work[weights_col].astype(float).fillna(1.0).clip(lower=1e-9)
    if len(work) < 30 or work[author_col].nunique() < 3:
        return _empty_result(
            len(work),
            int(work[author_col].nunique()) if len(work) else 0,
            "insufficient_obs_or_clusters",
        )
    n_cl = int(work[cluster_col].nunique())
    try:
        feols_kw: Dict[str, Any] = {"vcov": {"CRV1": cluster_col}}
        if w_series is not None:
            feols_kw["weights"] = w_series
        if entity_only:
            fit = feols(f"y ~ post | {author_col}", data=work, **feols_kw)
            return _feols_result_to_dict(
                fit, "post", len(work), n_cl, estimation_note="ok_entity_fe_only"
            )
        fit = feols(
            f"y ~ treat_post | {author_col} + {time_col}",
            data=work,
            **feols_kw,
        )
        return _feols_result_to_dict(fit, "treat_post", len(work), n_cl)
    except Exception:
        return _empty_result(len(work), n_cl, "estimation_error")


def _prep_panel(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str,
    time_col: str,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Function summary: drop missing and set MultiIndex for PanelOLS."""
    cols = [entity_col, time_col, y_col, "treat", "post"]
    if weights_col and weights_col in df.columns:
        cols.append(weights_col)
    work = df[cols].copy()
    work = work.rename(columns={y_col: "y", entity_col: "entity", time_col: "time"})
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    drop = ["y"]
    if weights_col and weights_col in work.columns:
        work[weights_col] = pd.to_numeric(work[weights_col], errors="coerce")
        drop.append(weights_col)
    work = work.dropna(subset=drop)
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
    weights_col: str | None = None,
) -> Dict[str, Any]:
    """Function summary: two-way FE DiD on treat×post with clustered SEs.

    Returns:
    - Dict with beta, se, ci_low, ci_high, pvalue, n_obs, n_clusters, estimation_note.
    """
    work = _prep_panel(df, y_col, entity_col, time_col, weights_col=weights_col)
    if len(work) < 30 or work.index.get_level_values(0).nunique() < 3:
        return _empty_result(
            len(work),
            work.index.get_level_values(0).nunique() if len(work) else 0,
            "insufficient_obs_or_clusters",
        )
    w = _panelols_weights(work, weights_col)
    try:
        kwargs: Dict[str, Any] = {"drop_absorbed": True}
        if w is not None:
            kwargs["weights"] = w
        mod = PanelOLS.from_formula(
            "y ~ treat_post + EntityEffects + TimeEffects",
            data=work,
            **kwargs,
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
    cond = design_matrix_condition_number(work)
    packed = _pack_result(beta, se, int(res.nobs), int(df[entity_col].nunique()), estimation_note="ok")
    packed["design_cond"] = cond
    return packed


def estimate_event_study(
    df: pd.DataFrame,
    y_col: str,
    rel_col: str = "rel_day",
    ref_day: int = -1,
    window: int = 30,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    weights_col: str | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: dynamic TWFE event study with lead/lag dummies × treat.

    Returns:
    - Summary dict (pretrend_F_p, etc.) and coefficient table.
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    drop = ["y"]
    if weights_col and weights_col in work.columns:
        work[weights_col] = pd.to_numeric(work[weights_col], errors="coerce")
        drop.append(weights_col)
    work = work.dropna(subset=drop)
    work = work[work[rel_col].between(-window, window)]
    work["treat"] = work["treat"].astype(float)
    interact_cols: List[str] = []
    col_to_k: Dict[str, int] = {}
    for k in sorted(work[rel_col].unique()):
        if int(k) == ref_day:
            continue
        col = _es_dummy_name(int(k))
        work[col] = ((work[rel_col] == k) * work["treat"]).astype(float)
        if work[col].sum() != 0:
            interact_cols.append(col)
            col_to_k[col] = int(k)
    if not interact_cols:
        return _empty_result(len(work), work[entity_col].nunique(), "no_event_study_variation"), pd.DataFrame()
    interact_rhs = " + ".join(interact_cols)
    work[time_col] = pd.to_datetime(work[time_col].astype(str))
    panel = work.set_index([entity_col, time_col])
    note = "ok"
    res = None
    for formula_rhs in (
        f"{interact_rhs} + EntityEffects + TimeEffects",
        f"{interact_rhs} + TimeEffects",
    ):
        try:
            w = _panelols_weights(panel, weights_col)
            kwargs: Dict[str, Any] = {"drop_absorbed": True}
            if w is not None:
                kwargs["weights"] = w
            mod = PanelOLS.from_formula(f"y ~ {formula_rhs}", data=panel, **kwargs)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            if formula_rhs.endswith("+ TimeEffects") and "EntityEffects" not in formula_rhs:
                note = "ok_time_fe_only"
            break
        except Exception:
            res = None
    if res is None:
        return _empty_result(len(work), work[entity_col].nunique(), "event_study_failed"), pd.DataFrame()
    pretrend_params = [
        p
        for p in res.params.index
        if p.startswith("es_")
        and (_parse_es_dummy_name(p) is not None and _parse_es_dummy_name(p) < ref_day)
    ]
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
        k = _parse_es_dummy_name(pname)
        if k is None:
            continue
        b = float(res.params[pname])
        se = float(res.std_errors[pname])
        if not np.isfinite(se):
            continue
        row = {
            rel_col: k,
            "gamma": b,
            "se": se,
            "ci_low": b - 1.96 * se,
            "ci_high": b + 1.96 * se,
            "pvalue": float(2 * (1 - stats.norm.cdf(abs(b / se)))) if se > 0 else float("nan"),
        }
        if rel_col != "rel_day":
            row["rel_day"] = k
        rows.append(row)
    es_df = pd.DataFrame(rows).sort_values(rel_col) if rows else pd.DataFrame()
    summary = _pack_result(
        float("nan"), float("nan"), int(res.nobs), int(work[entity_col].nunique()), estimation_note=note
    )
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
    weights_col: str | None = None,
) -> Dict[str, Any]:
    """Function summary: treat×post with entity FE only (author IT cohort; no time FE).

    Used when treat is constant so calendar time FE absorb the ban dummy.
    """
    cols = [entity_col, time_col, y_col, "treat", "post"]
    if weights_col and weights_col in df.columns:
        cols.append(weights_col)
    work = df[cols].copy()
    work = work.rename(columns={y_col: "y", entity_col: "entity", time_col: "time"})
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    drop = ["y"]
    if weights_col and weights_col in work.columns:
        work[weights_col] = pd.to_numeric(work[weights_col], errors="coerce")
        drop.append(weights_col)
    work = work.dropna(subset=drop)
    work["treat_post"] = work["treat"].astype(float) * work["post"].astype(float)
    work["time"] = pd.to_datetime(work["time"].astype(str))
    work = work.set_index(["entity", "time"])
    if len(work) < 30 or work.index.get_level_values(0).nunique() < 3:
        return _empty_result(
            len(work),
            work.index.get_level_values(0).nunique() if len(work) else 0,
            "insufficient_obs_or_clusters",
        )
    w = _panelols_weights(work, weights_col)
    try:
        kwargs: Dict[str, Any] = {"drop_absorbed": True}
        if w is not None:
            kwargs["weights"] = w
        mod = PanelOLS.from_formula(
            "y ~ treat_post + EntityEffects",
            data=work,
            **kwargs,
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
    panel_kind: str = "subreddit_day",
    weights: str | None = None,
) -> Dict[str, Any]:
    """Function summary: TWFE for one strategy/outcome combination."""
    if strategy.strategy_id == "within_italy_ddd" or strategy.strategy_id.startswith("within_italy"):
        italy = panel[panel["IT"].astype(int) == 1].copy()
        return estimate_ddd(italy, y_col, entity_col, time_col)
    sample = filter_strategy_sample(panel, strategy, window_days=window_days)
    if sample.empty:
        return _empty_result(0, 0, "empty_sample")
    author_col = "author" if "author" in sample.columns else entity_col
    if panel_kind == "comment":
        return estimate_comment_feols(
            sample,
            y_col,
            author_col=author_col,
            time_col=time_col,
            cluster_col=author_col,
            entity_only=is_entity_fe_only_strategy(strategy.strategy_id),
            weights_col=weights,
        )
    if panel_kind == "author_day":
        entity_col = author_col
        cluster_col = author_col
    if is_entity_fe_only_strategy(strategy.strategy_id):
        return estimate_twfe_entity_only(sample, y_col, entity_col, time_col, weights_col=weights)
    if _insufficient_panel(sample, entity_col, time_col):
        return _empty_result(
            len(sample),
            int(sample[entity_col].nunique()),
            "insufficient_panel",
        )
    if sample["treat"].nunique() < 2:
        return _empty_result(len(sample), int(sample[entity_col].nunique()), "no_treat_variation")
    return estimate_twfe(sample, y_col, entity_col, time_col, cluster_col, weights_col=weights)


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
