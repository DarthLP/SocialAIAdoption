"""
Comment-level bucket event-study estimation: paper Eq.1 static, bin-interaction ES, DDD.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.did.estimate import _empty_result, _es_dummy_name, _pack_result


def _prep_y(df: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    """Function summary: drop missing y, author, time_id; build post:IT interaction."""
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author", "time_id"])
    work["author"] = work["author"].astype(str)
    work["time_id"] = work["time_id"].astype(str)
    work["post"] = work["post"].astype(float)
    work["IT"] = work["IT"].astype(float)
    work["post_IT"] = work["post"] * work["IT"]
    return work


def prep_static_design(
    df: pd.DataFrame,
    y_col: str = "y",
    cluster_col: str = "subreddit",
) -> pd.DataFrame:
    """Function summary: minimal static-DiD design matrix for repeated feols/bootstrap.

    Parameters:
    - df: comment panel with post, IT, author, time_id, outcome.
    - y_col: outcome column name.
    - cluster_col: cluster id for SEs and wild bootstrap.

    Returns:
    - DataFrame with y, post, post_IT, author, time_id, and cluster_col if present.
    """
    work = _prep_y(df, y_col)
    keep = ["y", "post", "post_IT", "author", "time_id"]
    if cluster_col in work.columns:
        keep.append(cluster_col)
    return work[keep]


def _feols_fit(
    formula: str,
    data: pd.DataFrame,
    coef_name: str,
    cluster_col: str = "subreddit",
    weights_col: str | None = None,
) -> Dict[str, Any]:
    """Function summary: pyfixest feols with CRV1 cluster SE."""
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_result(len(data), 0, "pyfixest_missing")
    if len(data) < 30 or data["author"].nunique() < 3:
        return _empty_result(
            len(data),
            int(data[cluster_col].nunique()) if cluster_col in data.columns else 0,
            "insufficient_obs",
        )
    n_cl = int(data[cluster_col].nunique()) if cluster_col in data.columns else data["author"].nunique()
    vcov: Any = {"CRV1": cluster_col} if cluster_col in data.columns else "iid"
    try:
        fit_data = data
        feols_kw: Dict[str, Any] = {"vcov": vcov}
        if weights_col and weights_col in data.columns:
            fit_data = data.copy()
            fit_data[weights_col] = (
                pd.to_numeric(fit_data[weights_col], errors="coerce")
                .astype(float)
                .fillna(1.0)
                .clip(lower=1e-9)
            )
            feols_kw["weights"] = weights_col
        fit = feols(formula, data=fit_data, **feols_kw)
        coefs = fit.coef()
        beta = float(coefs.loc[coef_name]) if coef_name in coefs.index else float("nan")
        se_frame = fit.se()
        se = float(se_frame.loc[coef_name]) if coef_name in se_frame.index else float("nan")
        return _pack_result(beta, se, len(data), n_cl)
    except Exception:
        return _empty_result(len(data), n_cl, "estimation_error")


def feols_static_paper_eq1_prepped(
    work: pd.DataFrame,
    cluster_col: str = "subreddit",
) -> Dict[str, Any]:
    """Function summary: headline static DiD on a prepped design matrix (no copy).

    Parameters:
    - work: output of prep_static_design.
    - cluster_col: cluster column for CRV1 SEs.

    Returns:
    - Result dict for post:IT coefficient.
    """
    res = _feols_fit("y ~ post + post_IT | author", work, "post_IT", cluster_col)
    res["static_variant"] = "paper_eq1"
    res["coef_name"] = "post:IT"
    return res


def estimate_static_paper_eq1(
    df: pd.DataFrame,
    y_col: str = "y",
    cluster_col: str = "subreddit",
) -> Dict[str, Any]:
    """Function summary: headline static DiD — y ~ post + post:IT | author (no bin FE).

    Parameters:
    - df: comment panel with post, IT, author.
    - y_col: outcome column.
    - cluster_col: cluster variable for SEs.

    Returns:
    - Result dict for post:IT coefficient.
    """
    work = prep_static_design(df, y_col, cluster_col)
    return feols_static_paper_eq1_prepped(work, cluster_col)


def estimate_adopter_ddd_static(
    df: pd.DataFrame,
    y_col: str = "y",
    flag_col: str = "flag",
    cluster_col: str = "author",
    coef_name: str = "post_IT_flag",
) -> Dict[str, Any]:
    """Function summary: static adopter DDD — y ~ post:flag + post:IT:flag | author + topic_family×date.

    Parameters:
    - df: comment panel with post, IT, author, date_utc, topic_family, flag.
    - y_col: outcome column.
    - flag_col: adopter indicator (0/1).
    - cluster_col: cluster for SEs.
    - coef_name: coefficient to return (post_IT_flag = triple-diff).

    Returns:
    - Result dict for target coefficient plus cell size metadata.
    """
    work = _prep_y(df, y_col)
    if "topic_family" not in work.columns:
        return _empty_result(len(work), 0, "missing_topic_family")
    work["flag"] = work[flag_col].astype(float)
    work["post_flag"] = work["post"] * work["flag"]
    work["post_IT_flag"] = work["post"] * work["IT"] * work["flag"]
    work["country_date"] = (
        work["topic_family"].astype(str) + "_" + work["date_utc"].astype(str)
    )
    for need in ("post_flag", "post_IT_flag"):
        if work[need].nunique() < 2:
            return _empty_result(len(work), int(work[cluster_col].nunique()), "no_ddd_variation")
    cells = work.groupby(["IT", "post", "flag"], observed=True).size()
    meta = {
        "n_obs": len(work),
        "n_authors": int(work["author"].nunique()),
        "n_comments": len(work),
        "cell_IT0_post0_flag0": int(cells.get((0, 0, 0), 0)),
        "cell_IT0_post0_flag1": int(cells.get((0, 0, 1), 0)),
        "cell_IT0_post1_flag0": int(cells.get((0, 1, 0), 0)),
        "cell_IT0_post1_flag1": int(cells.get((0, 1, 1), 0)),
        "cell_IT1_post0_flag0": int(cells.get((1, 0, 0), 0)),
        "cell_IT1_post0_flag1": int(cells.get((1, 0, 1), 0)),
        "cell_IT1_post1_flag0": int(cells.get((1, 1, 0), 0)),
        "cell_IT1_post1_flag1": int(cells.get((1, 1, 1), 0)),
    }
    res = _feols_fit(
        "y ~ post_flag + post_IT_flag | author + country_date",
        work,
        coef_name,
        cluster_col,
    )
    res.update(meta)
    res["coef_name"] = coef_name
    return res


def estimate_static_full_time_fe(
    df: pd.DataFrame,
    y_col: str = "y",
    cluster_col: str = "subreddit",
) -> Dict[str, Any]:
    """Function summary: robustness static — y ~ post:IT | author + time_id (no standalone post).

    Parameters:
    - df: comment panel.
    - y_col: outcome.
    - cluster_col: cluster for SEs.

    Returns:
    - Result dict for post:IT.
    """
    work = _prep_y(df, y_col)
    res = _feols_fit("y ~ post_IT | author + time_id", work, "post_IT", cluster_col)
    res["static_variant"] = "full_time_fe"
    res["coef_name"] = "post:IT"
    return res


def _build_es_interactions(
    df: pd.DataFrame,
    rel_col: str,
    ref_period: int,
    window: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Function summary: add es_k:IT dummies omitting reference period (legacy / tests)."""
    work = df.copy()
    work = work[work[rel_col].between(-window, window)]
    interact_cols: List[str] = []
    col_to_k: Dict[str, int] = {}
    for k in sorted(work[rel_col].unique()):
        ki = int(k)
        if ki == ref_period:
            continue
        col = f"{_es_dummy_name(ki)}_IT"
        work[col] = ((work[rel_col] == ki) * work["IT"]).astype(float)
        if work[col].sum() != 0:
            interact_cols.append(col)
            col_to_k[col] = ki
    return work, interact_cols, col_to_k


def _rel_period_from_coef_name(name: str, rel_col: str = "rel_period") -> Optional[int]:
    """Function summary: parse event-time bin k from pyfixest i() coefficient names."""
    s = str(name)
    m = re.search(rf"{re.escape(rel_col)}::(-?\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(rf"{re.escape(rel_col)}\[(-?\d+)\]", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+):IT", s)
    if m:
        return int(m.group(1))
    m = re.search(r"es_(-?\d+)_IT", s)
    if m:
        return int(m.group(1))
    return None


def _es_rows_from_fit(
    fit: Any,
    rel_col: str,
    ref_period: int,
    bin_days: int = 3,
    gamma_col: str = "gamma",
) -> List[Dict[str, Any]]:
    """Function summary: map pyfixest coef/se frames to event-study output rows."""
    rows: List[Dict[str, Any]] = []
    coefs = fit.coef()
    se_frame = fit.se()
    for col in coefs.index:
        k = _rel_period_from_coef_name(str(col), rel_col)
        if k is None or k == ref_period:
            continue
        b = float(coefs.loc[col])
        se = float(se_frame.loc[col]) if col in se_frame.index else float("nan")
        rows.append(
            {
                rel_col: k,
                "rel_day": int(k) * bin_days if rel_col == "rel_period" else k,
                gamma_col: b,
                "se": se,
                "ci_low": b - 1.96 * se if np.isfinite(se) else float("nan"),
                "ci_high": b + 1.96 * se if np.isfinite(se) else float("nan"),
                "pvalue": float(2 * (1 - stats.norm.cdf(abs(b / se)))) if se and se > 0 else float("nan"),
                "coef_name": f"bin_{k}:IT",
            }
        )
    return rows


def _prep_panel_y(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str,
    time_col: str = "time_id",
) -> pd.DataFrame:
    """Function summary: drop missing y/entity/time; ensure IT and rel columns for panel ES.

    Parameters:
    - df: subreddit-day (or entity-day) panel.
    - y_col: outcome column.
    - entity_col: entity FE key (e.g. subreddit).
    - time_col: calendar time FE key.

    Returns:
    - Cleaned copy with y, entity_col, time_id, IT as float.
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", entity_col, time_col, "IT"])
    work[entity_col] = work[entity_col].astype(str)
    work[time_col] = work[time_col].astype(str)
    work["IT"] = work["IT"].astype(float)
    return work


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Function summary: weighted mean ignoring NaNs with positive weights."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return 0.0
    return float(np.average(values[ok], weights=weights[ok]))


def _twfe_within_transform(
    values: pd.Series,
    entity: pd.Series,
    time: pd.Series,
    *,
    weights: pd.Series | None = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> np.ndarray:
    """Function summary: alternating-projection TWFE within transform for one column.

    Parameters:
    - values: outcome or regressor series.
    - entity: entity id series aligned to values.
    - time: time id series aligned to values.
    - weights: optional positive weights for weighted TWFE absorption.
    - max_iter: maximum demeaning iterations.
    - tol: convergence tolerance on max abs change.

    Returns:
    - Within-transformed numpy array.
    """
    v = pd.to_numeric(values, errors="coerce").astype(float).values.copy()
    ent = entity.astype(str).values
    tim = time.astype(str).values
    w = None
    if weights is not None:
        w = (
            pd.to_numeric(weights, errors="coerce")
            .astype(float)
            .fillna(1.0)
            .clip(lower=1e-9)
            .values
        )
    for _ in range(max_iter):
        v_old = v.copy()
        for e in np.unique(ent):
            mask = ent == e
            if w is None:
                v[mask] -= np.nanmean(v[mask])
            else:
                v[mask] -= _weighted_mean(v[mask], w[mask])
        for t in np.unique(tim):
            mask = tim == t
            if w is None:
                v[mask] -= np.nanmean(v[mask])
            else:
                v[mask] -= _weighted_mean(v[mask], w[mask])
        if np.nanmax(np.abs(v - v_old)) < tol:
            break
    return v


def manual_panel_it_event_study(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "subreddit",
    time_col: str = "time_id",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "subreddit",
    bin_days: int = 3,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Function summary: TWFE event study via within-demean y then OLS on IT×rel_bin dummies.

    Parameters:
    - df: panel with IT, rel_col, entity_col, time_col, outcome.
    - y_col: outcome column.
    - entity_col: entity FE key.
    - time_col: time FE key.
    - rel_col: event-time bin column.
    - ref_period: omitted reference bin.
    - window: trim rel_col to [-window, window].
    - cluster_col: cluster for CRV1 SEs.
    - bin_days: rel_day display multiplier.
    - weights_col: optional WLS weights aligned with pyfixest run.

    Returns:
    - Coefficient DataFrame (rel_col, gamma, se, ci_*, pvalue, rel_day).
    """
    work = _prep_panel_y(df, y_col, entity_col, time_col)
    if rel_col not in work.columns:
        work[rel_col] = (work["rel_day"] // bin_days).astype(int)
    work = work[work[rel_col].between(-window, window)]
    if work.empty or work["IT"].nunique() < 2:
        return pd.DataFrame()
    w_series = None
    if weights_col and weights_col in work.columns:
        w_series = (
            pd.to_numeric(work[weights_col], errors="coerce")
            .astype(float)
            .fillna(1.0)
            .clip(lower=1e-9)
        )
    y_tilde = _twfe_within_transform(
        work["y"], work[entity_col], work[time_col], weights=w_series
    )
    interact_cols: List[str] = []
    col_to_k: Dict[str, int] = {}
    for k in sorted(work[rel_col].unique()):
        ki = int(k)
        if ki == ref_period:
            continue
        col = f"{_es_dummy_name(ki)}_IT"
        work[col] = ((work[rel_col] == ki) * work["IT"]).astype(float)
        if work[col].sum() == 0:
            continue
        interact_cols.append(col)
        col_to_k[col] = ki
    if not interact_cols:
        return pd.DataFrame()
    x_tilde = np.column_stack(
        [
            _twfe_within_transform(work[c], work[entity_col], work[time_col], weights=w_series)
            for c in interact_cols
        ]
    )
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()
    model = sm.OLS(y_tilde, x_tilde)
    if cluster_col in work.columns:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": work[cluster_col].astype(str)})
    else:
        fit = model.fit()
    rows: List[Dict[str, Any]] = []
    for i, col in enumerate(interact_cols):
        k = col_to_k[col]
        b = float(fit.params[i])
        se = float(fit.bse[i])
        rows.append(
            {
                rel_col: k,
                "rel_day": int(k) * bin_days if rel_col == "rel_period" else k,
                "gamma": b,
                "se": se,
                "ci_low": b - 1.96 * se if np.isfinite(se) else float("nan"),
                "ci_high": b + 1.96 * se if np.isfinite(se) else float("nan"),
                "pvalue": float(2 * (1 - stats.norm.cdf(abs(b / se)))) if se and se > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(rel_col) if rows else pd.DataFrame()


def estimate_panel_it_event_study(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str = "subreddit",
    time_col: str = "time_id",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "subreddit",
    bin_days: int = 3,
    weights_col: str | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: y ~ i(rel_period, IT, ref) | entity + time_id with CRV1 clustering.

    Parameters:
    - df: subreddit-day panel with rel_period, IT, entity_col, time_col.
    - y_col: outcome column.
    - entity_col: entity FE key (e.g. subreddit).
    - time_col: calendar time FE key.
    - rel_col: event-time bin column.
    - ref_period: omitted reference bin.
    - window: trim rel_col to [-window, window].
    - cluster_col: cluster for SEs.
    - bin_days: rel_day display multiplier.
    - weights_col: optional regression weights (e.g. n_comments).

    Returns:
    - Tuple (summary dict, coefficient DataFrame with rel_period, gamma, se, ci).
    """
    work = _prep_panel_y(df, y_col, entity_col, time_col)
    if rel_col not in work.columns:
        if "rel_day" in work.columns:
            work[rel_col] = (work["rel_day"] // bin_days).astype(int)
        else:
            return _empty_result(len(work), 0, "no_rel_col"), pd.DataFrame()
    work = work[work[rel_col].between(-window, window)]
    if work.empty or work["IT"].nunique() < 2:
        return _empty_result(len(work), 0, "no_event_study_variation"), pd.DataFrame()
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_result(len(work), 0, "pyfixest_missing"), pd.DataFrame()
    n_cl = int(work[cluster_col].nunique()) if cluster_col in work.columns else int(work[entity_col].nunique())
    vcov: Any = {"CRV1": cluster_col} if cluster_col in work.columns else "iid"
    formula = f"y ~ i({rel_col}, IT, ref={ref_period}) | {entity_col} + {time_col}"
    try:
        feols_kw: Dict[str, Any] = {"vcov": vcov}
        fit_work = work
        if weights_col and weights_col in work.columns:
            fit_work = work.copy()
            fit_work[weights_col] = (
                pd.to_numeric(fit_work[weights_col], errors="coerce").astype(float).fillna(1.0).clip(lower=1e-9)
            )
            feols_kw["weights"] = weights_col
        fit = feols(formula, data=fit_work, **feols_kw)
    except Exception:
        return _empty_result(len(work), n_cl, "event_study_failed"), pd.DataFrame()
    rows = _es_rows_from_fit(fit, rel_col, ref_period, bin_days=bin_days, gamma_col="gamma")
    es_df = pd.DataFrame(rows).sort_values(rel_col) if rows else pd.DataFrame()
    summary = _pack_result(float("nan"), float("nan"), len(work), n_cl, estimation_note="ok")
    return summary, es_df


def estimate_comment_it_event_study(
    df: pd.DataFrame,
    y_col: str = "y",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "subreddit",
    bin_days: int = 3,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: y ~ i(rel_period, IT, ref) | author + time_id; omit ref bin.

    Parameters:
    - df: comment panel with rel_period, IT, author, time_id.
    - y_col: outcome.
    - rel_col: event-time bin column.
    - ref_period: omitted reference bin.
    - window: trim rel_col to [-window, window].
    - cluster_col: cluster for SEs.
    - bin_days: multiplier for rel_day display.

    Returns:
    - Tuple (summary dict, coefficient DataFrame with rel_period, gamma, se, ci).
    """
    work = _prep_y(df, y_col)
    if rel_col not in work.columns:
        work["rel_period"] = (work["rel_day"] // bin_days).astype(int)
        rel_col = "rel_period"
    work = work[work[rel_col].between(-window, window)]
    if work.empty or work["IT"].nunique() < 2:
        return _empty_result(len(work), 0, "no_event_study_variation"), pd.DataFrame()
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_result(len(work), 0, "pyfixest_missing"), pd.DataFrame()
    n_cl = int(work[cluster_col].nunique()) if cluster_col in work.columns else work["author"].nunique()
    vcov: Any = {"CRV1": cluster_col} if cluster_col in work.columns else "iid"
    formula = f"y ~ i({rel_col}, IT, ref={ref_period}) | author + time_id"
    try:
        fit = feols(formula, data=work, vcov=vcov)
    except Exception:
        return _empty_result(len(work), work["author"].nunique(), "event_study_failed"), pd.DataFrame()
    rows = _es_rows_from_fit(fit, rel_col, ref_period, bin_days=bin_days, gamma_col="gamma")
    es_df = pd.DataFrame(rows).sort_values(rel_col) if rows else pd.DataFrame()
    summary = _pack_result(float("nan"), float("nan"), len(work), n_cl, estimation_note="ok")
    return summary, es_df


def estimate_comment_it_ddd_event_study(
    df: pd.DataFrame,
    liberal_label: str,
    conservative_label: str,
    y_col: str = "y",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "subreddit",
    bin_days: int = 3,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: stacked DDD — y ~ i(rel_period, IT, liberal, ref) | author + time_id.

    Parameters:
    - df: comments with bucket in {liberal, conservative}.
    - liberal_label: liberal bucket string.
    - conservative_label: conservative bucket string.
    - y_col, rel_col, ref_period, window, cluster_col: as in event study.
    - bin_days: rel_day display multiplier.

    Returns:
    - Tuple (summary, DDD coefficient table).
    """
    work = df[df["bucket"].astype(str).isin([liberal_label, conservative_label])]
    if work.empty:
        return _empty_result(0, 0, "no_ddd_variation"), pd.DataFrame()
    work = work.copy()
    work["liberal"] = (work["bucket"].astype(str) == liberal_label).astype(float)
    work = _prep_y(work, y_col)
    if rel_col not in work.columns:
        work["rel_period"] = (work["rel_day"] // bin_days).astype(int)
    work = work[work[rel_col].between(-window, window)]
    if work.empty:
        return _empty_result(0, 0, "no_ddd_variation"), pd.DataFrame()
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_result(0, 0, "pyfixest_missing"), pd.DataFrame()
    n_cl = int(work[cluster_col].nunique()) if cluster_col in work.columns else 0
    vcov: Any = {"CRV1": cluster_col} if cluster_col in work.columns else "iid"
    # pyfixest: three-way i(rel, IT, liberal) fails; interact liberal with IT×bin dummies.
    formula = f"y ~ i({rel_col}, IT, ref={ref_period}):liberal | author + time_id"
    try:
        fit = feols(formula, data=work, vcov=vcov)
    except Exception:
        return _empty_result(len(work), n_cl, "ddd_failed"), pd.DataFrame()
    rows = _es_rows_from_fit(fit, rel_col, ref_period, bin_days=bin_days, gamma_col="ddd_gamma")
    if not rows:
        import logging

        logging.getLogger(__name__).warning(
            "ddd coef parse empty; raw coef names: %s",
            list(fit.coef().index),
        )
        summary = _pack_result(
            float("nan"), float("nan"), len(work), n_cl, estimation_note="ddd_coef_parse_failed"
        )
        return summary, pd.DataFrame()
    for row in rows:
        row["coef_name"] = f"bin_{row[rel_col]}:IT:liberal"
    ddd_df = pd.DataFrame(rows).sort_values(rel_col)
    summary = _pack_result(float("nan"), float("nan"), len(work), n_cl, estimation_note="ok")
    return summary, ddd_df


def _median_finite(values: List[float]) -> float:
    """Function summary: median of finite floats, or NaN if none."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def combine_split_sample_static(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Function summary: average post:IT across cross-fit splits; SE from split dispersion.

    Parameters:
    - results: list of per-split estimate dicts.

    Returns:
    - Combined result dict with beta_mean, se_between_splits, and merged inference metadata.
    """
    betas = [r["beta"] for r in results if np.isfinite(r.get("beta", np.nan))]
    if not betas:
        return _empty_result(0, 0, "no_splits")
    arr = np.asarray(betas)
    mean_b = float(arr.mean())
    beta_sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1 and beta_sd == 0.0:
        se_split = float("nan")
        combine_note = "no_cross_split_variation"
    else:
        se_split = float(beta_sd / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        combine_note = "combined_splits"
    n_cl = max((int(r.get("n_clusters", 0) or 0) for r in results), default=0)
    out = _pack_result(mean_b, se_split, sum(r.get("n_obs", 0) for r in results), n_cl)
    out["n_splits"] = len(betas)
    out["beta_sd_across_splits"] = beta_sd
    out["estimation_note"] = combine_note if len(betas) > 1 else str(results[0].get("estimation_note", "ok"))
    out["static_variant"] = "paper_eq1"
    out["coef_name"] = "post:IT"
    out["inference_role"] = results[0].get("inference_role", "descriptive")
    out["sample"] = results[0].get("sample")
    out["pvalue_cluster"] = _median_finite([float(r.get("pvalue_cluster", r.get("pvalue", np.nan))) for r in results])
    p_placebo = _median_finite([float(r.get("p_placebo_space", np.nan)) for r in results])
    out["p_placebo_space"] = p_placebo
    out["perm_p"] = p_placebo
    out["p_wild"] = float("nan")
    return out


def combine_split_sample_es(es_frames: List[pd.DataFrame], rel_col: str = "rel_period") -> pd.DataFrame:
    """Function summary: mean gamma_k across splits with SE from cross-split SD."""
    if not es_frames:
        return pd.DataFrame()
    keys = sorted(set().union(*[set(f[rel_col].astype(int).tolist()) for f in es_frames if not f.empty]))
    rows: List[Dict[str, Any]] = []
    for k in keys:
        gammas = []
        for f in es_frames:
            sub = f[f[rel_col].astype(int) == k]
            if not sub.empty and np.isfinite(sub["gamma"].iloc[0]):
                gammas.append(float(sub["gamma"].iloc[0]))
        if not gammas:
            continue
        arr = np.asarray(gammas)
        m = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        rows.append(
            {
                rel_col: k,
                "gamma": m,
                "se": se,
                "ci_low": m - 1.96 * se if np.isfinite(se) else float("nan"),
                "ci_high": m + 1.96 * se if np.isfinite(se) else float("nan"),
                "n_splits": len(gammas),
            }
        )
    return pd.DataFrame(rows)


def compute_trajectory_means(
    df: pd.DataFrame,
    rel_col: str,
    series_id: str,
    series_label: str,
    y_col: str = "y",
    bin_days: int = 3,
) -> pd.DataFrame:
    """Function summary: mean outcome by event bin for one trajectory series.

    Parameters:
    - df: filtered comments for this series.
    - rel_col: rel_period column.
    - series_id: series identifier.
    - series_label: human label.
    - y_col: outcome.
    - bin_days: for rel_day display.

    Returns:
    - DataFrame with rel_period, mean_y, n_comments, series_id, series_label.
    """
    if df.empty:
        return pd.DataFrame()
    work = df
    if rel_col not in work.columns:
        if "rel_day" not in work.columns:
            return pd.DataFrame()
        work = work.copy()
        work[rel_col] = (work["rel_day"] // int(bin_days)).astype(int)
    if y_col not in work.columns or work[y_col].dtype != np.float32:
        y = pd.to_numeric(work[y_col], errors="coerce")
    else:
        y = work[y_col]
    g = pd.DataFrame({rel_col: work[rel_col], "y": y}).groupby(rel_col, observed=True)["y"]
    out = g.agg(["mean", "count"]).reset_index()
    out.columns = [rel_col, "mean_y", "n_comments"]
    out["rel_day"] = out[rel_col].astype(int) * int(bin_days)
    out["series_id"] = series_id
    out["series_label"] = series_label
    return out


def filter_trajectory_series(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: apply trajectory_series filter dict from config."""
    filt = spec.get("filter") or {}
    work = df
    for key, val in filt.items():
        if key not in work.columns:
            continue
        if isinstance(val, int):
            work = work[work[key].astype(int) == int(val)]
        else:
            work = work[work[key].astype(str) == str(val)]
    return work
