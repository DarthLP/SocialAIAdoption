"""
Placebo-in-space, restricted wild cluster bootstrap, and inference routing for DiD.
"""

from __future__ import annotations

import gc
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.did.estimate import estimate_twfe
from src.did.specs import (
    CONTROL_FAMILIES,
    ITALY_FAMILIES,
    StrategySpec,
    filter_strategy_sample,
    is_cross_country_strategy,
    is_entity_fe_only_strategy,
    is_wcb_eligible_strategy,
)

logger = logging.getLogger(__name__)

DEFAULT_WCB_DRAWS = 9999


@dataclass(frozen=True)
class PlaceboInSpaceResult:
    """Function summary: placebo-in-space p-values (coefficient and t-stat) and metadata."""

    p: float
    p_t: float
    p_floor: float
    n_placebo_draws: int
    beta_italy: float
    t_italy: float
    placebo_betas: Tuple[float, ...]
    placebo_ts: Tuple[float, ...]

    @property
    def perm_p_beta(self) -> float:
        """Function summary: alias for coefficient-based permutation p."""
        return self.p

    @property
    def perm_p_t(self) -> float:
        """Function summary: alias for t-statistic permutation p."""
        return self.p_t


def entity_country_id(row: pd.Series) -> str:
    """Function summary: map one panel row to country hub id (it, de, eu, uk, us).

    Parameters:
    - row: panel row with language_hub, topic_family, or IT.

    Returns:
    - Country id string.
    """
    if "language_hub" in row.index and pd.notna(row.get("language_hub")):
        hub = str(row["language_hub"])
        if hub == "it" or hub in ITALY_FAMILIES:
            return "it"
        if hub in CONTROL_FAMILIES:
            return hub
    if "topic_family" in row.index and pd.notna(row.get("topic_family")):
        fam = str(row["topic_family"])
        if fam in ITALY_FAMILIES:
            return "it"
        if fam in CONTROL_FAMILIES:
            return fam
    if "IT" in row.index and int(row.get("IT", 0) or 0) == 1:
        return "it"
    return ""


def assign_entity_country_series(
    df: pd.DataFrame,
    entity_col: str = "entity_id",
) -> pd.Series:
    """Function summary: per-entity country id from first row per entity.

    Parameters:
    - df: panel with country columns.
    - entity_col: entity identifier column.

    Returns:
    - Series indexed like df with country id per row.
    """
    ent = df[entity_col].astype(str)
    meta = df.assign(_ent=ent).groupby("_ent", observed=True).first()
    cmap = {str(k): entity_country_id(meta.loc[k]) for k in meta.index}
    return ent.map(cmap).astype(str)


def _is_italy_country(country: str) -> bool:
    """Function summary: True if country id is Italian."""
    return country == "it" or country in ITALY_FAMILIES


def _control_countries_in_sample(
    sample: pd.DataFrame,
    entity_col: str,
    strategy: StrategySpec,
) -> List[str]:
    """Function summary: control country ids present after dropping Italy."""
    countries = assign_entity_country_series(sample, entity_col)
    present = sorted({c for c in countries.unique() if c in CONTROL_FAMILIES})
    if strategy.control_family and strategy.control_family in present:
        return [strategy.control_family]
    return present


def _apply_placebo_treat(
    controls: pd.DataFrame,
    fake_country: str,
    entity_col: str,
) -> pd.DataFrame:
    """Function summary: set treat=1 for fake_country entities, 0 otherwise (control-only sample)."""
    out = controls.copy()
    ent_country = assign_entity_country_series(out, entity_col)
    treat = (ent_country == fake_country).astype(int)
    out["treat"] = treat
    if treat.nunique() < 2:
        raise ValueError(f"placebo {fake_country}: no treat variation in control sample")
    return out


def placebo_in_space_p(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    window_days: Optional[int] = None,
) -> PlaceboInSpaceResult:
    """Function summary: reassign treatment to each control country vs remaining controls (no Italy).

    Real Italy estimate uses filter_strategy_sample; placebos drop Italy and rotate fake-treated country.
    p = (#{|β_placebo| >= |β_Italy|}) / (n_control_countries + 1); floor = 1/(n_control+1).

    Parameters:
    - panel: analysis panel.
    - strategy: cross-country strategy spec.
    - y_col: outcome column.
    - entity_col, time_col: panel keys.
    - window_days: optional event window passed to filter_strategy_sample.

    Returns:
    - PlaceboInSpaceResult with p-value and metadata.
    """
    def _empty_pis() -> PlaceboInSpaceResult:
        return PlaceboInSpaceResult(
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            (),
            (),
        )

    if not is_cross_country_strategy(strategy.strategy_id):
        return _empty_pis()

    italy_sample = filter_strategy_sample(panel, strategy, window_days=window_days)
    if italy_sample.empty or italy_sample["treat"].nunique() < 2:
        return _empty_pis()

    base = estimate_twfe(italy_sample, y_col, entity_col, time_col)
    b_italy = base.get("beta", np.nan)
    se_italy = base.get("se", np.nan)
    t_italy = float(b_italy / se_italy) if se_italy and se_italy > 0 and np.isfinite(b_italy) else float("nan")
    if not np.isfinite(b_italy):
        return _empty_pis()

    ent_country = assign_entity_country_series(italy_sample, entity_col)
    controls = italy_sample[~ent_country.map(_is_italy_country)].copy()
    if controls.empty:
        return PlaceboInSpaceResult(
            float("nan"), float("nan"), float("nan"), 0, b_italy, t_italy, (), ()
        )

    control_countries = _control_countries_in_sample(controls, entity_col, strategy)
    if not control_countries:
        return PlaceboInSpaceResult(
            float("nan"), float("nan"), float("nan"), 0, b_italy, t_italy, (), ()
        )

    placebo_betas: List[float] = []
    placebo_ts: List[float] = []
    baseline_treat = controls["treat"].astype(int).tolist()
    for fake_c in control_countries:
        try:
            pl = _apply_placebo_treat(controls, fake_c, entity_col)
        except ValueError:
            continue
        assert pl["treat"].astype(int).tolist() != baseline_treat, (
            f"placebo-in-space degenerate for {fake_c}: treat unchanged"
        )
        r = estimate_twfe(pl, y_col, entity_col, time_col)
        b = r.get("beta", np.nan)
        se = r.get("se", np.nan)
        if np.isfinite(b):
            placebo_betas.append(float(b))
            if se and se > 0:
                placebo_ts.append(float(b / se))

    n_placebo = len(control_countries)
    p_floor = 1.0 / (n_placebo + 1)
    if not placebo_betas:
        return PlaceboInSpaceResult(
            float("nan"), float("nan"), p_floor, n_placebo, b_italy, t_italy, tuple(), tuple()
        )

    n_ge_b = sum(1 for b in placebo_betas if abs(b) >= abs(b_italy))
    p_beta = float((n_ge_b + 1) / (n_placebo + 1))
    p_beta = min(1.0, max(p_floor, p_beta))

    p_t = float("nan")
    if np.isfinite(t_italy) and placebo_ts:
        n_ge_t = sum(1 for t in placebo_ts if abs(t) >= abs(t_italy))
        p_t = float((n_ge_t + 1) / (len(placebo_ts) + 1))
        p_t = min(1.0, max(p_floor, p_t))

    return PlaceboInSpaceResult(
        p_beta,
        p_t,
        p_floor,
        n_placebo,
        float(b_italy),
        t_italy,
        tuple(placebo_betas),
        tuple(placebo_ts),
    )


def permutation_test_p(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    n_perm: int = 199,
    seed: int = 42,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> float:
    """Function summary: alias for placebo-in-space p (stratum permutation removed).

    Parameters:
    - panel, strategy, y_col: passed to placebo_in_space_p.
    - n_perm, seed: ignored (deterministic placebos); kept for API compatibility.

    Returns:
    - Two-sided placebo-in-space p-value.
    """
    del n_perm, seed
    warnings.warn(
        "permutation_test_p now uses placebo-in-space (not within-stratum permutation).",
        DeprecationWarning,
        stacklevel=2,
    )
    return placebo_in_space_p(
        panel, strategy, y_col, entity_col=entity_col, time_col=time_col
    ).p


def _prep_twfe_feols(
    sample: pd.DataFrame,
    y_col: str,
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Function summary: long data for pyfixest TWFE wild bootstrap."""
    work = sample[[entity_col, time_col, y_col, "treat", "post"]].copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    work[entity_col] = work[entity_col].astype(str)
    work[time_col] = work[time_col].astype(str)
    work["treat_post"] = work["treat"].astype(float) * work["post"].astype(float)
    return work


def _prep_ddd_feols(
    sample: pd.DataFrame,
    y_col: str,
    entity_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, str]:
    """Function summary: long data for within-Italy DDD pyfixest mirror (IT constant)."""
    work = sample.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y"])
    if "political_universe" not in work.columns:
        work["political_universe"] = (
            work["universe_slice"].astype(str) == "in_political_tree"
        ).astype(float)
    work["post_pol"] = work["post"].astype(float) * work["political_universe"].astype(float)
    work[entity_col] = work[entity_col].astype(str)
    work[time_col] = work[time_col].astype(str)
    return work[[entity_col, time_col, "y", "post_pol", "political_universe"]], "post_pol"


def _coef_from_fit(fit: Any, param: str) -> float:
    """Function summary: extract one coefficient from pyfixest fit."""
    coefs = fit.coef()
    if param in coefs.index:
        return float(coefs.loc[param])
    return float("nan")


def _drop_singleton_entities(work: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Function summary: drop entities with one row so pyfixest resid aligns with data."""
    counts = work.groupby(entity_col, observed=True).size()
    keep = counts[counts >= 2].index.astype(str)
    out = work[work[entity_col].astype(str).isin(keep)].copy()
    return out


def _rademacher_weight_matrix(n_clust: int, n_draws: int, seed: int) -> Tuple[np.ndarray, bool, float]:
    """Function summary: bootstrap sign weights; enumerate when G is small.

    Parameters:
    - n_clust: number of clusters G.
    - n_draws: requested random draws (ignored when enumerating).
    - seed: RNG seed for sampled draws.

    Returns:
    - Tuple (weights matrix shape (n_reps, G), enumerated flag, p_floor).
    """
    enum_threshold = 12
    if n_clust <= enum_threshold:
        n_enum = 2**n_clust
        bits = np.arange(n_enum, dtype=np.uint64)[:, None] >> np.arange(n_clust, dtype=np.uint64)
        signs = np.where(bits & 1, 1.0, -1.0).astype(np.float64)
        return signs, True, 1.0 / n_enum
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_draws, n_clust)), False, float("nan")


def _restricted_wcb_pyfixest(
    formula_full: str,
    formula_restricted: str,
    data: pd.DataFrame,
    param: str,
    cluster_col: str,
    n_draws: int,
    seed: int,
    min_draws: int = 20,
) -> float:
    """Function summary: restricted wild cluster bootstrap (H0 on param; Rademacher clusters).

    Rebuilds y* = fitted_restricted + v_g * resid_restricted per Cameron–Gelbach–Miller / Roodman.
    Uses pyfixest feols (wildboottest fails on absorbed FE designs).
    When G <= 12, enumerates all 2^G sign vectors (deterministic/exact).
    """
    result = _restricted_wcb_pyfixest_detail(
        formula_full,
        formula_restricted,
        data,
        param,
        cluster_col,
        n_draws,
        seed,
        min_draws=min_draws,
    )
    return float(result.get("pvalue", float("nan")))


def _restricted_wcb_pyfixest_detail(
    formula_full: str,
    formula_restricted: str,
    data: pd.DataFrame,
    param: str,
    cluster_col: str,
    n_draws: int,
    seed: int,
    min_draws: int = 20,
) -> Dict[str, Any]:
    """Function summary: restricted WCB with p-value floor and enumeration metadata.

    Parameters:
    - formula_full: unrestricted pyfixest formula.
    - formula_restricted: null-imposed formula.
    - data: estimation sample with y and cluster_col.
    - param: coefficient name under test.
    - cluster_col: cluster column.
    - n_draws: bootstrap replications when G > 12.
    - seed: RNG seed.
    - min_draws: minimum successful draws required.

    Returns:
    - Dict with pvalue, p_floor, enumerated, n_clusters, n_draws_used, beta.
    """
    empty: Dict[str, Any] = {
        "pvalue": float("nan"),
        "p_floor": float("nan"),
        "enumerated": False,
        "n_clusters": 0,
        "n_draws_used": 0,
        "beta": float("nan"),
    }
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return empty
    if len(data) < 30 or data[cluster_col].nunique() < 2:
        return empty
    try:
        data = _drop_singleton_entities(data, cluster_col)
        if len(data) < 30 or data[cluster_col].nunique() < 2:
            return empty
        fit_u = feols(formula_full, data=data, vcov={"CRV1": cluster_col})
        fit_r = feols(formula_restricted, data=data, vcov={"CRV1": cluster_col})
        b0 = _coef_from_fit(fit_u, param)
        if not np.isfinite(b0):
            return empty
        y_hat = np.asarray(fit_r.predict(), dtype=np.float64)
        resid = np.asarray(fit_r.resid(), dtype=np.float64)
        codes, _ = pd.factorize(data[cluster_col].astype(str))
        n_clust = int(codes.max()) + 1 if len(codes) else 0
        if n_clust < 2:
            return empty
        weight_matrix, enumerated, p_floor = _rademacher_weight_matrix(n_clust, n_draws, seed)
        betas: List[float] = []
        base = data.copy()
        for row in weight_matrix:
            boot = base.copy()
            boot["y"] = y_hat + resid * row[codes]
            fit_b = feols(formula_full, data=boot, vcov={"CRV1": cluster_col})
            b = _coef_from_fit(fit_b, param)
            if np.isfinite(b):
                betas.append(b)
        min_required = 1 if enumerated else min_draws
        if len(betas) < min_required:
            return empty
        arr = np.asarray(betas)
        p = float(min(1.0, max(0.0, np.mean(np.abs(arr) >= abs(b0)))))
        if enumerated and np.isfinite(p_floor):
            p = max(p_floor, p)
        return {
            "pvalue": p,
            "p_floor": p_floor if enumerated else float("nan"),
            "enumerated": enumerated,
            "n_clusters": n_clust,
            "n_draws_used": int(len(betas)),
            "beta": float(b0),
        }
    except Exception as exc:
        logger.debug("restricted WCB failed: %s", exc)
    return empty


def wild_cluster_bootstrap_p(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    n_draws: int = DEFAULT_WCB_DRAWS,
    seed: int = 42,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
    window_days: Optional[int] = None,
) -> float:
    """Function summary: restricted wild cluster bootstrap p for eligible strategies only.

    Parameters:
    - panel: analysis panel.
    - strategy: strategy spec (not cross_country_*).
    - y_col: outcome column.
    - n_draws: bootstrap replications (default 9999).
    - seed: RNG seed.
    - entity_col, time_col: panel keys.
    - window_days: optional window for filter_strategy_sample.

    Returns:
    - Two-sided WCB p-value, or NaN if ineligible or estimation fails.
    """
    if is_cross_country_strategy(strategy.strategy_id):
        logger.warning(
            "wild_cluster_bootstrap_p skipped for %s: one treated country cluster "
            "(MacKinnon–Webb); use placebo-in-space instead.",
            strategy.strategy_id,
        )
        return float("nan")

    if not is_wcb_eligible_strategy(strategy.strategy_id):
        return float("nan")

    if strategy.strategy_id == "within_italy_ddd":
        sample = panel[panel["IT"].astype(int) == 1].copy() if "IT" in panel.columns else panel.copy()
        if sample.empty:
            return float("nan")
        data, param = _prep_ddd_feols(sample, y_col, entity_col, time_col)
        full = f"y ~ post_pol + political_universe | {entity_col} + {time_col}"
        rest = f"y ~ political_universe | {entity_col} + {time_col}"
        return _restricted_wcb_pyfixest(full, rest, data, param, entity_col, n_draws, seed)

    sample = filter_strategy_sample(panel, strategy, window_days=window_days)
    if sample.empty:
        return float("nan")

    author_col = "author" if "author" in sample.columns else entity_col
    cluster_col = author_col

    if is_entity_fe_only_strategy(strategy.strategy_id):
        work = _prep_twfe_feols(sample, y_col, entity_col, time_col)
        if work.empty or work[entity_col].nunique() < 3:
            return float("nan")
        full = f"y ~ post | {entity_col}"
        rest = f"y ~ 1 | {entity_col}"
        return _restricted_wcb_pyfixest(full, rest, work, "post", cluster_col, n_draws, seed)

    work = _prep_twfe_feols(sample, y_col, entity_col, time_col)
    if work.empty or work["treat_post"].nunique() < 2:
        return float("nan")
    full = f"y ~ treat_post | {entity_col} + {time_col}"
    rest = f"y ~ 1 | {entity_col} + {time_col}"
    return _restricted_wcb_pyfixest(full, rest, work, "treat_post", cluster_col, n_draws, seed)


def wild_cluster_bootstrap_cross_language_static(
    df: pd.DataFrame,
    treat_col: str = "is_english",
    y_col: str = "y",
    cluster_col: str = "subreddit",
    n_draws: int = DEFAULT_WCB_DRAWS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Function summary: subreddit WCB for cross_language static post:EN during ban.

    Full: y ~ post + IT + post_IT | author + time_id
    Restricted: y ~ post + IT | author + time_id
    Uses adaptive Rademacher enumeration when G <= 12.

    Parameters:
    - df: comment panel with post, treat_col, author, time_id, subreddit, outcome.
    - treat_col: language treatment column (is_english).
    - y_col: outcome column.
    - cluster_col: subreddit cluster column.
    - n_draws: bootstrap draws when G > 12.
    - seed: RNG seed.

    Returns:
    - Dict with pvalue, p_floor, enumerated, n_clusters, beta, estimation_note.
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author", "time_id"])
    work["author"] = work["author"].astype(str)
    work["time_id"] = work["time_id"].astype(str)
    work["post"] = work["post"].astype(float)
    work["IT"] = pd.to_numeric(work[treat_col], errors="coerce").fillna(0).astype(float)
    work["post_IT"] = work["post"] * work["IT"]
    if cluster_col not in work.columns or len(work) < 30:
        return {
            "pvalue": float("nan"),
            "p_floor": float("nan"),
            "enumerated": False,
            "n_clusters": 0,
            "beta": float("nan"),
            "estimation_note": "insufficient_obs",
        }
    full = "y ~ post + IT + post_IT | author + time_id"
    rest = "y ~ post + IT | author + time_id"
    detail = _restricted_wcb_pyfixest_detail(
        full, rest, work, "post_IT", cluster_col, n_draws, seed
    )
    detail["estimation_note"] = "ok" if np.isfinite(detail.get("pvalue", np.nan)) else "wcb_failed"
    return detail


def wild_cluster_bootstrap_static_prepped(
    work: pd.DataFrame,
    b0: float,
    n_draws: int = DEFAULT_WCB_DRAWS,
    seed: int = 42,
    cluster_col: str = "subreddit",
    *,
    cross_country_it: bool = True,
) -> float:
    """Function summary: restricted WCB for pyfixest static post:IT (bucket / comment).

    Parameters:
    - work: prep_static_design output.
    - b0: baseline coefficient (unused; null imposed in bootstrap).
    - n_draws: bootstrap replications.
    - seed: RNG seed.
    - cluster_col: cluster column.
    - cross_country_it: if True, return NaN (IT vs controls = one treated cluster).

    Returns:
    - Two-sided WCB p-value.
    """
    del b0
    if cross_country_it:
        logger.warning(
            "wild_cluster_bootstrap_static_prepped skipped for cross-country post:IT; "
            "use placebo-in-space."
        )
        return float("nan")
    if cluster_col not in work.columns or len(work) < 30:
        return float("nan")
    return _restricted_wcb_pyfixest(
        "y ~ post + post_IT | author",
        "y ~ post | author",
        work,
        "post_IT",
        cluster_col,
        n_draws,
        seed,
    )


def _comment_placebo_country_col(df: pd.DataFrame) -> Optional[str]:
    """Function summary: column used to assign control countries on comment panels."""
    if "topic_family" in df.columns:
        return "topic_family"
    if "language_hub" in df.columns:
        return "language_hub"
    return None


def placebo_in_space_comment_p(
    df: pd.DataFrame,
    y_col: str = "y",
    cluster_col: str = "subreddit",
    author_col: str = "author",
    time_col: str = "time_id",
    beta_italy: Optional[float] = None,
) -> float:
    """Function summary: placebo-in-space on comment panel for post:IT (Italy vs controls).

    Matches headline static spec ``y ~ post + post_IT | author`` (pyfixest), not TWFE.

    Parameters:
    - df: comment panel with IT, post, topic_family or language_hub, outcome.
    - y_col: outcome column name in df.
    - cluster_col, author_col, time_col: design columns.
    - beta_italy: optional Italy beta from estimate_static_paper_eq1 (skips refit).

    Returns:
    - Placebo-in-space p-value for Italy vs pooled controls.
    """
    from src.did.bucket_estimate import feols_static_paper_eq1_prepped, prep_static_design

    country_col = _comment_placebo_country_col(df)
    if country_col is None:
        return float("nan")

    cluster_use = cluster_col if cluster_col in df.columns else author_col
    b_italy = beta_italy
    if b_italy is None or not np.isfinite(b_italy):
        work_it = prep_static_design(df, y_col, cluster_use)
        b_italy = feols_static_paper_eq1_prepped(work_it, cluster_use).get("beta", np.nan)
        del work_it
    if not np.isfinite(b_italy):
        return float("nan")

    y_name = y_col if y_col in df.columns else "y"
    keep = [y_name, "post", "IT", author_col, time_col, country_col]
    if cluster_use in df.columns:
        keep.append(cluster_use)
    controls = df.loc[df["IT"].astype(int) == 0, [c for c in keep if c in df.columns]]
    if controls.empty:
        return float("nan")

    countries = sorted(CONTROL_FAMILIES)
    placebo_betas: List[float] = []
    pl = controls[[c for c in controls.columns]].copy()
    baseline_it = pl["IT"].astype(int).tolist()
    country_vals = pl[country_col].astype(str)
    for fake_c in countries:
        it_fake = (country_vals == fake_c).astype(float)
        new_it = it_fake.astype(int).tolist()
        if new_it == baseline_it:
            continue
        assert new_it != baseline_it, f"degenerate comment placebo for {fake_c}"
        pl["IT"] = it_fake
        w = prep_static_design(pl, y_name, cluster_use)
        r = feols_static_paper_eq1_prepped(w, cluster_use)
        del w
        gc.collect()
        if np.isfinite(r.get("beta", np.nan)):
            placebo_betas.append(float(r["beta"]))
    n_placebo = len(countries)
    if not placebo_betas:
        return float("nan")
    n_ge = sum(1 for b in placebo_betas if abs(b) >= abs(b_italy))
    p_floor = 1.0 / (n_placebo + 1)
    return min(1.0, max(p_floor, (n_ge + 1) / (n_placebo + 1)))


def wild_cluster_bootstrap_comment_coef(
    df: pd.DataFrame,
    fit_fn: Any,
    coef_name: str = "post_IT",
    cluster_col: str = "subreddit",
    n_draws: int = DEFAULT_WCB_DRAWS,
    seed: int = 42,
    y_col: str = "y",
) -> float:
    """Function summary: WCB p for comment static coef (cross-country uses placebo, not WCB).

    Parameters:
    - df: comment panel.
    - fit_fn: unused; kept for API compatibility.
    - coef_name: target coefficient name.
    - cluster_col: cluster column.
    - n_draws, seed: bootstrap settings.
    - y_col: outcome column.

    Returns:
    - p-value (NaN for cross-country post:IT).
    """
    del fit_fn, coef_name
    work = df
    from src.did.bucket_estimate import prep_static_design

    prepped = prep_static_design(work, y_col, cluster_col)
    return wild_cluster_bootstrap_static_prepped(
        prepped,
        float("nan"),
        n_draws=n_draws,
        seed=seed,
        cluster_col=cluster_col,
        cross_country_it=True,
    )
