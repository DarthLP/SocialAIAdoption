"""
Script summary:
Validation metrics for LLM index leave-one-out and marginal (single-feature) robustness.

Functionality:
- Leave-one-out: ρ(primary, index without feature f); capture among joint hits.
- Marginal: ρ(feature, primary), ρ(only_f index, primary), lift in primary when feature hit.
- Ranks frozen-weight candidates (relative weights may exceed 1.0).

How to apply/run:
- Imported by scripts/diagnostics/validate_style_index_llm_ablations.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.style_index_llm import (
    ABLATION_DROP_FEATURES,
    LLM_BASE_FEATURES,
    ONLY_FEATURES,
    PRIMARY_COL,
    ablation_column_name,
    only_column_name,
)

# Map feature -> binary hit column on prepared validation frame.
FEATURE_HIT_COL: Dict[str, str] = {
    "ai_style_rate_100w": "ai_hit",
    "em_dash_rate_100w": "em_dash_hit",
    "semicolon_colon_rate_100w": "semicolon_hit",
    "hedging_phrase_rate_100w": "hedging_hit",
    "exclamation_rate_100w": "exclamation_hit",
    "caps_word_share": "caps_hit",
}

RHO_MARGINAL_MIN = 0.15
LIFT_PRIMARY_MIN = 1.05
from src.style_index_validation import MIN_PAIRS, spearman_corr

CAPTURE_RATIO_MIN = 0.65
RHO_REDUNDANCY_MIN = 0.85


def _finite_series(s: pd.Series) -> pd.Series:
    """Function summary: numeric series with NaNs dropped for pairing."""
    return pd.to_numeric(s, errors="coerce")


def spearman_pair(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Function summary: Spearman between two series."""
    return spearman_corr(_finite_series(a), _finite_series(b))


def capture_ratio(
    df: pd.DataFrame,
    *,
    primary_col: str,
    ablated_col: str,
    mask: pd.Series,
) -> float:
    """Function summary: mean(ablated)/mean(primary) among masked rows.

    Parameters:
    - df: frame with index columns.
    - primary_col, ablated_col: column names.
    - mask: boolean row selector (e.g. semicolon_hit & ai_hit).

    Returns:
    - Ratio or NaN.
    """
    sub = df.loc[mask, [primary_col, ablated_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < MIN_PAIRS:
        return float("nan")
    p = float(sub[primary_col].mean())
    a = float(sub[ablated_col].mean())
    if not np.isfinite(p) or p == 0:
        return float("nan")
    return float(a / p)


def ablation_metric_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    primary_col: str = PRIMARY_COL,
    ablation_cols: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Function summary: one row per ablation with redundancy and capture metrics.

    Parameters:
    - df: prepared validation frame (hits, length bins).
    - subset: label for output rows.
    - primary_col: full composite column.
    - ablation_cols: leave-one-out columns; default inferred from ABLATION_DROP_FEATURES.

    Returns:
    - List of metric dicts for CSV export.
    """
    if primary_col not in df.columns:
        return []
    cols = list(ablation_cols or [ablation_column_name(d) for d in ABLATION_DROP_FEATURES])
    cols = [c for c in cols if c in df.columns]
    rows: List[Dict[str, Any]] = []
    ai = _finite_series(df.get("ai_style_rate_100w", pd.Series(dtype=float)))

    for col in cols:
        ab = _finite_series(df[col])
        rho_red, n_red = spearman_pair(df[primary_col], ab)
        rho_ai, n_ai = spearman_pair(ab, ai)
        drop = col.replace("style_index_llm_no_", "")
        cap_semi = float("nan")
        cap_em = float("nan")
        cap_ai = float("nan")
        if "semicolon_hit" in df.columns and "ai_hit" in df.columns:
            m = (df["semicolon_hit"] == 1) & (df["ai_hit"] == 1)
            cap_semi = capture_ratio(df, primary_col=primary_col, ablated_col=col, mask=m)
        if "em_dash_hit" in df.columns and "ai_hit" in df.columns:
            m = (df["em_dash_hit"] == 1) & (df["ai_hit"] == 1)
            cap_em = capture_ratio(df, primary_col=primary_col, ablated_col=col, mask=m)
        if "ai_hit" in df.columns:
            m = df["ai_hit"] == 1
            cap_ai = capture_ratio(df, primary_col=primary_col, ablated_col=col, mask=m)
        rows.append(
            {
                "subset": subset,
                "dropped_feature": drop,
                "ablation_col": col,
                "rho_primary_vs_ablated": rho_red,
                "rho_ablated_vs_ai_rate": rho_ai,
                "n_pairs_redundancy": n_red,
                "capture_semicolon_and_ai_hit": cap_semi,
                "capture_em_dash_and_ai_hit": cap_em,
                "capture_ai_hit": cap_ai,
                "pass_redundancy": bool(np.isfinite(rho_red) and rho_red >= RHO_REDUNDANCY_MIN),
                "pass_capture_semicolon": bool(
                    np.isfinite(cap_semi) and cap_semi >= CAPTURE_RATIO_MIN
                ),
            }
        )
    return rows


def feature_rate_vs_own_ablation_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    features: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Function summary: ρ(feature_rate, style_index_llm_no_<feature>) for each feature.

    Tests whether the ablated index (with that feature removed from the composite)
    still correlates with the raw feature rate — should be low for features that
    truly enter the index (e.g. semicolon removed → ρ(semicolon, index_no_semi) drops).

    Parameters:
    - df: prepared validation frame with ablation columns.
    - subset: label for CSV rows.
    - features: feature names to test (default ABLATION_DROP_FEATURES in df).

    Returns:
    - One row per feature with Spearman rho and n pairs.
    """
    rows: List[Dict[str, Any]] = []
    feat_list = list(features or ABLATION_DROP_FEATURES)
    for feat in feat_list:
        if feat not in df.columns:
            continue
        ab_col = ablation_column_name(feat)
        if ab_col not in df.columns:
            continue
        rate = _finite_series(df[feat])
        ab = _finite_series(df[ab_col])
        rho, n = spearman_pair(rate, ab)
        rho_pri, _ = spearman_pair(rate, df[PRIMARY_COL]) if PRIMARY_COL in df.columns else (float("nan"), 0)
        rows.append(
            {
                "subset": subset,
                "feature": feat,
                "ablation_col": ab_col,
                "rho_feature_vs_own_ablation": rho,
                "rho_feature_vs_primary": rho_pri,
                "n_pairs": n,
                "delta_rho_primary_minus_ablation": (
                    float(rho_pri - rho) if np.isfinite(rho_pri) and np.isfinite(rho) else float("nan")
                ),
                "pass_nonnegative_own_ablation": bool(np.isfinite(rho) and rho >= 0),
            }
        )
    return rows


def feature_rate_vs_ablation_matrix_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    ablation_col: str,
    features: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Function summary: ρ(each feature_rate, one fixed ablation index column).

    Example: ρ(semicolon_rate, style_index_llm_no_semicolon_colon) plus same for
    ai_rate, em_dash_rate, etc. — all variables against one leave-one-out index.

    Parameters:
    - df: prepared frame.
    - subset: row label.
    - ablation_col: e.g. style_index_llm_no_semicolon_colon.
    - features: rates to correlate (default LLM_BASE_FEATURES in df).

    Returns:
    - One row per feature rate.
    """
    if ablation_col not in df.columns:
        return []
    ab = _finite_series(df[ablation_col])
    rows: List[Dict[str, Any]] = []
    for feat in list(features or LLM_BASE_FEATURES):
        if feat not in df.columns:
            continue
        rate = _finite_series(df[feat])
        rho, n = spearman_pair(rate, ab)
        rows.append(
            {
                "subset": subset,
                "ablation_col": ablation_col,
                "feature": feat,
                "rho_feature_vs_ablation": rho,
                "n_pairs": n,
            }
        )
    return rows


def feature_rate_vs_all_ablations_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    features: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Function summary: full matrix — each feature rate vs each leave-one-out index."""
    rows: List[Dict[str, Any]] = []
    ab_cols = [ablation_column_name(d) for d in ABLATION_DROP_FEATURES if ablation_column_name(d) in df.columns]
    for ab_col in ab_cols:
        rows.extend(
            feature_rate_vs_ablation_matrix_rows(
                df, subset=subset, ablation_col=ab_col, features=features
            )
        )
    return rows


def marginal_influence_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    primary_col: str = PRIMARY_COL,
    features: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Function summary: reverse robustness — does each feature move the composite?

    Parameters:
    - df: prepared frame with primary, feature rates, hit flags, only_* columns.
    - subset: row label.
    - primary_col: full LLM index.
    - features: features to test (default ONLY_FEATURES present in df).

    Returns:
    - Rows with ρ(feature, primary), ρ(only_feature, primary), lift when hit, etc.
    """
    if primary_col not in df.columns:
        return []
    feat_list = list(features or ONLY_FEATURES)
    primary = _finite_series(df[primary_col])
    rows: List[Dict[str, Any]] = []
    for feat in feat_list:
        if feat not in df.columns:
            continue
        rate = _finite_series(df[feat])
        rho_fp, n_fp = spearman_pair(rate, primary)
        only_col = only_column_name(feat)
        rho_only = float("nan")
        n_only = 0
        if only_col in df.columns:
            rho_only, n_only = spearman_pair(df[only_col], primary)
        hit_col = FEATURE_HIT_COL.get(feat)
        delta_hit = float("nan")
        lift = float("nan")
        n_hit = 0
        if hit_col and hit_col in df.columns:
            hit = df[hit_col].fillna(0).astype(int) == 1
            n_hit = int(hit.sum())
            if n_hit >= 10 and (~hit).sum() >= 10:
                m_hit = float(primary[hit].mean())
                m_no = float(primary[~hit].mean())
                delta_hit = m_hit - m_no
                if np.isfinite(m_no) and m_no != 0:
                    lift = m_hit / m_no
        rows.append(
            {
                "subset": subset,
                "feature": feat,
                "only_col": only_col if only_col in df.columns else "",
                "rho_feature_vs_primary": rho_fp,
                "rho_only_feature_vs_primary": rho_only,
                "n_pairs": n_fp,
                "n_feature_hit": n_hit,
                "delta_primary_hit_minus_miss": delta_hit,
                "lift_primary_hit_over_miss": lift,
                "pass_marginal_rho": bool(np.isfinite(rho_fp) and rho_fp >= RHO_MARGINAL_MIN),
                "pass_only_vs_primary": bool(
                    np.isfinite(rho_only) and rho_only >= RHO_MARGINAL_MIN
                ),
                "pass_lift": bool(np.isfinite(lift) and lift >= LIFT_PRIMARY_MIN),
            }
        )
    return rows


def score_candidate(
    df: pd.DataFrame,
    *,
    primary_col: str = PRIMARY_COL,
    signs: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Function summary: scalar score for ranking frozen-weight candidates (higher = better).

    Parameters:
    - df: IT ge20 validation frame with primary + ablation columns.

    Returns:
    - Dict with score components and total score.
    """
    rho_ai, n_ai = spearman_pair(df[primary_col], df.get("ai_style_rate_100w", pd.Series(dtype=float)))
    rho_len, _ = spearman_pair(df[primary_col], df.get("log_len", pd.Series(dtype=float)))
    score = 0.0
    if np.isfinite(rho_ai):
        score += float(rho_ai) * 2.0
    if np.isfinite(rho_len) and abs(rho_len) < 0.5:
        score += 0.5

    ab_rows = ablation_metric_rows(df, primary_col=primary_col)
    for row in ab_rows:
        drop = str(row["dropped_feature"])
        rho_red = row.get("rho_primary_vs_ablated", float("nan"))
        if "ai_style" in drop:
            if np.isfinite(rho_red) and rho_red < 0.95:
                score += 1.0
        else:
            if np.isfinite(rho_red) and rho_red >= RHO_REDUNDANCY_MIN:
                score += 1.5
                cap = row.get("capture_semicolon_and_ai_hit", float("nan"))
                if "semicolon" in drop and np.isfinite(cap) and cap >= CAPTURE_RATIO_MIN:
                    score += 1.0
                cap_em = row.get("capture_em_dash_and_ai_hit", float("nan"))
                if "em_dash" in drop and np.isfinite(cap_em) and cap_em >= CAPTURE_RATIO_MIN:
                    score += 1.0

    for mrow in marginal_influence_rows(df, primary_col=primary_col):
        feat = str(mrow.get("feature", ""))
        if feat == "ai_style_rate_100w":
            continue
        if mrow.get("pass_marginal_rho"):
            score += 0.75
        if mrow.get("pass_only_vs_primary"):
            score += 0.5
        if mrow.get("pass_lift") and mrow.get("n_feature_hit", 0) >= 20:
            score += 0.5

    sign_map = dict(signs or {})
    own_rows = feature_rate_vs_own_ablation_rows(df)
    n_negative_own = 0
    min_own_rho = float("nan")
    for row in own_rows:
        feat = str(row.get("feature", ""))
        feat_sign = int(sign_map.get(feat, 1))
        rho_own = row.get("rho_feature_vs_own_ablation", float("nan"))
        if np.isfinite(rho_own):
            if not np.isfinite(min_own_rho) or rho_own < min_own_rho:
                min_own_rho = float(rho_own)
        if feat_sign > 0 and np.isfinite(rho_own) and rho_own < 0:
            score -= 8.0
            n_negative_own += 1
        elif feat_sign > 0 and np.isfinite(rho_own) and rho_own < 0.05:
            score -= 1.0
        elif feat_sign > 0 and np.isfinite(rho_own) and rho_own >= 0.05:
            score += 0.4

    return {
        "score": score,
        "rho_primary_vs_ai_rate": rho_ai,
        "rho_primary_vs_log_len": rho_len,
        "n_pairs_ai": n_ai,
        "n_ablation_rows": len(ab_rows),
        "n_negative_own_ablation": n_negative_own,
        "min_rho_feature_vs_own_ablation": min_own_rho,
        "pass_own_ablation_nonneg": n_negative_own == 0,
    }
