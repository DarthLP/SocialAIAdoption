"""
Script summary:
Pure metrics for formula style-index construct validation (no scoring changes).

Functionality:
- Spearman/Pearson correlations, length-stratified AI-hit separation, optional partial correlation.
- Used by scripts/diagnostics/validate_style_index_gates.py and unit tests.

How to apply/run:
- Imported only; run validate_style_index_gates.py after compute_style_index_on_shards.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.comment_style import resolve_em_dash_count

MIN_PAIRS = 30
SPEARMAN_AI_STYLE_HEURISTIC = 0.3
SPEARMAN_AI_STYLE_V2_IT = 0.15
LENGTH_CONFOUND_REVIEW = 0.7
LENGTH_CONFOUND_V2_REVIEW = 0.5
INDEX_COL_LLM = "style_index_llm"
LENGTH_BIN_EDGES = (0, 20, 50, 100, 10_000)


def _to_num(s: pd.Series) -> pd.Series:
    """Function summary: coerce series to float with NaN for invalid."""
    return pd.to_numeric(s, errors="coerce")


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, int]:
    """Function summary: Spearman rho and pair count for two aligned series.

    Parameters:
    - x, y: numeric series.

    Returns:
    - Tuple (rho, n_pairs); rho is NaN if n < MIN_PAIRS.
    """
    xv = _to_num(x)
    yv = _to_num(y)
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < MIN_PAIRS:
        return float("nan"), n
    return float(xv[mask].corr(yv[mask], method="spearman")), n


def pearson_corr(x: pd.Series, y: pd.Series) -> Tuple[float, int]:
    """Function summary: Pearson r and pair count for two aligned series."""
    xv = _to_num(x)
    yv = _to_num(y)
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < MIN_PAIRS:
        return float("nan"), n
    return float(xv[mask].corr(yv[mask], method="pearson")), n


def partial_corr_spearman(
    x: pd.Series, y: pd.Series, z: pd.Series
) -> Tuple[float, int]:
    """Function summary: Spearman correlation of OLS residuals of x and y on z.

    Parameters:
    - x, y, z: aligned numeric series (e.g. index, ai_rate, log_len).

    Returns:
    - Tuple (partial_rho, n_pairs).
    """
    frame = pd.DataFrame({"x": _to_num(x), "y": _to_num(y), "z": _to_num(z)}).dropna()
    if len(frame) < MIN_PAIRS:
        return float("nan"), len(frame)

    def _resid(col: str) -> pd.Series:
        zv = frame["z"].values
        cv = frame[col].values
        zc = zv - zv.mean()
        if np.sum(zc * zc) <= 0:
            return pd.Series(np.zeros(len(frame)))
        beta = float(np.sum(zc * (cv - cv.mean())) / np.sum(zc * zc))
        return pd.Series(cv - (cv.mean() + beta * zc))

    rx = _resid("x")
    ry = _resid("y")
    return spearman_corr(rx, ry)


def prepare_validation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add derived columns for validation metrics.

    Parameters:
    - df: comment-level frame with style_index_llm, optional n_words, ai_style_rate_100w.

    Returns:
    - Copy with log_len, ai_hit, length_bin, n_words_filled.
    """
    out = df.copy()
    if "n_words" in out.columns:
        nw = _to_num(out["n_words"])
    else:
        nw = pd.Series(np.nan, index=out.index)
    out["n_words"] = nw
    out["log_len"] = np.log1p(nw.clip(lower=0))
    ai = _to_num(out.get("ai_style_rate_100w", pd.Series(dtype=float)))
    out["ai_style_rate_100w"] = ai
    out["ai_hit"] = (ai > 0).astype(int)
    out[INDEX_COL_LLM] = _to_num(out.get(INDEX_COL_LLM, pd.Series(dtype=float)))
    for col in (
        "style_index_llm_no_ai_style",
        "style_index_llm_no_em_dash",
        "style_index_llm_no_semicolon_colon",
        "style_index_llm_no_hedging_phrase",
        "style_index_llm_no_exclamation",
    ):
        if col in out.columns:
            out[col] = _to_num(out[col])
    em = _to_num(out.get("em_dash_rate_100w", pd.Series(dtype=float)))
    if "em_dash_rate_100w" in out.columns:
        out["em_dash_hit"] = (em > 0).astype(int)
    elif "em_dash_count" in out.columns or "em_dash_extended_count" in out.columns:
        counts = out.apply(
            lambda r: resolve_em_dash_count(
                r.get("em_dash_count"),
                r.get("em_dash_extended_count") if "em_dash_extended_count" in out.columns else None,
            ),
            axis=1,
        )
        out["em_dash_hit"] = (counts > 0).astype(int)
    sc = _to_num(out.get("semicolon_colon_rate_100w", pd.Series(dtype=float)))
    if "semicolon_colon_rate_100w" in out.columns:
        out["semicolon_hit"] = (sc > 0).astype(int)
    if "hedging_phrase_rate_100w" in out.columns:
        out["hedging_hit"] = (_to_num(out["hedging_phrase_rate_100w"]) > 0).astype(int)
    if "exclamation_rate_100w" in out.columns:
        out["exclamation_hit"] = (_to_num(out["exclamation_rate_100w"]) > 0).astype(int)
    if "caps_word_share" in out.columns:
        out["caps_hit"] = (_to_num(out["caps_word_share"]) > 0).astype(int)
    out["length_bin"] = pd.cut(
        nw,
        bins=list(LENGTH_BIN_EDGES),
        labels=["0_19", "20_49", "50_99", "100_plus"],
        right=False,
    )
    nw_pos = nw.fillna(0).clip(lower=0)
    if "hedging_phrase_hits" in out.columns and "hedging_phrase_rate_100w" not in out.columns:
        h = _to_num(out["hedging_phrase_hits"]).fillna(0)
        out["hedging_phrase_rate_100w"] = np.where(nw_pos > 0, 100.0 * h / nw_pos, np.nan)
    if "em_dash_rate_100w" not in out.columns and (
        "em_dash_count" in out.columns or "em_dash_extended_count" in out.columns
    ):
        e = out.apply(
            lambda r: resolve_em_dash_count(
                r.get("em_dash_count"),
                r.get("em_dash_extended_count") if "em_dash_extended_count" in out.columns else None,
            ),
            axis=1,
        )
        out["em_dash_rate_100w"] = np.where(nw_pos > 0, 100.0 * e / nw_pos, np.nan)
    return out


def convergence_correlation_rows(
    df: pd.DataFrame,
    *,
    subset: str = "all",
    index_col: str = INDEX_COL_LLM,
) -> List[Dict[str, Any]]:
    """Function summary: Spearman/Pearson of an index column vs benchmark columns.

    Parameters:
    - df: prepared validation frame.
    - subset: label for this slice (all, n_words_ge_20, lang=it, ...).
    - index_col: outcome column to correlate (default style_index_llm).

    Returns:
    - List of metric rows for CSV export.
    """
    if df.empty or index_col not in df.columns:
        return []
    si = df[index_col]
    benchmarks: Sequence[Tuple[str, str]] = (
        ("ai_style_rate_100w", "lexicon_ai_style_rate"),
        ("log_len", "log_comment_length"),
    )
    for col in (
        "hedging_phrase_rate_100w",
        "em_dash_rate_100w",
        "semicolon_colon_rate_100w",
    ):
        if col in df.columns:
            benchmarks = (*benchmarks, (col, col))
    rows: List[Dict[str, Any]] = []
    for col, label in benchmarks:
        if col not in df.columns:
            continue
        rho, n = spearman_corr(si, df[col])
        r, _ = pearson_corr(si, df[col])
        status = "info"
        if label == "lexicon_ai_style_rate" and np.isfinite(rho):
            status = "pass_heuristic" if rho > SPEARMAN_AI_STYLE_HEURISTIC else "review"
        if label == "log_comment_length" and np.isfinite(rho):
            status = "pass" if abs(rho) < LENGTH_CONFOUND_REVIEW else "review_length_confound"
        rows.append(
            {
                "subset": subset,
                "index_col": index_col,
                "benchmark": label,
                "column": col,
                "spearman": rho,
                "pearson": r,
                "n_pairs": n,
                "status": status,
                "threshold_note": (
                    f"heuristic pass if spearman>{SPEARMAN_AI_STYLE_HEURISTIC}"
                    if label == "lexicon_ai_style_rate"
                    else (
                        f"review if |spearman|>{LENGTH_CONFOUND_REVIEW}"
                        if label == "log_comment_length"
                        else ""
                    )
                ),
            }
        )
    if "log_len" in df.columns:
        prho, pn = partial_corr_spearman(si, df["ai_style_rate_100w"], df["log_len"])
        rows.append(
            {
                "subset": subset,
                "index_col": index_col,
                "benchmark": "ai_style_rate_partial_log_len",
                "column": "ai_style_rate_100w|log_len",
                "spearman": prho,
                "pearson": float("nan"),
                "n_pairs": pn,
                "status": "info",
                "threshold_note": "partial: index vs ai_rate after residualizing log_len",
            }
        )
    return rows


def length_stratified_ai_rows(
    df: pd.DataFrame, *, index_col: str = INDEX_COL_LLM
) -> List[Dict[str, Any]]:
    """Function summary: mean index for ai_hit vs no-hit within length bins.

    Parameters:
    - df: prepared validation frame.

    Returns:
    - Rows per length_bin with mean index by ai_hit group.
    """
    rows: List[Dict[str, Any]] = []
    if df.empty or index_col not in df.columns:
        return rows
    mean_key = f"mean_{index_col}"
    med_key = f"median_{index_col}"
    for bin_label, grp in df.groupby("length_bin", observed=True):
        if grp.empty:
            continue
        for hit, sub in grp.groupby("ai_hit", observed=True):
            si = sub[index_col].dropna()
            rows.append(
                {
                    "length_bin": str(bin_label),
                    "index_col": index_col,
                    "ai_hit": int(hit),
                    "n": len(sub),
                    "n_index_nonnull": int(si.notna().sum()),
                    mean_key: float(si.mean()) if len(si) else float("nan"),
                    med_key: float(si.median()) if len(si) else float("nan"),
                }
            )
        hit1 = grp.loc[grp["ai_hit"] == 1, index_col].dropna()
        hit0 = grp.loc[grp["ai_hit"] == 0, index_col].dropna()
        delta = float("nan")
        if len(hit1) >= 5 and len(hit0) >= 5:
            delta = float(hit1.mean() - hit0.mean())
        rows.append(
            {
                "length_bin": str(bin_label),
                "index_col": index_col,
                "ai_hit": "delta_hit_minus_nohit",
                "n": int(len(grp)),
                "n_index_nonnull": int(grp[index_col].notna().sum()),
                mean_key: delta,
                med_key: float("nan"),
            }
        )
    return rows


def length_stratified_delta(df: pd.DataFrame, bin_label: str, *, index_col: str = INDEX_COL_LLM) -> float:
    """Function summary: hit-minus-no-hit mean index delta for one length_bin."""
    for r in length_stratified_ai_rows(df, index_col=index_col):
        if r.get("length_bin") == bin_label and r.get("ai_hit") == "delta_hit_minus_nohit":
            return float(r.get(f"mean_{index_col}", float("nan")))
    return float("nan")


def by_language_correlation_rows(
    df: pd.DataFrame, *, index_col: str = INDEX_COL_LLM
) -> List[Dict[str, Any]]:
    """Function summary: convergence metrics per primary_lexicon."""
    if "primary_lexicon" not in df.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for lang, grp in df.groupby("primary_lexicon", observed=True):
        rows.extend(convergence_correlation_rows(grp, subset=f"lang_{lang}", index_col=index_col))
    return rows


def compare_indices_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Function summary: pairwise Spearman across style_index_llm and ablation columns."""
    cols = [
        c
        for c in (
            INDEX_COL_LLM,
            "style_index_llm_no_ai_style",
            "style_index_llm_no_em_dash",
            "style_index_llm_no_semicolon_colon",
        )
        if c in df.columns
    ]
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rho, n = spearman_corr(df[a], df[b])
            rows.append({"index_a": a, "index_b": b, "spearman": rho, "n_pairs": n})
    return rows


def joint_signal_bucket_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Function summary: mean style_index_llm by ai_hit x em_dash_hit within length bins."""
    rows: List[Dict[str, Any]] = []
    if df.empty or "em_dash_hit" not in df.columns:
        return rows
    index_cols = [c for c in (INDEX_COL_LLM,) if c in df.columns]
    if not index_cols:
        return rows
    for bin_label, grp in df.groupby("length_bin", observed=True):
        if str(bin_label) not in ("20_49", "50_99", "100_plus"):
            continue
        for (ai_hit, em_hit), sub in grp.groupby(["ai_hit", "em_dash_hit"], observed=True):
            row: Dict[str, Any] = {
                "length_bin": str(bin_label),
                "ai_hit": int(ai_hit),
                "em_dash_hit": int(em_hit),
                "n": int(len(sub)),
            }
            for ic in index_cols:
                s = sub[ic].dropna()
                row[f"mean_{ic}"] = float(s.mean()) if len(s) else float("nan")
            rows.append(row)
    return rows


def build_joint_lex_em_review_sample(df: pd.DataFrame, n_each: int = 20) -> pd.DataFrame:
    """Function summary: high lexicon + em dash in 20-49 word bin for manual review."""
    work = df.copy()
    if "length_bin" not in work.columns:
        work = prepare_validation_frame(work)
    mask = (
        (work["length_bin"].astype(str) == "20_49")
        & (work.get("ai_hit", 0) == 1)
        & (work.get("em_dash_hit", 0) == 1)
    )
    pool = work.loc[mask].dropna(subset=[INDEX_COL_LLM, "ai_style_rate_100w"])
    if pool.empty:
        return pd.DataFrame()
    n_each = min(n_each, len(pool))
    return pool.nlargest(n_each, INDEX_COL_LLM).assign(
        review_bucket="joint_high_lex_em_20_49"
    )


def by_subreddit_summary_rows(
    df: pd.DataFrame, *, top_n: int = 15
) -> List[Dict[str, Any]]:
    """Function summary: subreddit-level mean index and ai rate (all forums kept).

    Parameters:
    - df: prepared frame with subreddit column.
    - top_n: rows each for highest/lowest mean index subreddits.

    Returns:
    - Summary rows for inspection (not exclusion list).
    """
    if "subreddit" not in df.columns or df.empty:
        return []
    agg = (
        df.groupby("subreddit", observed=True)
        .agg(
            n=(INDEX_COL_LLM, "count"),
            mean_style_index=(INDEX_COL_LLM, "mean"),
            mean_ai_style=("ai_style_rate_100w", "mean"),
            share_ai_hit=("ai_hit", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values("mean_style_index", ascending=False)
    rows: List[Dict[str, Any]] = []
    for _, r in pd.concat([agg.head(top_n), agg.tail(top_n)]).drop_duplicates("subreddit").iterrows():
        rows.append(
            {
                "subreddit": r["subreddit"],
                "n": int(r["n"]),
                "mean_style_index_llm": float(r["mean_style_index"]),
                "mean_ai_style_rate_100w": float(r["mean_ai_style"]),
                "share_ai_hit": float(r["share_ai_hit"]),
                "tail": "high_index" if r["mean_style_index"] >= agg["mean_style_index"].median() else "low_index",
            }
        )
    return rows


def build_ai_rate_review_sample(
    df: pd.DataFrame, n_each: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Function summary: stratified manual review sample by ai_style_rate, not global index.

    Parameters:
    - df: prepared validation frame.
    - n_each: comments per bucket (high / low ai rate among scored rows).
    - seed: RNG seed.

    Returns:
    - DataFrame with review_bucket in {high_ai_rate, low_ai_rate}.
    """
    work = df.dropna(subset=[INDEX_COL_LLM, "ai_style_rate_100w"]).copy()
    if work.empty:
        return pd.DataFrame()
    work = work[work["n_words"].fillna(0) >= 20] if "n_words" in work.columns else work
    if len(work) < n_each * 2:
        n_each = max(1, len(work) // 2)
    high = work.nlargest(n_each, "ai_style_rate_100w").assign(review_bucket="high_ai_rate")
    low = work.nsmallest(n_each, "ai_style_rate_100w").assign(review_bucket="low_ai_rate")
    return pd.concat([high, low], ignore_index=True)


def gate_status_from_metrics(
    rho_ai: float, rho_len: float, length_delta_20_49: float
) -> Dict[str, str]:
    """Function summary: map key metrics to pass/review/info labels for gates_summary.

    Parameters:
    - rho_ai: Spearman index vs ai_style_rate (all or ge20).
    - rho_len: Spearman index vs log_len.
    - length_delta_20_49: mean index hit1 - hit0 in 20-49 word bin.

    Returns:
    - Dict of gate_name -> status.
    """
    out: Dict[str, str] = {}
    if np.isfinite(rho_ai):
        out["spearman_vs_ai_style_rate_100w"] = (
            "pass" if rho_ai > SPEARMAN_AI_STYLE_HEURISTIC else "review"
        )
    else:
        out["spearman_vs_ai_style_rate_100w"] = "insufficient_n"
    if np.isfinite(rho_len):
        out["spearman_vs_log_len"] = (
            "pass" if abs(rho_len) < LENGTH_CONFOUND_REVIEW else "review_length_confound"
        )
    if np.isfinite(length_delta_20_49):
        out["length_stratified_ai_20_49"] = (
            "pass" if length_delta_20_49 > 0 else "review_no_separation"
        )
    return out


def gate_status_from_metrics_v2(
    rho_ai_it: float, rho_len_it: float, length_delta_20_49: float
) -> Dict[str, str]:
    """Function summary: v2 lexical gates for Italian subset (IT threshold 0.15)."""
    out: Dict[str, str] = {}
    if np.isfinite(rho_ai_it):
        out["v2_it_spearman_lexical_vs_ai_style_rate"] = (
            "pass" if rho_ai_it > SPEARMAN_AI_STYLE_V2_IT else "review"
        )
    if np.isfinite(rho_len_it):
        out["v2_it_spearman_lexical_vs_log_len"] = (
            "pass" if abs(rho_len_it) < LENGTH_CONFOUND_V2_REVIEW else "review_length_confound"
        )
    if np.isfinite(length_delta_20_49):
        out["v2_length_stratified_ai_20_49"] = (
            "pass" if length_delta_20_49 > 0 else "review_no_separation"
        )
    return out
