"""
Author ideology bucket assignment: lexical (net_ideology) vs semantic (sem_axis_ideology).

Pre-ban word-weighted scores and per-primary_lexicon tertiles (conservative / neutral / liberal-leaning).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config_utils import tables_subdir, user_week_section


UNCLASSIFIED = "unclassified"
DEFAULT_BUCKET_LABELS = {
    "low": "conservative_leaning",
    "mid": "neutral",
    "high": "liberal_leaning",
}


@dataclass(frozen=True)
class IdeologyBucketConfig:
    """Function summary: YAML-driven thresholds for author ideology buckets."""

    method: str
    min_pre_words: int
    min_pre_weeks: int
    bucket_labels: Tuple[str, str, str]


def ideology_bucket_config(config: Dict[str, Any]) -> IdeologyBucketConfig:
    """Function summary: parse user_week.author_ideology_buckets from study YAML.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - IdeologyBucketConfig with tertile labels and pre-ban word/week floors.
    """
    raw = user_week_section(config).get("author_ideology_buckets", {})
    if not isinstance(raw, dict):
        raw = {}
    labels_raw = raw.get("bucket_labels", {})
    if not isinstance(labels_raw, dict):
        labels_raw = {}
    low = str(labels_raw.get("low", DEFAULT_BUCKET_LABELS["low"]))
    mid = str(labels_raw.get("mid", DEFAULT_BUCKET_LABELS["mid"]))
    high = str(labels_raw.get("high", DEFAULT_BUCKET_LABELS["high"]))
    return IdeologyBucketConfig(
        method=str(raw.get("method", "tertiles_within_lexicon")),
        min_pre_words=int(raw.get("min_pre_words", 400)),
        min_pre_weeks=int(raw.get("min_pre_weeks", 4)),
        bucket_labels=(low, mid, high),
    )


def label_pre_post_weeks(
    panel: pd.DataFrame,
    launch_iso_week: str,
    drop_ban_week: bool,
) -> pd.DataFrame:
    """Function summary: tag user-week rows as pre, post, or launch relative to ban ISO week.

    Parameters:
    - panel: user-week panel with iso_week_start.
    - launch_iso_week: YYYY-MM-DD Monday of launch week.
    - drop_ban_week: if True, remove launch-week rows.

    Returns:
    - Copy with period column.
    """
    out = panel.copy()
    out["iso_week_start"] = out["iso_week_start"].astype(str)
    out["period"] = "pre"
    out.loc[out["iso_week_start"] > launch_iso_week, "period"] = "post"
    out.loc[out["iso_week_start"] == launch_iso_week, "period"] = "launch"
    if drop_ban_week:
        out = out[out["period"] != "launch"].copy()
    return out


def load_semantic_orientation_multipliers(config: Dict[str, Any]) -> Dict[str, float]:
    """Function summary: per-language sign to align sem_axis with net_ideology (left positive).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Dict primary_lexicon -> +1 or -1 (default +1 when report missing).
    """
    path = tables_subdir(config, "semantic_axis") / "ideology_axis_orientation_report.csv"
    out: Dict[str, float] = {"it": 1.0, "en": 1.0, "de": 1.0}
    if not path.is_file():
        return out
    df = pd.read_csv(path)
    if "lang" not in df.columns:
        return out
    for _, row in df.iterrows():
        lang = str(row["lang"]).strip().lower()
        if lang.startswith("_"):
            continue
        flag = str(row.get("orientation_flag", "ok"))
        corr = row.get("corr_ideology_comment_pearson", float("nan"))
        flip = flag == "negative_corr" or (
            isinstance(corr, (int, float)) and np.isfinite(corr) and corr < 0
        )
        out[lang] = -1.0 if flip else 1.0
    return out


def semantic_orientation_flip(lexicon: str, multipliers: Dict[str, float]) -> float:
    """Function summary: return orientation multiplier for one primary lexicon code.

    Parameters:
    - lexicon: it, en, or de.
    - multipliers: from load_semantic_orientation_multipliers.

    Returns:
    - +1.0 or -1.0.
    """
    return float(multipliers.get(str(lexicon).strip().lower(), 1.0))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Function summary: word-weighted mean; NaN when total weight is zero."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    w = w[mask]
    v = v[mask]
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(v * w) / total)


def preban_author_scores(
    panel: pd.DataFrame,
    authors: Optional[Sequence[str]],
    min_pre_words: int,
    min_pre_weeks: int,
    semantic_multipliers: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Function summary: word-weighted pre-ban lexical and semantic scores per author.

    Parameters:
    - panel: labelled panel with period, n_words, net_ideology_mean, sem_axis_ideology_mean.
    - authors: optional author allow-list; None keeps all with enough pre words.
    - min_pre_words: minimum total pre-ban words to classify.
    - min_pre_weeks: minimum pre-ban weeks with n_words > 0.
    - semantic_multipliers: per-lexicon sign for sem_axis (unused if no lexicon col).

    Returns:
    - DataFrame author, n_words_pre, n_pre_weeks, lexical_score, semantic_score.
    """
    if panel.empty:
        return pd.DataFrame(
            columns=["author", "n_words_pre", "n_pre_weeks", "lexical_score", "semantic_score"]
        )
    pre = panel[panel["period"].astype(str) == "pre"].copy()
    if pre.empty:
        return pd.DataFrame(
            columns=["author", "n_words_pre", "n_pre_weeks", "lexical_score", "semantic_score"]
        )
    pre["author"] = pre["author"].astype(str)
    if authors is not None:
        allow = set(str(a) for a in authors)
        pre = pre[pre["author"].isin(allow)].copy()

    lex_col = "net_ideology_mean" if "net_ideology_mean" in pre.columns else "net_ideology"
    sem_col = "sem_axis_ideology_mean" if "sem_axis_ideology_mean" in pre.columns else "sem_axis_ideology"
    rows: List[Dict[str, Any]] = []
    for author, grp in pre.groupby("author", sort=False):
        nw = grp["n_words"].astype(float)
        n_words_pre = float(nw.sum())
        n_pre_weeks = int((nw > 0).sum())
        if n_words_pre < float(min_pre_words) or n_pre_weeks < int(min_pre_weeks):
            continue
        lexical = _weighted_mean(
            grp[lex_col].astype(float).values if lex_col in grp.columns else np.array([]),
            nw.values,
        )
        sem_vals = grp[sem_col].astype(float).values if sem_col in grp.columns else np.array([])
        if semantic_multipliers and "assigned_primary_lexicon" in grp.columns:
            lex = str(grp["assigned_primary_lexicon"].iloc[0]).lower()
            mult = semantic_orientation_flip(lex, semantic_multipliers)
            sem_vals = sem_vals * mult
        semantic = _weighted_mean(sem_vals, nw.values)
        rows.append(
            {
                "author": str(author),
                "n_words_pre": n_words_pre,
                "n_pre_weeks": n_pre_weeks,
                "lexical_score": lexical,
                "semantic_score": semantic,
            }
        )
    return pd.DataFrame(rows)


def assign_tertile_buckets(
    df: pd.DataFrame,
    lexicon_col: str,
    score_col: str,
    bucket_labels: Tuple[str, str, str],
    out_col: str,
) -> pd.DataFrame:
    """Function summary: assign low/mid/high tertile bucket within each primary lexicon.

    Parameters:
    - df: rows with lexicon and score (finite).
    - lexicon_col: language assignment column.
    - score_col: continuous score to tertile.
    - bucket_labels: (low, mid, high) names — low=conservative, high=liberal.
    - out_col: output bucket column name.

    Returns:
    - Copy with out_col; unclassified where qcut fails (too few per lang).
    """
    out = df.copy()
    out[out_col] = UNCLASSIFIED
    low_l, mid_l, high_l = bucket_labels
    for lex, grp in out.groupby(out[lexicon_col].astype(str), sort=False):
        idx = grp.index
        s = grp[score_col].astype(float)
        valid = s.notna()
        if valid.sum() < 3:
            continue
        try:
            buckets = pd.qcut(
                s[valid],
                q=3,
                labels=[low_l, mid_l, high_l],
                duplicates="drop",
            )
            out.loc[idx[valid], out_col] = buckets.astype(str).values
        except ValueError:
            continue
    return out


def bucket_cross_label(lexical_bucket: str, semantic_bucket: str) -> str:
    """Function summary: compact crosswalk label for lexical vs semantic bucket pair."""
    lx = str(lexical_bucket).replace("_leaning", "").replace("_", "")[:6]
    sm = str(semantic_bucket).replace("_leaning", "").replace("_", "")[:6]
    return f"{lx}_lex__{sm}_sem"


def build_author_ideology_buckets(
    panel_labelled: pd.DataFrame,
    assignment: pd.DataFrame,
    bucket_cfg: IdeologyBucketConfig,
    cohort_authors: Optional[Sequence[str]],
    semantic_multipliers: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Function summary: full author table with lexical/semantic scores and tertile buckets.

    Parameters:
    - panel_labelled: user-week panel with period column and optional assignment merged.
    - assignment: author, assigned_primary_lexicon.
    - bucket_cfg: thresholds and label names.
    - cohort_authors: optional allow-list.
    - semantic_multipliers: per-lexicon sem_axis sign.

    Returns:
    - Author-level bucket table with buckets_agree and bucket_cross.
    """
    assign = assignment[["author", "assigned_primary_lexicon"]].copy()
    assign["author"] = assign["author"].astype(str)
    merged_panel = panel_labelled.merge(assign, on="author", how="inner")
    scores = preban_author_scores(
        merged_panel,
        cohort_authors,
        bucket_cfg.min_pre_words,
        bucket_cfg.min_pre_weeks,
        semantic_multipliers=semantic_multipliers,
    )
    if scores.empty:
        return pd.DataFrame()
    out = scores.merge(assign, on="author", how="left")
    out["lexical_bucket"] = UNCLASSIFIED
    out["semantic_bucket"] = UNCLASSIFIED
    mask_lex = out["assigned_primary_lexicon"].notna() & out["lexical_score"].notna()
    if mask_lex.any():
        lex_part = assign_tertile_buckets(
            out.loc[mask_lex].copy(),
            "assigned_primary_lexicon",
            "lexical_score",
            bucket_cfg.bucket_labels,
            "lexical_bucket",
        )
        out.loc[mask_lex, "lexical_bucket"] = lex_part["lexical_bucket"].values
    mask_sem = mask_lex & out["semantic_score"].notna()
    if mask_sem.any():
        sem_part = assign_tertile_buckets(
            out.loc[mask_sem].copy(),
            "assigned_primary_lexicon",
            "semantic_score",
            bucket_cfg.bucket_labels,
            "semantic_bucket",
        )
        out.loc[mask_sem, "semantic_bucket"] = sem_part["semantic_bucket"].values
    classified = (out["lexical_bucket"] != UNCLASSIFIED) & (out["semantic_bucket"] != UNCLASSIFIED)
    out["buckets_agree"] = False
    out.loc[classified, "buckets_agree"] = (
        out.loc[classified, "lexical_bucket"] == out.loc[classified, "semantic_bucket"]
    )
    out["bucket_cross"] = [
        bucket_cross_label(lx, sm) for lx, sm in zip(out["lexical_bucket"], out["semantic_bucket"])
    ]
    return out


# --- Agreement metrics (Phase 2) ---

BUCKET_ORDER = {
    "conservative_leaning": 0,
    "neutral": 1,
    "liberal_leaning": 2,
}


def cohens_kappa(y1: pd.Series, y2: pd.Series, labels: Sequence[str]) -> float:
    """Function summary: Cohen's kappa for two nominal columns over a fixed label set.

    Parameters:
    - y1, y2: aligned bucket columns.
    - labels: category list.

    Returns:
    - Kappa in [-1, 1], or NaN if undefined.
    """
    lab = list(labels)
    n = len(y1)
    if n == 0:
        return float("nan")
    mat = np.zeros((len(lab), len(lab)), dtype=float)
    idx1 = {l: i for i, l in enumerate(lab)}
    for a, b in zip(y1.astype(str), y2.astype(str)):
        if a not in idx1 or b not in idx1:
            continue
        mat[idx1[a], idx1[b]] += 1
    mat = mat / mat.sum() if mat.sum() > 0 else mat
    p_o = float(np.trace(mat))
    p_e = float(mat.sum(axis=0) @ mat.sum(axis=1))
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def pct_exact_match(y1: pd.Series, y2: pd.Series) -> float:
    """Function summary: fraction of rows where two bucket columns match exactly."""
    if len(y1) == 0:
        return float("nan")
    return float((y1.astype(str) == y2.astype(str)).mean())


def pct_adjacent_match(y1: pd.Series, y2: pd.Series) -> float:
    """Function summary: fraction where bucket ranks differ by at most one step."""
    if len(y1) == 0:
        return float("nan")
    ok = 0
    for a, b in zip(y1.astype(str), y2.astype(str)):
        ra = BUCKET_ORDER.get(a)
        rb = BUCKET_ORDER.get(b)
        if ra is None or rb is None:
            continue
        if abs(ra - rb) <= 1:
            ok += 1
    return ok / len(y1)


def confusion_table(
    df: pd.DataFrame,
    row_col: str = "lexical_bucket",
    col_col: str = "semantic_bucket",
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Function summary: cross-tab counts for lexical vs semantic buckets.

    Parameters:
    - df: classified authors only.
    - row_col, col_col: bucket columns.
    - labels: ordered bucket names.

    Returns:
    - DataFrame cross-tab (rows=lexical).
    """
    if labels is None:
        labels = list(DEFAULT_BUCKET_LABELS.values())
    sub = df[
        df[row_col].isin(labels)
        & df[col_col].isin(labels)
    ].copy()
    if sub.empty:
        return pd.DataFrame(0, index=labels, columns=labels)
    ct = pd.crosstab(sub[row_col], sub[col_col], dropna=False)
    return ct.reindex(index=labels, columns=labels, fill_value=0)


def agreement_summary_rows(
    df: pd.DataFrame,
    labels: Sequence[str],
    group_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Function summary: build agreement metric rows overall or per lexicon group.

    Parameters:
    - df: author bucket table.
    - labels: bucket label list.
    - group_col: if set, one row per group value plus overall handled separately.

    Returns:
    - List of metric dicts.
    """
    from scipy.stats import spearmanr

    classified = df[
        df["lexical_bucket"].isin(labels) & df["semantic_bucket"].isin(labels)
    ].copy()

    def _one_block(sub: pd.DataFrame, scope: str) -> Dict[str, Any]:
        if sub.empty:
            return {
                "scope": scope,
                "n_classified": 0,
                "pct_exact_match": float("nan"),
                "pct_adjacent_match": float("nan"),
                "cohens_kappa": float("nan"),
                "spearman_rho": float("nan"),
            }
        rho = float("nan")
        if sub["lexical_score"].notna().sum() >= 10:
            rho_val, _ = spearmanr(
                sub["lexical_score"].astype(float),
                sub["semantic_score"].astype(float),
                nan_policy="omit",
            )
            rho = float(rho_val) if np.isfinite(rho_val) else float("nan")
        return {
            "scope": scope,
            "n_classified": int(len(sub)),
            "pct_exact_match": pct_exact_match(sub["lexical_bucket"], sub["semantic_bucket"]),
            "pct_adjacent_match": pct_adjacent_match(sub["lexical_bucket"], sub["semantic_bucket"]),
            "cohens_kappa": cohens_kappa(sub["lexical_bucket"], sub["semantic_bucket"], labels),
            "spearman_rho": rho,
        }

    rows: List[Dict[str, Any]] = []
    if group_col and group_col in classified.columns:
        for val, grp in classified.groupby(classified[group_col].astype(str), sort=True):
            rows.append(_one_block(grp, str(val)))
    rows.append(_one_block(classified, "overall"))
    return rows


def load_cohort_authors_from_shift(tables_dir: Path, cohort: str) -> List[str]:
    """Function summary: author ids from existing shift_per_user polarization export.

    Parameters:
    - tables_dir: user_week tables root.
    - cohort: strict or loose.

    Returns:
    - List of author strings (empty if file missing).
    """
    path = tables_dir / f"shift_per_user_{cohort}_polarization.csv"
    if not path.is_file():
        path = tables_dir / f"shift_per_user_{cohort}.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path, usecols=["author"])
    return df["author"].astype(str).drop_duplicates().tolist()
