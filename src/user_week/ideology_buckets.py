"""
Author ideology bucket assignment: lexical (net_ideology) vs semantic (sem_axis_ideology).

Supports asymmetric_v2 (pre-ban L/R hits + semantic tail-weeks) and legacy tertiles_within_lexicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config_utils import tables_subdir, user_week_section
from src.semantic_axis_stats import _percentile_threshold, percentile_lookup_from_csv


UNCLASSIFIED = "unclassified"
SEMANTICALLY_UNSCORED = "semantically_unscored"
DEFAULT_BUCKET_LABELS = {
    "low": "conservative_leaning",
    "mid": "neutral",
    "high": "liberal_leaning",
}
IDEOLOGY_AXIS = "ideology"


@dataclass(frozen=True)
class IdeologyBucketConfig:
    """Function summary: YAML-driven thresholds for author ideology buckets."""

    method: str
    min_pre_words: int
    min_pre_weeks: int
    bucket_labels: Tuple[str, str, str]
    min_share_scored: float
    percentile_thresholds_path: Optional[str]
    tail_percentile_low: int
    tail_percentile_high: int


def ideology_bucket_config(config: Dict[str, Any]) -> IdeologyBucketConfig:
    """Function summary: parse user_week.author_ideology_buckets from study YAML.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - IdeologyBucketConfig with method, floors, and optional semantic paths.
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
    pct_path = raw.get("percentile_thresholds_path")
    tail_raw = raw.get("tail_percentiles", [25, 75])
    if isinstance(tail_raw, (list, tuple)) and len(tail_raw) >= 2:
        tail_low, tail_high = int(tail_raw[0]), int(tail_raw[1])
    else:
        tail_low, tail_high = 25, 75
    return IdeologyBucketConfig(
        method=str(raw.get("method", "asymmetric_v2")),
        min_pre_words=int(raw.get("min_pre_words", 400)),
        min_pre_weeks=int(raw.get("min_pre_weeks", 4)),
        bucket_labels=(low, mid, high),
        min_share_scored=float(raw.get("min_share_scored", 0.5)),
        percentile_thresholds_path=str(pct_path) if pct_path else None,
        tail_percentile_low=tail_low,
        tail_percentile_high=tail_high,
    )


def resolve_percentile_thresholds_path(
    config: Dict[str, Any],
    bucket_cfg: IdeologyBucketConfig,
    project_root: Optional[Path] = None,
) -> Path:
    """Function summary: resolve calibrated p10/p90 CSV path from YAML or default tables layout.

    Parameters:
    - config: study YAML.
    - bucket_cfg: parsed bucket config.
    - project_root: optional repo root for relative paths.

    Returns:
    - Absolute or cwd-relative Path to thresholds CSV.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    if bucket_cfg.percentile_thresholds_path:
        p = Path(bucket_cfg.percentile_thresholds_path)
        if p.is_absolute():
            return p
        return root / p
    return tables_subdir(config, "semantic_axis") / "semantic_axis_lexicon_percentile_thresholds.csv"


def load_percentile_thresholds(
    config: Dict[str, Any],
    bucket_cfg: IdeologyBucketConfig,
    project_root: Optional[Path] = None,
) -> Dict[Tuple[str, str, int], float]:
    """Function summary: load semantic_axis_lexicon_percentile_thresholds.csv or raise.

    Parameters:
    - config: study YAML.
    - bucket_cfg: bucket config with optional path override.
    - project_root: repo root for relative paths.

    Returns:
    - Lookup (lexicon, axis, percentile) -> threshold.

    Raises:
    - FileNotFoundError when CSV is missing (run prepare_semantic_axis_descriptives.py).
    """
    path = resolve_percentile_thresholds_path(config, bucket_cfg, project_root=project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing semantic percentile thresholds at {path}. "
            "Run scripts/diagnostics/prepare_semantic_axis_descriptives.py first."
        )
    lookup = percentile_lookup_from_csv(path)
    if not lookup:
        raise FileNotFoundError(f"Empty or unreadable percentile thresholds at {path}.")
    return lookup


def require_tail_percentiles_in_lookup(
    lookup: Mapping[Tuple[str, str, int], float],
    tail_low: int,
    tail_high: int,
    lexicons: Sequence[str] = ("it", "en", "de"),
) -> None:
    """Function summary: verify calibrated thresholds exist for tail percentiles per language.

    Parameters:
    - lookup: percentile calibration map.
    - tail_low, tail_high: required percentile integers (e.g. 20, 80).
    - lexicons: primary lexicon codes to check.

    Raises:
    - FileNotFoundError with regeneration hint when any pair is missing.
    """
    missing: List[str] = []
    for lex in lexicons:
        for pct in (tail_low, tail_high):
            key = (str(lex).lower(), IDEOLOGY_AXIS, int(pct))
            val = lookup.get(key, float("nan"))
            if not np.isfinite(val):
                missing.append(f"{lex}/ideology/p{pct}")
    if missing:
        raise FileNotFoundError(
            f"Missing ideology tail thresholds for: {', '.join(missing)}. "
            "Set semantic_axis.pole_percentiles to include those values and run "
            "scripts/diagnostics/prepare_semantic_axis_descriptives.py."
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


def _coverage_column(panel: pd.DataFrame) -> Optional[str]:
    """Function summary: pick semantic coverage column present on user-week panel."""
    for col in ("sem_axis_coverage_mean", "sem_axis_coverage", "sem_axis_ideology_coverage"):
        if col in panel.columns:
            return col
    return None


def collect_preban_author_features(
    panel: pd.DataFrame,
    authors: Optional[Sequence[str]],
    min_pre_words: int,
    min_pre_weeks: int,
    semantic_multipliers: Optional[Dict[str, float]],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
    tail_percentile_low: int = 25,
    tail_percentile_high: int = 75,
) -> pd.DataFrame:
    """Function summary: pre-ban scores, L/R hits, coverage, and semantic tail-week counts per author.

    Parameters:
    - panel: labelled panel with period, hits, sem_axis means, assigned_primary_lexicon.
    - authors: optional allow-list.
    - min_pre_words, min_pre_weeks: classification floors.
    - semantic_multipliers: per-lexicon orientation for sem_axis.
    - percentile_lookup: (lexicon, axis, percentile) thresholds for tail weeks.
    - tail_percentile_low, tail_percentile_high: e.g. 20 and 80 for wider tails than p10/p90.

    Returns:
    - Author-level feature table for bucket assignment.
    """
    cols = [
        "author",
        "n_words_pre",
        "n_pre_weeks",
        "lexical_score",
        "semantic_score",
        "left_hits_pre",
        "right_hits_pre",
        "share_scored",
        "n_sem_left_tail_weeks",
        "n_sem_right_tail_weeks",
    ]
    if panel.empty:
        return pd.DataFrame(columns=cols)
    pre = panel[panel["period"].astype(str) == "pre"].copy()
    if pre.empty:
        return pd.DataFrame(columns=cols)
    pre["author"] = pre["author"].astype(str)
    if authors is not None:
        allow = set(str(a) for a in authors)
        pre = pre[pre["author"].isin(allow)].copy()

    lex_col = "net_ideology_mean" if "net_ideology_mean" in pre.columns else "net_ideology"
    sem_col = "sem_axis_ideology_mean" if "sem_axis_ideology_mean" in pre.columns else "sem_axis_ideology"
    cov_col = _coverage_column(pre)
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
        left_hits_pre = float(grp["left_hits"].astype(float).sum()) if "left_hits" in grp.columns else 0.0
        right_hits_pre = float(grp["right_hits"].astype(float).sum()) if "right_hits" in grp.columns else 0.0

        lex = str(grp["assigned_primary_lexicon"].iloc[0]).lower() if "assigned_primary_lexicon" in grp.columns else "it"
        mult = semantic_orientation_flip(lex, semantic_multipliers or {})
        p_low = _percentile_threshold(percentile_lookup, lex, IDEOLOGY_AXIS, tail_percentile_low)
        p_high = _percentile_threshold(percentile_lookup, lex, IDEOLOGY_AXIS, tail_percentile_high)

        if cov_col is not None:
            cov_vals = grp[cov_col].astype(float).values
            share_scored = _weighted_mean(cov_vals, nw.values)
        else:
            sem_raw = grp[sem_col].astype(float).values if sem_col in grp.columns else np.array([])
            scored_mask = np.isfinite(sem_raw)
            share_scored = float(scored_mask.mean()) if len(sem_raw) else 0.0

        n_left_tail = 0
        n_right_tail = 0
        if sem_col in grp.columns and np.isfinite(p_low) and np.isfinite(p_high):
            for _, wk in grp.iterrows():
                sem_v = wk.get(sem_col)
                if not np.isfinite(sem_v):
                    continue
                if cov_col is not None:
                    cov_v = float(wk.get(cov_col, 0.0))
                    if cov_v <= 0:
                        continue
                oriented = float(sem_v) * mult
                if oriented < p_low:
                    n_left_tail += 1
                elif oriented > p_high:
                    n_right_tail += 1

        sem_vals = grp[sem_col].astype(float).values if sem_col in grp.columns else np.array([])
        sem_vals_oriented = sem_vals * mult if len(sem_vals) else sem_vals
        semantic = _weighted_mean(sem_vals_oriented, nw.values)

        rows.append(
            {
                "author": str(author),
                "n_words_pre": n_words_pre,
                "n_pre_weeks": n_pre_weeks,
                "lexical_score": lexical,
                "semantic_score": semantic,
                "left_hits_pre": left_hits_pre,
                "right_hits_pre": right_hits_pre,
                "share_scored": share_scored,
                "n_sem_left_tail_weeks": n_left_tail,
                "n_sem_right_tail_weeks": n_right_tail,
            }
        )
    return pd.DataFrame(rows)


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
    empty = pd.DataFrame(
        columns=["author", "n_words_pre", "n_pre_weeks", "lexical_score", "semantic_score"]
    )
    if panel.empty:
        return empty
    lookup: Dict[Tuple[str, str, int], float] = {
        ("it", IDEOLOGY_AXIS, 10): -1.0,
        ("it", IDEOLOGY_AXIS, 90): 1.0,
        ("en", IDEOLOGY_AXIS, 10): -1.0,
        ("en", IDEOLOGY_AXIS, 90): 1.0,
        ("de", IDEOLOGY_AXIS, 10): -1.0,
        ("de", IDEOLOGY_AXIS, 90): 1.0,
    }
    feat = collect_preban_author_features(
        panel,
        authors,
        min_pre_words,
        min_pre_weeks,
        semantic_multipliers,
        lookup,
    )
    if feat.empty:
        return empty
    return feat[
        ["author", "n_words_pre", "n_pre_weeks", "lexical_score", "semantic_score"]
    ].copy()


def assign_lexical_buckets(
    df: pd.DataFrame,
    bucket_labels: Tuple[str, str, str],
) -> pd.Series:
    """Function summary: neutral when no pre-ban L/R hits; else pole from lexical score sign.

    Parameters:
    - df: rows with left_hits_pre, right_hits_pre, lexical_score.
    - bucket_labels: (conservative, neutral, liberal).

    Returns:
    - Series of lexical_bucket labels aligned to df.index.
    """
    low_l, mid_l, high_l = bucket_labels
    buckets: List[str] = []
    for _, row in df.iterrows():
        if not np.isfinite(row.get("lexical_score", float("nan"))):
            buckets.append(UNCLASSIFIED)
            continue
        lh = float(row.get("left_hits_pre", 0.0) or 0.0)
        rh = float(row.get("right_hits_pre", 0.0) or 0.0)
        if lh + rh <= 0:
            buckets.append(mid_l)
            continue
        score = float(row["lexical_score"])
        if score > 0:
            buckets.append(high_l)
        elif score < 0:
            buckets.append(low_l)
        else:
            buckets.append(mid_l)
    return pd.Series(buckets, index=df.index)


def assign_semantic_tail_buckets(
    df: pd.DataFrame,
    bucket_labels: Tuple[str, str, str],
    min_share_scored: float,
) -> pd.Series:
    """Function summary: tail-week semantic buckets; semantically_unscored when coverage low.

    Parameters:
    - df: rows with share_scored, tail week counts, semantic_score.
    - bucket_labels: pole labels.
    - min_share_scored: minimum word-weighted coverage to classify.

    Returns:
    - Series semantic_bucket aligned to df.index.
    """
    low_l, mid_l, high_l = bucket_labels
    buckets: List[str] = []
    for _, row in df.iterrows():
        share = float(row.get("share_scored", 0.0) or 0.0)
        if share < min_share_scored:
            buckets.append(SEMANTICALLY_UNSCORED)
            continue
        n_left = int(row.get("n_sem_left_tail_weeks", 0) or 0)
        n_right = int(row.get("n_sem_right_tail_weeks", 0) or 0)
        if n_left + n_right == 0:
            buckets.append(mid_l)
            continue
        if n_left > n_right:
            buckets.append(high_l)
        elif n_right > n_left:
            buckets.append(low_l)
        else:
            score = float(row.get("semantic_score", 0.0) or 0.0)
            if score > 0:
                buckets.append(high_l)
            elif score < 0:
                buckets.append(low_l)
            else:
                buckets.append(mid_l)
    return pd.Series(buckets, index=df.index)


def assign_semantic_mag_band_buckets(
    df: pd.DataFrame,
    bucket_labels: Tuple[str, str, str],
    min_share_scored: float,
    lexicon_col: str = "assigned_primary_lexicon",
) -> pd.Series:
    """Function summary: neutral band from language p25 of |semantic_score| among scorable authors.

    Parameters:
    - df: author features with share_scored and semantic_score.
    - bucket_labels: pole labels.
    - min_share_scored: coverage gate.
    - lexicon_col: language grouping column.

    Returns:
    - Series semantic_bucket_mag_band.
    """
    low_l, mid_l, high_l = bucket_labels
    out = pd.Series(UNCLASSIFIED, index=df.index, dtype=object)
    scorable = df[df["share_scored"].astype(float) >= min_share_scored].copy()
    if scorable.empty:
        return out
    for lex, grp in scorable.groupby(scorable[lexicon_col].astype(str), sort=False):
        scores = grp["semantic_score"].astype(float)
        valid = scores.notna()
        if valid.sum() < 1:
            continue
        abs_s = scores[valid].abs()
        p25 = float(abs_s.quantile(0.25))
        for idx in grp.index:
            s = float(df.at[idx, "semantic_score"])
            if not np.isfinite(s):
                continue
            if abs(s) < p25:
                out.at[idx] = mid_l
            elif s > 0:
                out.at[idx] = high_l
            elif s < 0:
                out.at[idx] = low_l
            else:
                out.at[idx] = mid_l
    return out


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
    for _, grp in out.groupby(out[lexicon_col].astype(str), sort=False):
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


def _build_tertile_buckets(
    out: pd.DataFrame,
    bucket_cfg: IdeologyBucketConfig,
) -> pd.DataFrame:
    """Function summary: legacy tertile assignment within primary lexicon."""
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
    out["semantic_bucket_mag_band"] = UNCLASSIFIED
    return out


def _build_asymmetric_buckets(
    out: pd.DataFrame,
    bucket_cfg: IdeologyBucketConfig,
) -> pd.DataFrame:
    """Function summary: asymmetric lexical hits + semantic tail-week buckets."""
    mask_lex = out["assigned_primary_lexicon"].notna() & out["lexical_score"].notna()
    out["lexical_bucket"] = UNCLASSIFIED
    if mask_lex.any():
        out.loc[mask_lex, "lexical_bucket"] = assign_lexical_buckets(
            out.loc[mask_lex], bucket_cfg.bucket_labels
        ).values
    mask_sem = mask_lex & out["semantic_score"].notna()
    out["semantic_bucket"] = UNCLASSIFIED
    if mask_sem.any():
        out.loc[mask_sem, "semantic_bucket"] = assign_semantic_tail_buckets(
            out.loc[mask_sem],
            bucket_cfg.bucket_labels,
            bucket_cfg.min_share_scored,
        ).values
    out["semantic_bucket_mag_band"] = UNCLASSIFIED
    if mask_sem.any():
        out.loc[mask_sem, "semantic_bucket_mag_band"] = assign_semantic_mag_band_buckets(
            out.loc[mask_sem],
            bucket_cfg.bucket_labels,
            bucket_cfg.min_share_scored,
        ).values
    return out


def build_author_ideology_buckets(
    panel_labelled: pd.DataFrame,
    assignment: pd.DataFrame,
    bucket_cfg: IdeologyBucketConfig,
    cohort_authors: Optional[Sequence[str]],
    semantic_multipliers: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    percentile_lookup: Optional[Dict[Tuple[str, str, int], float]] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Function summary: full author table with lexical/semantic buckets and agreement flags.

    Parameters:
    - panel_labelled: user-week panel with period column.
    - assignment: author, assigned_primary_lexicon.
    - bucket_cfg: thresholds and label names.
    - cohort_authors: optional allow-list.
    - semantic_multipliers: per-lexicon sem_axis sign.
    - config: study YAML (required for asymmetric_v2 percentile load).
    - percentile_lookup: optional pre-loaded thresholds.
    - project_root: repo root for relative paths.

    Returns:
    - Author-level bucket table with buckets_agree and bucket_cross.
    """
    assign = assignment[["author", "assigned_primary_lexicon"]].copy()
    assign["author"] = assign["author"].astype(str)
    merged_panel = panel_labelled.merge(assign, on="author", how="inner")

    if bucket_cfg.method == "asymmetric_v2":
        if percentile_lookup is None:
            if config is None:
                raise ValueError("config is required for asymmetric_v2 percentile thresholds")
            percentile_lookup = load_percentile_thresholds(
                config, bucket_cfg, project_root=project_root
            )
            require_tail_percentiles_in_lookup(
                percentile_lookup,
                bucket_cfg.tail_percentile_low,
                bucket_cfg.tail_percentile_high,
            )
        features = collect_preban_author_features(
            merged_panel,
            cohort_authors,
            bucket_cfg.min_pre_words,
            bucket_cfg.min_pre_weeks,
            semantic_multipliers,
            percentile_lookup,
            tail_percentile_low=bucket_cfg.tail_percentile_low,
            tail_percentile_high=bucket_cfg.tail_percentile_high,
        )
    else:
        features = preban_author_scores(
            merged_panel,
            cohort_authors,
            bucket_cfg.min_pre_words,
            bucket_cfg.min_pre_weeks,
            semantic_multipliers=semantic_multipliers,
        )

    if features.empty:
        return pd.DataFrame()
    out = features.merge(assign, on="author", how="left")

    if bucket_cfg.method == "asymmetric_v2":
        out = _build_asymmetric_buckets(out, bucket_cfg)
    else:
        out = _build_tertile_buckets(out, bucket_cfg)

    labels = set(bucket_cfg.bucket_labels)
    classified = (
        out["lexical_bucket"].isin(labels)
        & out["semantic_bucket"].isin(labels)
        & (out["semantic_bucket"] != SEMANTICALLY_UNSCORED)
    )
    out["buckets_agree"] = False
    out.loc[classified, "buckets_agree"] = (
        out.loc[classified, "lexical_bucket"] == out.loc[classified, "semantic_bucket"]
    )
    out["bucket_cross"] = [
        bucket_cross_label(lx, sm) for lx, sm in zip(out["lexical_bucket"], out["semantic_bucket"])
    ]
    return out


# --- Agreement metrics ---

BUCKET_ORDER = {
    "conservative_leaning": 0,
    "neutral": 1,
    "liberal_leaning": 2,
}


def filter_agreement_sample(
    df: pd.DataFrame,
    labels: Sequence[str],
    semantic_col: str = "semantic_bucket",
    lexical_col: str = "lexical_bucket",
    exclude_semantic_unscored: bool = True,
) -> pd.DataFrame:
    """Function summary: rows eligible for lexical vs semantic agreement metrics.

    Parameters:
    - df: author bucket table.
    - labels: pole bucket names.
    - semantic_col, lexical_col: columns to compare.
    - exclude_semantic_unscored: drop semantically_unscored rows.

    Returns:
    - Filtered copy.
    """
    sub = df[df[lexical_col].isin(labels) & df[semantic_col].isin(labels)].copy()
    if exclude_semantic_unscored:
        sub = sub[sub[semantic_col] != SEMANTICALLY_UNSCORED]
    return sub


def filter_pole_only_agreement_sample(
    df: pd.DataFrame,
    labels: Sequence[str],
    semantic_col: str = "semantic_bucket",
    lexical_col: str = "lexical_bucket",
) -> pd.DataFrame:
    """Function summary: authors non-neutral on both lexical and semantic buckets."""
    sub = filter_agreement_sample(df, labels, semantic_col=semantic_col, lexical_col=lexical_col)
    mid = labels[1] if len(labels) >= 2 else "neutral"
    return sub[(sub[lexical_col] != mid) & (sub[semantic_col] != mid)]


def marginal_bucket_counts(
    df: pd.DataFrame,
    col: str,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Function summary: count authors per bucket value including special labels.

    Parameters:
    - df: author table.
    - col: bucket column name.
    - labels: optional ordered pole labels to prepend in output.

    Returns:
    - DataFrame bucket, n, pct.
    """
    counts = df[col].astype(str).value_counts()
    rows = [{"bucket": k, "n": int(v), "pct": float(v) / len(df) if len(df) else 0.0} for k, v in counts.items()]
    out = pd.DataFrame(rows)
    if labels and not out.empty:
        order = {b: i for i, b in enumerate(list(labels) + [SEMANTICALLY_UNSCORED, UNCLASSIFIED])}
        out["_ord"] = out["bucket"].map(lambda b: order.get(b, 999))
        out = out.sort_values("_ord").drop(columns=["_ord"])
    return out


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
    sub = df[df[row_col].isin(labels) & df[col_col].isin(labels)].copy()
    if sub.empty:
        return pd.DataFrame(0, index=labels, columns=labels)
    ct = pd.crosstab(sub[row_col], sub[col_col], dropna=False)
    return ct.reindex(index=labels, columns=labels, fill_value=0)


def agreement_summary_rows(
    df: pd.DataFrame,
    labels: Sequence[str],
    group_col: Optional[str] = None,
    semantic_col: str = "semantic_bucket",
    lexical_col: str = "lexical_bucket",
    exclude_semantic_unscored: bool = True,
    scope_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Function summary: build agreement metric rows overall or per lexicon group.

    Parameters:
    - df: author bucket table.
    - labels: bucket label list.
    - group_col: if set, one row per group value plus overall.
    - semantic_col, lexical_col: columns compared.
    - exclude_semantic_unscored: drop low-coverage semantic rows.
    - scope_prefix: prepended to scope names (e.g. tail_ vs mag_band_).

    Returns:
    - List of metric dicts.
    """
    from scipy.stats import spearmanr

    classified = filter_agreement_sample(
        df,
        labels,
        semantic_col=semantic_col,
        lexical_col=lexical_col,
        exclude_semantic_unscored=exclude_semantic_unscored,
    )

    def _scope_name(name: str) -> str:
        if scope_prefix:
            return f"{scope_prefix}{name}"
        return name

    def _one_block(sub: pd.DataFrame, scope: str) -> Dict[str, Any]:
        if sub.empty:
            return {
                "scope": _scope_name(scope),
                "semantic_col": semantic_col,
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
            "scope": _scope_name(scope),
            "semantic_col": semantic_col,
            "n_classified": int(len(sub)),
            "pct_exact_match": pct_exact_match(sub[lexical_col], sub[semantic_col]),
            "pct_adjacent_match": pct_adjacent_match(sub[lexical_col], sub[semantic_col]),
            "cohens_kappa": cohens_kappa(sub[lexical_col], sub[semantic_col], labels),
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
