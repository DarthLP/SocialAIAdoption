"""
Semantic-axis panel statistics: lexicon calibration, pole bucket specs, and validity checks.

Functionality:
- Map topic_family to primary_lexicon for homogeneous panel cells.
- Calibrate per-lexicon score percentiles from enriched shards.
- Build pole-bucket specifications (absolute per lexicon + percentile cutoffs).
- Ideology-axis orientation report vs lexical net_ideology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.embeddings import ALL_AXIS_NAMES

# All scored axes (legacy + extended issue dimensions).
SEMANTIC_AXES: Tuple[str, ...] = ALL_AXIS_NAMES

# topic_family -> primary_lexicon for panel cells without primary_lexicon column
FAMILY_TO_LEXICON: Dict[str, str] = {
    "de": "de",
    "eu": "en",
    "us": "en",
    "uk": "en",
    "it_political": "it",
    "it_others": "it",
    "it_pure_political": "it",
}


@dataclass(frozen=True)
class PoleBucketSpec:
    """One pole-bucket counter set (high and/or low masks) with output column suffix."""

    axis: str
    high_label: str
    low_label: str
    suffix: str
    kind: str  # symmetric_absolute | high_percentile | low_percentile


def topic_family_primary_lexicon(topic_family: str) -> str:
    """Function summary: map topic_family slug to scoring lexicon (it, en, de)."""
    return FAMILY_TO_LEXICON.get(str(topic_family), "en")


def group_primary_lexicon(grp: pd.DataFrame) -> str:
    """Function summary: infer lexicon for a panel aggregation group from its comments."""
    if "primary_lexicon" in grp.columns and grp["primary_lexicon"].notna().any():
        return str(grp["primary_lexicon"].dropna().iloc[0])
    if "topic_family" in grp.columns and grp["topic_family"].notna().any():
        return topic_family_primary_lexicon(str(grp["topic_family"].iloc[0]))
    return "en"


def absolute_threshold(
    sem_cfg: Mapping[str, Any],
    lexicon: str,
    axis: str,
) -> float:
    """Function summary: per-lexicon absolute cosine cutoff for one axis."""
    by_lex = sem_cfg.get("pole_thresholds_by_lexicon") or {}
    lex = str(lexicon).lower()
    if isinstance(by_lex.get(lex), dict):
        thr = by_lex[lex].get(axis)
        if thr is not None:
            return float(thr)
    legacy = sem_cfg.get("pole_thresholds") or {}
    if isinstance(legacy, dict) and axis in legacy:
        return float(legacy[axis])
    cutoffs = sem_cfg.get("pole_cutoffs") or [0.25]
    return float(cutoffs[0]) if cutoffs else 0.25


def tau_suffix(tau: float) -> str:
    """Function summary: column suffix for an absolute cutoff (e.g. 0.25 -> tau25)."""
    return f"tau{int(round(float(tau) * 100))}"


def build_pole_bucket_specs(sem_cfg: Mapping[str, Any]) -> List[PoleBucketSpec]:
    """Function summary: pole bucket definitions from config (absolute + percentile kinds)."""
    specs: List[PoleBucketSpec] = []
    for axis in SEMANTIC_AXES:
        use_lr = axis == "ideology"
        high_label = "right" if use_lr else "pos"
        low_label = "left" if use_lr else "neg"
        specs.append(
            PoleBucketSpec(
                axis=axis,
                high_label=high_label,
                low_label=low_label,
                suffix="abs",
                kind="symmetric_absolute",
            )
        )
    for pct in sem_cfg.get("pole_percentiles") or []:
        p = int(pct)
        for axis in SEMANTIC_AXES:
            use_lr = axis == "ideology"
            high_label = "right" if use_lr else "pos"
            low_label = "left" if use_lr else "neg"
            if p >= 50:
                specs.append(
                    PoleBucketSpec(
                        axis=axis,
                        high_label=high_label,
                        low_label=low_label,
                        suffix=f"above_p{p}",
                        kind="high_percentile",
                    )
                )
            else:
                specs.append(
                    PoleBucketSpec(
                        axis=axis,
                        high_label=high_label,
                        low_label=low_label,
                        suffix=f"below_p{p}",
                        kind="low_percentile",
                    )
                )
    return specs


def percentile_lookup_from_csv(path: Path) -> Dict[Tuple[str, str, int], float]:
    """Function summary: load calibration CSV into (lexicon, axis, percentile) -> threshold."""
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: Dict[Tuple[str, str, int], float] = {}
    for _, row in df.iterrows():
        key = (
            str(row["primary_lexicon"]).lower(),
            str(row["axis"]),
            int(row["percentile"]),
        )
        out[key] = float(row["threshold"])
    return out


def _percentile_threshold(
    lookup: Mapping[Tuple[str, str, int], float],
    lexicon: str,
    axis: str,
    percentile: int,
) -> float:
    """Function summary: get calibrated threshold or NaN if missing."""
    return float(lookup.get((str(lexicon).lower(), axis, int(percentile)), float("nan")))


def pole_column_prefix(axis: str) -> str:
    """Function summary: column name prefix for one semantic axis."""
    return f"sem_axis_{axis}"


def calibrate_lexicon_percentiles(
    shard_paths: Sequence[Path],
    read_columns: Sequence[str],
    sem_cfg: Mapping[str, Any],
    *,
    read_shard_fn,
    preban_only: bool = False,
    launch_day: Optional[str] = None,
) -> pd.DataFrame:
    """Function summary: sample scored comments and compute per-lexicon score percentiles.

    Parameters:
    - shard_paths: enriched parquet paths.
    - read_columns: columns to read from each shard.
    - sem_cfg: semantic_axis config block.
    - read_shard_fn: callable(path, columns) -> DataFrame | None.
    - preban_only: when True, restrict calibration sample to date_utc < launch_day.
    - launch_day: ban launch date (YYYY-MM-DD) for pre-ban filter.

    Returns:
    - DataFrame with primary_lexicon, axis, percentile, threshold, n_sample.
    """
    cal = sem_cfg.get("percentile_calibration") or {}
    if not cal.get("enabled", True):
        return pd.DataFrame()
    max_per_lang = int(cal.get("max_comments_per_lang", 50000))
    percentiles = [int(p) for p in (sem_cfg.get("pole_percentiles") or [10, 90])]
    use_preban = preban_only
    if launch_day is None and use_preban:
        use_preban = False

    samples: Dict[str, Dict[str, List[float]]] = {
        lang: {axis: [] for axis in SEMANTIC_AXES} for lang in ("it", "en", "de")
    }
    counts = {lang: 0 for lang in samples}

    read_cols = list(read_columns)
    if use_preban and "date_utc" not in read_cols:
        read_cols.append("date_utc")

    for path in shard_paths:
        if all(counts[lang] >= max_per_lang for lang in counts):
            break
        df = read_shard_fn(path, read_cols)
        if df is None or df.empty:
            continue
        if "has_sem_axis" not in df.columns:
            continue
        scored = df[df["has_sem_axis"].astype(float) > 0]
        if use_preban and launch_day and "date_utc" in scored.columns:
            scored = scored[scored["date_utc"].astype(str) < str(launch_day)]
        if scored.empty or "primary_lexicon" not in scored.columns:
            continue
        for lex in scored["primary_lexicon"].astype(str).unique():
            lex_l = lex.lower()
            if lex_l not in samples or counts[lex_l] >= max_per_lang:
                continue
            sub = scored[scored["primary_lexicon"].astype(str) == lex]
            remain = max_per_lang - counts[lex_l]
            if len(sub) > remain:
                sub = sub.sample(n=remain, random_state=42)
            counts[lex_l] += len(sub)
            for axis in SEMANTIC_AXES:
                col = pole_column_prefix(axis)
                if col not in sub.columns:
                    continue
                samples[lex_l][axis].extend(sub[col].astype(float).tolist())

    rows: List[Dict[str, Any]] = []
    for lex, axes in samples.items():
        for axis, vals in axes.items():
            if len(vals) < 10:
                continue
            arr = np.asarray(vals, dtype=float)
            for pct in percentiles:
                rows.append(
                    {
                        "primary_lexicon": lex,
                        "axis": axis,
                        "percentile": pct,
                        "threshold": float(np.percentile(arr, pct)),
                        "n_sample": len(vals),
                    }
                )
    return pd.DataFrame(rows)


def ideology_orientation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: correlate sem_axis_ideology with net_ideology by language.

    Parameters:
    - df: comment-level frame with primary_lexicon, sem_axis_ideology, net_ideology, has_sem_axis.

    Returns:
    - Summary table with correlation signs and sparse-zero rates for face-validity review.
    """
    rows: List[Dict[str, Any]] = []
    scored_all = df[df["has_sem_axis"].astype(float) > 0] if "has_sem_axis" in df.columns else df
    if scored_all.empty:
        return pd.DataFrame()

    def _corr(a: pd.Series, b: pd.Series) -> float:
        if len(a) < 10:
            return float("nan")
        return float(a.corr(b))

    for lang, grp in scored_all.groupby(scored_all["primary_lexicon"].astype(str)):
        r_comment = _corr(grp["sem_axis_ideology"].astype(float), grp["net_ideology"].astype(float))
        if "subreddit" in grp.columns and "date_utc" in grp.columns:
            daily = grp.groupby(["subreddit", "date_utc"], as_index=False).agg(
                sem=("sem_axis_ideology", "mean"),
                net=("net_ideology", "mean"),
            )
            r_day = _corr(daily["sem"], daily["net"])
            frac_zero_net_day = float((daily["net"] == 0).mean())
        else:
            r_day = float("nan")
            frac_zero_net_day = float("nan")
        rows.append(
            {
                "lang": lang,
                "n_comments": len(grp),
                "corr_ideology_comment_pearson": r_comment,
                "corr_ideology_subreddit_day_pearson": r_day,
                "frac_net_ideology_zero_comment": float((grp["net_ideology"].astype(float) == 0).mean()),
                "frac_net_ideology_zero_subreddit_day": frac_zero_net_day,
                "orientation_flag": (
                    "negative_corr"
                    if r_comment == r_comment and r_comment < 0
                    else ("weak" if r_comment == r_comment and abs(r_comment) < 0.05 else "ok")
                ),
            }
        )

    pooled = scored_all
    r_pool = _corr(
        pooled["sem_axis_ideology"].astype(float),
        pooled["net_ideology"].astype(float),
    )
    rows.append(
        {
            "lang": "_pooled",
            "n_comments": len(pooled),
            "corr_ideology_comment_pearson": r_pool,
            "corr_ideology_subreddit_day_pearson": float("nan"),
            "frac_net_ideology_zero_comment": float(
                (pooled["net_ideology"].astype(float) == 0).mean()
            ),
            "frac_net_ideology_zero_subreddit_day": float("nan"),
            "orientation_flag": (
                "negative_corr"
                if r_pool == r_pool and r_pool < 0
                else ("weak" if r_pool == r_pool and abs(r_pool) < 0.05 else "ok")
            ),
        }
    )
    return pd.DataFrame(rows)
