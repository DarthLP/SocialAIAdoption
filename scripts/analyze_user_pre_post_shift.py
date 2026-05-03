"""
Script summary:
This script consumes the per-author per-ISO-week style panel produced by
`scripts/prepare_user_week_style_panel.py` and produces the per-user pre-vs-post
shift analysis described in the within-user pre/post style shift plan. It
answers, for each user, "is the post-launch level unusual relative to that
user's own pre-launch wiggle (weekly view) AND relative to a precision-aware
pooled-comments view that scales standard errors with how much the user
actually wrote?".

Two parallel comparisons are computed for every user and every feature:
  1. Weekly view: word-weighted weekly mean / SD across pre weeks vs post weeks,
     standardized delta with a winsorized SD floor, robust MAD variant, and a
     Welch-style across-weeks t.
  2. Pooled-comments view: pre and post pooled directly (rate features =
     sum(hits)/sum(words); mean features = sum/n with sumsq variance), SE,
     delta with 95% CI, and a per-user t.

A composite `ai_likeness_user_week` (z(ai_word) + z(formality_balance) +
z(assistant_tone) + z(list_structure) - z(contraction)) is built with z-scales
frozen on the pre-launch user-week pool and applied to all user-weeks.

Cohort thresholds are split into weekly thresholds (good weeks must have
>= --min_words_per_week, >= --min_pre_weeks and >= --min_post_weeks) and
pooled thresholds (>= --min_total_words_pre, --min_total_words_post). Hard
pre-launch requirement: a user enters the comparison only with both pre and
post coverage above thresholds. Excluded users are surfaced in the side audit.

Aggregate-level outputs include tail shares, Wilcoxon and sign tests, an
inverse-variance weighted pooled effect, an agreement diagnostic between
weekly and pooled views, a topic-stable sub-cohort, topic-stratified summaries,
and a placebo run with the launch shifted by --placebo_offset_weeks.

How to apply/run:
- Default (both strict and loose cohorts):
  `.venv/bin/python scripts/analyze_user_pre_post_shift.py --config config/political_forums_setup.yaml`
- Strict only with explicit thresholds:
  `.venv/bin/python scripts/analyze_user_pre_post_shift.py --config config/political_forums_setup.yaml --cohort strict --min_words_per_week 100 --min_pre_weeks 4 --min_post_weeks 4 --min_total_words_pre 400 --min_total_words_post 400`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


# ----------------------------- Constants -----------------------------

# Composite components and their signs (matches add_ai_likeness_index in prepare_event_time_metrics.py).
COMPOSITE_COMPONENTS: List[Tuple[str, int]] = [
    ("ai_word_rate_100w", +1),
    ("formality_balance_100w", +1),
    ("assistant_tone_rate_100w", +1),
    ("list_structure_intensity", +1),
    ("contraction_rate_100w", -1),
]
COMPOSITE_NAME = "ai_likeness_user_week"

# Per-feature spec: how to recover pooled pre/post values and SE from the panel.
# kind in {"rate_100w", "binary_mean", "mean"}.
# rate_100w: panel has <feat>=hits_per_100_words, raw_hits_col integer hits; SE Poisson on hits.
# binary_mean: panel has <feat> = mean of 0/1 flag, plus <flag_sum_col> raw sum; SE binomial p(1-p)/n.
# mean: panel has <feat>_mean, <feat>_sum, <feat>_sumsq, <feat>_n; SE = sqrt(var/n) where
#       var is the sample variance from sumsq/sum/n.
FEATURE_SPECS: Dict[str, Dict[str, str]] = {
    "ai_word_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "strict_ai_word_hits_total",
    },
    "ai_word_extended_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "extended_ai_word_hits_total",
    },
    "assistant_tone_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "assistant_tone_phrase_count",
    },
    "contraction_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "contraction_count",
    },
    "full_form_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "full_form_count",
    },
    "passive_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "passive_count",
    },
    "toxic_lexicon_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "toxic_lexicon_hits",
    },
    "semicolon_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "semicolon_count",
    },
    "em_dash_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "em_dash_count",
    },
    "en_dash_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "en_dash_count",
    },
    "ascii_double_hyphen_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "ascii_double_hyphen_count",
    },
    "colon_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "colon_count",
    },
    "open_paren_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "open_paren_count",
    },
    "curly_quote_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "curly_quote_count",
    },
    "markdown_bold_pair_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "markdown_bold_pair_count",
    },
    "markdown_heading_line_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "markdown_heading_line_count",
    },
    "hedging_phrase_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "hedging_phrase_hits",
    },
    "polite_closer_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "polite_closer_hits",
    },
    "signposting_phrase_rate_100w": {
        "kind": "rate_100w",
        "raw_hits_col": "signposting_phrase_hits",
    },
    "formality_balance_100w": {
        # Special: = (full_form_count - contraction_count) / n_words * 100. We treat as a rate-100w
        # with raw_hits being the SIGNED difference; SE is the sum of two Poisson SEs added in quadrature
        # (full_form ~ Poisson, contraction ~ Poisson) since they are different lexical events; this is
        # a conservative approximation.
        "kind": "rate_100w_signed",
    },
    "list_structure_intensity": {
        "kind": "binary_mean",
        "flag_sum_col": "list_structure_flag_sum",
    },
    "comment_length_words": {
        "kind": "mean",
    },
    "avg_words_per_sentence_comment": {
        "kind": "mean",
    },
    "complexity_index": {
        # Ratio of sums (not a per-comment mean). Pooled value is recomputed from
        # weekly totals; SE is left NaN unless --bootstrap_reps>0 (deferred). Plotted but
        # not part of the composite SE.
        "kind": "complexity",
    },
}

# Default features to report (keep concise; composite components first).
DEFAULT_FEATURES: List[str] = [
    "ai_word_rate_100w",
    "formality_balance_100w",
    "assistant_tone_rate_100w",
    "contraction_rate_100w",
    "list_structure_intensity",
    "ai_word_extended_rate_100w",
    "comment_length_words",
    "em_dash_rate_100w",
    "hedging_phrase_rate_100w",
    "signposting_phrase_rate_100w",
    "avg_words_per_sentence_comment",
    "complexity_index",
]


# ----------------------------- CLI / dataclasses -----------------------------


@dataclass
class CohortThresholds:
    """Function summary: cohort threshold bundle used to gate weekly and pooled views."""

    label: str
    min_words_per_week: int
    min_pre_weeks: int
    min_post_weeks: int
    min_total_words_pre: int
    min_total_words_post: int


@dataclass
class RuntimePaths:
    """Function summary: store resolved input and output locations for the analysis script."""

    panel_path: Path
    user_week_tables_dir: Path
    user_week_logs_dir: Path


def parse_args() -> argparse.Namespace:
    """Function summary: parse command line options for cohort thresholds, placebo, and IO control."""
    parser = argparse.ArgumentParser(description="Within-user pre/post style shift analysis from user-week panel.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default="both",
        choices=["both", "strict", "loose"],
        help="Which cohort to analyze. 'both' produces strict and loose outputs.",
    )
    parser.add_argument("--min_words_per_week_strict", type=int, default=100)
    parser.add_argument("--min_pre_weeks_strict", type=int, default=4)
    parser.add_argument("--min_post_weeks_strict", type=int, default=4)
    parser.add_argument("--min_total_words_pre_strict", type=int, default=400)
    parser.add_argument("--min_total_words_post_strict", type=int, default=400)

    parser.add_argument("--min_words_per_week_loose", type=int, default=30)
    parser.add_argument("--min_pre_weeks_loose", type=int, default=2)
    parser.add_argument("--min_post_weeks_loose", type=int, default=2)
    parser.add_argument("--min_total_words_pre_loose", type=int, default=100)
    parser.add_argument("--min_total_words_post_loose", type=int, default=100)

    parser.add_argument(
        "--drop_launch_week",
        action="store_true",
        help="Drop the ISO week containing the launch date itself as a buffer between pre and post.",
    )
    parser.add_argument(
        "--sd_winsor_pct",
        type=float,
        default=5.0,
        help="Floor for pre_sd_w in std_delta_weekly: percentile across users in the cohort.",
    )
    parser.add_argument(
        "--placebo_offset_weeks",
        type=int,
        default=8,
        help="Run a placebo analysis with the launch date shifted back by this many weeks.",
    )
    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=0,
        help="Reserved: number of comment-level bootstrap reps for nonparametric CIs (deferred; default 0).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Optional comma-separated subset of features to analyze (default uses DEFAULT_FEATURES).",
    )
    return parser.parse_args()


def build_paths(config: Dict[str, Any]) -> RuntimePaths:
    """Function summary: resolve panel input path and ensure user-week output / log folders exist."""
    tables_dir = Path(config["paths"]["tables_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    user_week_tables_dir = tables_dir / "user_week"
    user_week_logs_dir = logs_dir / "user_week"
    user_week_tables_dir.mkdir(parents=True, exist_ok=True)
    user_week_logs_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        panel_path=user_week_tables_dir / "user_week_panel.parquet",
        user_week_tables_dir=user_week_tables_dir,
        user_week_logs_dir=user_week_logs_dir,
    )


def make_thresholds_from_args(args: argparse.Namespace) -> List[CohortThresholds]:
    """Function summary: assemble the requested cohort threshold bundles based on the --cohort flag."""
    strict = CohortThresholds(
        label="strict",
        min_words_per_week=int(args.min_words_per_week_strict),
        min_pre_weeks=int(args.min_pre_weeks_strict),
        min_post_weeks=int(args.min_post_weeks_strict),
        min_total_words_pre=int(args.min_total_words_pre_strict),
        min_total_words_post=int(args.min_total_words_post_strict),
    )
    loose = CohortThresholds(
        label="loose",
        min_words_per_week=int(args.min_words_per_week_loose),
        min_pre_weeks=int(args.min_pre_weeks_loose),
        min_post_weeks=int(args.min_post_weeks_loose),
        min_total_words_pre=int(args.min_total_words_pre_loose),
        min_total_words_post=int(args.min_total_words_post_loose),
    )
    if args.cohort == "strict":
        return [strict]
    if args.cohort == "loose":
        return [loose]
    return [strict, loose]


# ----------------------------- Utilities -----------------------------


def iso_week_monday(d: datetime) -> datetime:
    """Function summary: snap a datetime to the Monday that begins its ISO week (UTC date arithmetic)."""
    monday_date = d.date() - timedelta(days=d.date().weekday())
    return datetime(monday_date.year, monday_date.month, monday_date.day, tzinfo=timezone.utc)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Function summary: word-weighted mean of weekly values; returns NaN when total weight is zero."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(v * w) / total)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Function summary: weighted sample SD with effective-N correction; NaN when effective df < 1."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    total_w = float(w.sum())
    if total_w <= 0:
        return float("nan")
    mean = float(np.sum(v * w) / total_w)
    sumsq_w = float(np.sum(w * (v - mean) ** 2))
    # Effective sample size correction (Kish-style) so single dominant week does not give SD=0.
    sum_w_sq = float(np.sum(w ** 2))
    if sum_w_sq <= 0:
        return float("nan")
    n_eff = (total_w ** 2) / sum_w_sq
    if n_eff <= 1.0:
        return float("nan")
    return float(math.sqrt(sumsq_w / total_w * (n_eff / (n_eff - 1.0))))


# ----------------------------- Pre/post labeling -----------------------------


def label_pre_post(panel: pd.DataFrame, launch_iso_week: str, drop_launch_week: bool) -> pd.DataFrame:
    """Function summary: tag each user-week as pre or post relative to the ISO Monday of the launch week and optionally drop the launch week itself."""
    out = panel.copy()
    out["iso_week_start"] = out["iso_week_start"].astype(str)
    out["period"] = "pre"
    out.loc[out["iso_week_start"] > launch_iso_week, "period"] = "post"
    out.loc[out["iso_week_start"] == launch_iso_week, "period"] = "launch"
    if drop_launch_week:
        out = out[out["period"] != "launch"].copy()
    return out


def launch_iso_week_str(launch_dt_utc: datetime) -> str:
    """Function summary: return the YYYY-MM-DD ISO Monday string for the launch datetime."""
    return iso_week_monday(launch_dt_utc).date().isoformat()


# ----------------------------- Composite frozen-pre z-scaling -----------------------------


def freeze_composite_zscale(panel_pre: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Function summary: compute word-weighted means and SDs for composite components on the pre-launch user-week pool."""
    scales: Dict[str, Dict[str, float]] = {}
    if panel_pre.empty:
        return scales
    weights = panel_pre["n_words"].astype(float).values
    for component, _sign in COMPOSITE_COMPONENTS:
        if component not in panel_pre.columns:
            continue
        values = pd.to_numeric(panel_pre[component], errors="coerce").fillna(0.0).values
        mean = weighted_mean(values, weights)
        sd = weighted_std(values, weights)
        if not np.isfinite(sd) or sd == 0:
            sd = float("nan")
        scales[component] = {"mean": float(mean) if np.isfinite(mean) else 0.0, "sd": float(sd) if np.isfinite(sd) else 1.0}
    return scales


def add_composite_to_panel(panel: pd.DataFrame, scales: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Function summary: add the composite ai_likeness_user_week column using frozen-pre z-scales for each component."""
    if panel.empty or not scales:
        out = panel.copy()
        out[COMPOSITE_NAME] = float("nan")
        return out
    out = panel.copy()
    composite = pd.Series(0.0, index=out.index)
    any_component = False
    for component, sign in COMPOSITE_COMPONENTS:
        if component not in out.columns or component not in scales:
            continue
        mean = float(scales[component]["mean"])
        sd = float(scales[component]["sd"])
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (pd.to_numeric(out[component], errors="coerce").fillna(0.0) - mean) / sd
        composite = composite + sign * z
        any_component = True
    out[COMPOSITE_NAME] = composite if any_component else float("nan")
    return out


# ----------------------------- Per-user metric computations -----------------------------


def panel_value_column_for_feature(feature: str, panel_columns: Iterable[str]) -> str:
    """Function summary: resolve which panel column carries the per-week display value for a feature (handles _mean suffix for mean-kind features)."""
    cols = set(panel_columns)
    if feature in cols:
        return feature
    mean_alias = f"{feature}_mean"
    if mean_alias in cols:
        return mean_alias
    return feature


# ----------------------------- Per-user table builder -----------------------------


def _build_audit_df(panel: pd.DataFrame, thresholds: CohortThresholds) -> pd.DataFrame:
    """Function summary: vectorized audit categorization (panel / pre_only / post_only / below_thresholds) per author."""
    if panel.empty:
        return pd.DataFrame()
    n_words = panel["n_words"].astype(float)
    is_pre = panel["period"].values == "pre"
    is_post = panel["period"].values == "post"
    is_good = (n_words >= float(thresholds.min_words_per_week)).values
    df = pd.DataFrame(
        {
            "author": panel["author"].astype(str).values,
            "n_pre_any_w": is_pre.astype(int),
            "n_post_any_w": is_post.astype(int),
            "n_good_pre_w": (is_pre & is_good).astype(int),
            "n_good_post_w": (is_post & is_good).astype(int),
            "good_pre_words": np.where(is_pre & is_good, n_words.values, 0.0),
            "good_post_words": np.where(is_post & is_good, n_words.values, 0.0),
            "n_comments_total": panel["n_comments"].astype(float).values,
            "n_words_total": n_words.values,
        }
    )
    user = df.groupby("author", sort=False).sum().reset_index()
    user.rename(
        columns={
            "n_pre_any_w": "n_pre_weeks_any",
            "n_post_any_w": "n_post_weeks_any",
            "n_good_pre_w": "n_good_pre_weeks",
            "n_good_post_w": "n_good_post_weeks",
            "good_pre_words": "pre_words_total_good",
            "good_post_words": "post_words_total_good",
        },
        inplace=True,
    )
    is_panel = (
        (user["n_good_pre_weeks"] >= thresholds.min_pre_weeks)
        & (user["n_good_post_weeks"] >= thresholds.min_post_weeks)
        & (user["pre_words_total_good"] >= thresholds.min_total_words_pre)
        & (user["post_words_total_good"] >= thresholds.min_total_words_post)
    )
    is_pre_only = (user["n_pre_weeks_any"] > 0) & (user["n_post_weeks_any"] == 0) & ~is_panel
    is_post_only = (user["n_pre_weeks_any"] == 0) & (user["n_post_weeks_any"] > 0) & ~is_panel
    user["audit_category"] = "below_thresholds"
    user.loc[is_panel, "audit_category"] = "panel"
    user.loc[is_pre_only, "audit_category"] = "pre_only"
    user.loc[is_post_only, "audit_category"] = "post_only"
    return user


def _per_user_top_label(
    rows: pd.DataFrame,
    label_col: str,
    weight_col: str = "n_words",
) -> pd.Series:
    """Function summary: vectorized top-1 label per author by aggregated word weight (returns Series indexed by author)."""
    if rows.empty or label_col not in rows.columns:
        return pd.Series(dtype="object")
    weights = rows[weight_col].astype(float)
    grp = rows.assign(__w=weights).groupby(["author", label_col], dropna=False)["__w"].sum().reset_index()
    if grp.empty:
        return pd.Series(dtype="object")
    grp = grp.sort_values(["author", "__w"], ascending=[True, False])
    return grp.drop_duplicates("author").set_index("author")[label_col].astype(str)


def _per_user_concentration(
    rows: pd.DataFrame,
    label_col: str,
    weight_col: str = "n_words",
) -> pd.Series:
    """Function summary: per-author Herfindahl concentration of weight shares across labels for one period."""
    if rows.empty or label_col not in rows.columns:
        return pd.Series(dtype="float64")
    weights = rows[weight_col].astype(float)
    grp = rows.assign(__w=weights).groupby(["author", label_col], dropna=False)["__w"].sum().reset_index()
    if grp.empty:
        return pd.Series(dtype="float64")
    totals = grp.groupby("author")["__w"].sum().rename("__total")
    grp = grp.merge(totals, on="author", how="left")
    grp["__share2"] = (grp["__w"] / grp["__total"].where(grp["__total"] > 0, np.nan)) ** 2
    return grp.groupby("author")["__share2"].sum().astype(float)


def _kish_corrected_sd_from_sums(
    sum_w: pd.Series,
    sum_w2: pd.Series,
    sum_v_w: pd.Series,
    sum_v2_w: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Function summary: recover word-weighted mean and Kish-corrected sample SD from per-author sufficient sums."""
    sum_w = sum_w.astype(float)
    sum_w2 = sum_w2.astype(float)
    sum_v_w = sum_v_w.astype(float)
    sum_v2_w = sum_v2_w.astype(float)
    safe_w = sum_w.where(sum_w > 0, np.nan)
    mean = sum_v_w / safe_w
    var_pop = (sum_v2_w / safe_w) - mean ** 2
    var_pop = var_pop.clip(lower=0)
    safe_w2 = sum_w2.where(sum_w2 > 0, np.nan)
    n_eff = (sum_w ** 2) / safe_w2
    correction = n_eff / (n_eff - 1.0)
    correction = correction.where(n_eff > 1, np.nan)
    sd = np.sqrt(var_pop * correction)
    return mean, sd


def _build_panel_user_features(
    panel: pd.DataFrame,
    panel_authors: List[str],
    scales: Dict[str, Dict[str, float]],
    features: List[str],
) -> pd.DataFrame:
    """Function summary: vectorized weekly + pooled view per (panel user, feature) using sufficient sums per period."""
    if not panel_authors:
        return pd.DataFrame()
    sub = panel[panel["author"].astype(str).isin(panel_authors)].copy()
    if sub.empty:
        return pd.DataFrame()

    n_words = sub["n_words"].astype(float).values
    is_pre = (sub["period"].values == "pre") & (n_words >= 1.0)
    is_post = (sub["period"].values == "post") & (n_words >= 1.0)

    # Sufficient sums for weekly word-weighted mean + Kish SD per (author, period, feature).
    base_cols: Dict[str, np.ndarray] = {
        "__pre_w": np.where(is_pre, n_words, 0.0),
        "__post_w": np.where(is_post, n_words, 0.0),
        "__pre_w2": np.where(is_pre, n_words ** 2, 0.0),
        "__post_w2": np.where(is_post, n_words ** 2, 0.0),
        "__pre_n_w": is_pre.astype(int),
        "__post_n_w": is_post.astype(int),
        "__pre_n_comments_pooled": np.where(is_pre, sub["n_comments"].astype(float).values, 0.0),
        "__post_n_comments_pooled": np.where(is_post, sub["n_comments"].astype(float).values, 0.0),
    }

    # Per-feature value columns for weekly view (handles _mean alias for mean-kind features).
    feature_value_columns: Dict[str, str] = {}
    for feat in features + [COMPOSITE_NAME]:
        value_col = panel_value_column_for_feature(feat, sub.columns)
        feature_value_columns[feat] = value_col
        if value_col not in sub.columns:
            continue
        v = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0).astype(float).values
        base_cols[f"__pre_v_w_{feat}"] = np.where(is_pre, n_words * v, 0.0)
        base_cols[f"__post_v_w_{feat}"] = np.where(is_post, n_words * v, 0.0)
        base_cols[f"__pre_v2_w_{feat}"] = np.where(is_pre, n_words * v * v, 0.0)
        base_cols[f"__post_v2_w_{feat}"] = np.where(is_post, n_words * v * v, 0.0)

    # Per-feature pooled-view raw fields.
    for feat in features:
        spec = FEATURE_SPECS.get(feat)
        if spec is None:
            continue
        kind = spec["kind"]
        if kind == "rate_100w":
            col = spec.get("raw_hits_col", "")
            if col in sub.columns:
                hits = sub[col].astype(float).values
                base_cols[f"__pre_hits_{feat}"] = np.where(is_pre, hits, 0.0)
                base_cols[f"__post_hits_{feat}"] = np.where(is_post, hits, 0.0)
                base_cols[f"__pre_words_{feat}"] = np.where(is_pre, n_words, 0.0)
                base_cols[f"__post_words_{feat}"] = np.where(is_post, n_words, 0.0)
        elif kind == "rate_100w_signed":
            if {"full_form_count", "contraction_count"}.issubset(sub.columns):
                ff = sub["full_form_count"].astype(float).values
                cn = sub["contraction_count"].astype(float).values
                base_cols[f"__pre_ff_{feat}"] = np.where(is_pre, ff, 0.0)
                base_cols[f"__pre_cn_{feat}"] = np.where(is_pre, cn, 0.0)
                base_cols[f"__post_ff_{feat}"] = np.where(is_post, ff, 0.0)
                base_cols[f"__post_cn_{feat}"] = np.where(is_post, cn, 0.0)
                base_cols[f"__pre_words_{feat}"] = np.where(is_pre, n_words, 0.0)
                base_cols[f"__post_words_{feat}"] = np.where(is_post, n_words, 0.0)
        elif kind == "binary_mean":
            col = spec.get("flag_sum_col", "")
            if col in sub.columns:
                fl = sub[col].astype(float).values
                base_cols[f"__pre_flag_{feat}"] = np.where(is_pre, fl, 0.0)
                base_cols[f"__post_flag_{feat}"] = np.where(is_post, fl, 0.0)
                base_cols[f"__pre_n_comments_{feat}"] = np.where(is_pre, sub["n_comments"].astype(float).values, 0.0)
                base_cols[f"__post_n_comments_{feat}"] = np.where(is_post, sub["n_comments"].astype(float).values, 0.0)
        elif kind == "mean":
            for suf in ("sum", "sumsq", "n"):
                col = f"{feat}_{suf}"
                if col in sub.columns:
                    arr = sub[col].astype(float).values
                    base_cols[f"__pre_{suf}_{feat}"] = np.where(is_pre, arr, 0.0)
                    base_cols[f"__post_{suf}_{feat}"] = np.where(is_post, arr, 0.0)
        elif kind == "complexity":
            for col in ("n_words", "total_word_chars_comment", "sentence_count_comment"):
                if col in sub.columns:
                    arr = sub[col].astype(float).values
                    base_cols[f"__pre_{col}_{feat}"] = np.where(is_pre, arr, 0.0)
                    base_cols[f"__post_{col}_{feat}"] = np.where(is_post, arr, 0.0)

    flat = pd.DataFrame(base_cols)
    flat["author"] = sub["author"].astype(str).values
    user_sums = flat.groupby("author", sort=False).sum()

    out_cols: Dict[str, pd.Series] = {}

    # Weekly view per feature.
    for feat in features + [COMPOSITE_NAME]:
        if f"__pre_v_w_{feat}" not in user_sums.columns:
            continue
        pre_mean, pre_sd = _kish_corrected_sd_from_sums(
            user_sums["__pre_w"], user_sums["__pre_w2"],
            user_sums[f"__pre_v_w_{feat}"], user_sums[f"__pre_v2_w_{feat}"],
        )
        post_mean, post_sd = _kish_corrected_sd_from_sums(
            user_sums["__post_w"], user_sums["__post_w2"],
            user_sums[f"__post_v_w_{feat}"], user_sums[f"__post_v2_w_{feat}"],
        )
        delta = post_mean - pre_mean
        pre_n = user_sums["__pre_n_w"].astype(float)
        post_n = user_sums["__post_n_w"].astype(float)
        denom = np.sqrt((pre_sd ** 2) / pre_n.where(pre_n > 0, np.nan) + (post_sd ** 2) / post_n.where(post_n > 0, np.nan))
        welch = delta / denom

        out_cols[f"pre_n_weeks_{feat}"] = pre_n.astype(int)
        out_cols[f"post_n_weeks_{feat}"] = post_n.astype(int)
        out_cols[f"pre_total_words_{feat}"] = user_sums["__pre_w"].astype(float)
        out_cols[f"post_total_words_{feat}"] = user_sums["__post_w"].astype(float)
        out_cols[f"pre_mean_w_{feat}"] = pre_mean.astype(float)
        out_cols[f"post_mean_w_{feat}"] = post_mean.astype(float)
        out_cols[f"pre_sd_w_{feat}"] = pre_sd.astype(float)
        out_cols[f"post_sd_w_{feat}"] = post_sd.astype(float)
        out_cols[f"delta_weekly_{feat}"] = delta.astype(float)
        out_cols[f"personal_t_weekly_{feat}"] = welch.astype(float)

    # Pooled view per feature.
    for feat in features:
        spec = FEATURE_SPECS.get(feat)
        if spec is None:
            continue
        kind = spec["kind"]
        pre_rate = pd.Series(np.nan, index=user_sums.index, dtype=float)
        post_rate = pd.Series(np.nan, index=user_sums.index, dtype=float)
        pre_se = pd.Series(np.nan, index=user_sums.index, dtype=float)
        post_se = pd.Series(np.nan, index=user_sums.index, dtype=float)

        if kind == "rate_100w" and f"__pre_hits_{feat}" in user_sums.columns:
            pw = user_sums[f"__pre_words_{feat}"].astype(float)
            ph = user_sums[f"__pre_hits_{feat}"].astype(float)
            qw = user_sums[f"__post_words_{feat}"].astype(float)
            qh = user_sums[f"__post_hits_{feat}"].astype(float)
            pre_rate = (ph / pw.where(pw > 0, np.nan)) * 100.0
            post_rate = (qh / qw.where(qw > 0, np.nan)) * 100.0
            pre_se = np.sqrt(ph.clip(lower=0)) / pw.where(pw > 0, np.nan) * 100.0
            post_se = np.sqrt(qh.clip(lower=0)) / qw.where(qw > 0, np.nan) * 100.0
        elif kind == "rate_100w_signed" and f"__pre_ff_{feat}" in user_sums.columns:
            pre_ff = user_sums[f"__pre_ff_{feat}"].astype(float)
            pre_cn = user_sums[f"__pre_cn_{feat}"].astype(float)
            post_ff = user_sums[f"__post_ff_{feat}"].astype(float)
            post_cn = user_sums[f"__post_cn_{feat}"].astype(float)
            pw = user_sums[f"__pre_words_{feat}"].astype(float)
            qw = user_sums[f"__post_words_{feat}"].astype(float)
            pre_rate = ((pre_ff - pre_cn) / pw.where(pw > 0, np.nan)) * 100.0
            post_rate = ((post_ff - post_cn) / qw.where(qw > 0, np.nan)) * 100.0
            pre_se = np.sqrt(pre_ff.clip(lower=0) + pre_cn.clip(lower=0)) / pw.where(pw > 0, np.nan) * 100.0
            post_se = np.sqrt(post_ff.clip(lower=0) + post_cn.clip(lower=0)) / qw.where(qw > 0, np.nan) * 100.0
        elif kind == "binary_mean" and f"__pre_flag_{feat}" in user_sums.columns:
            pf = user_sums[f"__pre_flag_{feat}"].astype(float)
            pn = user_sums[f"__pre_n_comments_{feat}"].astype(float)
            qf = user_sums[f"__post_flag_{feat}"].astype(float)
            qn = user_sums[f"__post_n_comments_{feat}"].astype(float)
            pre_rate = pf / pn.where(pn > 0, np.nan)
            post_rate = qf / qn.where(qn > 0, np.nan)
            pre_se = np.sqrt((pre_rate * (1 - pre_rate)).clip(lower=0) / pn.where(pn > 0, np.nan))
            post_se = np.sqrt((post_rate * (1 - post_rate)).clip(lower=0) / qn.where(qn > 0, np.nan))
        elif kind == "mean" and f"__pre_sum_{feat}" in user_sums.columns:
            ps = user_sums[f"__pre_sum_{feat}"].astype(float)
            pq = user_sums[f"__pre_sumsq_{feat}"].astype(float)
            pn = user_sums[f"__pre_n_{feat}"].astype(float)
            qs = user_sums[f"__post_sum_{feat}"].astype(float)
            qq = user_sums[f"__post_sumsq_{feat}"].astype(float)
            qn = user_sums[f"__post_n_{feat}"].astype(float)
            pre_rate = ps / pn.where(pn > 0, np.nan)
            post_rate = qs / qn.where(qn > 0, np.nan)
            pre_var = ((pq - pn * pre_rate ** 2) / (pn - 1).where(pn > 1, np.nan)).clip(lower=0)
            post_var = ((qq - qn * post_rate ** 2) / (qn - 1).where(qn > 1, np.nan)).clip(lower=0)
            pre_se = np.sqrt(pre_var / pn.where(pn > 0, np.nan))
            post_se = np.sqrt(post_var / qn.where(qn > 0, np.nan))
        elif kind == "complexity":
            words_pre = user_sums.get(f"__pre_n_words_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            chars_pre = user_sums.get(f"__pre_total_word_chars_comment_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            sents_pre = user_sums.get(f"__pre_sentence_count_comment_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            words_post = user_sums.get(f"__post_n_words_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            chars_post = user_sums.get(f"__post_total_word_chars_comment_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            sents_post = user_sums.get(f"__post_sentence_count_comment_{feat}", pd.Series(0.0, index=user_sums.index)).astype(float)
            pre_rate = 0.5 * (words_pre / sents_pre.where(sents_pre > 0, 1.0)) + 0.5 * (chars_pre / words_pre.where(words_pre > 0, np.nan))
            post_rate = 0.5 * (words_post / sents_post.where(sents_post > 0, 1.0)) + 0.5 * (chars_post / words_post.where(words_post > 0, np.nan))
            # SE deferred for complexity (NaN unless bootstrap is enabled later).

        delta = post_rate - pre_rate
        se_combined = np.sqrt(pre_se ** 2 + post_se ** 2)
        t_pooled = delta / se_combined.where(se_combined > 0, np.nan)
        ci_low = delta - 1.96 * se_combined
        ci_high = delta + 1.96 * se_combined

        out_cols[f"rate_pooled_pre_{feat}"] = pre_rate.astype(float)
        out_cols[f"rate_pooled_post_{feat}"] = post_rate.astype(float)
        out_cols[f"se_pooled_pre_{feat}"] = pre_se.astype(float)
        out_cols[f"se_pooled_post_{feat}"] = post_se.astype(float)
        out_cols[f"delta_pooled_{feat}"] = delta.astype(float)
        out_cols[f"se_pooled_delta_{feat}"] = se_combined.astype(float)
        out_cols[f"t_user_pooled_{feat}"] = t_pooled.astype(float)
        out_cols[f"ci95_low_{feat}"] = ci_low.astype(float)
        out_cols[f"ci95_high_{feat}"] = ci_high.astype(float)

    # Composite pooled-view via weighted sum of component pooled rates.
    pre_val = pd.Series(0.0, index=user_sums.index)
    post_val = pd.Series(0.0, index=user_sums.index)
    var_pre = pd.Series(0.0, index=user_sums.index)
    var_post = pd.Series(0.0, index=user_sums.index)
    any_component = False
    for component, sign in COMPOSITE_COMPONENTS:
        if f"rate_pooled_pre_{component}" not in out_cols:
            continue
        sd = float(scales.get(component, {}).get("sd", float("nan"))) if scales else float("nan")
        mean = float(scales.get(component, {}).get("mean", 0.0)) if scales else 0.0
        if not np.isfinite(sd) or sd == 0:
            continue
        any_component = True
        pre_v = (out_cols[f"rate_pooled_pre_{component}"] - mean) / sd
        post_v = (out_cols[f"rate_pooled_post_{component}"] - mean) / sd
        pre_val = pre_val + sign * pre_v.fillna(0.0)
        post_val = post_val + sign * post_v.fillna(0.0)
        var_pre = var_pre + (out_cols[f"se_pooled_pre_{component}"].fillna(0.0) / sd) ** 2
        var_post = var_post + (out_cols[f"se_pooled_post_{component}"].fillna(0.0) / sd) ** 2
    if any_component:
        pre_se_c = np.sqrt(var_pre)
        post_se_c = np.sqrt(var_post)
        delta_c = post_val - pre_val
        se_delta_c = np.sqrt(var_pre + var_post)
        t_c = delta_c / se_delta_c.where(se_delta_c > 0, np.nan)
        ci_low_c = delta_c - 1.96 * se_delta_c
        ci_high_c = delta_c + 1.96 * se_delta_c
        out_cols[f"rate_pooled_pre_{COMPOSITE_NAME}"] = pre_val.astype(float)
        out_cols[f"rate_pooled_post_{COMPOSITE_NAME}"] = post_val.astype(float)
        out_cols[f"se_pooled_pre_{COMPOSITE_NAME}"] = pre_se_c.astype(float)
        out_cols[f"se_pooled_post_{COMPOSITE_NAME}"] = post_se_c.astype(float)
        out_cols[f"delta_pooled_{COMPOSITE_NAME}"] = delta_c.astype(float)
        out_cols[f"se_pooled_delta_{COMPOSITE_NAME}"] = se_delta_c.astype(float)
        out_cols[f"t_user_pooled_{COMPOSITE_NAME}"] = t_c.astype(float)
        out_cols[f"ci95_low_{COMPOSITE_NAME}"] = ci_low_c.astype(float)
        out_cols[f"ci95_high_{COMPOSITE_NAME}"] = ci_high_c.astype(float)

    out = pd.DataFrame(out_cols, index=user_sums.index)
    return out.reset_index().rename(columns={"index": "author"})


def _per_user_metadata(panel: pd.DataFrame, panel_authors: List[str], thresholds: CohortThresholds) -> pd.DataFrame:
    """Function summary: vectorized topic / subreddit / counts metadata for panel users (pre and post separately)."""
    if not panel_authors:
        return pd.DataFrame()
    sub = panel[panel["author"].astype(str).isin(panel_authors)].copy()
    n_words = sub["n_words"].astype(float)
    is_good = (n_words >= float(thresholds.min_words_per_week)).values
    is_pre = (sub["period"].values == "pre") & is_good
    is_post = (sub["period"].values == "post") & is_good

    pre_rows = sub.iloc[np.where(is_pre)[0]]
    post_rows = sub.iloc[np.where(is_post)[0]]

    top_topic_pre = _per_user_top_label(pre_rows, "top_topic")
    top_topic_post = _per_user_top_label(post_rows, "top_topic")
    top_subreddit_pre = _per_user_top_label(pre_rows, "top_subreddit")
    top_subreddit_post = _per_user_top_label(post_rows, "top_subreddit")
    sub_conc_pre = _per_user_concentration(pre_rows, "top_subreddit")
    sub_conc_post = _per_user_concentration(post_rows, "top_subreddit")

    n_comments_pre = pre_rows.groupby("author")["n_comments"].sum()
    n_comments_post = post_rows.groupby("author")["n_comments"].sum()

    meta = pd.DataFrame(index=pd.Index(panel_authors, name="author"))
    meta["top_topic_pre"] = top_topic_pre.reindex(meta.index).fillna("")
    meta["top_topic_post"] = top_topic_post.reindex(meta.index).fillna("")
    meta["top_subreddit_pre"] = top_subreddit_pre.reindex(meta.index).fillna("")
    meta["top_subreddit_post"] = top_subreddit_post.reindex(meta.index).fillna("")
    meta["subreddit_concentration_pre"] = sub_conc_pre.reindex(meta.index).astype(float)
    meta["subreddit_concentration_post"] = sub_conc_post.reindex(meta.index).astype(float)
    meta["n_comments_pre"] = n_comments_pre.reindex(meta.index).fillna(0).astype(int)
    meta["n_comments_post"] = n_comments_post.reindex(meta.index).fillna(0).astype(int)
    meta["topic_changed_pre_post"] = (
        (meta["top_topic_pre"] != meta["top_topic_post"]) & meta["top_topic_pre"].astype(bool) & meta["top_topic_post"].astype(bool)
    )
    return meta.reset_index()


def _per_user_pre_mad(panel: pd.DataFrame, panel_authors: List[str], features: List[str], thresholds: CohortThresholds) -> pd.DataFrame:
    """Function summary: per-panel-user pre-period MAD across good weeks (sensitivity column for std_delta_weekly)."""
    if not panel_authors:
        return pd.DataFrame({"author": pd.Series(dtype=str)})
    sub = panel[panel["author"].astype(str).isin(panel_authors)].copy()
    n_words = sub["n_words"].astype(float)
    is_pre_good = (sub["period"].values == "pre") & (n_words >= float(thresholds.min_words_per_week)).values
    pre_rows = sub.iloc[np.where(is_pre_good)[0]]
    if pre_rows.empty:
        return pd.DataFrame({"author": panel_authors})
    out = pd.DataFrame(index=pd.Index(panel_authors, name="author"))
    for feat in features + [COMPOSITE_NAME]:
        value_col = panel_value_column_for_feature(feat, pre_rows.columns)
        if value_col not in pre_rows.columns:
            out[f"pre_mad_{feat}"] = np.nan
            continue
        vals = pd.to_numeric(pre_rows[value_col], errors="coerce").fillna(0.0)
        median_per_user = pre_rows.assign(__v=vals).groupby("author")["__v"].median()
        merged = pre_rows.assign(__v=vals).merge(median_per_user.rename("__med"), left_on="author", right_index=True)
        merged["__abs_dev"] = (merged["__v"] - merged["__med"]).abs()
        mad = merged.groupby("author")["__abs_dev"].median()
        out[f"pre_mad_{feat}"] = mad.reindex(out.index).astype(float)
    return out.reset_index()


def per_user_summary(
    panel: pd.DataFrame,
    thresholds: CohortThresholds,
    scales: Dict[str, Dict[str, float]],
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function summary: vectorized per-user shift table (panel cohort) and full-user audit table."""
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    audit_df = _build_audit_df(panel, thresholds=thresholds)
    panel_authors = audit_df.loc[audit_df["audit_category"] == "panel", "author"].astype(str).tolist()
    if not panel_authors:
        return pd.DataFrame(), audit_df

    feature_user_df = _build_panel_user_features(panel=panel, panel_authors=panel_authors, scales=scales, features=features)
    metadata_df = _per_user_metadata(panel=panel, panel_authors=panel_authors, thresholds=thresholds)
    mad_df = _per_user_pre_mad(panel=panel, panel_authors=panel_authors, features=features, thresholds=thresholds)

    audit_lookup = audit_df[
        [
            "author",
            "audit_category",
            "n_good_pre_weeks",
            "n_good_post_weeks",
            "pre_words_total_good",
            "post_words_total_good",
        ]
    ]

    user_df = (
        audit_lookup[audit_lookup["audit_category"] == "panel"]
        .merge(metadata_df, on="author", how="left")
        .merge(feature_user_df, on="author", how="left")
        .merge(mad_df, on="author", how="left")
    )
    return user_df, audit_df


def apply_sd_winsor_floor(user_df: pd.DataFrame, features: List[str], pct: float) -> pd.DataFrame:
    """Function summary: floor pre_sd_w at the given percentile across the cohort and emit std_delta_weekly_<f> columns."""
    if user_df.empty:
        return user_df
    out = user_df.copy()
    for feature in features + [COMPOSITE_NAME]:
        sd_col = f"pre_sd_w_{feature}"
        if sd_col not in out.columns:
            continue
        sd_values = out[sd_col].astype(float)
        valid = sd_values[(sd_values > 0) & sd_values.notna()]
        if valid.empty:
            floor_val = float("nan")
        else:
            floor_val = float(np.nanpercentile(valid.values, pct))
        out[f"sd_floor_{feature}"] = floor_val
        sd_clipped = sd_values.clip(lower=floor_val) if np.isfinite(floor_val) else sd_values
        delta = out.get(f"delta_weekly_{feature}", pd.Series(float("nan"), index=out.index))
        out[f"std_delta_weekly_{feature}"] = delta.astype(float) / sd_clipped
        # Robust variant (MAD-based, no winsor floor; if pre MAD = 0, leave NaN).
        mad_col = f"pre_mad_{feature}"
        if mad_col in out.columns:
            mad = out[mad_col].astype(float).where(out[mad_col].astype(float) > 0)
            out[f"robust_std_delta_weekly_{feature}"] = delta.astype(float) / mad
    return out


# ----------------------------- Aggregate summary -----------------------------


def aggregate_one(rows: pd.DataFrame, label: str, features: List[str]) -> Dict[str, Any]:
    """Function summary: aggregate per-user shift columns across a (sub-)cohort into one summary row."""
    summary: Dict[str, Any] = {"summary_label": label, "n_users": int(len(rows))}
    if rows.empty:
        return summary

    weekly_col = f"std_delta_weekly_{COMPOSITE_NAME}"
    pooled_col = f"t_user_pooled_{COMPOSITE_NAME}"
    delta_pooled_col = f"delta_pooled_{COMPOSITE_NAME}"
    se_pooled_col = f"se_pooled_delta_{COMPOSITE_NAME}"
    delta_weekly_col = f"delta_weekly_{COMPOSITE_NAME}"

    for thr in (1.0, 2.0):
        if weekly_col in rows.columns:
            v = rows[weekly_col].astype(float).dropna()
            if not v.empty:
                summary[f"share_weekly_gt_p{int(thr)}"] = float((v > thr).mean())
                summary[f"share_weekly_lt_n{int(thr)}"] = float((v < -thr).mean())
        if pooled_col in rows.columns:
            v = rows[pooled_col].astype(float).dropna()
            if not v.empty:
                summary[f"share_pooled_t_gt_p{int(thr)}"] = float((v > thr).mean())
                summary[f"share_pooled_t_lt_n{int(thr)}"] = float((v < -thr).mean())

    # Wilcoxon and sign tests via simple computation (avoid scipy dependency).
    if delta_weekly_col in rows.columns:
        v = rows[delta_weekly_col].astype(float).dropna()
        if not v.empty:
            summary["mean_delta_weekly_composite"] = float(v.mean())
            summary["median_delta_weekly_composite"] = float(v.median())
            summary["sign_test_pos_share_weekly"] = float((v > 0).mean())

    if delta_pooled_col in rows.columns:
        v = rows[delta_pooled_col].astype(float).dropna()
        if not v.empty:
            summary["mean_delta_pooled_composite"] = float(v.mean())
            summary["median_delta_pooled_composite"] = float(v.median())
            summary["sign_test_pos_share_pooled"] = float((v > 0).mean())

    # Inverse-variance weighted pooled effect on composite delta.
    if delta_pooled_col in rows.columns and se_pooled_col in rows.columns:
        df_ivw = rows[[delta_pooled_col, se_pooled_col]].dropna()
        df_ivw = df_ivw[df_ivw[se_pooled_col].astype(float) > 0]
        if not df_ivw.empty:
            w = 1.0 / (df_ivw[se_pooled_col].astype(float).values ** 2)
            d = df_ivw[delta_pooled_col].astype(float).values
            wsum = float(w.sum())
            ivw_mean = float(np.sum(d * w) / wsum) if wsum > 0 else float("nan")
            ivw_se = float(1.0 / math.sqrt(wsum)) if wsum > 0 else float("nan")
            summary["ivw_delta_pooled_composite"] = ivw_mean
            summary["ivw_se_pooled_composite"] = ivw_se
            summary["ivw_z_pooled_composite"] = ivw_mean / ivw_se if ivw_se and ivw_se > 0 else float("nan")
            summary["ivw_n_users"] = int(len(df_ivw))

    # Agreement diagnostic.
    if weekly_col in rows.columns and pooled_col in rows.columns:
        df_agree = rows[[weekly_col, pooled_col]].dropna()
        if not df_agree.empty:
            same_sign = (np.sign(df_agree[weekly_col]) == np.sign(df_agree[pooled_col])).mean()
            corr = float(df_agree.corr().iloc[0, 1]) if df_agree.shape[0] > 1 else float("nan")
            summary["agree_sign_share_weekly_vs_pooled"] = float(same_sign)
            summary["corr_std_delta_weekly_vs_t_pooled"] = corr

    summary["n_users_in_summary"] = int(len(rows))
    return summary


def aggregate_audit_rows(audit_df: pd.DataFrame, panel: pd.DataFrame) -> List[Dict[str, Any]]:
    """Function summary: count and characterize panel / pre_only / post_only / below_thresholds users for the audit table."""
    rows: List[Dict[str, Any]] = []
    if audit_df.empty:
        return rows
    by_cat = audit_df.groupby("audit_category")
    panel_authors_by_cat = {cat: set(grp["author"].astype(str).tolist()) for cat, grp in by_cat}
    for cat, grp in by_cat:
        author_set = panel_authors_by_cat.get(cat, set())
        cat_panel = panel[panel["author"].astype(str).isin(author_set)]
        n_users = int(len(grp))
        n_comments = int(cat_panel["n_comments"].astype(float).sum()) if not cat_panel.empty else 0
        n_words = int(cat_panel["n_words"].astype(float).sum()) if not cat_panel.empty else 0
        topic_dist: Dict[str, int] = {}
        if not cat_panel.empty and "top_topic" in cat_panel.columns:
            agg = cat_panel.groupby("top_topic")["n_words"].sum().sort_values(ascending=False)
            for topic, words in agg.items():
                topic_dist[str(topic)] = int(words)
        rows.append(
            {
                "summary_label": f"audit_{cat}",
                "n_users": n_users,
                "n_comments": n_comments,
                "n_words": n_words,
                "topic_word_distribution_json": json.dumps(topic_dist, sort_keys=True),
            }
        )
    return rows


# ----------------------------- Cohort runner -----------------------------


def run_one_cohort(
    panel: pd.DataFrame,
    thresholds: CohortThresholds,
    launch_iso_week: str,
    drop_launch_week: bool,
    sd_winsor_pct: float,
    features: List[str],
    placebo_offset_weeks: int,
    paths: RuntimePaths,
) -> Dict[str, Any]:
    """Function summary: run one cohort (strict or loose) including main analysis, audits, topic splits, and placebo, and write outputs."""
    label = thresholds.label
    print(
        f"[analyze_user_pre_post_shift] cohort={label} thresholds={thresholds.__dict__}",
        flush=True,
    )

    labelled = label_pre_post(panel, launch_iso_week=launch_iso_week, drop_launch_week=drop_launch_week)
    pre_pool = labelled[(labelled["period"] == "pre") & (labelled["n_words"].astype(float) >= float(thresholds.min_words_per_week))]
    scales = freeze_composite_zscale(pre_pool)
    print(
        f"[analyze_user_pre_post_shift] cohort={label} pre_pool_user_weeks={int(len(pre_pool))} components_in_scales={list(scales.keys())}",
        flush=True,
    )
    panel_with_composite = add_composite_to_panel(labelled, scales)

    # Persist scales (one file shared across cohorts; last cohort wins, but they should be identical
    # if input panel is the same — we keep one file per cohort to be unambiguous).
    scales_path = paths.user_week_tables_dir / f"composite_zscale_pre_{label}.json"
    scales_path.write_text(json.dumps(scales, indent=2, sort_keys=True), encoding="utf-8")

    user_df, audit_df = per_user_summary(panel_with_composite, thresholds=thresholds, scales=scales, features=features)
    user_df = apply_sd_winsor_floor(user_df, features=features, pct=float(sd_winsor_pct))

    summary_rows: List[Dict[str, Any]] = []
    summary_rows.append(aggregate_one(user_df, label="panel", features=features))

    # Topic-stable sub-cohort: top_topic_pre == top_topic_post and both non-empty.
    if not user_df.empty:
        topic_stable_mask = (user_df["top_topic_pre"] == user_df["top_topic_post"]) & user_df["top_topic_pre"].astype(bool)
        topic_stable_df = user_df[topic_stable_mask]
        summary_rows.append(aggregate_one(topic_stable_df, label="panel_topic_stable", features=features))
        # Topic-stratified summaries.
        for topic, group in user_df.groupby("top_topic_pre", sort=True):
            topic_str = str(topic) if topic is not None else "unknown"
            summary_rows.append(aggregate_one(group, label=f"panel_topic={topic_str}", features=features))

    # Audit rows describing pre_only / post_only / below_thresholds.
    summary_rows.extend(aggregate_audit_rows(audit_df, panel=panel))

    # Placebo: shift launch by N weeks and rerun analysis.
    placebo_summary: Dict[str, Any] = {}
    if int(placebo_offset_weeks) > 0:
        placebo_iso_week_dt = datetime.fromisoformat(launch_iso_week + "T00:00:00+00:00") - timedelta(weeks=int(placebo_offset_weeks))
        placebo_iso_week_str = placebo_iso_week_dt.date().isoformat()
        placebo_labelled = label_pre_post(panel, launch_iso_week=placebo_iso_week_str, drop_launch_week=drop_launch_week)
        placebo_pre = placebo_labelled[
            (placebo_labelled["period"] == "pre") & (placebo_labelled["n_words"].astype(float) >= float(thresholds.min_words_per_week))
        ]
        placebo_scales = freeze_composite_zscale(placebo_pre)
        placebo_panel = add_composite_to_panel(placebo_labelled, placebo_scales)
        placebo_user_df, _placebo_audit = per_user_summary(
            placebo_panel, thresholds=thresholds, scales=placebo_scales, features=features
        )
        placebo_user_df = apply_sd_winsor_floor(placebo_user_df, features=features, pct=float(sd_winsor_pct))
        placebo_summary = aggregate_one(placebo_user_df, label=f"placebo_offset_weeks={int(placebo_offset_weeks)}", features=features)
        summary_rows.append(placebo_summary)

    # Write outputs.
    out_user = paths.user_week_tables_dir / f"shift_per_user_{label}.csv"
    out_summary = paths.user_week_tables_dir / f"shift_summary_{label}.csv"
    out_audit = paths.user_week_tables_dir / f"shift_audit_per_user_{label}.csv"
    user_df.to_csv(out_user, index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    audit_df.to_csv(out_audit, index=False)
    print(
        f"[analyze_user_pre_post_shift] cohort={label} wrote: per_user={out_user.name} ({len(user_df)} rows), "
        f"summary={out_summary.name} ({len(summary_rows)} rows), audit={out_audit.name} ({len(audit_df)} rows)",
        flush=True,
    )

    return {"label": label, "n_users_panel": int(len(user_df)), "summary_rows": summary_rows}


# ----------------------------- main -----------------------------


def main() -> None:
    """Function summary: load the user-week panel and execute the requested cohort analyses end-to-end."""
    args = parse_args()
    config = load_config(args.config)
    paths = build_paths(config)

    if not paths.panel_path.exists():
        raise FileNotFoundError(
            f"User-week panel not found: {paths.panel_path}. Run scripts/prepare_user_week_style_panel.py first."
        )

    panel = pd.read_parquet(paths.panel_path)
    if panel.empty:
        print("[analyze_user_pre_post_shift] panel_empty: nothing to analyze", flush=True)
        return

    # Resolve features.
    if args.features:
        feats = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        feats = list(DEFAULT_FEATURES)
    feats = [f for f in feats if f in FEATURE_SPECS]
    if not feats:
        feats = list(DEFAULT_FEATURES)

    # Launch ISO week.
    launch_iso_str = str(config["event_window"]["launch_day_utc"])
    launch_dt = datetime.fromisoformat(launch_iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    launch_iso_week = launch_iso_week_str(launch_dt)
    print(
        f"[analyze_user_pre_post_shift] launch_day_utc={launch_iso_str} launch_iso_week_start={launch_iso_week} "
        f"drop_launch_week={bool(args.drop_launch_week)} features={feats}",
        flush=True,
    )

    cohorts = make_thresholds_from_args(args)
    results: List[Dict[str, Any]] = []
    for thresholds in cohorts:
        # Suppress spurious numpy warnings (NaN comparisons in MAD with sparse data).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = run_one_cohort(
                panel=panel,
                thresholds=thresholds,
                launch_iso_week=launch_iso_week,
                drop_launch_week=bool(args.drop_launch_week),
                sd_winsor_pct=float(args.sd_winsor_pct),
                features=feats,
                placebo_offset_weeks=int(args.placebo_offset_weeks),
                paths=paths,
            )
            results.append(result)

    write_methods_note(paths.user_week_tables_dir / "shift_methods_note.txt", launch_iso_str=launch_iso_str)

    # One terse log line per cohort for quick eyeballing.
    summary_log = paths.user_week_logs_dir / "analyze_user_pre_post_shift.log"
    with summary_log.open("a", encoding="utf-8") as handle:
        for r in results:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} cohort={r['label']} n_users_panel={r['n_users_panel']} "
                f"summary_rows={len(r['summary_rows'])}\n"
            )
    print("[analyze_user_pre_post_shift] done", flush=True)


def write_methods_note(path: Path, launch_iso_str: str) -> None:
    """Function summary: write a short methods file documenting design choices, caveats, and what each output contains."""
    lines = [
        "Within-User Pre/Post Shift: Methods Note",
        "========================================",
        "",
        f"Launch anchor (UTC): {launch_iso_str}",
        "Time bin: ISO week (UTC, Monday start). The week containing the launch date is",
        "treated as 'launch'; pass --drop_launch_week to exclude it as a buffer.",
        "",
        "Two parallel comparisons per user, per feature:",
        "1. Weekly view: word-weighted weekly mean and SD across pre vs post weeks; std_delta",
        "   = delta / pre_sd, with pre_sd floored at the cohort's --sd_winsor_pct percentile.",
        "   robust_std_delta uses pre MAD instead of SD as a sensitivity column.",
        "   personal_t_weekly is a Welch-style across-weeks t.",
        "2. Pooled-comments view: pre and post pooled directly. Rate features use Poisson SE on raw",
        "   integer hits; binary-mean features use binomial p(1-p)/n; mean features use sumsq-derived",
        "   variance. Composite SE is the quadrature sum of component SE/sd contributions",
        "   (independence approximation, called out here).",
        "",
        "Composite ai_likeness_user_week:",
        "+ z(ai_word_rate_100w) + z(formality_balance_100w) + z(assistant_tone_rate_100w)",
        "+ z(list_structure_intensity) - z(contraction_rate_100w)",
        "Z-scales are frozen on the pre-launch user-week pool (word-weighted) and persisted to",
        "composite_zscale_pre_<cohort>.json so the same scaling is applied to all weeks.",
        "",
        "Cohort gating:",
        "Hard pre-launch + post-launch requirement. Users who lack good pre or good post weeks",
        "are excluded from the comparison and surfaced in shift_audit_per_user_<cohort>.csv and",
        "in the audit_* rows of shift_summary_<cohort>.csv.",
        "",
        "Forum / topic differentiation (forum-pooled per user as headline):",
        "shift_summary_<cohort>.csv contains additional rows for the topic-stable sub-cohort",
        "(top_topic_pre == top_topic_post) and per-topic stratifications (panel_topic=...).",
        "",
        "Placebo:",
        "If --placebo_offset_weeks > 0, the same analysis is rerun with the launch shifted back",
        "by that many ISO weeks; result appears as the placebo_offset_weeks=<n> row.",
        "",
        "What this does and does not say:",
        "It quantifies how unusual each user's post-launch level is relative to (a) their own",
        "weekly wiggle and (b) a pooled standard error tied to actual writing volume. Neither",
        "view proves causal use of ChatGPT.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
