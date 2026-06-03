"""
Author lean buckets for bucket-then-comment DiD (asymmetric lexical rules on labeling-window comments).

Labeling schemes: split_sample (cross-fit), holdout_2wk, naive_full_march.
Uses the same lexical bucket logic as user_week (no L/R hits -> neutral; else pole from mean net_ideology).
Separate from user_week semantic tail-week buckets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.descriptives_util import event_dates_from_config
from src.did.specs import CONTROL_FAMILIES, ITALY_FAMILIES
from src.user_week.ideology_buckets import UNCLASSIFIED, assign_lexical_buckets

MARCH_PREFIX = "2023-03"


@dataclass(frozen=True)
class BucketEventStudyConfig:
    """Function summary: parsed did.bucket_event_study YAML block."""

    schemes: Tuple[str, ...]
    outcome: str
    political_universe_only: bool
    treated_families: frozenset[str]
    bin_days: int
    ref_rel_period: int
    event_window_days: int
    bucket_method: str
    bucket_labels: Tuple[str, str, str]
    min_selection_comments: int
    split_method: str
    n_splits: int
    split_seed: int
    min_half_comments: int
    holdout_label_start: str
    holdout_label_end: str
    holdout_estimate_start: str
    naive_label_month: str
    outcome_scale: str
    standardization_window: str
    control_variants: Tuple[str, ...]
    ddd_control_variants: Tuple[str, ...]
    ddd_buckets: Tuple[str, str]
    run_pooled: bool
    run_descriptive_trajectories: bool
    balanced_panel: bool
    static_full_time_fe: bool
    bootstrap_draws: int
    bootstrap_seed: int
    cluster_robustness: Tuple[str, ...]
    trajectory_series: Tuple[Dict[str, Any], ...]


def bucket_event_study_config(config: Dict[str, Any]) -> BucketEventStudyConfig:
    """Function summary: parse did.bucket_event_study from study YAML.

    Parameters:
    - config: loaded project config.

    Returns:
    - BucketEventStudyConfig with defaults for omitted keys.
    """
    raw = (config.get("did") or {}).get("bucket_event_study") or {}
    if not isinstance(raw, dict):
        raw = {}
    labels_raw = raw.get("bucket_labels") or {}
    low = str(labels_raw.get("low", "conservative_leaning"))
    mid = str(labels_raw.get("mid", "neutral"))
    high = str(labels_raw.get("high", "liberal_leaning"))
    split_raw = raw.get("split_sample") or {}
    hold_raw = raw.get("holdout_2wk") or {}
    naive_raw = raw.get("naive_full_march") or {}
    inf_raw = raw.get("inference") or {}
    rob_raw = raw.get("static_robustness") or {}
    schemes = tuple(str(s) for s in (raw.get("schemes") or ["split_sample", "holdout_2wk", "naive_full_march"]))
    ddd_b = raw.get("ddd_buckets") or [high, low]
    if len(ddd_b) >= 2:
        ddd_buckets = (str(ddd_b[0]), str(ddd_b[1]))
    else:
        ddd_buckets = (high, low)
    traj = tuple(raw.get("trajectory_series") or [])
    return BucketEventStudyConfig(
        schemes=schemes,
        outcome=str(raw.get("outcome", "net_ideology")),
        political_universe_only=bool(raw.get("political_universe_only", True)),
        treated_families=frozenset(str(x) for x in (raw.get("treated_families") or ["it_political", "it_others"])),
        bin_days=int(raw.get("bin_days", 3)),
        ref_rel_period=int(raw.get("ref_rel_period", -1)),
        event_window_days=int(raw.get("event_window_days", 30)),
        bucket_method=str(raw.get("bucket_method", "asymmetric_lexical")),
        bucket_labels=(low, mid, high),
        min_selection_comments=int(raw.get("min_selection_comments", 5)),
        split_method=str(split_raw.get("split_method", "odd_even")),
        n_splits=int(split_raw.get("n_splits", 5)),
        split_seed=int(split_raw.get("seed", 42)),
        min_half_comments=int(split_raw.get("min_half_comments", 3)),
        holdout_label_start=str(hold_raw.get("label_start", "2023-03-01")),
        holdout_label_end=str(hold_raw.get("label_end", "2023-03-14")),
        holdout_estimate_start=str(hold_raw.get("estimate_start", "2023-03-15")),
        naive_label_month=str(naive_raw.get("label_month", MARCH_PREFIX)),
        outcome_scale=str(raw.get("outcome_scale", "raw")),
        standardization_window=str(raw.get("standardization_window", "march_pre")),
        control_variants=tuple(str(v) for v in (raw.get("control_variants") or ["all_controls_pooled"])),
        ddd_control_variants=tuple(
            str(v) for v in (raw.get("ddd_control_variants") or raw.get("control_variants") or ["all_controls_pooled"])
        ),
        ddd_buckets=ddd_buckets,
        run_pooled=bool(raw.get("run_pooled", True)),
        run_descriptive_trajectories=bool(raw.get("run_descriptive_trajectories", True)),
        balanced_panel=bool(raw.get("balanced_panel", True)),
        static_full_time_fe=bool(rob_raw.get("full_time_fe", True)),
        bootstrap_draws=int(inf_raw.get("bootstrap_draws", 199)),
        bootstrap_seed=int(inf_raw.get("bootstrap_seed", 42)),
        cluster_robustness=tuple(str(c) for c in (inf_raw.get("cluster_robustness") or ["subreddit", "author"])),
        trajectory_series=traj,
    )


def assert_net_ideology_sign(df: pd.DataFrame, outcome_col: str = "net_ideology") -> None:
    """Function summary: verify positive net_ideology aligns with left/liberal lexicon hits.

    Parameters:
    - df: comment rows with net_ideology and optional left/right hit columns.
    - outcome_col: ideology outcome column name.

    Raises:
    - AssertionError if sign convention fails on a scorable subsample.
    """
    if df.empty or outcome_col not in df.columns:
        return
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if "left_hits" in df.columns and "right_hits" in df.columns:
        left = pd.to_numeric(df["left_hits"], errors="coerce").fillna(0)
        right = pd.to_numeric(df["right_hits"], errors="coerce").fillna(0)
        mask_lr = (left > right) & y.notna()
        mask_rl = (right > left) & y.notna()
        if mask_lr.sum() >= 5 and mask_rl.sum() >= 5:
            assert float(y[mask_lr].mean()) > float(y[mask_rl].mean()), (
                "net_ideology sign check failed: expected higher net_ideology when left_hits > right_hits"
            )
            return
    if "left_rate_100w" in df.columns and "right_rate_100w" in df.columns:
        diff = pd.to_numeric(df["left_rate_100w"], errors="coerce") - pd.to_numeric(
            df["right_rate_100w"], errors="coerce"
        )
        mask = diff.notna() & y.notna()
        if mask.sum() >= 20:
            corr = float(diff[mask].corr(y[mask]))
            assert corr > 0, f"net_ideology should correlate positively with left-right rate diff (r={corr})"
            return
    pos = y[y > 0]
    neg = y[y < 0]
    if len(pos) >= 5 and len(neg) >= 5:
        assert float(pos.mean()) > float(neg.mean()), "net_ideology: positive values should exceed negative on average"


def _is_march(date_utc: pd.Series) -> pd.Series:
    """Function summary: True for calendar dates in March 2023."""
    return date_utc.astype(str).str.startswith(MARCH_PREFIX)


def labeling_window_mask(df: pd.DataFrame, scheme: str, bcfg: BucketEventStudyConfig) -> pd.Series:
    """Function summary: row mask for comments used to label author buckets.

    Parameters:
    - df: annotated comments with date_utc.
    - scheme: split_sample | holdout_2wk | naive_full_march.
    - bcfg: bucket event-study config.

    Returns:
    - Boolean Series aligned to df.index.
    """
    dates = df["date_utc"].astype(str)
    if scheme == "holdout_2wk":
        return (dates >= bcfg.holdout_label_start) & (dates <= bcfg.holdout_label_end)
    if scheme == "naive_full_march":
        return _is_march(dates)
    if scheme == "split_sample":
        return _is_march(dates)
    raise ValueError(f"Unknown labeling scheme: {scheme}")


def estimation_window_mask(
    df: pd.DataFrame,
    scheme: str,
    bcfg: BucketEventStudyConfig,
    config: Dict[str, Any],
    *,
    split_half_b_ids: Optional[pd.Index] = None,
) -> pd.Series:
    """Function summary: estimation window mask using event_window end from config."""
    dates = df["date_utc"].astype(str)
    _, end_excl, _, _ = event_dates_from_config(config)
    if scheme == "holdout_2wk":
        return (dates >= bcfg.holdout_estimate_start) & (dates < end_excl)
    if scheme == "naive_full_march":
        return dates < end_excl
    if scheme == "split_sample":
        if split_half_b_ids is None:
            raise ValueError("split_sample estimation requires split_half_b_ids")
        in_b = df["id"].astype(str).isin(split_half_b_ids.astype(str))
        april = (dates >= "2023-04-01") & (dates < end_excl)
        return in_b | april
    raise ValueError(f"Unknown scheme: {scheme}")


def assert_holdout_windows_disjoint(bcfg: BucketEventStudyConfig) -> None:
    """Function summary: assert holdout label and estimation windows do not overlap."""
    assert bcfg.holdout_label_end < bcfg.holdout_estimate_start, (
        f"holdout label end {bcfg.holdout_label_end} must be before estimate start {bcfg.holdout_estimate_start}"
    )


def split_march_halves(
    march_df: pd.DataFrame,
    method: str,
    seed: int,
    split_id: int,
) -> Tuple[pd.Index, pd.Index]:
    """Function summary: split March comments into disjoint halves for cross-fitting.

    Parameters:
    - march_df: March comments with id column.
    - method: odd_even | random.
    - seed: RNG base seed.
    - split_id: split index 0..K-1 for random rotations.

    Returns:
    - Tuple (half_a_ids, half_b_ids) as comment id indexes.
    """
    ids = march_df["id"].astype(str)
    if method == "odd_even":
        numeric = pd.to_numeric(march_df["id"], errors="coerce")
        if numeric.notna().all():
            half_a = ids[numeric % 2 == 0]
            half_b = ids[numeric % 2 == 1]
        else:
            parity = pd.util.hash_pandas_object(ids, index=False).astype(np.int64) % 2
            half_a = ids[parity == 0]
            half_b = ids[parity == 1]
    elif method == "random":
        rng = np.random.default_rng(seed + int(split_id))
        perm = rng.permutation(len(march_df))
        mid = len(perm) // 2
        idx_a = march_df.index[perm[:mid]]
        idx_b = march_df.index[perm[mid:]]
        half_a = march_df.loc[idx_a, "id"].astype(str)
        half_b = march_df.loc[idx_b, "id"].astype(str)
    else:
        raise ValueError(f"Unknown split_method: {method}")
    assert not set(half_a).intersection(set(half_b)), "split_sample halves must not share comment ids"
    return pd.Index(half_a), pd.Index(half_b)


def author_lean_features(
    df: pd.DataFrame,
    label_mask: pd.Series,
    outcome_col: str = "net_ideology",
) -> pd.DataFrame:
    """Function summary: per-author mean ideology and comment counts in labeling window.

    Parameters:
    - df: comment panel.
    - label_mask: rows used for labeling.
    - outcome_col: outcome column.

    Returns:
    - DataFrame indexed by author with lean_mean, n_label_comments, primary_lexicon.
    """
    sub = df.loc[label_mask]
    if sub.empty:
        return pd.DataFrame(columns=["author", "lean_mean", "n_label_comments", "primary_lexicon"])
    sub = sub.assign(y=pd.to_numeric(sub[outcome_col], errors="coerce"))
    g = sub.groupby("author", observed=True)
    def _lex_mode(s: pd.Series) -> str:
        m = s.astype(str).mode()
        return str(m.iloc[0]) if len(m) else "unknown"

    agg_parts: Dict[str, Any] = {
        "lean_mean": ("y", "mean"),
        "n_label_comments": ("author", "size"),
        "primary_lexicon": ("primary_lexicon", _lex_mode),
    }
    if "left_hits" in sub.columns:
        agg_parts["left_hits_sum"] = ("left_hits", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum()))
    if "right_hits" in sub.columns:
        agg_parts["right_hits_sum"] = ("right_hits", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum()))
    agg = g.agg(**agg_parts).reset_index()
    agg["author"] = agg["author"].astype(str)
    keep = ["author", "lean_mean", "n_label_comments", "primary_lexicon"]
    for col in ("left_hits_sum", "right_hits_sum"):
        if col in agg.columns:
            keep.append(col)
    return agg[keep]


def assign_lean_buckets(
    features: pd.DataFrame,
    bcfg: BucketEventStudyConfig,
) -> pd.Series:
    """Function summary: assign liberal/neutral/conservative buckets from labeling-window features.

    Parameters:
    - features: author_lean_features output (lean_mean, optional left/right hit sums).
    - bcfg: bucket config (bucket_method, bucket_labels).

    Returns:
    - Series bucket labels indexed like features (author index).
    """
    if features.empty:
        return pd.Series(dtype=str)
    work = features.copy()
    if "author" in work.columns:
        work = work.set_index(work["author"].astype(str))
    work = work[work["n_label_comments"] >= bcfg.min_selection_comments]
    if work.empty:
        return pd.Series(dtype=str)
    if bcfg.bucket_method == "asymmetric_lexical":
        if "left_hits_sum" not in work.columns or "right_hits_sum" not in work.columns:
            warnings.warn(
                "asymmetric_lexical: left_hits/right_hits missing on comment panel; "
                "all authors treated as zero hits (mostly neutral). "
                "Re-run prepare_did_comment_panel.py to project left_hits and right_hits.",
                stacklevel=2,
            )
        lex_in = work.rename(
            columns={
                "lean_mean": "lexical_score",
                "left_hits_sum": "left_hits_pre",
                "right_hits_sum": "right_hits_pre",
            }
        )
        if "left_hits_pre" not in lex_in.columns:
            lex_in["left_hits_pre"] = 0.0
        if "right_hits_pre" not in lex_in.columns:
            lex_in["right_hits_pre"] = 0.0
        return assign_lexical_buckets(lex_in, bcfg.bucket_labels)
    if bcfg.bucket_method == "fixed_threshold":
        low, mid, high = bcfg.bucket_labels
        buckets = pd.Series(UNCLASSIFIED, index=work.index, dtype=str)
        s = work["lean_mean"].astype(float)
        buckets.loc[s > 0] = high
        buckets.loc[s < 0] = low
        buckets.loc[s == 0] = mid
        return buckets
    if bcfg.bucket_method == "tertile_within_language":
        raise ValueError(
            "bucket_method=tertile_within_language is removed (peer ranking within language). "
            "Use asymmetric_lexical (default) or fixed_threshold in did.bucket_event_study."
        )
    raise ValueError(f"Unknown bucket_method: {bcfg.bucket_method}")


def build_lean_bucket_table(
    df: pd.DataFrame,
    scheme: str,
    bcfg: BucketEventStudyConfig,
    config: Dict[str, Any],
    *,
    split_id: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: author bucket assignments for one scheme (and optional split).

    Parameters:
    - df: full annotated comment panel.
    - scheme: labeling scheme.
    - bcfg: config.
    - config: full YAML for window asserts.
    - split_id: for split_sample, which cross-fit split.

    Returns:
    - DataFrame with scheme, author, bucket, lean_mean, n_label_comments, primary_lexicon, split_id.
    """
    if scheme == "holdout_2wk":
        assert_holdout_windows_disjoint(bcfg)
    label_mask = labeling_window_mask(df, scheme, bcfg)
    if scheme == "split_sample":
        if split_id is None:
            split_id = 0
        march = df.loc[label_mask]
        half_a, half_b = split_march_halves(march, bcfg.split_method, bcfg.split_seed, split_id)
        label_mask = label_mask & df["id"].astype(str).isin(half_a.astype(str))
        ha_n = march["id"].astype(str).isin(half_a.astype(str))
        hb_n = march["id"].astype(str).isin(half_b.astype(str))
        assert not set(march.loc[ha_n, "id"].astype(str)).intersection(
            set(march.loc[hb_n, "id"].astype(str))
        ), "split halves must be disjoint"
        eligible_authors: set[str] = set()
        for author, grp in march.groupby("author", observed=True):
            na = int(ha_n.loc[grp.index].sum())
            nb = int(hb_n.loc[grp.index].sum())
            if na >= bcfg.min_half_comments and nb >= bcfg.min_half_comments:
                eligible_authors.add(str(author))
        feats = author_lean_features(df, label_mask, bcfg.outcome)
        if not feats.empty:
            feats = feats[feats["author"].astype(str).isin(eligible_authors)]
    else:
        feats = author_lean_features(df, label_mask, bcfg.outcome)
    if feats.empty:
        return pd.DataFrame()
    buckets = assign_lean_buckets(feats, bcfg)
    out = feats.set_index("author").join(buckets.rename("bucket"), how="inner")
    out = out.reset_index()
    out["scheme"] = scheme
    if split_id is not None:
        out["split_id"] = int(split_id)
    return out


def march_standardization_moments(
    df: pd.DataFrame,
    config: Dict[str, Any],
    bcfg: BucketEventStudyConfig,
    outcome_col: str = "net_ideology",
) -> pd.DataFrame:
    """Function summary: per-language mean and std from March pre-period for z-scoring.

    Parameters:
    - df: comments with date_utc, primary_lexicon, outcome.
    - config: study config.
    - bcfg: bucket config.

    Returns:
    - DataFrame with primary_lexicon, mu, sigma, n_comments.
    """
    if bcfg.standardization_window == "march_pre":
        work = df.loc[df["rel_day"].astype(int) < 0]
    else:
        work = df.loc[_is_march(df["date_utc"].astype(str))]
    rows: List[Dict[str, Any]] = []
    for lex, grp in work.groupby(work["primary_lexicon"].astype(str), observed=True):
        y = pd.to_numeric(grp[outcome_col], errors="coerce").dropna()
        if len(y) < 2:
            rows.append({"primary_lexicon": lex, "mu": 0.0, "sigma": 1.0, "n_comments": len(y)})
            continue
        sigma = float(y.std(ddof=0))
        rows.append(
            {
                "primary_lexicon": lex,
                "mu": float(y.mean()),
                "sigma": sigma if sigma > 1e-9 else 1.0,
                "n_comments": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def apply_outcome_scale(
    df: pd.DataFrame,
    moments: pd.DataFrame,
    bcfg: BucketEventStudyConfig,
    outcome_col: str = "net_ideology",
    out_col: str = "y",
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Function summary: add estimation outcome column y (raw or standardized).

    Parameters:
    - df: comment panel.
    - moments: march_standardization_moments table.
    - bcfg: config.
    - outcome_col: raw outcome.
    - out_col: output column name.
    - copy: when False, write y into df without copying (caller must own df).

    Returns:
    - DataFrame with y column.
    """
    out = df.copy() if copy else df
    raw = pd.to_numeric(out[outcome_col], errors="coerce")
    if bcfg.outcome_scale != "standardized":
        out[out_col] = raw.astype("float32")
        return out
    mu_map = moments.set_index("primary_lexicon")["mu"].to_dict()
    sig_map = moments.set_index("primary_lexicon")["sigma"].to_dict()
    lex = out["primary_lexicon"].astype(str)
    mu = lex.map(mu_map).fillna(0.0)
    sig = lex.map(sig_map).fillna(1.0).replace(0, 1.0)
    out[out_col] = ((raw - mu) / sig).astype("float32")
    return out


def control_variant_mask(
    df: pd.DataFrame,
    variant_id: str,
    bcfg: BucketEventStudyConfig,
) -> pd.Series:
    """Function summary: boolean mask for IT vs control rows per control_variant id.

    Parameters:
    - df: comment panel with topic_family (and primary_lexicon for vs_en).
    - variant_id: control variant key.
    - bcfg: bucket config.

    Returns:
    - Boolean Series aligned to df.index.
    """
    fam = df["topic_family"].astype(str)
    it_mask = fam.isin(bcfg.treated_families)
    if variant_id == "all_controls_pooled":
        return it_mask | fam.isin(CONTROL_FAMILIES)
    if variant_id == "vs_de":
        return it_mask | fam.eq("de")
    if variant_id == "vs_en":
        lex = df.get("primary_lexicon", pd.Series("en", index=df.index)).astype(str)
        return it_mask | (lex.eq("en") & ~it_mask)
    if variant_id == "vs_uk":
        return it_mask | fam.eq("uk")
    if variant_id == "it_political_vs_it_others":
        return fam.eq("it_political") | fam.eq("it_others")
    raise ValueError(f"Unknown control_variant: {variant_id}")


def filter_control_variant(df: pd.DataFrame, variant_id: str, bcfg: BucketEventStudyConfig) -> pd.DataFrame:
    """Function summary: restrict sample to IT vs control per control_variant id.

    Parameters:
    - df: comment panel with IT, topic_family, primary_lexicon.
    - variant_id: control variant key.
    - bcfg: bucket config.

    Returns:
    - Filtered DataFrame.
    """
    mask = control_variant_mask(df, variant_id, bcfg)
    if variant_id == "it_political_vs_it_others":
        out = df.loc[mask].copy()
        out["IT"] = df.loc[mask, "topic_family"].astype(str).eq("it_political").astype(int)
        return out
    return df.loc[mask]


def balanced_author_set(df: pd.DataFrame) -> frozenset[str]:
    """Function summary: authors with at least one pre- and post-ban comment.

    Parameters:
    - df: comment panel with author and post or rel_day.

    Returns:
    - Frozenset of author id strings.
    """
    if "post" in df.columns:
        pre_auth = set(df.loc[df["post"].astype(int) == 0, "author"].astype(str))
        post_auth = set(df.loc[df["post"].astype(int) == 1, "author"].astype(str))
    else:
        pre_auth = set(df.loc[df["rel_day"].astype(int) < 0, "author"].astype(str))
        post_auth = set(df.loc[df["rel_day"].astype(int) >= 0, "author"].astype(str))
    return frozenset(pre_auth & post_auth)


def filter_balanced_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep authors with at least one pre- and post-ban comment.

    Parameters:
    - df: estimation sample with author, rel_day or post.

    Returns:
    - Subset of df for balanced panel.
    """
    keep = balanced_author_set(df)
    return df[df["author"].astype(str).isin(keep)]


def merge_buckets(
    df: pd.DataFrame,
    buckets: pd.DataFrame,
    scheme: str,
    split_id: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: attach frozen bucket labels to comment rows.

    Parameters:
    - df: comments.
    - buckets: build_lean_bucket_table output.
    - scheme: scheme name filter.
    - split_id: optional split filter.

    Returns:
    - df with bucket column; unlabeled authors dropped.
    """
    b = buckets[buckets["scheme"].astype(str) == scheme].copy()
    if split_id is not None and "split_id" in b.columns:
        b = b[b["split_id"].astype(int) == int(split_id)]
    b = b.drop_duplicates(subset=["author"], keep="first")
    out = df.merge(b[["author", "bucket", "lean_mean", "n_label_comments", "primary_lexicon"]], on="author", how="inner")
    return out[out["bucket"].astype(str) != UNCLASSIFIED].copy()


def write_lean_buckets_csv(path: Path, table: pd.DataFrame) -> None:
    """Function summary: write lean bucket assignment table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def estimation_sample_mask(
    df: pd.DataFrame,
    scheme: str,
    bcfg: BucketEventStudyConfig,
    config: Dict[str, Any],
    split_id: Optional[int] = None,
) -> pd.Series:
    """Function summary: boolean mask for comments in the estimation window.

    Parameters:
    - df: full comment panel.
    - scheme: labeling scheme.
    - bcfg: bucket config.
    - config: study YAML.
    - split_id: required for split_sample.

    Returns:
    - Boolean Series.
    """
    if scheme == "split_sample":
        if split_id is None:
            split_id = 0
        march = df[_is_march(df["date_utc"].astype(str))]
        _, half_b = split_march_halves(march, bcfg.split_method, bcfg.split_seed, split_id)
        return estimation_window_mask(df, scheme, bcfg, config, split_half_b_ids=half_b)
    return estimation_window_mask(df, scheme, bcfg, config)


def build_all_lean_buckets(
    df: pd.DataFrame,
    bcfg: BucketEventStudyConfig,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: concatenate bucket tables for all schemes and split_sample splits.

    Parameters:
    - df: comment panel.
    - bcfg: config.
    - config: study YAML.

    Returns:
    - Combined lean bucket assignment DataFrame.
    """
    parts: List[pd.DataFrame] = []
    for scheme in bcfg.schemes:
        if scheme == "split_sample":
            for sid in range(bcfg.n_splits):
                t = build_lean_bucket_table(df, scheme, bcfg, config, split_id=sid)
                if not t.empty:
                    parts.append(t)
        else:
            t = build_lean_bucket_table(df, scheme, bcfg, config)
            if not t.empty:
                parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
