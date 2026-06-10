"""
Event-study level-leak robustness diagnostic (Italy AI-ban bucket event study).

PURPOSE
-------
The headline bucket event study estimates
    y ~ i(rel_period, IT, ref=-1) | author + time_id
(`src/did/bucket_estimate.py::estimate_comment_it_event_study`). Because `IT`
is author-invariant and the comment panel is unbalanced (many authors are not
observed in the reference bin), the *permanent* Italy-vs-control level gap leaks
into every non-reference event-time coefficient `gamma_k`, while bin -1 is pinned
to 0. The result is a flat negative plateau with an artificial "spike" at bin -1,
most severe for high level-gap outcomes (all semantic axes, aggression/negative/
extremity) and negligible for net_ideology. The existing `balanced_panel: true`
does NOT fix this: `balanced_author_set` only requires one pre- and one post-ban
comment, not presence in the reference bin.

This script re-estimates the comment-level event study under three specifications,
for every outcome, labeling scheme, and semantic ideology bucket (`all`,
`liberal_leaning`, `conservative_leaning`, `neutral`), at 1-day or 3-day bins:

  B       baseline TWFE   : production spec on the balanced sample (shows the leak).
  FD_ref  first-diff ref  : author bin means differenced vs reference bin (-1).
  FD_mean first-diff mean : author bin means differenced vs comment-weighted pre-ban mean.

Spec A (reference-anchored TWFE) was dropped — the initial 3d pooled run showed it
does not remove the leak. Baseline TWFE (B) is kept only as a visual contrast;
**FD_ref** and **FD_mean** are the leak-proof dynamic estimates.

It re-uses the production sample-construction helpers so spec B matches the main
`event_study_{scheme}.csv` rows per bucket; it is read-only with respect to the
main pipeline and writes tables to `robustness/` and figures to `FD/` and `baseline/`.

HOW TO RUN
----------
    python scripts/diagnostics/event_study_level_robustness.py \\
        --config config/italy_polarization_setup.yaml --bin-days 3 --stratification semantic

Optional filters:
    --outcomes sem_axis_anti_establishment,net_ideology
    --schemes naive_full_march,holdout_2wk
    --buckets all,liberal_leaning
    --fd-baselines ref,preban_mean
    --no-figures

OUTPUTS
-------
    results/tables/.../did/bucket_event_study/{bin}d/strat_{strat}/robustness/
        event_study_level_robustness_by_bucket.csv
        level_leak_summary_by_bucket.csv
    results/figures/.../did/bucket_event_study/{bin}d/
        FD/ref/{scheme}/{bucket}/{outcome}.png
        FD/preban_mean/{scheme}/{bucket}/{outcome}.png
        baseline/{scheme}/{bucket}/{outcome}.png

Replot figures from existing CSV (no re-estimation):
    python scripts/diagnostics/event_study_level_robustness.py \\
        --config config/italy_polarization_setup.yaml --bin-days 3 --figures-only
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FDBaseline = Literal["ref", "preban_mean"]

DEFAULT_BUCKETS = ("all", "liberal_leaning", "conservative_leaning", "neutral")
DEFAULT_FD_BASELINES: Tuple[FDBaseline, ...] = ("ref", "preban_mean")
SPEC_B = "B"
SPEC_FD_REF = "FD_ref"
SPEC_FD_MEAN = "FD_mean"


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute repo root Path (also injected into sys.path by the bootstrap).
    """
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.bucket_estimate import (  # noqa: E402
    _es_rows_from_fit,
    estimate_comment_it_event_study,
)
from src.did.lean_buckets import (  # noqa: E402
    apply_outcome_scale,
    balanced_author_set,
    build_all_semantic_buckets,
    bucket_event_study_config,
    bucket_event_study_outcomes,
    control_variant_mask,
    estimation_sample_mask,
    march_standardization_moments,
    merge_buckets,
)
from src.did.outputs import plot_event_study  # noqa: E402
from src.did.panels import load_comment_panel  # noqa: E402
from src.did.paths import did_root  # noqa: E402

CONTROL_VARIANT = "all_controls_pooled"
TIDY_COLS = [
    "bin_days",
    "stratification",
    "outcome",
    "scheme",
    "bucket",
    "spec",
    "rel_period",
    "rel_day",
    "gamma",
    "se",
    "ci_low",
    "ci_high",
    "pvalue",
    "n_obs",
    "n_authors",
]


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for the level-leak robustness diagnostic.

    Returns:
    - argparse.Namespace with config, bin_days, stratification, bucket/spec filters, no_figures.
    """
    p = argparse.ArgumentParser(description="Event-study level-leak robustness diagnostic.")
    p.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    p.add_argument(
        "--bin-days",
        type=int,
        choices=(1, 3),
        default=3,
        help="Event calendar bin width (1 or 3); needs matching prepared comment panel.",
    )
    p.add_argument(
        "--stratification",
        type=str,
        choices=("semantic", "lexical"),
        default="semantic",
        help="Bucket stratification for per-bucket runs (default: semantic).",
    )
    p.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="Comma-separated buckets (default: all,liberal_leaning,conservative_leaning,neutral).",
    )
    p.add_argument(
        "--fd-baselines",
        type=str,
        default=None,
        help="Comma-separated FD baselines: ref, preban_mean (default: both).",
    )
    p.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated subset of outcomes (default: config outcome + additional_outcomes).",
    )
    p.add_argument(
        "--schemes",
        type=str,
        default=None,
        help="Comma-separated subset of labeling schemes (default: all config schemes).",
    )
    p.add_argument("--no-figures", action="store_true", help="Skip event-study figures.")
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Replot figures from existing by_bucket CSV; skip panel load and estimation.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="After estimation, spot-check spec B vs existing event_study_{scheme}.csv rows.",
    )
    return p.parse_args()


def robustness_tables_dir(config: Dict[str, Any], bin_days: int, stratification: str) -> Path:
    """Function summary: tables output root for combined by-bucket robustness CSVs.

    Parameters:
    - config: study YAML.
    - bin_days: 1 or 3.
    - stratification: semantic or lexical.

    Returns:
    - Path ending in .../robustness/ under the stratification tag.
    """
    base = did_root(config) / "bucket_event_study" / f"{int(bin_days)}d"
    if stratification == "semantic":
        return base / "strat_semantic" / "robustness"
    return base / "robustness"


def fd_figures_root(config: Dict[str, Any], bin_days: int) -> Path:
    """Function summary: figures root for FD and baseline event-study PNGs.

    Parameters:
    - config: study YAML.
    - bin_days: 1 or 3.

    Returns:
    - Path to .../bucket_event_study/{bin}d/ under figures_dir.
    """
    return figures_subdir(config, "did") / "bucket_event_study" / f"{int(bin_days)}d"


def figure_path_for_spec(
    root: Path,
    spec: str,
    scheme: str,
    bucket: str,
    outcome: str,
) -> Path:
    """Function summary: output PNG path for one spec under FD/ or baseline/.

    Parameters:
    - root: fd_figures_root path.
    - spec: B, FD_ref, or FD_mean.
    - scheme: labeling scheme.
    - bucket: ideology bucket label.
    - outcome: outcome column id.

    Returns:
    - Full path ending in {outcome}.png under the spec folder tree.
    """
    if spec == SPEC_B:
        return root / "baseline" / scheme / bucket / f"{outcome}.png"
    if spec == SPEC_FD_REF:
        return root / "FD" / "ref" / scheme / bucket / f"{outcome}.png"
    if spec == SPEC_FD_MEAN:
        return root / "FD" / "preban_mean" / scheme / bucket / f"{outcome}.png"
    raise ValueError(f"unknown spec for figure path: {spec}")


def tidy_group_to_es_df(grp: pd.DataFrame) -> pd.DataFrame:
    """Function summary: convert tidy CSV rows for one cell into an es_df for plotting.

    Parameters:
    - grp: rows sharing outcome, scheme, bucket, and spec.

    Returns:
    - DataFrame with rel_period, rel_day, gamma, se, ci_low, ci_high, pvalue.
    """
    if grp.empty:
        return pd.DataFrame()
    cols = ["rel_period", "rel_day", "gamma", "se", "ci_low", "ci_high", "pvalue"]
    out = grp[cols].copy()
    out["rel_period"] = out["rel_period"].astype(int)
    return out.sort_values("rel_period").reset_index(drop=True)


def write_single_spec_figure(
    es_df: pd.DataFrame,
    outcome: str,
    scheme: str,
    bucket: str,
    out_path: Path,
) -> bool:
    """Function summary: write one single-series event-study PNG.

    Parameters:
    - es_df: coefficient frame with rel_period, gamma, se.
    - outcome: outcome id for the title.
    - scheme: labeling scheme.
    - bucket: ideology bucket label.
    - out_path: PNG output path.

    Returns:
    - True if a figure was written, False if skipped (<2 finite SEs or empty).
    """
    if es_df.empty:
        return False
    se = pd.to_numeric(es_df.get("se"), errors="coerce")
    if int((se.notna() & (se > 1e-12)).sum()) < 2:
        return False
    plot_event_study(
        es_df,
        outcome,
        out_path,
        rel_col="rel_period",
        title=f"{outcome} / {scheme} / {bucket}",
    )
    return True


def write_figures_from_tidy(
    tidy_df: pd.DataFrame,
    config: Dict[str, Any],
    bin_days: int,
    *,
    outcomes: Optional[Sequence[str]] = None,
    schemes: Optional[Sequence[str]] = None,
    buckets: Optional[Sequence[str]] = None,
) -> int:
    """Function summary: replot all single-spec figures from a tidy by_bucket CSV.

    Parameters:
    - tidy_df: full event_study_level_robustness_by_bucket.csv contents.
    - config: study YAML.
    - bin_days: 1 or 3.
    - outcomes, schemes, buckets: optional filters.

    Returns:
    - Count of PNG files written.
    """
    root = fd_figures_root(config, bin_days)
    work = tidy_df[tidy_df["bin_days"].astype(int) == int(bin_days)].copy()
    if outcomes:
        work = work[work["outcome"].astype(str).isin(outcomes)]
    if schemes:
        work = work[work["scheme"].astype(str).isin(schemes)]
    if buckets:
        work = work[work["bucket"].astype(str).isin(buckets)]

    n_written = 0
    group_cols = ["outcome", "scheme", "bucket", "spec"]
    for keys, grp in work.groupby(group_cols, observed=True):
        outcome, scheme, bucket, spec = (str(k) for k in keys)
        if spec not in (SPEC_B, SPEC_FD_REF, SPEC_FD_MEAN):
            continue
        es_df = tidy_group_to_es_df(grp)
        out_path = figure_path_for_spec(root, spec, scheme, bucket, outcome)
        if write_single_spec_figure(es_df, outcome, scheme, bucket, out_path):
            n_written += 1
    return n_written


def production_event_study_path(
    config: Dict[str, Any],
    bin_days: int,
    stratification: str,
    outcome: str,
    scheme: str,
) -> Path:
    """Function summary: path to the production event_study_{scheme}.csv for validation.

    Parameters:
    - config: study YAML.
    - bin_days: 1 or 3.
    - stratification: semantic or lexical.
    - outcome: outcome column.
    - scheme: labeling scheme.

    Returns:
    - Path to event_study_{scheme}.csv under the production tables tree.
    """
    base = did_root(config) / "bucket_event_study" / f"{int(bin_days)}d"
    if stratification == "semantic":
        root = base / "strat_semantic" / outcome
    elif outcome == "net_ideology":
        root = base
    else:
        root = base / "strat_lexical" / outcome
    return root / f"event_study_{scheme}.csv"


def build_pool(
    df: pd.DataFrame,
    scheme: str,
    bcfg: Any,
    config: Dict[str, Any],
    moments: pd.DataFrame,
    balanced_authors: Optional[frozenset],
    split_id: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: reproduce the production pooled all_controls_pooled sample.

    Mirrors `bucket_event_study._estimation_frame` + the pooled control-variant
    filter so spec B matches the existing event_study_{scheme}.csv rows.

    Parameters:
    - df: full prepared comment panel.
    - scheme: labeling scheme (split_sample uses split_id window).
    - bcfg: bucket event-study config (with outcome already set).
    - config: study YAML.
    - moments: march standardization moments (used only when outcome_scale=standardized).
    - balanced_authors: precomputed balanced author set, or None when not balancing.
    - split_id: split_sample half id (default 0 when scheme is split_sample).

    Returns:
    - Pooled IT-vs-control estimation sample with outcome column y attached.
    """
    sid = split_id if scheme == "split_sample" else None
    if scheme == "split_sample" and sid is None:
        sid = 0
    mask = estimation_sample_mask(df, scheme, bcfg, config, split_id=sid)
    pool = apply_outcome_scale(df.loc[mask], moments, bcfg, bcfg.outcome, "y")
    if balanced_authors is not None:
        pool = pool[pool["author"].astype(str).isin(balanced_authors)]
    pool = pool.loc[control_variant_mask(pool, CONTROL_VARIANT, bcfg)]
    return pool


def build_pool_bucket(
    df: pd.DataFrame,
    scheme: str,
    bcfg: Any,
    config: Dict[str, Any],
    moments: pd.DataFrame,
    balanced_authors: Optional[frozenset],
    buckets: pd.DataFrame,
    bucket: str,
    split_id: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: estimation sample for one ideology bucket (or pooled all).

    Parameters:
    - df: full prepared comment panel.
    - scheme: labeling scheme.
    - bcfg: bucket config with outcome set.
    - config: study YAML.
    - moments: standardization moments.
    - balanced_authors: balanced author set or None.
    - buckets: semantic (or lexical) bucket assignment table.
    - bucket: "all" or a bucket label.
    - split_id: split_sample id (0 for split_sample schemes).

    Returns:
    - Estimation sample restricted to the requested bucket.
    """
    pool = build_pool(df, scheme, bcfg, config, moments, balanced_authors, split_id=split_id)
    if bucket == "all":
        return pool
    sid = split_id if scheme == "split_sample" else None
    if scheme == "split_sample" and sid is None:
        sid = 0
    merged = merge_buckets(pool, buckets, scheme, sid)
    return merged[merged["bucket"].astype(str) == bucket].copy()


def _rel_period_series(df: pd.DataFrame, bin_days: int) -> pd.Series:
    """Function summary: integer rel_period series, derived from rel_day if absent.

    Parameters:
    - df: comment sample with rel_period or rel_day.
    - bin_days: bin width for rel_day // bin_days fallback.

    Returns:
    - Integer-valued rel_period Series aligned to df.index.
    """
    if "rel_period" in df.columns:
        return df["rel_period"].astype(int)
    return (df["rel_day"].astype(int) // int(bin_days)).astype(int)


def _author_preban_mean(
    work: pd.DataFrame,
    bin_days: int,
) -> pd.DataFrame:
    """Function summary: comment-weighted pre-ban mean y per author.

    Parameters:
    - work: comment sample with author, y, rel_period/rel_day.
    - bin_days: bin width for rel_period derivation.

    Returns:
    - DataFrame with columns author, preban_mean.
    """
    rel = _rel_period_series(work, bin_days)
    pre = work.loc[rel < 0].copy()
    if pre.empty:
        return pd.DataFrame(columns=["author", "preban_mean"])
    parts = []
    for author, g in pre.groupby("author", observed=True):
        y = g["y"].astype(float)
        parts.append({"author": str(author), "preban_mean": float(y.mean())})
    return pd.DataFrame(parts)


def estimate_first_difference_event_study(
    pool: pd.DataFrame,
    ref_period: int,
    window: int,
    bin_days: int,
    baseline: FDBaseline = "ref",
    y_col: str = "y",
    cluster_col: str = "author",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: author first-difference event study.

    Collapses comments to author x rel_period means, differences each author's
    bin mean against either the reference bin (baseline="ref") or the author's
    comment-weighted pre-ban mean (baseline="preban_mean"), then estimates
        dy ~ i(rel_period, IT) | rel_period
    weighted by per-cell comment count and clustered by author.

    Parameters:
    - pool: pooled estimation sample with author, IT, rel_period/rel_day, y.
    - ref_period: reference bin for baseline="ref" (dropped from output).
    - window: trim rel_period to [-window, window].
    - bin_days: bin width for rel_day display and rel_period fallback.
    - baseline: "ref" (vs bin -1) or "preban_mean" (vs author pre-ban mean).
    - y_col: outcome column.
    - cluster_col: cluster id for CRV1 SEs (author after the collapse).

    Returns:
    - Tuple (meta dict with n_obs/n_authors, es_df with rel_period/gamma/se/ci/pvalue).
    """
    work = pool.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author"])
    work["author"] = work["author"].astype(str)
    work["IT"] = work["IT"].astype(float)
    work["rel_period"] = _rel_period_series(work, bin_days)
    work = work[work["rel_period"].between(-window, window)]
    meta_empty = {"n_obs": int(len(work)), "n_authors": 0}
    if work.empty or work["IT"].nunique() < 2:
        return {**meta_empty, "estimation_note": "no_variation"}, pd.DataFrame()

    if baseline == "preban_mean":
        preban = _author_preban_mean(work, bin_days)
        if preban.empty:
            return {**meta_empty, "estimation_note": "no_preban_authors"}, pd.DataFrame()
        work = work.merge(preban, on="author", how="inner")
        baseline_col = "preban_mean"
        omit_ref_for_parse = -9999
    else:
        ref = (
            work.loc[work["rel_period"] == int(ref_period), ["author", "y"]]
            .groupby("author", observed=True)["y"]
            .mean()
            .reset_index(name="ref_y")
        )
        if ref.empty:
            return {**meta_empty, "estimation_note": "no_ref_bin_authors"}, pd.DataFrame()
        work = work.merge(ref, on="author", how="inner")
        baseline_col = "ref_y"
        omit_ref_for_parse = int(ref_period)

    cell = (
        work.groupby(["author", "rel_period"], observed=True)
        .agg(mean_y=("y", "mean"), n=("y", "size"), IT=("IT", "first"), baseline=(baseline_col, "first"))
        .reset_index()
    )
    cell["dy"] = cell["mean_y"] - cell["baseline"]
    if baseline == "ref":
        cell = cell[cell["rel_period"] != int(ref_period)].copy()
    n_authors = int(cell["author"].nunique())
    meta = {
        "n_obs": int(work.shape[0]) if baseline == "preban_mean" else int(work[work["rel_period"] != int(ref_period)].shape[0]),
        "n_authors": n_authors,
    }
    if cell.empty or cell["IT"].nunique() < 2 or cell["rel_period"].nunique() < 2:
        return {**meta, "estimation_note": "no_fd_variation"}, pd.DataFrame()

    try:
        from pyfixest.estimation import feols
    except ImportError:
        return {**meta, "estimation_note": "pyfixest_missing"}, pd.DataFrame()
    vcov: Any = {"CRV1": cluster_col} if cluster_col in cell.columns else "iid"
    cell["w"] = cell["n"].astype(float).clip(lower=1e-9)
    try:
        fit = feols(
            "dy ~ i(rel_period, IT) | rel_period",
            data=cell,
            vcov=vcov,
            weights="w",
        )
    except Exception:
        return {**meta, "estimation_note": "fd_estimation_error"}, pd.DataFrame()
    rows = _es_rows_from_fit(
        fit, "rel_period", omit_ref_for_parse, bin_days=bin_days, gamma_col="gamma"
    )
    es_df = pd.DataFrame(rows).sort_values("rel_period") if rows else pd.DataFrame()
    meta["estimation_note"] = "ok"
    meta["fd_baseline"] = baseline
    return meta, es_df


def plateau_summary(es_df: pd.DataFrame) -> Tuple[float, float]:
    """Function summary: mean |gamma| over pre-ban and post-ban bins (artifact size).

    Parameters:
    - es_df: event-study coefficient frame with rel_period and gamma.

    Returns:
    - Tuple (pre_plateau_mean_abs_gamma, post_plateau_mean_abs_gamma); NaN if absent.
    """
    if es_df.empty or "gamma" not in es_df.columns:
        return float("nan"), float("nan")
    rel = es_df["rel_period"].astype(int)
    g = es_df["gamma"].astype(float).abs()
    pre = float(g[rel < 0].mean()) if (rel < 0).any() else float("nan")
    post = float(g[rel >= 0].mean()) if (rel >= 0).any() else float("nan")
    return pre, post


def _tidy_rows(
    es_df: pd.DataFrame,
    *,
    bin_days: int,
    stratification: str,
    outcome: str,
    scheme: str,
    bucket: str,
    spec: str,
    n_obs: int,
    n_authors: int,
) -> List[Dict[str, Any]]:
    """Function summary: convert an es_df to tidy output rows with run metadata.

    Parameters:
    - es_df: event-study coefficient frame.
    - bin_days, stratification, outcome, scheme, bucket, spec: run identifiers.
    - n_obs, n_authors: sample sizes for this spec.

    Returns:
    - List of dict rows restricted to TIDY_COLS.
    """
    if es_df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for _, r in es_df.iterrows():
        out.append(
            {
                "bin_days": int(bin_days),
                "stratification": stratification,
                "outcome": outcome,
                "scheme": scheme,
                "bucket": bucket,
                "spec": spec,
                "rel_period": int(r["rel_period"]),
                "rel_day": int(r.get("rel_day", int(r["rel_period"]) * bin_days)),
                "gamma": float(r["gamma"]),
                "se": float(r.get("se", float("nan"))),
                "ci_low": float(r.get("ci_low", float("nan"))),
                "ci_high": float(r.get("ci_high", float("nan"))),
                "pvalue": float(r.get("pvalue", float("nan"))),
                "n_obs": int(n_obs),
                "n_authors": int(n_authors),
            }
        )
    return out


def run_outcome_scheme_bucket(
    df: pd.DataFrame,
    outcome: str,
    scheme: str,
    bucket: str,
    bcfg: Any,
    config: Dict[str, Any],
    balanced_authors: Optional[frozenset],
    buckets: pd.DataFrame,
    stratification: str,
    fd_baselines: Sequence[FDBaseline],
    split_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Function summary: estimate B + FD specs for one outcome x scheme x bucket.

    Parameters:
    - df: full prepared comment panel.
    - outcome: estimation outcome column.
    - scheme: labeling scheme.
    - bucket: ideology bucket label or "all".
    - bcfg: base bucket config (outcome overridden inside).
    - config: study YAML.
    - balanced_authors: precomputed balanced author set or None.
    - buckets: bucket assignment table for merge_buckets.
    - stratification: semantic or lexical (metadata only).
    - fd_baselines: which FD baselines to run (ref and/or preban_mean).
    - split_id: split_sample id (0 for split_sample).

    Returns:
    - Tuple (tidy_rows, summary_rows, es_by_spec) where es_by_spec maps spec->es_df.
    """
    bcfg_oc = replace(bcfg, outcome=outcome)
    moments = march_standardization_moments(df, config, bcfg_oc, outcome_col=outcome)
    pool = build_pool_bucket(
        df, scheme, bcfg_oc, config, moments, balanced_authors, buckets, bucket, split_id
    )
    ref = int(bcfg_oc.ref_rel_period)
    window = int(bcfg_oc.event_window_days)
    bin_days = int(bcfg_oc.bin_days)

    tidy: List[Dict[str, Any]] = []
    summ: List[Dict[str, Any]] = []
    es_by_spec: Dict[str, pd.DataFrame] = {}
    if pool.empty:
        return tidy, summ, es_by_spec

    # B: baseline full (balanced) sample — production TWFE (shows leak).
    _, es_b = estimate_comment_it_event_study(pool, ref_period=ref, window=window, bin_days=bin_days)
    nb_obs, nb_auth = int(len(pool)), int(pool["author"].astype(str).nunique())
    es_by_spec[SPEC_B] = es_b
    tidy.extend(
        _tidy_rows(
            es_b,
            bin_days=bin_days,
            stratification=stratification,
            outcome=outcome,
            scheme=scheme,
            bucket=bucket,
            spec=SPEC_B,
            n_obs=nb_obs,
            n_authors=nb_auth,
        )
    )
    pre, post = plateau_summary(es_b)
    summ.append(
        {
            "bin_days": bin_days,
            "stratification": stratification,
            "outcome": outcome,
            "scheme": scheme,
            "bucket": bucket,
            "spec": SPEC_B,
            "pre_plateau_mean_abs_gamma": pre,
            "post_plateau_mean_abs_gamma": post,
            "n_obs": nb_obs,
            "n_authors": nb_auth,
        }
    )

    fd_specs: List[Tuple[str, FDBaseline]] = []
    if "ref" in fd_baselines:
        fd_specs.append((SPEC_FD_REF, "ref"))
    if "preban_mean" in fd_baselines:
        fd_specs.append((SPEC_FD_MEAN, "preban_mean"))

    for spec_label, baseline in fd_specs:
        fd_meta, es_fd = estimate_first_difference_event_study(
            pool, ref, window, bin_days, baseline=baseline
        )
        n_obs = int(fd_meta.get("n_obs", 0))
        n_auth = int(fd_meta.get("n_authors", 0))
        es_by_spec[spec_label] = es_fd
        tidy.extend(
            _tidy_rows(
                es_fd,
                bin_days=bin_days,
                stratification=stratification,
                outcome=outcome,
                scheme=scheme,
                bucket=bucket,
                spec=spec_label,
                n_obs=n_obs,
                n_authors=n_auth,
            )
        )
        pre, post = plateau_summary(es_fd)
        summ.append(
            {
                "bin_days": bin_days,
                "stratification": stratification,
                "outcome": outcome,
                "scheme": scheme,
                "bucket": bucket,
                "spec": spec_label,
                "pre_plateau_mean_abs_gamma": pre,
                "post_plateau_mean_abs_gamma": post,
                "n_obs": n_obs,
                "n_authors": n_auth,
            }
        )
    return tidy, summ, es_by_spec


def write_cell_figures(
    es_by_spec: Dict[str, pd.DataFrame],
    outcome: str,
    scheme: str,
    bucket: str,
    fig_root: Path,
) -> int:
    """Function summary: write separate B / FD_ref / FD_mean PNGs for one cell.

    Parameters:
    - es_by_spec: mapping spec -> es_df (with rel_period, gamma, se).
    - outcome: outcome id for the title.
    - scheme: labeling scheme.
    - bucket: ideology bucket label.
    - fig_root: fd_figures_root path.

    Returns:
    - Count of PNG files written.
    """
    n = 0
    for spec in (SPEC_B, SPEC_FD_REF, SPEC_FD_MEAN):
        es_df = es_by_spec.get(spec)
        if es_df is None or es_df.empty:
            continue
        out_path = figure_path_for_spec(fig_root, spec, scheme, bucket, outcome)
        if write_single_spec_figure(es_df, outcome, scheme, bucket, out_path):
            n += 1
    return n


def validate_spec_b(
    tidy_df: pd.DataFrame,
    config: Dict[str, Any],
    bin_days: int,
    stratification: str,
    *,
    atol: float = 1e-4,
) -> List[str]:
    """Function summary: compare spec B rows to production event_study CSVs.

    Parameters:
    - tidy_df: combined tidy output with spec B rows.
    - config: study YAML.
    - bin_days: 1 or 3.
    - stratification: semantic or lexical.
    - atol: absolute tolerance on gamma.

    Returns:
    - List of validation message strings (empty if all checks pass).
    """
    messages: List[str] = []
    spec_b = tidy_df[tidy_df["spec"] == SPEC_B].copy()
    if spec_b.empty:
        messages.append("validate: no spec B rows to check")
        return messages

    for (outcome, scheme, bucket), grp in spec_b.groupby(["outcome", "scheme", "bucket"], observed=True):
        prod_path = production_event_study_path(config, bin_days, stratification, str(outcome), str(scheme))
        if not prod_path.is_file():
            messages.append(f"validate: missing production CSV {prod_path}")
            continue
        prod = pd.read_csv(prod_path)
        prod = prod[
            (prod["control_variant"].astype(str) == CONTROL_VARIANT)
            & (prod["bucket"].astype(str) == str(bucket))
        ]
        if str(scheme) == "split_sample" and "split_id" in prod.columns:
            prod = prod[prod["split_id"].astype(int) == 0]
        if prod.empty:
            messages.append(f"validate: no production rows outcome={outcome} scheme={scheme} bucket={bucket}")
            continue
        merged = grp.merge(
            prod[["rel_period", "gamma"]].rename(columns={"gamma": "gamma_prod"}),
            on="rel_period",
            how="inner",
        )
        if merged.empty:
            messages.append(f"validate: no overlapping rel_period outcome={outcome} scheme={scheme} bucket={bucket}")
            continue
        diff = (merged["gamma"] - merged["gamma_prod"]).abs()
        bad = diff[diff > atol]
        if not bad.empty:
            messages.append(
                f"validate FAIL outcome={outcome} scheme={scheme} bucket={bucket}: "
                f"max |Δgamma|={float(diff.max()):.6f} n_bad={len(bad)}"
            )
        else:
            messages.append(
                f"validate OK outcome={outcome} scheme={scheme} bucket={bucket}: "
                f"max |Δgamma|={float(diff.max()):.6f} n={len(merged)}"
            )
    return messages


def _parse_bucket_list(raw: Optional[str]) -> Tuple[str, ...]:
    """Function summary: parse comma-separated bucket CLI argument.

    Parameters:
    - raw: comma-separated bucket names or None.

    Returns:
    - Tuple of bucket labels.
    """
    if not raw:
        return DEFAULT_BUCKETS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _parse_fd_baselines(raw: Optional[str]) -> Tuple[FDBaseline, ...]:
    """Function summary: parse comma-separated FD baseline CLI argument.

    Parameters:
    - raw: comma-separated ref/preban_mean or None.

    Returns:
    - Tuple of FD baseline mode strings.
    """
    if not raw:
        return DEFAULT_FD_BASELINES
    out: List[FDBaseline] = []
    for part in raw.split(","):
        p = part.strip()
        if p in ("ref", "preban_mean"):
            out.append(p)  # type: ignore[arg-type]
    return tuple(out) if out else DEFAULT_FD_BASELINES


def main() -> None:
    """Function summary: run estimation and/or replot figures from by_bucket CSV."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    bcfg = bucket_event_study_config(config)
    bcfg = replace(bcfg, bin_days=int(args.bin_days))
    stratification = str(args.stratification)

    outcomes = (
        tuple(s.strip() for s in args.outcomes.split(",") if s.strip())
        if args.outcomes
        else bucket_event_study_outcomes(bcfg)
    )
    schemes = (
        tuple(s.strip() for s in args.schemes.split(",") if s.strip())
        if args.schemes
        else tuple(bcfg.schemes)
    )
    bucket_list = _parse_bucket_list(args.buckets)

    tables_dir = robustness_tables_dir(config, int(bcfg.bin_days), stratification)
    tidy_path = tables_dir / "event_study_level_robustness_by_bucket.csv"

    if args.figures_only:
        if not tidy_path.is_file():
            print(f"[level_robustness] missing tidy CSV: {tidy_path}", flush=True)
            return
        print(
            f"[level_robustness] figures-only from {tidy_path} ({bcfg.bin_days}d)...",
            flush=True,
        )
        tidy_df = pd.read_csv(tidy_path)
        n = write_figures_from_tidy(
            tidy_df,
            config,
            int(bcfg.bin_days),
            outcomes=outcomes if args.outcomes else None,
            schemes=schemes if args.schemes else None,
            buckets=bucket_list if args.buckets else None,
        )
        print(f"[level_robustness] wrote {n} figures under {fd_figures_root(config, int(bcfg.bin_days))}", flush=True)
        return

    fd_baselines = _parse_fd_baselines(args.fd_baselines)

    print(
        f"[level_robustness] loading prepared comment panel ({bcfg.bin_days}d) "
        f"stratification={stratification} buckets={bucket_list} fd={fd_baselines}...",
        flush=True,
    )
    df = load_comment_panel(config, bin_days=int(bcfg.bin_days))
    if df.empty:
        print("[level_robustness] empty panel; aborting", flush=True)
        return
    balanced_authors = balanced_author_set(df) if bcfg.balanced_panel else None

    if stratification == "semantic":
        buckets = build_all_semantic_buckets(bcfg, config)
    else:
        from src.did.lean_buckets import build_all_lean_buckets

        buckets = build_all_lean_buckets(df, bcfg, config)
    if buckets.empty and any(b != "all" for b in bucket_list):
        print("[level_robustness] empty bucket table; only 'all' bucket will run", flush=True)

    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_root = fd_figures_root(config, int(bcfg.bin_days))

    tidy_all: List[Dict[str, Any]] = []
    summ_all: List[Dict[str, Any]] = []
    n_figs = 0
    for outcome in outcomes:
        if outcome not in df.columns:
            print(f"[level_robustness] skip outcome={outcome}: column missing", flush=True)
            continue
        for scheme in schemes:
            split_id = 0 if scheme == "split_sample" else None
            for bucket in bucket_list:
                if bucket != "all" and buckets.empty:
                    continue
                print(
                    f"[level_robustness] outcome={outcome} scheme={scheme} bucket={bucket}",
                    flush=True,
                )
                tidy, summ, es_by_spec = run_outcome_scheme_bucket(
                    df,
                    outcome,
                    scheme,
                    bucket,
                    bcfg,
                    config,
                    balanced_authors,
                    buckets,
                    stratification,
                    fd_baselines,
                    split_id=split_id,
                )
                tidy_all.extend(tidy)
                summ_all.extend(summ)
                if not args.no_figures and es_by_spec:
                    n_figs += write_cell_figures(es_by_spec, outcome, scheme, bucket, fig_root)

    tidy_df = pd.DataFrame(tidy_all, columns=TIDY_COLS)
    summ_df = pd.DataFrame(summ_all)
    summ_path = tables_dir / "level_leak_summary_by_bucket.csv"
    tidy_df.to_csv(tidy_path, index=False)
    summ_df.to_csv(summ_path, index=False)
    print(f"[level_robustness] wrote {len(tidy_df)} coef rows -> {tidy_path}", flush=True)
    print(f"[level_robustness] wrote {len(summ_df)} summary rows -> {summ_path}", flush=True)
    if not args.no_figures:
        print(f"[level_robustness] wrote {n_figs} figures under {fig_root}", flush=True)

    if args.validate and not tidy_df.empty:
        for msg in validate_spec_b(tidy_df, config, int(bcfg.bin_days), stratification):
            print(f"[level_robustness] {msg}", flush=True)


if __name__ == "__main__":
    main()
