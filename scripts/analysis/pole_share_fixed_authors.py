"""
Script summary:
Fixed-author robustness check for headline pole_share (Italy ChatGPT-ban DiD).

Functionality:
- Loads Mar–Apr comment shards with ideology hit columns (left/center/right_hits).
- Defines a pre-ban author roster: authors with >=1 ideology-lexicon hit in 2023-03-01..2023-03-30.
- Rebuilds subreddit-day pole_share for (a) all authors and (b) fixed-author subset.
- Drops subreddit-days with fewer than min_scored comments (default 3).
- Estimates cross_country_all TWFE statics (full_ban, early_ban_7d) and 3-day event studies (ref=-1).
- Event study uses pyfixest i(rel_period, IT) | subreddit + time_id with manual two-step cross-check.
- Writes summary + event-study CSVs under did/pole_share_fixed_authors/ (read-only w.r.t. main pipeline).

How to apply/run:
  .venv/bin/python scripts/analysis/pole_share_fixed_authors.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/pole_share_fixed_authors.py --min-scored 1 --no-figures
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PRE_START = "2023-03-01"
PRE_END = "2023-03-30"
BIN_DAYS = 3
REF_REL_PERIOD = -1
MIN_CELL_COUNT = 5
MIN_FORUM_DAYS = 30
CROSS_EST_TOL = 1e-6
STATIC_TOL = 0.03


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from scripts.diagnostics.prepare_did_subreddit_panel import (  # noqa: E402
    _add_treatment_flags,
    _annotate_subreddit_panel,
)
from scripts.diagnostics.prepare_polarization_descriptives import (  # noqa: E402
    daily_subreddit_table,
    load_comment_frame,
    validate_feature_columns_present,
)
from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    load_polarization_config,
    require_dominant_v1_ideology_scoring,
    resolve_primary_subreddits,
    subreddit_family_map,
    tables_subdir,
)
from src.did.bucket_estimate import (  # noqa: E402
    estimate_panel_it_event_study,
    manual_panel_it_event_study,
)
from src.did.english_quality import DEFAULT_BOT_AUTHORS  # noqa: E402
from src.did.estimate import estimate_pretrend_f, run_strategy_twfe  # noqa: E402
from src.did.event_study_panels import prepare_subreddit_event_study_panel  # noqa: E402
from src.did.outputs import plot_event_study  # noqa: E402
from src.did.specs import (  # noqa: E402
    EVENT_WINDOW_DAYS_BY_BIN,
    StrategySpec,
    filter_strategy_sample,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for fixed-author pole_share robustness."""
    parser = argparse.ArgumentParser(description="Fixed-author pole_share DiD robustness.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--min-scored",
        type=int,
        default=3,
        help="Minimum scored comments per subreddit-day (sensitivity at 1 or 5).",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip standalone ES figure.")
    return parser.parse_args()


def _filter_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: drop deleted and known bot authors.

    Parameters:
    - df: comment frame with author column.

    Returns:
    - Filtered copy.
    """
    out = df.copy()
    out["author"] = out["author"].astype(str)
    bad = DEFAULT_BOT_AUTHORS | {"[deleted]", "AutoModerator", ""}
    return out[~out["author"].isin(bad)]


def _ideology_hit_mask(df: pd.DataFrame) -> pd.Series:
    """Function summary: True when comment has any ideology lexicon hit.

    Parameters:
    - df: frame with left/center/right_hits.

    Returns:
    - Boolean series.
    """
    left = pd.to_numeric(df.get("left_hits", 0), errors="coerce").fillna(0)
    center = pd.to_numeric(df.get("center_hits", 0), errors="coerce").fillna(0)
    right = pd.to_numeric(df.get("right_hits", 0), errors="coerce").fillna(0)
    return (left + center + right) > 0


def _fixed_author_set(df: pd.DataFrame) -> set[str]:
    """Function summary: authors with >=1 ideology hit in pre-ban window.

    Parameters:
    - df: filtered comment frame in event window.

    Returns:
    - Set of author ids.
    """
    pre = df[(df["date_utc"] >= PRE_START) & (df["date_utc"] <= PRE_END)]
    hit = pre[_ideology_hit_mask(pre)]
    return set(hit["author"].astype(str).unique())


def _build_subreddit_day_panel(
    df: pd.DataFrame,
    pol_cfg: Dict[str, Any],
    launch: str,
    end_excl: str,
    min_scored: int,
) -> pd.DataFrame:
    """Function summary: aggregate comments to subreddit-day pole_share with min-n gate.

    Parameters:
    - df: comment-level frame.
    - pol_cfg: polarization config.
    - launch: ban launch date.
    - end_excl: corpus end exclusive.
    - min_scored: minimum comments per subreddit-day.

    Returns:
    - Annotated subreddit-day panel.
    """
    daily = daily_subreddit_table(df, pol_cfg)
    if daily.empty:
        return daily
    daily = daily[daily["n_comments"].astype(int) >= int(min_scored)].copy()
    daily = _annotate_subreddit_panel(daily, launch, end_excl)
    daily["entity_id"] = daily["subreddit"].astype(str)
    daily["time_id"] = daily["date_utc"].astype(str)
    return daily


def _rel_bin_cell_counts(es_sample: pd.DataFrame) -> pd.DataFrame:
    """Function summary: count IT and control forum-days per rel_period with valid pole_share.

    Parameters:
    - es_sample: binned ES estimation sample.

    Returns:
    - DataFrame with rel_period, n_it, n_control.
    """
    work = es_sample.dropna(subset=["pole_share"]).copy()
    work["rel_period"] = pd.to_numeric(work["rel_period"], errors="coerce").astype("Int64")
    work["IT"] = pd.to_numeric(work["IT"], errors="coerce").fillna(0).astype(int)
    rows: List[Dict[str, Any]] = []
    for rp in sorted(work["rel_period"].dropna().unique()):
        sub = work[work["rel_period"] == rp]
        rows.append(
            {
                "rel_period": int(rp),
                "n_it": int((sub["IT"] == 1).sum()),
                "n_control": int((sub["IT"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _print_cell_counts(
    counts: pd.DataFrame,
    sample_label: str,
    *,
    repair_note: str = "",
) -> List[int]:
    """Function summary: log IT/control cell counts and return sparse rel_period ids.

    Parameters:
    - counts: output of _rel_bin_cell_counts.
    - sample_label: all_authors or fixed_authors.
    - repair_note: optional repair description.

    Returns:
    - rel_period values where either n_it or n_control < MIN_CELL_COUNT.
    """
    header = f"[pole_share_fixed_authors] cell counts ({sample_label})"
    if repair_note:
        header = f"{header} [{repair_note}]"
    print(header, flush=True)
    sparse: List[int] = []
    for _, row in counts.iterrows():
        rp = int(row["rel_period"])
        n_it = int(row["n_it"])
        n_ctrl = int(row["n_control"])
        flag = ""
        if n_it < MIN_CELL_COUNT or n_ctrl < MIN_CELL_COUNT:
            sparse.append(rp)
            flag = " ** SPARSE"
        print(f"  rel_period={rp:3d}  IT={n_it:4d}  control={n_ctrl:4d}{flag}", flush=True)
    return sparse


def _repair_sparse_bins(
    es_sample: pd.DataFrame,
    sparse_bins: Sequence[int],
    sample_label: str,
) -> Tuple[pd.DataFrame, str]:
    """Function summary: drop sparse boundary bins or restrict to forums with dense coverage.

    Parameters:
    - es_sample: binned ES sample before repair.
    - sparse_bins: rel_period ids flagged sparse.
    - sample_label: sample name for logging.

    Returns:
    - Tuple (repaired sample, repair description).
    """
    if not sparse_bins:
        return es_sample, ""
    work = es_sample.copy()
    drop = sorted(set(sparse_bins))
    repaired = work[~work["rel_period"].isin(drop)].copy()
    counts_after = _rel_bin_cell_counts(repaired)
    still_sparse = [
        int(r["rel_period"])
        for _, r in counts_after.iterrows()
        if int(r["n_it"]) < MIN_CELL_COUNT or int(r["n_control"]) < MIN_CELL_COUNT
    ]
    if not still_sparse:
        note = f"dropped sparse boundary bins {drop}"
        print(f"[pole_share_fixed_authors] {sample_label}: {note}", flush=True)
        return repaired, note
    day_counts = (
        repaired.dropna(subset=["pole_share"])
        .groupby("subreddit", observed=True)
        .size()
    )
    keep_subs = day_counts[day_counts >= MIN_FORUM_DAYS].index.astype(str)
    restricted = repaired[repaired["subreddit"].astype(str).isin(keep_subs)].copy()
    note = f"dropped bins {drop}; restricted to {len(keep_subs)} forums (>={MIN_FORUM_DAYS} days)"
    print(f"[pole_share_fixed_authors] {sample_label}: {note}", flush=True)
    return restricted, note


def _bin_weights(es_sample: pd.DataFrame) -> pd.Series:
    """Function summary: n_comments sum per rel_period for ES aggregation weights."""
    return (
        es_sample.dropna(subset=["pole_share"])
        .groupby("rel_period", observed=True)["n_comments"]
        .sum()
    )


def _weighted_gamma_mean(
    es_df: pd.DataFrame,
    weights: pd.Series,
    *,
    rel_min: Optional[int] = None,
    rel_max: Optional[int] = None,
) -> float:
    """Function summary: n_comments-weighted mean gamma over selected rel_period range.

    Parameters:
    - es_df: event-study coefficients.
    - weights: n_comments per rel_period.
    - rel_min: inclusive lower rel_period bound (optional).
    - rel_max: inclusive upper rel_period bound (optional).

    Returns:
    - Weighted mean gamma, or NaN when no rows qualify.
    """
    rel = pd.to_numeric(es_df["rel_period"], errors="coerce")
    sub = es_df.copy()
    if rel_min is not None:
        sub = sub[rel >= rel_min]
    if rel_max is not None:
        sub = sub[rel <= rel_max]
    if sub.empty:
        return float("nan")
    vals: List[float] = []
    wts: List[float] = []
    for _, row in sub.iterrows():
        rp = int(row["rel_period"])
        g = float(row["gamma"])
        w = float(weights.get(rp, np.nan))
        if np.isfinite(g) and np.isfinite(w) and w > 0:
            vals.append(g)
            wts.append(w)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=wts))


def _validate_es_gates(
    es_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    static_beta: float,
    es_sample: pd.DataFrame,
    sample_label: str,
    repair_note: str,
) -> None:
    """Function summary: hard sanity gates; raise SystemExit on any violation.

    Parameters:
    - es_df: pyfixest event-study coefficients.
    - manual_df: manual two-step coefficients.
    - static_beta: full_ban static TWFE beta from same sample.
    - es_sample: estimation sample for post-period weighting.
    - sample_label: all_authors or fixed_authors.
    - repair_note: sparse-bin repair description for logging.

    Returns:
    - None; raises SystemExit when a gate fails.
    """
    if es_df.empty:
        raise SystemExit(f"[pole_share_fixed_authors] gate fail ({sample_label}): empty ES")

    # Gate (a): cross-estimator agreement
    merged = es_df.merge(
        manual_df,
        on="rel_period",
        how="outer",
        suffixes=("_py", "_man"),
    )
    for _, row in merged.iterrows():
        rp = int(row["rel_period"])
        g_py = float(row.get("gamma_py", np.nan))
        g_man = float(row.get("gamma_man", np.nan))
        if not np.isfinite(g_py) or not np.isfinite(g_man):
            raise SystemExit(
                f"[pole_share_fixed_authors] gate (a) fail ({sample_label}): "
                f"missing coef at rel_period={rp}"
            )
        if abs(g_py - g_man) > CROSS_EST_TOL:
            raise SystemExit(
                f"[pole_share_fixed_authors] gate (a) fail ({sample_label}): "
                f"rel_period={rp} pyfixest={g_py:.8f} manual={g_man:.8f} "
                f"diff={abs(g_py - g_man):.2e} > {CROSS_EST_TOL}"
            )

    gamma = pd.to_numeric(es_df["gamma"], errors="coerce")
    se = pd.to_numeric(es_df["se"], errors="coerce")
    rel = pd.to_numeric(es_df["rel_period"], errors="coerce")

    # Gate (c): magnitude bounds
    if (gamma.abs() >= 0.5).any():
        bad = es_df.loc[gamma.abs() >= 0.5, "rel_period"].tolist()
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): |gamma|>=0.5 at {bad}"
        )
    if (se >= 0.5).any():
        bad = es_df.loc[se >= 0.5, "rel_period"].tolist()
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): SE>=0.5 at {bad}"
        )

    early = es_df[rel.isin([0, 1, 2])]
    if len(early) < 3:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): missing early bins 0–2"
        )
    early_mean = float(early["gamma"].mean())
    if abs(early_mean) >= 0.03:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): "
            f"early bins 0–2 mean gamma={early_mean:.4f} (need |mean|<0.03)"
        )

    late = es_df[rel.isin([8, 9, 10])]
    if late.empty:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): no late bins 8–10"
        )
    late_max = float(late["gamma"].max())
    late_mean = float(late["gamma"].mean())
    if late_max <= 0.04:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): "
            f"late peak max gamma={late_max:.4f} <= 0.04"
        )
    if late_mean < 0.02:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (c) fail ({sample_label}): "
            f"late bins 8–10 mean gamma={late_mean:.4f} < 0.02"
        )

    # Gate (b): static consistency via implied post-minus-pre ATT
    weights = _bin_weights(es_sample)
    post_wmean = _weighted_gamma_mean(es_df, weights, rel_min=0)
    pre_wmean = _weighted_gamma_mean(es_df, weights, rel_max=-2)
    implied_att = post_wmean - pre_wmean
    if not np.isfinite(implied_att) or abs(implied_att - static_beta) > STATIC_TOL:
        raise SystemExit(
            f"[pole_share_fixed_authors] gate (b) fail ({sample_label}): "
            f"implied ATT={implied_att:.4f} (post={post_wmean:.4f}, pre={pre_wmean:.4f}) "
            f"vs static beta={static_beta:.4f} (tol={STATIC_TOL})"
        )

    # Gate (d): diagnostics logged
    print(
        f"[pole_share_fixed_authors] gates OK ({sample_label}): "
        f"repair={repair_note or 'none'}; implied_ATT={implied_att:.4f} "
        f"(post_wmean={post_wmean:.4f}, pre_wmean={pre_wmean:.4f}) static={static_beta:.4f}",
        flush=True,
    )


def _estimate_specs(
    panel: pd.DataFrame,
    config: Dict[str, Any],
    sample_label: str,
    window_days: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Function summary: run cross_country_all statics and 3d event study for pole_share.

    Parameters:
    - panel: annotated subreddit-day panel (daily).
    - sample_label: all_authors or fixed_authors.
    - window_days: calendar-day ES window for 3d binning.

    Returns:
    - Tuple (summary rows list, event-study coefficient dataframe).
    """
    meta = panel[["subreddit", "topic_family"]].drop_duplicates("subreddit")
    binned = prepare_subreddit_event_study_panel(
        panel, config, BIN_DAYS, entity_cols=("subreddit",)
    )
    binned = binned.merge(meta, on="subreddit", how="left")
    binned = _add_treatment_flags(binned)

    summary_rows: List[Dict[str, Any]] = []
    static_beta = float("nan")

    for spec_id in ("full_ban", "early_ban_7d"):
        spec_strat = StrategySpec("cross_country_all", post_mode=spec_id)
        static_res = run_strategy_twfe(
            panel,
            spec_strat,
            "pole_share",
            window_days=None,
            entity_col="entity_id",
            time_col="time_id",
            cluster_col="entity_id",
        )
        static_sample = filter_strategy_sample(panel, spec_strat, window_days=None)
        pret_p, _ = estimate_pretrend_f(
            static_sample,
            "pole_share",
            entity_col="entity_id",
            time_col="time_id",
            rel_col="rel_day",
        )
        summary_rows.append(
            {
                "sample": sample_label,
                "strategy_id": "cross_country_all",
                "spec": spec_id,
                "beta": static_res.get("beta"),
                "se": static_res.get("se"),
                "pvalue": static_res.get("pvalue"),
                "n_obs": static_res.get("n_obs"),
                "n_clusters": static_res.get("n_clusters"),
                "pretrend_F_p": pret_p,
                "estimation_note": static_res.get("estimation_note", "ok"),
            }
        )
        if spec_id == "full_ban":
            static_beta = float(static_res.get("beta", float("nan")))

    es_strat = StrategySpec("cross_country_all", post_mode="full_ban")
    es_sample = filter_strategy_sample(binned, es_strat, window_days=window_days)
    es_sample = es_sample.dropna(subset=["pole_share"]).copy()

    counts = _rel_bin_cell_counts(es_sample)
    sparse = _print_cell_counts(counts, sample_label)
    repair_note = ""
    if sparse:
        es_sample, repair_note = _repair_sparse_bins(es_sample, sparse, sample_label)
        counts = _rel_bin_cell_counts(es_sample)
        _print_cell_counts(counts, sample_label, repair_note=repair_note or "after repair")

    _, es_df = estimate_panel_it_event_study(
        es_sample,
        "pole_share",
        entity_col="subreddit",
        time_col="time_id",
        rel_col="rel_period",
        ref_period=REF_REL_PERIOD,
        window=window_days,
        cluster_col="subreddit",
        bin_days=BIN_DAYS,
    )
    manual_df = manual_panel_it_event_study(
        es_sample,
        "pole_share",
        entity_col="subreddit",
        time_col="time_id",
        rel_col="rel_period",
        ref_period=REF_REL_PERIOD,
        window=window_days,
        cluster_col="subreddit",
        bin_days=BIN_DAYS,
    )

    if es_df.empty or manual_df.empty:
        raise SystemExit(
            f"[pole_share_fixed_authors] ES estimation failed for {sample_label}"
        )

    es_df = es_df.copy()
    es_df["rel_day"] = es_df["rel_period"] * BIN_DAYS
    es_df["strategy_id"] = "cross_country_all"
    es_df["sample"] = sample_label

    _validate_es_gates(es_df, manual_df, static_beta, es_sample, sample_label, repair_note)
    return summary_rows, es_df


def _write_readme(out_dir: Path, n_fixed: int, n_all: int, min_scored: int) -> None:
    """Function summary: short README describing construction and interpretation."""
    text = f"""# pole_share fixed-author robustness

## Construction
- Pre-ban author roster: authors with >=1 ideology-lexicon hit ({PRE_START}..{PRE_END}).
- Excludes [deleted] and known bots ({len(DEFAULT_BOT_AUTHORS)} default bot accounts).
- Subreddit-day pole_share = (left+right)/(left+center+right) pooled over comments.
- Minimum {min_scored} scored comments per subreddit-day.
- Fixed-author roster size: {n_fixed:,}; all-author comment rows: {n_all:,}.

## Estimation
- Strategy: cross_country_all (IT political + IT others vs DE/EU/US/UK controls).
- TWFE: y ~ treat×post | subreddit + day, SE clustered by subreddit.
- Event study: 3-day bins, reference period -1.

## Interpretation
- If fixed-author full-ban beta stays ~+0.06, composition is exonerated (differential trend caveat remains).
- If fixed-author beta collapses toward 0, the headline is driven by author composition change.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Function summary: run fixed-author pole_share robustness and write outputs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    require_dominant_v1_ideology_scoring(config)
    pol_cfg = load_polarization_config(config)
    start, end_excl, launch, _ = event_dates_from_config(config)

    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    validate_feature_columns_present(shard_root)
    subs = resolve_primary_subreddits(config)

    df = load_comment_frame(shard_root, subs)
    if df.empty:
        raise SystemExit("[pole_share_fixed_authors] no parquet data found")
    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    if "topic_family" not in df.columns:
        df["topic_family"] = df["subreddit"].map(subreddit_family_map(config))

    df = _filter_authors(df)
    fixed_authors = _fixed_author_set(df)
    df_fixed = df[df["author"].astype(str).isin(fixed_authors)].copy()

    print(
        f"[pole_share_fixed_authors] comments all={len(df):,} fixed={len(df_fixed):,} "
        f"authors_fixed={len(fixed_authors):,}",
        flush=True,
    )

    panel_all = _build_subreddit_day_panel(df, pol_cfg, launch, end_excl, args.min_scored)
    panel_fixed = _build_subreddit_day_panel(df_fixed, pol_cfg, launch, end_excl, args.min_scored)

    window_days = EVENT_WINDOW_DAYS_BY_BIN.get(BIN_DAYS, 30)
    rows_all, es_all = _estimate_specs(panel_all, config, "all_authors", window_days)
    rows_fixed, es_fixed = _estimate_specs(panel_fixed, config, "fixed_authors", window_days)

    summary = pd.DataFrame(rows_all + rows_fixed)
    event_study = pd.concat([es_all, es_fixed], ignore_index=True)

    out_dir = tables_subdir(config, "did") / "pole_share_fixed_authors"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    event_study.to_csv(out_dir / "event_study.csv", index=False)
    _write_readme(out_dir, len(fixed_authors), len(df), args.min_scored)

    base_row = summary[(summary["sample"] == "all_authors") & (summary["spec"] == "full_ban")]
    if not base_row.empty:
        beta = float(base_row["beta"].iloc[0])
        print(
            f"[pole_share_fixed_authors] all-authors full_ban beta={beta:.4f} "
            f"(target ~+0.062)",
            flush=True,
        )

    if not args.no_figures and not es_fixed.empty:
        fig_dir = figures_subdir(config, "did") / "pole_share_fixed_authors"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_event_study(
            es_fixed,
            "pole_share",
            fig_dir / "pole_share_fixed_authors_es.png",
            rel_col="rel_period",
            title="pole_share event study (pre-ban author set)",
        )

    print(f"[pole_share_fixed_authors] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
