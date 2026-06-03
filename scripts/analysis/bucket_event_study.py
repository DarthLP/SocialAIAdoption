"""
Script summary:
Bucket-then-comment-level event study for the Italy ChatGPT ban (Mar–Apr 2023).

Functionality:
- Label authors into liberal/neutral/conservative buckets via asymmetric lexical rules on each scheme's
  labeling-window comments (split_sample, holdout_2wk, naive_full_march).
- Headline static DiD (Table 1): net_ideology ~ Post + Post:IT | AuthorFE (no bin FE).
- Event study: net_ideology ~ sum_k (bin_k:IT) | AuthorFE + binFE (3-day bins, ref k=-1).
- Pooled + by-bucket runs, control variants, stacked DDD (liberal vs conservative gamma_k).
- Descriptive trajectories (Italy vs controls) per bucket; subreddit wild-cluster bootstrap.

Interpretation guardrail:
The ban REMOVES AI access. Under an "AI increases polarization" prior, expect CONVERGENCE in
treated Italy during the ban (liberals and conservatives drift toward center), rebounding after
the 28-Apr lift — the mirror of an AI-access result. Scheme C (overlapping March labeling) can
mimic mean reversion; compare to schemes A/B. Standardized outcomes assume parallel trends in
z-space after within-language March normalization.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_comment_panel.py \\
    --config config/italy_polarization_setup.yaml --bin-days 3
  .venv/bin/python scripts/analysis/bucket_event_study.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/bucket_event_study.py --bin-days 1  # -> did/bucket_event_study/1d/
  .venv/bin/python scripts/analysis/bucket_event_study.py --max-shards 2 --no-bootstrap --no-figures
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


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
from scripts.diagnostics.prepare_did_comment_panel import (  # noqa: E402
    COMMENT_COLUMNS,
    _annotate_comments,
    _apply_3d_bins,
    _iter_shard_paths,
    _read_shard,
)
from src.config_utils import load_config, resolve_primary_subreddits  # noqa: E402
from src.did.bucket_estimate import (  # noqa: E402
    combine_split_sample_static,
    compute_trajectory_means,
    estimate_comment_it_ddd_event_study,
    estimate_comment_it_event_study,
    estimate_static_full_time_fe,
    estimate_static_paper_eq1,
    feols_static_paper_eq1_prepped,
    filter_trajectory_series,
    prep_static_design,
)
from src.did.inference import placebo_in_space_comment_p  # noqa: E402
from src.did.lean_buckets import (  # noqa: E402
    UNCLASSIFIED,
    apply_outcome_scale,
    assert_net_ideology_sign,
    balanced_author_set,
    bucket_event_study_config,
    build_all_lean_buckets,
    control_variant_mask,
    estimation_sample_mask,
    filter_control_variant,
    march_standardization_moments,
    merge_buckets,
    write_lean_buckets_csv,
)
from src.did.outputs import apply_event_study_axes_style, plot_event_study  # noqa: E402
from src.did.panel_dtypes import compact_comment_panel_dtypes  # noqa: E402
from src.did.panels import comment_panel_available, load_comment_panel as load_prepared_panel  # noqa: E402
from src.did.paths import (  # noqa: E402
    bucket_event_study_figures_dir,
    did_bucket_event_study_dir,
    did_lean_buckets_dir,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for bucket event-study pipeline."""
    p = argparse.ArgumentParser(description="Bucket-then-comment event study (Italy ban).")
    p.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--scheme", type=str, default=None, help="Run one scheme only.")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--balanced-only", action="store_true")
    p.add_argument(
        "--bin-days",
        type=int,
        choices=(1, 3),
        default=None,
        help="Event calendar bin width (1 or 3). Overrides did.bucket_event_study.bin_days; "
        "requires matching did_comment_panel_{1,3}d from prepare_did_comment_panel.py.",
    )
    return p.parse_args()


def _load_comment_panel_from_shards(
    config: Dict[str, Any],
    bcfg: Any,
    max_shards: Optional[int],
) -> pd.DataFrame:
    """Function summary: stream enriched shards into one annotated comment DataFrame.

    Parameters:
    - config: study YAML.
    - bcfg: bucket event-study config.
    - max_shards: optional per-subreddit cap.

    Returns:
    - Annotated comments with rel_day, rel_period, IT, post, time_id.
    """
    start, end_excl, launch, _ = event_dates_from_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = resolve_primary_subreddits(config)
    shards = _iter_shard_paths(interim_dir, subs, max_shards)
    frames: List[pd.DataFrame] = []
    for path in shards:
        raw = _read_shard(path, COMMENT_COLUMNS)
        if raw is None or raw.empty:
            continue
        raw["date_utc"] = raw["date_utc"].astype(str)
        raw = raw[(raw["date_utc"] >= start) & (raw["date_utc"] < end_excl)]
        if bcfg.political_universe_only and "comment_in_political_universe" in raw.columns:
            raw = raw[raw["comment_in_political_universe"].astype(bool)]
        if raw.empty:
            continue
        ann = _annotate_comments(raw, launch, end_excl)
        ann = _apply_3d_bins(ann, launch, bcfg.bin_days)
        if "rel_period" not in ann.columns:
            ann["rel_period"] = (ann["rel_day"] // bcfg.bin_days).astype(int)
        frames.append(ann)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_bucket_comment_panel(
    config: Dict[str, Any],
    bcfg: Any,
    max_shards: Optional[int],
) -> pd.DataFrame:
    """Function summary: load comments from prepared panel or enriched shards.

    Parameters:
    - config: study YAML.
    - bcfg: bucket event-study config.
    - max_shards: when set, always stream shards (smoke tests).

    Returns:
    - Annotated comment panel.
    """
    bin_days = int(bcfg.bin_days)
    if max_shards is None and comment_panel_available(config, bin_days):
        print(
            f"[bucket_event_study] loading prepared comment panel ({bin_days}d bins)...",
            flush=True,
        )
        return load_prepared_panel(config, bin_days=bin_days)
    if max_shards is None:
        print(
            "[bucket_event_study] hint: run prepare_did_comment_panel.py --bin-days "
            f"{bin_days} for faster reruns",
            flush=True,
        )
    print("[bucket_event_study] streaming enriched shards...", flush=True)
    return _load_comment_panel_from_shards(config, bcfg, max_shards)


def _lift_rel_period(config: Dict[str, Any], bin_days: int) -> int:
    """Function summary: event-time bin index for ban lift date."""
    _, _, launch, lift = event_dates_from_config(config)
    rel_day = int((pd.Timestamp(lift) - pd.Timestamp(launch)).days)
    return rel_day // int(bin_days)


def _plot_trajectories(
    traj_df: pd.DataFrame,
    out_path: Path,
    title: str,
    bin_days: int,
    lift_period: int,
) -> None:
    """Function summary: multi-series mean outcome by event bin."""
    if traj_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, grp in traj_df.groupby("series_label", sort=False):
        ax.plot(grp["rel_day"], grp["mean_y"], marker="o", label=label, linewidth=1.2)
    apply_event_study_axes_style(ax, xlabel=f"days rel. to ban ({bin_days}-day bins)")
    ax.axvline(int(lift_period * bin_days), color="gray", linestyle=":", linewidth=0.9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ddd_es(ddd_df: pd.DataFrame, out_path: Path, bin_days: int, lift_period: int) -> None:
    """Function summary: DDD event-study coefficients with CIs."""
    if ddd_df.empty:
        return
    work = ddd_df.copy()
    work["event_time"] = work["rel_period"].astype(int) * bin_days
    ref = -1
    if ref not in work["rel_period"].astype(int).tolist():
        work = pd.concat(
            [
                work,
                pd.DataFrame(
                    [{"rel_period": ref, "event_time": ref * bin_days, "ddd_gamma": 0.0, "se": 0.0}]
                ),
            ],
            ignore_index=True,
        )
    work = work.sort_values("event_time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = work["se"].fillna(0) > 0
    ax.errorbar(
        work.loc[mask, "event_time"],
        work.loc[mask, "ddd_gamma"],
        yerr=1.96 * work.loc[mask, "se"],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.plot(work["event_time"], work["ddd_gamma"], "o", mfc="white", mec="black")
    apply_event_study_axes_style(ax, xlabel=f"days rel. to ban ({bin_days}-day bins)")
    ax.axvline(lift_period * bin_days, color="gray", linestyle=":", linewidth=0.9)
    ax.set_title("DDD: liberal − conservative (IT×bin)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _estimation_frame(
    df: pd.DataFrame,
    scheme: str,
    bcfg: Any,
    config: Dict[str, Any],
    moments: pd.DataFrame,
    split_id: Optional[int],
    balanced_only: bool,
    balanced_authors: Optional[frozenset[str]],
) -> pd.DataFrame:
    """Function summary: comments in estimation window with outcome y attached."""
    mask = estimation_sample_mask(df, scheme, bcfg, config, split_id=split_id)
    out = apply_outcome_scale(df.loc[mask], moments, bcfg, bcfg.outcome, "y")
    if balanced_only or bcfg.balanced_panel:
        if balanced_authors is None:
            from src.did.lean_buckets import filter_balanced_authors

            out = filter_balanced_authors(out)
        else:
            out = out[out["author"].astype(str).isin(balanced_authors)]
    return out


def _run_static_block(
    sample: pd.DataFrame,
    bcfg: Any,
    run_bootstrap: bool,
) -> List[Dict[str, Any]]:
    """Function summary: headline + optional robustness static DiD rows."""
    rows: List[Dict[str, Any]] = []
    res = estimate_static_paper_eq1(sample)
    res["inference_role"] = "descriptive"
    res["pvalue_cluster"] = res.get("pvalue", float("nan"))
    if run_bootstrap:
        res["p_placebo_space"] = placebo_in_space_comment_p(
            sample,
            y_col="y" if "y" in sample.columns else "net_ideology",
            cluster_col="subreddit" if "subreddit" in sample.columns else "author",
        )
        res["perm_p"] = res["p_placebo_space"]
        res["p_wild"] = float("nan")
    else:
        res["p_placebo_space"] = float("nan")
        res["perm_p"] = float("nan")
        res["p_wild"] = float("nan")
    rows.append(res)
    if bcfg.static_full_time_fe:
        rows.append(estimate_static_full_time_fe(sample))
    return rows


def _write_trajectories(
    sub: pd.DataFrame,
    bcfg: Any,
    tables_dir: Path,
    fig_dir: Path,
    scheme: str,
    bucket: str,
    write_figures: bool,
    config: Dict[str, Any],
) -> None:
    """Function summary: descriptive mean-y paths by series."""
    if not bcfg.run_descriptive_trajectories:
        return
    parts: List[pd.DataFrame] = []
    for spec in bcfg.trajectory_series:
        ser = filter_trajectory_series(sub, spec)
        if ser.empty:
            continue
        parts.append(
            compute_trajectory_means(
                ser,
                "rel_period",
                str(spec.get("id", "")),
                str(spec.get("label", "")),
                bin_days=bcfg.bin_days,
            )
        )
    if not parts:
        return
    traj_all = pd.concat(parts, ignore_index=True)
    traj_all.to_csv(tables_dir / f"trajectories_{scheme}_{bucket}.csv", index=False)
    if write_figures:
        _plot_trajectories(
            traj_all,
            fig_dir / scheme / bucket / "trajectories.png",
            f"{scheme} / {bucket}",
            bcfg.bin_days,
            _lift_rel_period(config, bcfg.bin_days),
        )


def run_scheme(
    df: pd.DataFrame,
    buckets: pd.DataFrame,
    scheme: str,
    bcfg: Any,
    config: Dict[str, Any],
    tables_dir: Path,
    fig_dir: Path,
    *,
    run_bootstrap: bool,
    write_figures: bool,
    balanced_only: bool,
) -> None:
    """Function summary: full estimation stack for one labeling scheme."""
    moments = march_standardization_moments(df, config, bcfg)
    moments_path = did_lean_buckets_dir(config) / "standardization_moments.csv"
    if not moments.empty:
        moments.to_csv(moments_path, index=False)

    split_ids: List[Optional[int]] = (
        list(range(bcfg.n_splits)) if scheme == "split_sample" else [None]
    )
    static_rows: List[Dict[str, Any]] = []
    es_frames: List[pd.DataFrame] = []
    bucket_names = sorted(
        b for b in buckets.loc[buckets["scheme"] == scheme, "bucket"].astype(str).unique() if b != UNCLASSIFIED
    )
    need_balance = balanced_only or bcfg.balanced_panel
    balanced_authors: Optional[frozenset[str]] = balanced_author_set(df) if need_balance else None
    bases_by_split: Dict[Optional[int], pd.DataFrame] = {}

    for split_id in split_ids:
        base = _estimation_frame(
            df, scheme, bcfg, config, moments, split_id, balanced_only, balanced_authors
        )
        if base.empty:
            continue
        bases_by_split[split_id] = base
        sid_val = -1 if split_id is None else int(split_id)
        cv_masks = {cv: control_variant_mask(base, cv, bcfg) for cv in bcfg.control_variants}
        base_merged = merge_buckets(base, buckets, scheme, split_id)

        for cv in bcfg.control_variants:
            if cv == "it_political_vs_it_others":
                pool = filter_control_variant(base, cv, bcfg)
                merged = merge_buckets(pool, buckets, scheme, split_id)
            else:
                pool = base.loc[cv_masks[cv]]
                merged = base_merged.loc[control_variant_mask(base_merged, cv, bcfg)]

            if pool.empty:
                continue

            if bcfg.run_pooled:
                meta = {
                    "scheme": scheme,
                    "split_id": sid_val,
                    "control_variant": cv,
                    "bucket": "all",
                    "sample": "balanced" if balanced_only else "full",
                }
                for sres in _run_static_block(pool, bcfg, run_bootstrap):
                    static_rows.append({**meta, **sres})
                _, es_df = estimate_comment_it_event_study(
                    pool,
                    ref_period=bcfg.ref_rel_period,
                    window=bcfg.event_window_days,
                    bin_days=bcfg.bin_days,
                )
                if not es_df.empty:
                    es_frames.append(es_df.assign(**meta))
                if write_figures and not es_df.empty:
                    plot_event_study(
                        es_df,
                        bcfg.outcome,
                        fig_dir / scheme / "all" / f"es_{cv}_split{sid_val}.png",
                        rel_col="rel_period",
                    )

            for bucket in bucket_names:
                sub = merged[merged["bucket"].astype(str) == bucket]
                if sub.empty:
                    continue
                bmeta = {
                    "scheme": scheme,
                    "split_id": sid_val,
                    "control_variant": cv,
                    "bucket": bucket,
                    "sample": "balanced" if balanced_only else "full",
                }
                for sres in _run_static_block(sub, bcfg, run_bootstrap):
                    static_rows.append({**bmeta, **sres})
                _, es_b = estimate_comment_it_event_study(
                    sub,
                    ref_period=bcfg.ref_rel_period,
                    window=bcfg.event_window_days,
                    bin_days=bcfg.bin_days,
                )
                if not es_b.empty:
                    es_frames.append(es_b.assign(**bmeta))
                if write_figures and not es_b.empty:
                    plot_event_study(
                        es_b,
                        bcfg.outcome,
                        fig_dir / scheme / bucket / f"es_{cv}_split{sid_val}.png",
                        rel_col="rel_period",
                    )
                _write_trajectories(sub, bcfg, tables_dir, fig_dir, scheme, bucket, write_figures, config)

    if scheme == "split_sample" and static_rows:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in static_rows:
            if r.get("static_variant") != "paper_eq1":
                continue
            key = f"{r.get('control_variant')}|{r.get('bucket')}"
            grouped.setdefault(key, []).append(r)
        extra: List[Dict[str, Any]] = []
        keep: List[Dict[str, Any]] = [r for r in static_rows if r.get("static_variant") != "paper_eq1"]
        for key, grp in grouped.items():
            if len(grp) > 1:
                c = combine_split_sample_static(grp)
                cv, bucket = key.split("|", 1)
                c.update({"scheme": scheme, "control_variant": cv, "bucket": bucket, "split_id": -1})
                extra.append(c)
            else:
                keep.extend(grp)
        static_rows = keep + extra

    if static_rows:
        pd.DataFrame(static_rows).to_csv(tables_dir / f"static_{scheme}.csv", index=False)
    if es_frames:
        pd.concat(es_frames, ignore_index=True).to_csv(tables_dir / f"event_study_{scheme}.csv", index=False)

    lib, con = bcfg.ddd_buckets
    sid_ddd: Optional[int] = 0 if scheme == "split_sample" else None
    base_ddd = bases_by_split.get(sid_ddd)
    if base_ddd is None:
        base_ddd = _estimation_frame(
            df, scheme, bcfg, config, moments, sid_ddd, balanced_only, balanced_authors
        )
    for cv in bcfg.ddd_control_variants:
        pool = filter_control_variant(base_ddd, cv, bcfg)
        pool = merge_buckets(pool, buckets, scheme, split_id=sid_ddd)
        pool = pool[pool["bucket"].astype(str).isin([lib, con])]
        if pool.empty:
            continue
        _, ddd_df = estimate_comment_it_ddd_event_study(
            pool,
            lib,
            con,
            ref_period=bcfg.ref_rel_period,
            window=bcfg.event_window_days,
            bin_days=bcfg.bin_days,
        )
        if ddd_df.empty:
            continue
        ddd_df.assign(scheme=scheme, control_variant=cv).to_csv(
            tables_dir / f"ddd_{scheme}_{cv}.csv", index=False
        )
        if write_figures:
            _plot_ddd_es(
                ddd_df,
                fig_dir / scheme / f"ddd_{cv}.png",
                bcfg.bin_days,
                _lift_rel_period(config, bcfg.bin_days),
            )


def main() -> None:
    """Function summary: CLI entry — load data, label buckets, estimate, write outputs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    bcfg = bucket_event_study_config(config)
    if args.bin_days is not None:
        yaml_days = bcfg.bin_days
        bcfg = replace(bcfg, bin_days=int(args.bin_days))
        if int(args.bin_days) != yaml_days:
            print(
                f"[bucket_event_study] --bin-days {args.bin_days} overrides config bin_days={yaml_days}",
                flush=True,
            )
    if not (config.get("did") or {}).get("bucket_event_study", {}).get("enabled", True):
        print("[bucket_event_study] disabled in config", flush=True)
        return

    df = load_bucket_comment_panel(config, bcfg, args.max_shards)
    if df.empty:
        print("[bucket_event_study] no comments loaded", flush=True)
        return
    compact_comment_panel_dtypes(df)
    assert_net_ideology_sign(df, bcfg.outcome)

    lean_dir = did_lean_buckets_dir(config)
    tables_dir = did_bucket_event_study_dir(config, bin_days=bcfg.bin_days)
    fig_dir = bucket_event_study_figures_dir(config, bin_days=bcfg.bin_days)
    lean_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    buckets = build_all_lean_buckets(df, bcfg, config)
    if buckets.empty:
        print("[bucket_event_study] no bucket assignments", flush=True)
        return
    write_lean_buckets_csv(lean_dir / "all_schemes.csv", buckets)
    for scheme in bcfg.schemes:
        sub = buckets[buckets["scheme"] == scheme]
        if not sub.empty:
            write_lean_buckets_csv(lean_dir / f"{scheme}.csv", sub)

    schemes = [args.scheme] if args.scheme else list(bcfg.schemes)
    run_bootstrap = not args.no_bootstrap
    for scheme in schemes:
        print(f"[bucket_event_study] scheme={scheme}", flush=True)
        run_scheme(
            df,
            buckets,
            scheme,
            bcfg,
            config,
            tables_dir,
            fig_dir,
            run_bootstrap=run_bootstrap,
            write_figures=not args.no_figures,
            balanced_only=args.balanced_only,
        )
    print(f"[bucket_event_study] done -> {tables_dir}", flush=True)


if __name__ == "__main__":
    main()
