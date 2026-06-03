"""
Script summary:
Compare lexical vs semantic author ideology buckets (agreement, κ, Spearman, confusion).

How to apply/run:
  .venv/bin/python scripts/user_week/compare_lexical_semantic_author_buckets.py \\
    --config config/italy_polarization_setup.yaml --cohort strict
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

from src.config_utils import load_config  # noqa: E402
from src.user_week.ideology_buckets import (  # noqa: E402
    agreement_summary_rows,
    confusion_table,
    filter_pole_only_agreement_sample,
    ideology_bucket_config,
    marginal_bucket_counts,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for bucket agreement analysis."""
    parser = argparse.ArgumentParser(description="Compare lexical vs semantic author ideology buckets.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default="both",
        choices=["strict", "loose", "both"],
    )
    return parser.parse_args()


def _plot_confusion(ct: pd.DataFrame, title: str, out_path: Path) -> None:
    """Function summary: heatmap for one confusion matrix."""
    if ct.empty or ct.values.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(ct.astype(int), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Semantic bucket")
    ax.set_ylabel("Lexical bucket")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, out_path: Path, cohort: str) -> None:
    """Function summary: lexical vs semantic continuous scores colored by lexicon."""
    sub = df.dropna(subset=["lexical_score", "semantic_score"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for lex, grp in sub.groupby(sub["assigned_primary_lexicon"].astype(str)):
        ax.scatter(
            grp["lexical_score"],
            grp["semantic_score"],
            alpha=0.35,
            s=12,
            label=lex,
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Pre-ban lexical score (net_ideology)")
    ax.set_ylabel("Pre-ban semantic score (oriented sem_axis_ideology)")
    ax.set_title(f"Author scores — {cohort}")
    ax.legend(title="Lexicon", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_agreement_bar(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: bar chart of exact match rate by scope."""
    sub = summary[summary["scope"] != "overall"].copy()
    if sub.empty:
        sub = summary.copy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(sub["scope"].astype(str), sub["pct_exact_match"].astype(float) * 100.0)
    ax.set_ylabel("% exact bucket match")
    ax.set_xlabel("Scope")
    ax.set_title("Lexical vs semantic agreement")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_readme(fig_dir: Path) -> None:
    """Function summary: short interpretation note in figure folder."""
    text = (
        "# Lexical vs semantic author buckets\n\n"
        "Buckets use **asymmetric** pre-ban rules: lexical neutral when no L/R lexicon hits; "
        "semantic neutral when no tail weeks (p25/p75 on oriented sem_axis by default). "
        "`semantic_bucket_mag_band` is exported for diagnostic agreement only. "
        "Mismatch between measures is expected; low κ alone is not a validity failure.\n"
    )
    (fig_dir / "README.md").write_text(text, encoding="utf-8")


def run_cohort(config: dict, cohort: str) -> None:
    """Function summary: agreement tables and figures for one cohort."""
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    bucket_path = tables_dir / f"author_ideology_buckets_{cohort}.csv"
    if not bucket_path.is_file():
        raise FileNotFoundError(f"Missing {bucket_path}; run assign_author_ideology_buckets.py")

    df = pd.read_csv(bucket_path)
    bucket_cfg = ideology_bucket_config(config)
    labels = list(bucket_cfg.bucket_labels)

    out_tables = tables_dir / f"ideology_bucket_agreement_{cohort}"
    out_tables.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(config["paths"]["figures_dir"]) / "user_week" / cohort / "ideology_bucket_agreement"
    fig_dir.mkdir(parents=True, exist_ok=True)

    marginal_bucket_counts(df, "lexical_bucket", labels).to_csv(
        out_tables / "marginal_lexical_bucket.csv", index=False
    )
    marginal_bucket_counts(df, "semantic_bucket", labels).to_csv(
        out_tables / "marginal_semantic_bucket_tail.csv", index=False
    )
    if "semantic_bucket_mag_band" in df.columns:
        marginal_bucket_counts(df, "semantic_bucket_mag_band", labels).to_csv(
            out_tables / "marginal_semantic_bucket_mag_band.csv", index=False
        )

    classified_tail = df[
        df["lexical_bucket"].isin(labels) & df["semantic_bucket"].isin(labels)
    ].copy()
    classified_tail = classified_tail[classified_tail["semantic_bucket"] != "semantically_unscored"]
    classified_tail.to_csv(out_tables / "author_crosswalk.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    summary_rows.extend(
        agreement_summary_rows(
            df,
            labels,
            group_col="assigned_primary_lexicon",
            semantic_col="semantic_bucket",
            scope_prefix="tail_",
        )
    )
    if "semantic_bucket_mag_band" in df.columns:
        summary_rows.extend(
            agreement_summary_rows(
                df,
                labels,
                group_col="assigned_primary_lexicon",
                semantic_col="semantic_bucket_mag_band",
                scope_prefix="mag_band_",
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_tables / "agreement_summary.csv", index=False)

    ct_all = confusion_table(classified_tail, labels=labels)
    ct_all.to_csv(out_tables / "confusion_overall_tail.csv")
    _plot_confusion(ct_all, f"Tail semantic — overall — {cohort}", fig_dir / "confusion_overall_tail.png")

    if "semantic_bucket_mag_band" in df.columns:
        classified_mag = df[
            df["lexical_bucket"].isin(labels) & df["semantic_bucket_mag_band"].isin(labels)
        ].copy()
        classified_mag = classified_mag[classified_mag["semantic_bucket_mag_band"] != "semantically_unscored"]
        ct_mag = confusion_table(
            classified_mag,
            col_col="semantic_bucket_mag_band",
            labels=labels,
        )
        ct_mag.to_csv(out_tables / "confusion_overall_mag_band.csv")
        _plot_confusion(
            ct_mag,
            f"Mag-band semantic — overall — {cohort}",
            fig_dir / "confusion_overall_mag_band.png",
        )

    for lex in sorted(classified_tail["assigned_primary_lexicon"].dropna().unique()):
        sub = classified_tail[classified_tail["assigned_primary_lexicon"].astype(str) == str(lex)]
        ct = confusion_table(sub, labels=labels)
        ct.to_csv(out_tables / f"confusion_by_lexicon_{lex}_tail.csv")
        _plot_confusion(ct, f"Tail — {lex} — {cohort}", fig_dir / f"confusion_by_lexicon_{lex}_tail.png")

    pole_only = filter_pole_only_agreement_sample(df, labels)
    pole_rows = agreement_summary_rows(
        pole_only,
        labels,
        group_col="assigned_primary_lexicon",
        semantic_col="semantic_bucket",
        exclude_semantic_unscored=False,
        scope_prefix="pole_only_tail_",
    )
    pd.DataFrame(pole_rows).to_csv(out_tables / "agreement_pole_only_tail.csv", index=False)

    tail_summary = summary[summary["scope"] == "tail_overall"].copy()
    _plot_scatter(classified_tail, fig_dir / "lexical_vs_semantic_scatter.png", cohort)
    by_scope = summary[
        summary["scope"].astype(str).str.startswith("tail_")
        & (summary["scope"].astype(str) != "tail_overall")
    ].copy()
    if not by_scope.empty:
        by_scope = by_scope.assign(scope=by_scope["scope"].astype(str).str.replace("tail_", "", 1))
        _plot_agreement_bar(by_scope, fig_dir / "pct_exact_match_by_scope.png")

    if "bucket_cross" in classified_tail.columns:
        top = (
            classified_tail[~classified_tail["buckets_agree"]]
            .groupby("bucket_cross", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(15)
        )
        top.to_csv(out_tables / "top_bucket_mismatches.csv", index=False)

    _write_readme(fig_dir)
    if not tail_summary.empty:
        row = tail_summary.iloc[0]
        print(
            f"[compare_lexical_semantic_author_buckets] cohort={cohort} n={row['n_classified']} "
            f"pct_exact={row['pct_exact_match']:.3f} kappa={row['cohens_kappa']:.3f} "
            f"rho={row['spearman_rho']:.3f}",
            flush=True,
        )


def main() -> None:
    """Function summary: run comparison for requested cohorts."""
    args = parse_args()
    config = load_config(args.config)
    cohorts: List[str] = ["strict", "loose"] if args.cohort == "both" else [args.cohort]
    for cohort in cohorts:
        run_cohort(config, cohort)
    print("[compare_lexical_semantic_author_buckets] done", flush=True)


if __name__ == "__main__":
    main()
