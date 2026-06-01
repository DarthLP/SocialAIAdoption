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
from typing import List

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
    ideology_bucket_config,
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
        "Buckets are **descriptive** tertiles within each primary lexicon (it/en/de), "
        "from **pre-ban** writing only. Mismatch between lexical (lexicon hits) and "
        "semantic (embedding axis) classification is expected when coverage differs or "
        "topics are not ideological. This is not a validity failure by itself.\n"
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

    classified = df[
        df["lexical_bucket"].isin(labels) & df["semantic_bucket"].isin(labels)
    ].copy()
    classified.to_csv(out_tables / "author_crosswalk.csv", index=False)

    summary_rows = agreement_summary_rows(classified, labels, group_col="assigned_primary_lexicon")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_tables / "agreement_summary.csv", index=False)

    ct_all = confusion_table(classified, labels=labels)
    ct_all.to_csv(out_tables / "confusion_overall.csv")
    _plot_confusion(ct_all, f"Overall — {cohort}", fig_dir / "confusion_overall.png")

    for lex in sorted(classified["assigned_primary_lexicon"].dropna().unique()):
        sub = classified[classified["assigned_primary_lexicon"].astype(str) == str(lex)]
        ct = confusion_table(sub, labels=labels)
        ct.to_csv(out_tables / f"confusion_by_lexicon_{lex}.csv")
        _plot_confusion(ct, f"{lex} — {cohort}", fig_dir / f"confusion_by_lexicon_{lex}.png")

    _plot_scatter(classified, fig_dir / "lexical_vs_semantic_scatter.png", cohort)
    _plot_agreement_bar(summary, fig_dir / "pct_exact_match_by_scope.png")

    if "bucket_cross" in classified.columns:
        top = (
            classified[~classified["buckets_agree"]]
            .groupby("bucket_cross", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(15)
        )
        top.to_csv(out_tables / "top_bucket_mismatches.csv", index=False)

    _write_readme(fig_dir)
    overall = summary[summary["scope"] == "overall"]
    if not overall.empty:
        row = overall.iloc[0]
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
