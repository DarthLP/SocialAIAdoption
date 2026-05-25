"""
Script summary:
Plot semantic-axis descriptives from prepared CSV tables.

Functionality:
- Family-level daily timeseries for ideology and emotion axes.
- Subreddit-day scatter of sem_axis_ideology vs net_ideology.
- Score density histograms by topic_family.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
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

from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    plot_reference_dates_calendar_utc,
    tables_subdir,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot semantic-axis descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: draw vertical ban reference dates."""
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def _plot_family_timeseries(
    panel: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    config: dict,
) -> None:
    """Function summary: daily family means for one semantic-axis metric."""
    if metric not in panel.columns:
        return
    fam = panel.groupby(["date_utc", "topic_family"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    for family, grp in fam.groupby("topic_family"):
        grp = grp.sort_values("date_utc")
        ax.plot(pd.to_datetime(grp["date_utc"]), grp[metric], label=family, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(title)
    ax.set_xlabel("date_utc")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: CLI entry for semantic-axis figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables = tables_subdir(config, "semantic_axis")
    panel_path = tables / "semantic_axis_panel.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Run prepare_semantic_axis_descriptives.py first: missing {panel_path}")
    panel = pd.read_csv(panel_path)
    fig_root = figures_subdir(config, "semantic_axis")

    _plot_family_timeseries(
        panel,
        "sem_axis_ideology_mean",
        "Embedding ideology axis — daily mean by topic family",
        fig_root / "ideology_axis_timeseries_by_family.png",
        config,
    )
    _plot_family_timeseries(
        panel,
        "sem_axis_emotion_mean",
        "Embedding emotion axis (affect vs cognition) — daily mean by topic family",
        fig_root / "emotion_axis_timeseries_by_family.png",
        config,
    )

    if {"sem_axis_ideology_mean", "net_ideology_mean"}.issubset(panel.columns):
        sub_day = panel.groupby(["subreddit", "date_utc"], as_index=False).agg(
            sem_axis_ideology_mean=("sem_axis_ideology_mean", "mean"),
            net_ideology_mean=("net_ideology_mean", "mean"),
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub_day["net_ideology_mean"], sub_day["sem_axis_ideology_mean"], alpha=0.35, s=12)
        ax.set_xlabel("Lexical net_ideology (subreddit-day mean)")
        ax.set_ylabel("Embedding sem_axis_ideology (subreddit-day mean)")
        ax.set_title("Semantic axis vs lexicon ideology")
        fig.tight_layout()
        fig.savefig(fig_root / "axis_vs_lexicon_scatter.png", dpi=150)
        plt.close(fig)

    shard_glob = list(
        (PROJECT_ROOT / config["paths"]["interim_dir"] / "cleaned_monthly_chunks").glob("*/*.parquet")
    )
    if shard_glob:
        from scripts.features._enriched_shard_runner import read_parquet_shard_safe

        sample = read_parquet_shard_safe(shard_glob[0])
        if sample is not None and "sem_axis_ideology" in sample.columns:
            cols = [
                c
                for c in (
                    "sem_axis_ideology",
                    "sem_axis_emotion",
                    "topic_family",
                )
                if c in sample.columns
            ]
            chunks = []
            for p in shard_glob[:40]:
                d = read_parquet_shard_safe(p)
                if d is not None and cols:
                    chunks.append(d[cols])
            if chunks:
                raw = pd.concat(chunks, ignore_index=True)
                scored = raw[raw["sem_axis_ideology"].notna()]
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, col in zip(axes, ("sem_axis_ideology", "sem_axis_emotion")):
                    if col not in scored.columns:
                        continue
                    for fam, grp in scored.groupby("topic_family"):
                        ax.hist(
                            grp[col].astype(float),
                            bins=40,
                            alpha=0.35,
                            density=True,
                            label=str(fam),
                        )
                    ax.set_title(col)
                    ax.legend(fontsize=7)
                fig.suptitle("Semantic axis score distributions by topic_family (sample shards)")
                fig.tight_layout()
                fig.savefig(fig_root / "score_distributions_by_family.png", dpi=150)
                plt.close(fig)

    print(f"[plot_semantic_axis_descriptives] wrote figures under {fig_root}", flush=True)


if __name__ == "__main__":
    main()
