"""
Script summary:
Plot dominant ideology, pair-framing, and v4 metadata descriptives from prepared CSV tables.

Functionality:
- Content-named folders under descriptives/ (ideology_dominant, pairs, stance, primary, trajectory_scatter).
- Primary vs exploratory titles; min-n gating on raw daily; rolling + ban windows as primary views.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
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

from scripts.diagnostics import descriptives_util as du


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
    load_polarization_config,
    plot_reference_dates_calendar_utc,
    require_dominant_v1_ideology_scoring,
    tables_subdir,
)

ITALIAN_FAMILIES = ("it_political", "it_pure_political", "it_others")


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot lexicon descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling_window", type=int, default=7)
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: draw vertical ban reference dates."""
    for d in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(d), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)


def plot_family_lines(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    config: dict,
    title_prefix: str,
    families: tuple[str, ...] = ITALIAN_FAMILIES,
    min_n: int = 30,
) -> None:
    """Function summary: multi-line daily/rolling plot by topic_family.

    Parameters:
    - df: daily table with metric_mean column.
    - metric: base metric name (suffix _mean added).
    - out_path: PNG path.
    - config: study YAML.
    - title_prefix: PRIMARY or EXPLORATORY label.
    - families: families to plot.
    - min_n: minimum comments per day.

    Returns:
    - None.
    """
    col = f"{metric}_mean"
    if df.empty or col not in df.columns:
        return
    d = du.apply_min_n_filter(df, min_n)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for fam in families:
        sub = d[d["topic_family"] == fam].sort_values("date_utc")
        if sub.empty:
            continue
        ax.plot(pd.to_datetime(sub["date_utc"]), sub[col], label=fam, linewidth=1.5)
    add_ref_lines(ax, config)
    ax.set_title(f"{title_prefix} {metric} — Italian families — Mar–Apr 2023")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(col)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ban_bars(df: pd.DataFrame, metric: str, out_path: Path, title_prefix: str) -> None:
    """Function summary: grouped bar chart pre vs post by window for one metric."""
    col = f"{metric}_mean"
    if df.empty or col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for wk in sorted(df["window"].unique()):
        for phase, color in (("pre", "steelblue"), ("post", "coral")):
            row = df[(df["window"] == wk) & (df["phase"] == phase)]
            if row.empty:
                continue
            ax.bar(f"{wk}_{phase}", float(row[col].iloc[0]), color=color, label=phase if wk == "W0" else "")
    ax.set_title(f"{title_prefix} {metric} — ban windows")
    ax.set_ylabel(col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_scatter(
    ideology_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    out_path: Path,
    launch: str,
    min_n: int,
) -> None:
    """Function summary: exploratory valence vs net_ideology scatter colored by days to launch.

    Parameters:
    - ideology_df: rolling or daily ideology table.
    - meta_df: metadata table with valence rates.
    - out_path: PNG path.
    - launch: launch date.
    - min_n: min comments.

    Returns:
    - None.
    """
    if ideology_df.empty or meta_df.empty:
        return
    d = ideology_df.merge(
        meta_df[["date_utc", "topic_family", "valence_positive_rate_100w_mean", "valence_negative_rate_100w_mean", "n_comments"]],
        on=["date_utc", "topic_family"],
        how="inner",
    )
    d = du.apply_min_n_filter(d, min_n)
    core = d[d["topic_family"] == "it_political"]
    if core.empty or "net_ideology_mean" not in core.columns:
        return
    core = core.copy()
    core["valence_index"] = (
        core["valence_positive_rate_100w_mean"].fillna(0) - core["valence_negative_rate_100w_mean"].fillna(0)
    )
    launch_ts = pd.Timestamp(launch)
    core["days_to_launch"] = (pd.to_datetime(core["date_utc"]) - launch_ts).dt.days
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        core["valence_index"],
        core["net_ideology_mean"],
        c=core["days_to_launch"],
        cmap="RdYlGn_r",
        s=40,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="Days to launch")
    ax.set_xlabel("Valence index (+ pos rate − neg rate)")
    ax.set_ylabel("net_ideology (dominant)")
    ax.set_title("[EXPLORATORY] Valence vs net ideology — it_political — Mar–Apr 2023")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: generate lexicon descriptives figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    require_dominant_v1_ideology_scoring(config)
    pol_cfg = load_polarization_config(config)
    min_n = int(pol_cfg.get("dip_min_n", 30))
    roll_days = int(args.rolling_window)
    _, _, launch, _ = du.event_dates_from_config(config)

    tables_dir = tables_subdir(config, "descriptives")
    fig_root = figures_subdir(config, "descriptives")

    ide_daily = pd.read_csv(tables_dir / "daily_ideology_dominant_by_topic_family.csv")
    ide_roll = pd.read_csv(tables_dir / "rolling_ideology_dominant_by_topic_family.csv")
    pair_daily = pd.read_csv(tables_dir / "daily_pair_framing_by_topic_family.csv")
    pair_roll = pd.read_csv(tables_dir / "rolling_pair_framing_by_topic_family.csv")
    meta_daily = pd.read_csv(tables_dir / "daily_v4_metadata_by_topic_family.csv")
    meta_roll = pd.read_csv(tables_dir / "rolling_v4_metadata_by_topic_family.csv")
    ban_launch = pd.read_csv(tables_dir / "ban_windows_launch_primary.csv")

    # Primary folder
    plot_family_lines(
        ide_roll,
        "net_ideology",
        fig_root / "primary" / "net_ideology_launch_w0_rolling.png",
        config,
        "[PRIMARY]",
        families=("it_political",),
        min_n=min_n,
    )
    plot_family_lines(
        pair_roll,
        "pair_framing_net_strict",
        fig_root / "primary" / "pair_framing_net_strict_launch_w0_rolling.png",
        config,
        "[PRIMARY]",
        families=("it_political",),
        min_n=min_n,
    )

    # Ideology dominant
    for metric in ("net_ideology", "left_rate_100w", "center_rate_100w", "right_rate_100w"):
        plot_family_lines(
            ide_roll,
            metric,
            fig_root / "ideology_dominant" / "rolling_daily" / "by_family" / f"{metric}.png",
            config,
            "[PRIMARY]",
            min_n=min_n,
        )
        plot_family_lines(
            ide_daily,
            metric,
            fig_root / "ideology_dominant" / "daily" / "by_family" / f"{metric}.png",
            config,
            "[EXPLORATORY]",
            min_n=min_n,
        )

    # Pairs
    for metric in ("pair_framing_net_strict", "pair_framing_rate_100w_strict", "pair_active_strict"):
        plot_family_lines(
            pair_roll,
            metric,
            fig_root / "pairs" / "strict_polarized" / "rolling_daily" / f"{metric}.png",
            config,
            "[PRIMARY]",
            min_n=min_n,
        )
    plot_ban_bars(
        ban_launch,
        "pair_framing_net_strict",
        fig_root / "pairs" / "strict_polarized" / "ban_windows_launch" / "pair_framing_net_strict.png",
        "[PRIMARY]",
    )

    # Metadata / stance / valence / polarized
    for metric, folder in (
        ("stance_contra_rate_100w", "stance"),
        ("stance_pro_rate_100w", "stance"),
        ("valence_negative_rate_100w", "valence"),
        ("valence_positive_rate_100w", "valence"),
        ("polarized_yes_rate_100w", "polarized"),
        ("relevance_weighted_contra_rate_100w", "relevance_high"),
    ):
        plot_family_lines(
            meta_roll,
            metric,
            fig_root / folder / "rolling_daily" / f"{metric}.png",
            config,
            "[PRIMARY]",
            min_n=min_n,
        )

    plot_trajectory_scatter(
        ide_roll,
        meta_roll,
        fig_root / "trajectory_scatter" / "by_topic_family" / "valence_vs_net_ideology_it_political.png",
        launch,
        min_n,
    )

    print(f"[plot_lexicon_descriptives] wrote figures under {fig_root}", flush=True)


if __name__ == "__main__":
    main()
