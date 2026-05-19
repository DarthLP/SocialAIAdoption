"""
Script summary:
Plot polarization and AI-use descriptives from prepared CSV tables.

Functionality:
- Family-level daily trends for primary polarization metrics.
- Country-panel overlay (Italy vs controls) for AI first-stage and net ideology.
- Vertical ban reference lines from config.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import pandas as pd


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, plot_reference_dates_calendar_utc  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot polarization descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: draw vertical reference dates on an axes.

    Parameters:
    - ax: matplotlib axes.
    - config: study YAML.
    """
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def plot_family_metric(family_df: pd.DataFrame, metric: str, out_path: Path, config: dict, title: str) -> None:
    """Function summary: line plot of one metric by topic_family over date.

    Parameters:
    - family_df: daily_by_topic_family table.
    - metric: column name.
    - out_path: PNG path.
    - config: YAML for reference lines.
    - title: plot title.
    """
    if family_df.empty or metric not in family_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work = family_df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    for family, grp in work.groupby("topic_family"):
        grp = grp.sort_values("date_utc")
        ax.plot(grp["date_utc"], grp[metric], label=family, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(title)
    ax.set_xlabel("date_utc")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_country_panel(country_df: pd.DataFrame, metric: str, out_path: Path, config: dict) -> None:
    """Function summary: plot metric by country_panel for ban-window comparison.

    Parameters:
    - country_df: daily_country_panel table.
    - metric: column to plot.
    - out_path: PNG output.
    - config: study config.
    """
    if country_df.empty or metric not in country_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    work = country_df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"])
    for panel, grp in work.groupby("country_panel"):
        grp = grp.sort_values("date_utc")
        ax.plot(grp["date_utc"], grp[metric], label=panel, alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(f"Country panel: {metric}")
    ax.set_xlabel("date_utc")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: generate descriptives figures from CSV tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables_dir = Path(config["paths"]["tables_dir"]) / "descriptives"
    fig_dir = Path(config["paths"]["figures_dir"]) / "descriptives"
    fig_dir.mkdir(parents=True, exist_ok=True)

    family_path = tables_dir / "daily_by_topic_family.csv"
    country_path = tables_dir / "daily_country_panel.csv"
    if not family_path.is_file():
        print("[plot_polarization_descriptives] run prepare_polarization_descriptives.py first", flush=True)
        return

    family_df = pd.read_csv(family_path)
    for metric, fname in [
        ("net_ideology_mean", "by_family_net_ideology.png"),
        ("extremity_mean", "by_family_extremity.png"),
        ("other_side_salience_rate_100w_mean", "by_family_other_side_salience.png"),
        ("aggression_rate_100w_mean", "by_family_aggression.png"),
        ("ai_style_rate_100w_mean", "by_family_ai_style.png"),
    ]:
        plot_family_metric(family_df, metric, fig_dir / fname, config, metric)

    if country_path.is_file():
        country_df = pd.read_csv(country_path)
        plot_country_panel(country_df, "ai_style_rate_100w_mean", fig_dir / "country_panel_ai_style.png", config)
        plot_country_panel(country_df, "net_ideology_mean", fig_dir / "country_panel_net_ideology.png", config)
        plot_country_panel(country_df, "esteban_ray_index", fig_dir / "country_panel_esteban_ray.png", config)

    print(f"[plot_polarization_descriptives] wrote figures to {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
