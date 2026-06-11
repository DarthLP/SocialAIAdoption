"""
Script summary:
Two-panel thesis figure for lexical pole_share: raw IT vs control trajectories and 3-day event study.

Functionality:
- Panel A: daily mean pole_share (extreme L+R) for Italian vs pooled control arms from the
  canonical subreddit-day DiD panel (did_subreddit_panel_1d.csv).
- Panel B: 3-day cross_country_all event-study coefficients from saved language_universe ES CSV.
- Applies shared thesis ban-window guides and standardized axis labels; no baked-in statistics.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_pole_share_eventstudy.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_DAYS = 3
POLE_SHARE_COL = "pole_share"
FIXED_AUTHORS_ES = "pole_share_fixed_authors/event_study.csv"


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

from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.did.panels import load_subreddit_panel  # noqa: E402
from src.did.specs import CONTROL_FAMILIES, ITALY_FAMILIES  # noqa: E402
from src.plotting.thesis_theme import (  # noqa: E402
    THESIS_CONTROL,
    THESIS_ITALY,
    shade_ban_window,
    xlabel_event_study,
    xlabel_event_study_days,
    ylabel_italy_bin_coefficient,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for pole_share two-panel thesis figure."""
    parser = argparse.ArgumentParser(description="Pole share raw + event-study thesis figure.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _launch_day(config: Dict[str, Any]) -> str:
    """Function summary: ban launch ISO date from config."""
    return str((config.get("event_window") or {}).get("launch_day_utc") or "2023-03-31")


def _load_trajectory_means(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: daily mean pole_share by arm (Italy vs pooled controls).

    Parameters:
    - config: study YAML.

    Returns:
    - DataFrame with columns rel_day, italy_mean, control_mean.
    """
    panel = load_subreddit_panel(config)
    if panel.empty or POLE_SHARE_COL not in panel.columns:
        raise SystemExit("[plot_pole_share_eventstudy] missing pole_share on subreddit panel")
    launch = pd.Timestamp(_launch_day(config))
    work = panel.dropna(subset=[POLE_SHARE_COL]).copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"], utc=True, errors="coerce")
    work = work[work["date_utc"].notna()]
    work["rel_day"] = (work["date_utc"].dt.normalize() - launch).dt.days.astype(int)
    work["arm"] = np.where(
        work["topic_family"].astype(str).isin(ITALY_FAMILIES),
        "italy",
        np.where(work["topic_family"].astype(str).isin(CONTROL_FAMILIES), "control", "other"),
    )
    work = work[work["arm"].isin(("italy", "control"))]
    agg = (
        work.groupby(["rel_day", "arm"], observed=True)[POLE_SHARE_COL]
        .mean()
        .unstack("arm")
        .reset_index()
    )
    if "italy" not in agg.columns:
        agg["italy"] = np.nan
    if "control" not in agg.columns:
        agg["control"] = np.nan
    return agg.rename(columns={"italy": "italy_mean", "control": "control_mean"})


def _load_event_study_csv(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load 3-day language_universe pole_share event-study coefficients.

    Parameters:
    - config: study YAML.

    Returns:
    - Sorted ES DataFrame with rel_period, gamma, ci_low, ci_high.
    """
    path = (
        tables_subdir(config, "did")
        / "estimates/lexical/event_study/language_universe/3d/cross_country_all/pole_share.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    es = pd.read_csv(path)
    es["rel_period"] = pd.to_numeric(es["rel_period"], errors="coerce").astype("Int64")
    es = es.sort_values("rel_period").reset_index(drop=True)
    if -1 not in set(es["rel_period"].dropna().astype(int)):
        ref = {
            "rel_period": -1,
            "gamma": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
        es = pd.concat([es, pd.DataFrame([ref])], ignore_index=True)
        es = es.sort_values("rel_period").reset_index(drop=True)
    return es


def _compare_fixed_authors_es(config: Dict[str, Any], es: pd.DataFrame) -> None:
    """Function summary: sanity-check panel B coefficients vs fixed-author all-authors ES.

    Parameters:
    - config: study YAML.
    - es: panel B coefficient table.

    Returns:
    - None; prints max absolute gamma difference at shared rel_periods.
    """
    path = tables_subdir(config, "did") / FIXED_AUTHORS_ES
    if not path.is_file():
        print(f"[plot_pole_share_eventstudy] skip ES compare: missing {path}", flush=True)
        return
    fixed = pd.read_csv(path)
    fixed = fixed[fixed["sample"].astype(str) == "all_authors"].copy()
    if fixed.empty:
        return
    merged = es.merge(
        fixed[["rel_period", "gamma"]].rename(columns={"gamma": "gamma_fixed"}),
        on="rel_period",
        how="inner",
    )
    if merged.empty:
        print("[plot_pole_share_eventstudy] ES compare: no overlapping rel_periods", flush=True)
        return
    diff = (merged["gamma"] - merged["gamma_fixed"]).abs()
    print(
        f"[plot_pole_share_eventstudy] ES vs fixed-authors all: "
        f"max|Δγ|={diff.max():.6f} n_periods={len(merged)}",
        flush=True,
    )


def _plot_panel_a(ax: plt.Axes, traj: pd.DataFrame) -> None:
    """Function summary: draw raw trajectory panel (Italy vs pooled controls).

    Parameters:
    - ax: matplotlib axes.
    - traj: daily means with rel_day, italy_mean, control_mean.

    Returns:
    - None; mutates ax in place.
    """
    ax.plot(
        traj["rel_day"],
        traj["italy_mean"],
        color=THESIS_ITALY,
        linewidth=2.2,
        label="Italy",
        zorder=4,
    )
    ax.plot(
        traj["rel_day"],
        traj["control_mean"],
        color=THESIS_CONTROL,
        linewidth=1.5,
        label="Pooled controls",
        zorder=4,
    )
    shade_ban_window(ax, mode="event_study", bin_days=1, x_scale="days", zorder=0)
    ax.set_title("Raw trajectories")
    ax.set_xlabel(xlabel_event_study_days())
    ax.set_ylabel("Pole share (extreme L+R)")
    ax.legend(loc="best", fontsize=8)


def _plot_panel_b(ax: plt.Axes, es: pd.DataFrame) -> None:
    """Function summary: draw 3-day event-study coefficient panel.

    Parameters:
    - ax: matplotlib axes.
    - es: event-study coefficients with rel_period and CI columns.

    Returns:
    - None; mutates ax in place.
    """
    x = es["rel_period"].astype(float) * BIN_DAYS
    y = pd.to_numeric(es["gamma"], errors="coerce")
    lo = pd.to_numeric(es["ci_low"], errors="coerce")
    hi = pd.to_numeric(es["ci_high"], errors="coerce")
    yerr_lo = (y - lo).clip(lower=0)
    yerr_hi = (hi - y).clip(lower=0)
    mask = y.notna() & lo.notna() & hi.notna()
    ax.errorbar(
        x[mask],
        y[mask],
        yerr=[yerr_lo[mask], yerr_hi[mask]],
        fmt="none",
        ecolor="black",
        capsize=3,
        zorder=5,
    )
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        zorder=6,
    )
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    shade_ban_window(ax, mode="event_study", bin_days=BIN_DAYS, x_scale="days", zorder=0)
    ax.set_title("Event study")
    ax.set_xlabel(xlabel_event_study(BIN_DAYS))
    ax.set_ylabel(ylabel_italy_bin_coefficient())


def plot_pole_share_eventstudy(config: Dict[str, Any]) -> Tuple[Path, Dict[str, str], Dict[str, str]]:
    """Function summary: build and save the two-panel pole_share thesis figure.

    Parameters:
    - config: study YAML.

    Returns:
    - Tuple of (output path, panel A label dict, panel B label dict).
    """
    traj = _load_trajectory_means(config)
    es = _load_event_study_csv(config)
    _compare_fixed_authors_es(config, es)

    out = (
        figures_subdir(config, "did")
        / "lexical"
        / "event_study"
        / "pole_share_eventstudy.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))
    _plot_panel_a(ax_a, traj)
    _plot_panel_b(ax_b, es)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    labels_a = {
        "x": ax_a.get_xlabel(),
        "y": ax_a.get_ylabel(),
        "title": ax_a.get_title(),
    }
    labels_b = {
        "x": ax_b.get_xlabel(),
        "y": ax_b.get_ylabel(),
        "title": ax_b.get_title(),
    }
    return out, labels_a, labels_b


def main() -> None:
    """Function summary: CLI entry — write pole_share_eventstudy.png and print labels."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out, labels_a, labels_b = plot_pole_share_eventstudy(config)
    print(f"[plot_pole_share_eventstudy] wrote {out.resolve()}", flush=True)
    print(
        f"  Panel A: x={labels_a['x']!r} y={labels_a['y']!r} title={labels_a['title']!r}",
        flush=True,
    )
    print(
        f"  Panel B: x={labels_b['x']!r} y={labels_b['y']!r} title={labels_b['title']!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
