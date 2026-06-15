"""
Script summary:
Two-panel thesis figure for the SEMANTIC pole share: raw IT vs control trajectories and 3-day event study.

Functionality:
- Panel A: daily mean sem_axis_ideology_pole_share_pct (extreme L+R tail share on the semantic
  ideology axis, PER-LEXICON pre-ban p10/p90 calibration — semantic scores are not comparable
  across languages, so each language gets its own cutoffs and every arm sits near ~0.2 pre-ban
  by construction) for Italian vs pooled control arms, from the same subreddit-day panel that
  feeds the DiD (load_subreddit_panel + merge_semantic_axis). The non-_pct base column uses
  absolute cutoffs that control-language comments essentially never cross (controls ~0) — do
  not plot that variant for a cross-arm figure.
- Panel B: 3-day cross_country_all event-study coefficients on the same _pct outcome from the
  language/subreddit/3d aggregated ES CSV (neutral #333333 markers).
- Shared thesis ban-window guides and standardized axis labels; no baked-in statistics.
- Sanity gate: aborts when any |gamma| > 0.5 in the event-study CSV (degenerate fit guard).

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_semantic_pole_share_eventstudy.py --config config/italy_polarization_setup.yaml
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
SEM_POLE_SHARE_COL = "sem_axis_ideology_pole_share_pct"
MAX_ABS_GAMMA = 0.5


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
from src.did.panels import load_subreddit_panel, merge_semantic_axis  # noqa: E402
from src.did.specs import CONTROL_FAMILIES, ITALY_FAMILIES  # noqa: E402
from src.plotting.thesis_theme import (  # noqa: E402
    THESIS_COEF_MARKER,
    THESIS_CONTROL,
    THESIS_ITALY,
    shade_ban_window,
    xlabel_event_study,
    xlabel_event_study_days,
    ylabel_italy_bin_coefficient,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for the semantic pole_share two-panel thesis figure."""
    parser = argparse.ArgumentParser(
        description="Semantic pole share raw + event-study thesis figure."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _launch_day(config: Dict[str, Any]) -> str:
    """Function summary: ban launch ISO date from config."""
    return str((config.get("event_window") or {}).get("launch_day_utc") or "2023-03-31")


def _load_trajectory_means(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: daily mean semantic pole share by arm (Italy vs pooled controls).

    Parameters:
    - config: study YAML.

    Returns:
    - DataFrame with columns rel_day, italy_mean, control_mean.
    """
    panel = merge_semantic_axis(load_subreddit_panel(config), config)
    if panel.empty or SEM_POLE_SHARE_COL not in panel.columns:
        raise SystemExit(
            f"[plot_semantic_pole_share_eventstudy] missing {SEM_POLE_SHARE_COL} on subreddit panel"
        )
    launch = pd.Timestamp(_launch_day(config))
    work = panel.dropna(subset=[SEM_POLE_SHARE_COL]).copy()
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
        work.groupby(["rel_day", "arm"], observed=True)[SEM_POLE_SHARE_COL]
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
    """Function summary: load 3-day language/subreddit semantic pole-share event-study coefficients.

    Parameters:
    - config: study YAML.

    Returns:
    - Sorted ES DataFrame with rel_period, gamma, ci_low, ci_high (ref bin appended at 0).

    Raises:
    - SystemExit: when any |gamma| exceeds MAX_ABS_GAMMA (degenerate fit guard).
    """
    path = (
        tables_subdir(config, "did")
        / "estimates/semantic_axis/event_study/language/subreddit/3d/cross_country_all"
        / f"{SEM_POLE_SHARE_COL}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    es = pd.read_csv(path)
    gmax = float(pd.to_numeric(es["gamma"], errors="coerce").abs().max())
    if gmax > MAX_ABS_GAMMA:
        raise SystemExit(
            f"[plot_semantic_pole_share_eventstudy] ABORT: max|gamma|={gmax:.4g} > "
            f"{MAX_ABS_GAMMA} in {path} — degenerate fit; regenerate the "
            "language/subreddit/3d bundle before plotting."
        )
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
    ax.set_ylabel("Semantic pole share (extreme L+R)")
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
        ecolor=THESIS_COEF_MARKER,
        capsize=3,
        zorder=5,
    )
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=THESIS_COEF_MARKER,
        markeredgewidth=1.0,
        zorder=6,
    )
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    shade_ban_window(ax, mode="event_study", bin_days=BIN_DAYS, x_scale="days", zorder=0)
    ax.set_title("Event study")
    ax.set_xlabel(xlabel_event_study(BIN_DAYS))
    ax.set_ylabel(ylabel_italy_bin_coefficient())


def plot_semantic_pole_share_eventstudy(
    config: Dict[str, Any],
) -> Tuple[Path, Dict[str, str], Dict[str, str], Tuple[float, float]]:
    """Function summary: build and save the two-panel semantic pole_share thesis figure.

    Parameters:
    - config: study YAML.

    Returns:
    - Tuple of (output path, panel A labels, panel B labels, (gamma_min, gamma_max)).
    """
    traj = _load_trajectory_means(config)
    es = _load_event_study_csv(config)

    out = (
        figures_subdir(config, "did")
        / "semantic_axis"
        / "event_study"
        / "semantic_pole_share_eventstudy.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))
    _plot_panel_a(ax_a, traj)
    _plot_panel_b(ax_b, es)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    labels_a = {"x": ax_a.get_xlabel(), "y": ax_a.get_ylabel(), "title": ax_a.get_title()}
    labels_b = {"x": ax_b.get_xlabel(), "y": ax_b.get_ylabel(), "title": ax_b.get_title()}
    gamma = pd.to_numeric(es["gamma"], errors="coerce")
    return out, labels_a, labels_b, (float(gamma.min()), float(gamma.max()))


def main() -> None:
    """Function summary: CLI entry — write semantic_pole_share_eventstudy.png and print audit."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out, labels_a, labels_b, (g_min, g_max) = plot_semantic_pole_share_eventstudy(config)
    print(f"[plot_semantic_pole_share_eventstudy] wrote {out.resolve()}", flush=True)
    print(
        f"  Panel A: x={labels_a['x']!r} y={labels_a['y']!r} title={labels_a['title']!r}",
        flush=True,
    )
    print(
        f"  Panel B: x={labels_b['x']!r} y={labels_b['y']!r} title={labels_b['title']!r} "
        f"gamma_min={g_min:.4f} gamma_max={g_max:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
