"""
Script summary:
Thesis dual-tail semantic ideology event-study figure (extreme-left vs extreme-right shares).

Functionality:
- Reads language/subreddit/3d/cross_country_all sem_axis_ideology_extreme_{left,right}.csv.
- Shape gate before plotting: |gamma| <= 0.12 on both tails AND post-ban left peak in
  [+0.03, +0.08] AND post-ban right trough in [-0.08, -0.03] (rejects degenerate
  Italian-only fits and the too-flat overlay_pooled scale alike).
- Fallback to language/hub_pooled/3d ONLY when the subreddit source fails the gate
  (prints a loud warning; hub aggregation must then be noted in the LaTeX caption).
- Dodged overlay (+-0.2 day), LEFT #34708F (blue), RIGHT #CC0000 (red);
  x axis in calendar days (rel_period x 3); shared ban-window guides.
- Prints an audit line (x/y labels, title, gamma min/max, output path).

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_semantic_leftright_eventstudy.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.plotting.thesis_theme import (  # noqa: E402
    shade_ban_window,
    xlabel_event_study,
    ylabel_italy_bin_coefficient,
)

BIN_DAYS = 3
DODGE_DAYS = 0.2
LEFT_COLOR = "#34708F"
RIGHT_COLOR = "#CC0000"
TITLE = "Extreme-left and extreme-right tail shares"

MAX_ABS_GAMMA = 0.12
LEFT_PEAK_RANGE = (0.03, 0.08)
RIGHT_TROUGH_RANGE = (-0.08, -0.03)

PRIMARY_SOURCE = ("language", "subreddit")
FALLBACK_SOURCE = ("language", "hub_pooled")


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for the semantic left/right tail event-study figure."""
    parser = argparse.ArgumentParser(description="Semantic leftright tail-share ES figure.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _csv_path(config: Dict[str, Any], panel_level: str, bundle: str, side: str) -> Path:
    """Function summary: strategy CSV path for one tail under a bundle."""
    return (
        tables_subdir(config, "did")
        / "estimates"
        / "semantic_axis"
        / "event_study"
        / panel_level
        / bundle
        / f"{BIN_DAYS}d"
        / "cross_country_all"
        / f"sem_axis_ideology_extreme_{side}.csv"
    )


def _load_pair(
    config: Dict[str, Any], panel_level: str, bundle: str
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Function summary: load (left, right) ES tables for one source, or None."""
    frames = []
    for side in ("left", "right"):
        path = _csv_path(config, panel_level, bundle, side)
        if not path.is_file():
            print(f"[plot_semantic_leftright_eventstudy] missing {path}", flush=True)
            return None
        df = pd.read_csv(path)
        if df.empty or "rel_period" not in df.columns:
            print(f"[plot_semantic_leftright_eventstudy] unusable {path}", flush=True)
            return None
        df["rel_period"] = pd.to_numeric(df["rel_period"], errors="coerce").astype(int)
        frames.append(df.sort_values("rel_period").reset_index(drop=True))
    return frames[0], frames[1]


def shape_gate(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[bool, str]:
    """Function summary: validate tail-share ES magnitudes and post-ban shape.

    Parameters:
    - left: extreme-left ES table (rel_period, gamma).
    - right: extreme-right ES table.

    Returns:
    - Tuple (passed, detail message).
    """
    gmax = max(float(left["gamma"].abs().max()), float(right["gamma"].abs().max()))
    if gmax > MAX_ABS_GAMMA:
        return False, f"max|gamma|={gmax:.4f} > {MAX_ABS_GAMMA} (degenerate fit)"
    left_post = left[left["rel_period"] >= 0]["gamma"]
    right_post = right[right["rel_period"] >= 0]["gamma"]
    if left_post.empty or right_post.empty:
        return False, "no post-ban coefficients"
    left_peak = float(left_post.max())
    right_trough = float(right_post.min())
    if not (LEFT_PEAK_RANGE[0] <= left_peak <= LEFT_PEAK_RANGE[1]):
        return (
            False,
            f"left post-ban peak {left_peak:.4f} outside {LEFT_PEAK_RANGE} "
            "(overlay_pooled-scale or corrupt series)",
        )
    if not (RIGHT_TROUGH_RANGE[0] <= right_trough <= RIGHT_TROUGH_RANGE[1]):
        return (
            False,
            f"right post-ban trough {right_trough:.4f} outside {RIGHT_TROUGH_RANGE} "
            "(overlay_pooled-scale or corrupt series)",
        )
    return True, f"max|gamma|={gmax:.4f} left_peak={left_peak:.4f} right_trough={right_trough:.4f}"


def _with_reference_row(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: append rel_period -1 at gamma=0 when absent."""
    if -1 in set(df["rel_period"].astype(int)):
        return df
    ref = {"rel_period": -1, "gamma": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return (
        pd.concat([df, pd.DataFrame([ref])], ignore_index=True)
        .sort_values("rel_period")
        .reset_index(drop=True)
    )


def plot_figure(left: pd.DataFrame, right: pd.DataFrame, out_path: Path) -> Dict[str, str]:
    """Function summary: render the dodged dual-tail figure; returns audit labels.

    Parameters:
    - left: extreme-left ES table.
    - right: extreme-right ES table.
    - out_path: destination PNG.

    Returns:
    - Dict with x/y labels and title used.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = (
        (left, LEFT_COLOR, "o", -DODGE_DAYS, "Extreme-left tail share (< p10)"),
        (right, RIGHT_COLOR, "s", DODGE_DAYS, "Extreme-right tail share (> p90)"),
    )
    for df, color, marker, dodge, label in styles:
        plot_df = _with_reference_row(df.copy())
        x = plot_df["rel_period"].astype(float) * BIN_DAYS + dodge
        se = pd.to_numeric(plot_df["se"], errors="coerce").fillna(0)
        mask = se > 0
        ax.errorbar(
            x[mask],
            plot_df.loc[mask, "gamma"],
            yerr=1.96 * se[mask],
            fmt="none",
            ecolor=color,
            capsize=3,
            elinewidth=1.0,
            zorder=5,
        )
        ax.plot(
            x,
            plot_df["gamma"],
            linestyle="none",
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            label=label,
            zorder=6,
        )
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    shade_ban_window(ax, mode="event_study", bin_days=BIN_DAYS, x_scale="days", zorder=0)
    ax.set_xlabel(xlabel_event_study(BIN_DAYS))
    ax.set_ylabel(ylabel_italy_bin_coefficient())
    ax.set_title(TITLE)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"x": ax.get_xlabel(), "y": ax.get_ylabel(), "title": ax.get_title()}


def main() -> None:
    """Function summary: CLI entry — gate, plot, and print the audit line."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    source = PRIMARY_SOURCE
    pair = _load_pair(config, *PRIMARY_SOURCE)
    detail = "missing CSVs"
    if pair is not None:
        ok, detail = shape_gate(*pair)
        if not ok:
            pair = None
    if pair is None:
        print(
            f"[plot_semantic_leftright_eventstudy] WARNING: primary source "
            f"{'/'.join(PRIMARY_SOURCE)}/{BIN_DAYS}d failed gate ({detail}); "
            f"falling back to {'/'.join(FALLBACK_SOURCE)} — note hub-pooled "
            "aggregation in the thesis caption.",
            flush=True,
        )
        source = FALLBACK_SOURCE
        pair = _load_pair(config, *FALLBACK_SOURCE)
        if pair is None:
            sys.exit("[plot_semantic_leftright_eventstudy] fallback CSVs missing — aborting")
        ok, detail = shape_gate(*pair)
        if not ok:
            sys.exit(
                f"[plot_semantic_leftright_eventstudy] fallback also failed gate ({detail}) "
                "— refusing to write a corrupt thesis figure"
            )

    left, right = pair
    out = (
        figures_subdir(config, "did")
        / "event_study"
        / source[0]
        / source[1]
        / f"{BIN_DAYS}d"
        / "semantic_leftright_eventstudy.png"
    )
    labels = plot_figure(left, right, out)
    gammas = pd.concat([left["gamma"], right["gamma"]])
    print(
        f"[plot_semantic_leftright_eventstudy] source={'/'.join(source)}/{BIN_DAYS}d "
        f"gate: {detail}",
        flush=True,
    )
    print(
        f"  x={labels['x']!r} y={labels['y']!r} title={labels['title']!r} "
        f"gamma_min={gammas.min():.4f} gamma_max={gammas.max():.4f} "
        f"ban_guides=onset+lift+shade -> {out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
