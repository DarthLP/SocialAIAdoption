"""
Script summary:
Within-user pole-margin decomposition figures linking forum-day pole_share DiD to author shifts.

Functionality:
- Reads shift_per_user_* CSVs from analyze_user_pre_post_shift.py (pole_share, left/right/center rates).
- Plots pooled pre/post shift distributions for ideology pole margins (strict and loose cohorts).

How to apply/run:
  .venv/bin/python scripts/user_week/plot_user_pole_decomposition.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLE_FEATURES: tuple[tuple[str, str], ...] = (
    ("left_rate_100w", "Left rate (per 100w)"),
    ("right_rate_100w", "Right rate (per 100w)"),
    ("center_rate_100w", "Center rate (per 100w)"),
    ("pole_share", "Pole share"),
)


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


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for pole decomposition figures."""
    parser = argparse.ArgumentParser(description="Plot within-user pole margin shifts.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohorts",
        type=str,
        default="strict,loose",
        help="Comma-separated cohort labels (must match shift_per_user_* files).",
    )
    parser.add_argument(
        "--composite-slug",
        type=str,
        default="polarization",
        help="Composite slug used in shift_per_user_<cohort>_<slug>.csv filenames.",
    )
    return parser.parse_args()


def _load_per_user(tables_dir: Path, cohort: str, composite_slug: str) -> pd.DataFrame:
    """Function summary: load per-user shift table for one cohort if present.

    Parameters:
    - tables_dir: user_week tables root.
    - cohort: strict or loose.
    - composite_slug: polarization or style.

    Returns:
    - Per-user DataFrame (possibly empty).
    """
    path = tables_dir / f"shift_per_user_{cohort}_{composite_slug}.csv"
    if not path.is_file():
        path = tables_dir / f"shift_per_user_{cohort}.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_pole_decomposition(
    per_user: pd.DataFrame,
    cohort: str,
    out_path: Path,
    launch_day: str,
) -> None:
    """Function summary: four-panel histogram of pooled deltas for pole margin features.

    Parameters:
    - per_user: shift_per_user table.
    - cohort: cohort label for title.
    - out_path: PNG destination.
    - launch_day: ban date string for subtitle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()
    any_panel = False
    for ax, (feat, label) in zip(axes_flat, POLE_FEATURES):
        col = f"delta_pooled_{feat}"
        if col not in per_user.columns:
            ax.set_visible(False)
            continue
        v = pd.to_numeric(per_user[col], errors="coerce").dropna()
        if v.empty:
            ax.set_title(f"{label}\n(no data)")
            continue
        any_panel = True
        ax.hist(v.values, bins=50, color="#457b9d", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="gray", linewidth=0.8)
        med = float(v.median())
        ax.axvline(med, color="#e76f51", linewidth=1.2, linestyle="--", label=f"median={med:.3g}")
        ax.set_title(label)
        ax.set_xlabel("Pooled post − pre")
        ax.set_ylabel("Users")
        ax.legend(fontsize=8)
    if not any_panel:
        plt.close(fig)
        return
    fig.suptitle(
        f"Within-user pole margins ({cohort} cohort; ban {launch_day})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: write pole decomposition PNGs per cohort under figures/user_week/."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    fig_root = Path(config["paths"]["figures_dir"]) / "user_week"
    launch = str(config["event_window"]["launch_day_utc"])
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    if not tables_dir.joinpath("user_week_panel.parquet").is_file():
        raise FileNotFoundError(
            f"Missing {tables_dir / 'user_week_panel.parquet'}. "
            "Run scripts/user_week/prepare_user_week_style_panel.py first."
        )
    for cohort in cohorts:
        per_user = _load_per_user(tables_dir, cohort, args.composite_slug)
        if per_user.empty:
            print(f"[plot_user_pole_decomposition] skip {cohort}: no shift_per_user file", flush=True)
            continue
        out = fig_root / cohort / "pole_decomposition" / "pole_margin_shifts.png"
        plot_pole_decomposition(per_user, cohort, out, launch)
        print(f"[plot_user_pole_decomposition] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
