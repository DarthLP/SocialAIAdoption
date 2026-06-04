"""
Script summary:
Overview dotplot of median within-person pooled deltas for headline lexical and semantic features.

Functionality:
- Reads shift_per_user_{strict}_polarization.csv and shift_per_user_{strict}_semantic.csv.
- Plots median delta_pooled_* with optional IQR whiskers for thesis summary figure.

How to apply/run:
  .venv/bin/python scripts/user_week/plot_user_week_overview.py \\
    --config config/italy_polarization_setup.yaml --cohort strict
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OVERVIEW_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("polarization", "delta_pooled_net_ideology", "Net ideology"),
    ("polarization", "delta_pooled_pole_share", "Pole share"),
    ("polarization", "delta_pooled_extremity", "Extremity"),
    ("semantic", "delta_pooled_sem_axis_ideology", "Sem. ideology"),
    ("semantic", "delta_pooled_sem_axis_emotion", "Sem. emotion"),
    ("semantic", "delta_pooled_sem_axis_aggression", "Sem. aggression"),
    ("semantic", "delta_pooled_sem_axis_economic", "Sem. economic"),
    ("semantic", "delta_pooled_sem_axis_cultural", "Sem. cultural"),
    ("semantic", "delta_pooled_sem_axis_nationalism", "Sem. nationalism"),
    ("semantic", "delta_pooled_sem_axis_anti_establishment", "Sem. anti-est."),
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
    """Function summary: CLI for overview figure."""
    parser = argparse.ArgumentParser(description="User-week headline shift overview.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--cohort", type=str, default="strict")
    return parser.parse_args()


def _median_iqr(series: pd.Series) -> Tuple[float, float, float]:
    """Function summary: median and 25/75 percentiles for error bars."""
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty:
        return float("nan"), float("nan"), float("nan")
    return float(v.median()), float(v.quantile(0.25)), float(v.quantile(0.75))


def main() -> None:
    """Function summary: write overview forest-style dotplot for one cohort."""
    args = parse_args()
    config = load_config(args.config)
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    fig_dir = Path(config["paths"]["figures_dir"]) / "user_week" / args.cohort / "overview"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    medians: List[float] = []
    err_low: List[float] = []
    err_high: List[float] = []
    colors: List[str] = []

    for slug, col, label in OVERVIEW_SPECS:
        path = tables_dir / f"shift_per_user_{args.cohort}_{slug}.csv"
        if not path.is_file() or col not in pd.read_csv(path, nrows=0).columns:
            continue
        df = pd.read_csv(path, usecols=[col])
        med, q25, q75 = _median_iqr(df[col])
        if not np.isfinite(med):
            continue
        labels.append(label)
        medians.append(med)
        err_low.append(med - q25 if np.isfinite(q25) else 0.0)
        err_high.append(q75 - med if np.isfinite(q75) else 0.0)
        colors.append("#e76f51" if slug == "semantic" else "#2a6f97")

    if not labels:
        print("[plot_user_week_overview] no data; run analyze_user_pre_post_shift.py", flush=True)
        return

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.55)))
    ax.errorbar(
        medians,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=3,
    )
    for i, c in enumerate(colors):
        ax.plot(medians[i], y_pos[i], "o", color=c, markersize=8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median pooled pre→post delta (users)")
    ax.set_title(f"Headline within-person shifts ({args.cohort} cohort)")
    fig.suptitle(
        "Descriptive Italy user-week shifts; blue=lexical, orange=semantic",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out = fig_dir / "headline_median_shifts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    readme = (
        "# User-week overview\n\n"
        "Median pooled within-person deltas (post − pre) for headline outcomes. "
        "Error bars span the 25th–75th percentile across users. "
        "Not cross-country DiD; complements `estimate_user_week_panel.py` post coefficients.\n"
    )
    (fig_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[plot_user_week_overview] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
