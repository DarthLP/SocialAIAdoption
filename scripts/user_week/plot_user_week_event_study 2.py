"""
Script summary:
Plot author×week event-study coefficients from estimate_user_week_panel.py exports.

Functionality:
- Reads event_study_{cohort}_{feature}.csv tables.
- Coefficient plot with vertical line at ban week (rel_week=0); ref week -1 omitted.

How to apply/run:
  .venv/bin/python scripts/user_week/plot_user_week_event_study.py \\
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

from src.config_utils import load_config  # noqa: E402

FEATURE_LABELS = {
    "net_ideology": "Net ideology (lexical)",
    "sem_axis_ideology": "Semantic ideology axis",
}


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for event-study figure generation."""
    parser = argparse.ArgumentParser(description="Plot user-week event-study coefficients.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--cohorts", type=str, default="strict,loose")
    return parser.parse_args()


def plot_event_study(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Function summary: coefficient plot with 95% CI error bars.

    Parameters:
    - df: event study table with rel_week, beta, se.
    - title: figure title.
    - out_path: PNG path.
    """
    if df.empty:
        return
    work = df.sort_values("rel_week")
    x = work["rel_week"].astype(int).values
    y = work["beta"].astype(float).values
    se = work["se"].astype(float).values
    ci = 1.96 * se
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, y, yerr=ci, fmt="o-", capsize=3, color="#2a6f97")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="#e76f51", linewidth=1.0, linestyle="--", label="Ban week")
    ax.set_xlabel("Weeks relative to ban (ISO weeks)")
    ax.set_ylabel("Coefficient (vs week −1)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Author FE event study on user×week panel (Italy; not cross-country DiD)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: write event-study PNGs per cohort and feature."""
    args = parse_args()
    config = load_config(args.config)
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    fig_root = Path(config["paths"]["figures_dir"]) / "user_week"

    for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        out_dir = fig_root / cohort / "event_study"
        for feat, label in FEATURE_LABELS.items():
            path = tables_dir / f"event_study_{cohort}_{feat}.csv"
            if not path.is_file():
                print(f"[plot_user_week_event_study] skip {cohort}/{feat}: no table", flush=True)
                continue
            df = pd.read_csv(path)
            plot_event_study(
                df,
                f"{label} — {cohort}",
                out_dir / f"{feat}.png",
            )
        print(f"[plot_user_week_event_study] cohort={cohort} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
