"""
Script summary:
Descriptive bridge plots comparing within-person lexical shifts across author language groups.

Functionality:
- Joins shift_per_user_*_polarization.csv with wordfish author assignment (primary lexicon).
- Violin plots of pooled lexical deltas by assigned_primary_lexicon (it/en/de).
- Not causal cross-country effects; complements forum-day lexical DiD.

How to apply/run:
  .venv/bin/python scripts/user_week/plot_user_lexical_by_lexicon.py \\
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LEXICAL_DELTA_COLS: Tuple[Tuple[str, str], ...] = (
    ("delta_pooled_net_ideology", "Net ideology"),
    ("delta_pooled_pole_share", "Pole share"),
    ("delta_pooled_extremity", "Extremity"),
    ("delta_pooled_polarization_composite_user_week", "Polarization composite"),
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

from scripts.diagnostics.prepare_did_author_semantic_week_panel import load_assignment  # noqa: E402
from src.config_utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for lexicon-group lexical shift figures."""
    parser = argparse.ArgumentParser(
        description="Plot within-person lexical shifts by author primary lexicon."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohorts",
        type=str,
        default="strict,loose",
        help="Comma-separated cohort labels.",
    )
    return parser.parse_args()


def _load_shift(tables_dir: Path, cohort: str) -> pd.DataFrame:
    """Function summary: load per-user polarization shift CSV for one cohort."""
    path = tables_dir / f"shift_per_user_{cohort}_polarization.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_by_lexicon(
    df: pd.DataFrame,
    delta_col: str,
    title: str,
    cohort: str,
    out_path: Path,
) -> None:
    """Function summary: violin plot of one delta column by primary lexicon."""
    if df.empty or delta_col not in df.columns:
        return
    work = df.dropna(subset=[delta_col, "assigned_primary_lexicon"]).copy()
    if work.empty:
        return
    order: List[str] = [x for x in ("it", "en", "de") if x in work["assigned_primary_lexicon"].unique()]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [work.loc[work["assigned_primary_lexicon"] == lex, delta_col].astype(float).values for lex in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmeans=True, showmedians=True)
    for body in parts.get("bodies", []):
        body.set_alpha(0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Pooled pre→post delta")
    ax.set_title(f"{title} — {cohort} cohort")
    fig.suptitle(
        "Within-person shift by author language (descriptive; not IT vs control DiD)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: write by_primary_lexicon figures for each cohort and lexical delta."""
    args = parse_args()
    config = load_config(args.config)
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    fig_root = Path(config["paths"]["figures_dir"]) / "user_week"
    assignment = load_assignment(config)

    for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        shifts = _load_shift(tables_dir, cohort)
        if shifts.empty:
            print(
                f"[plot_user_lexical_by_lexicon] skip cohort={cohort}: missing shift CSV",
                flush=True,
            )
            continue
        shifts["author"] = shifts["author"].astype(str)
        merged = shifts.merge(assignment, on="author", how="inner")
        out_dir = fig_root / cohort / "polarization" / "by_primary_lexicon"
        for col, label in LEXICAL_DELTA_COLS:
            stem = col.replace("delta_pooled_", "")
            plot_by_lexicon(
                merged,
                col,
                label,
                cohort,
                out_dir / f"{stem}_by_lexicon.png",
            )
        print(
            f"[plot_user_lexical_by_lexicon] cohort={cohort} users={len(merged)} out_dir={out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
