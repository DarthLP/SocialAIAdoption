"""
Script summary:
Plot ban-window within-person shifts by pre-ban ideology bucket (lexical vs semantic labels).

Functionality:
- Joins author_ideology_buckets with shift_per_user semantic and polarization exports.
- Violin plots of semantic (ideology, emotion, aggression) and lexical deltas grouped by
  lexical_bucket and by semantic_bucket.

How to apply/run:
  .venv/bin/python scripts/user_week/plot_user_shift_by_ideology_bucket.py \\
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SHIFT_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("semantic", "delta_pooled_sem_axis_ideology", "Ideology axis (semantic delta)"),
    ("semantic", "delta_pooled_sem_axis_emotion", "Emotion axis (semantic delta)"),
    ("semantic", "delta_pooled_sem_axis_aggression", "Aggression axis (semantic delta)"),
    ("semantic", "delta_pooled_sem_axis_economic", "Economic axis (semantic delta)"),
    ("semantic", "delta_pooled_sem_axis_cultural", "Cultural axis (semantic delta)"),
    ("semantic", "delta_pooled_sem_axis_nationalism", "Nationalism axis (semantic delta)"),
    (
        "semantic",
        "delta_pooled_sem_axis_anti_establishment",
        "Anti-establishment axis (semantic delta)",
    ),
    ("semantic", "delta_pooled_semantic_composite_user_week", "Semantic composite delta"),
    ("lexical", "delta_pooled_net_ideology", "Net ideology (lexical delta)"),
    ("lexical", "delta_pooled_pole_share", "Pole share (lexical delta)"),
    ("lexical", "delta_pooled_aggression_rate_100w", "Aggression rate (lexical delta)"),
    ("lexical", "delta_pooled_negative_rate_100w", "Negative affect rate (lexical delta)"),
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
from src.user_week.ideology_buckets import ideology_bucket_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for shift-by-bucket violin plots."""
    parser = argparse.ArgumentParser(description="Plot user shifts by ideology bucket.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohorts",
        type=str,
        default="strict,loose",
        help="Comma-separated cohort labels.",
    )
    return parser.parse_args()


def _load_shifts(tables_dir: Path, cohort: str, slug: str) -> pd.DataFrame:
    """Function summary: load per-user shift CSV for one composite slug."""
    path = tables_dir / f"shift_per_user_{cohort}_{slug}.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_shifts(tables_dir: Path, cohort: str) -> pd.DataFrame:
    """Function summary: combine semantic and polarization shift columns per author."""
    sem = _load_shifts(tables_dir, cohort, "semantic")
    pol = _load_shifts(tables_dir, cohort, "polarization")
    if sem.empty and pol.empty:
        return pd.DataFrame()
    if sem.empty:
        return pol
    if pol.empty:
        return sem
    cols_pol = [c for c in pol.columns if c not in sem.columns or c == "author"]
    return sem.merge(pol[cols_pol], on="author", how="outer")


def plot_violin_by_bucket(
    df: pd.DataFrame,
    delta_col: str,
    bucket_col: str,
    title: str,
    out_path: Path,
    bucket_order: Sequence[str],
) -> None:
    """Function summary: violin plot of one delta column by bucket column.

    Parameters:
    - df: merged shifts + buckets.
    - delta_col: pooled delta column name.
    - bucket_col: lexical_bucket or semantic_bucket.
    - title: plot title.
    - out_path: PNG path.
    - bucket_order: x-axis order.
    """
    if delta_col not in df.columns or bucket_col not in df.columns:
        return
    work = df[df[bucket_col].isin(bucket_order)].dropna(subset=[delta_col])
    if work.empty:
        return
    data = [
        work.loc[work[bucket_col] == b, delta_col].astype(float).values for b in bucket_order
    ]
    if not any(len(d) > 0 for d in data):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot(data, positions=range(len(bucket_order)), showmeans=True, showmedians=True)
    for body in parts.get("bodies", []):
        body.set_alpha(0.7)
    ax.set_xticks(range(len(bucket_order)))
    short = [b.replace("_leaning", "").replace("_", "\n") for b in bucket_order]
    ax.set_xticklabels(short, fontsize=8)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Pooled pre→post delta")
    ax.set_title(title)
    fig.suptitle(
        "Buckets from pre-ban tertiles; deltas are post-ban outcomes (descriptive)",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cohort(config: dict, cohort: str, bucket_order: List[str]) -> None:
    """Function summary: write all shift-by-bucket figures for one cohort."""
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    buckets_path = tables_dir / f"author_ideology_buckets_{cohort}.csv"
    if not buckets_path.is_file():
        print(
            f"[plot_user_shift_by_ideology_bucket] skip cohort={cohort}: missing buckets CSV",
            flush=True,
        )
        return
    buckets = pd.read_csv(buckets_path)
    buckets["author"] = buckets["author"].astype(str)
    shifts = _merge_shifts(tables_dir, cohort)
    if shifts.empty:
        print(
            f"[plot_user_shift_by_ideology_bucket] skip cohort={cohort}: missing shift CSVs",
            flush=True,
        )
        return
    shifts["author"] = shifts["author"].astype(str)
    merged = shifts.merge(buckets, on="author", how="inner")
    fig_root = Path(config["paths"]["figures_dir"]) / "user_week" / cohort / "by_ideology_bucket"

    for family, delta_col, label in SHIFT_SPECS:
        if delta_col not in merged.columns:
            continue
        for group_col in ("lexical_bucket", "semantic_bucket"):
            subdir = fig_root / f"{family}_shift_by_{group_col}"
            fname = f"{delta_col}_by_{group_col}.png"
            title = f"{label} — {cohort} — grouped by {group_col}"
            plot_violin_by_bucket(merged, delta_col, group_col, title, subdir / fname, bucket_order)

    readme = (
        "# Shifts by ideology bucket\n\n"
        "Bucket labels use **pre-ban** asymmetric rules (lexical: no L/R hits → neutral; "
        "semantic: tail-week p25/p75 on sem_axis_ideology). "
        "Outcomes include semantic axes (ideology, emotion, aggression) and lexical rates "
        "(net_ideology, pole_share, aggression_rate, negative_rate). "
        "Deltas are **within-person** post minus pre; not cross-country DiD.\n"
    )
    fig_root.mkdir(parents=True, exist_ok=True)
    (fig_root / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"[plot_user_shift_by_ideology_bucket] cohort={cohort} users={len(merged)} out={fig_root}",
        flush=True,
    )


def main() -> None:
    """Function summary: plot shift violins for each requested cohort."""
    args = parse_args()
    config = load_config(args.config)
    bucket_cfg = ideology_bucket_config(config)
    bucket_order = list(bucket_cfg.bucket_labels)
    for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        run_cohort(config, cohort, bucket_order)
    print("[plot_user_shift_by_ideology_bucket] done", flush=True)


if __name__ == "__main__":
    main()
