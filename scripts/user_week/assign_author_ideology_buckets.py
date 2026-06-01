"""
Script summary:
Assign pre-ban liberal / neutral / conservative-leaning buckets per author using lexical
(net_ideology) and semantic (sem_axis_ideology) tertiles within primary_lexicon.

How to apply/run:
  .venv/bin/python scripts/user_week/assign_author_ideology_buckets.py \\
    --config config/italy_polarization_setup.yaml --cohort strict
  .venv/bin/python scripts/user_week/assign_author_ideology_buckets.py \\
    --config config/italy_polarization_setup.yaml --cohort both
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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

from scripts.diagnostics.prepare_did_author_semantic_week_panel import load_assignment  # noqa: E402
from scripts.user_week.analyze_user_pre_post_shift import launch_iso_week_str  # noqa: E402
from src.config_utils import load_config, user_week_drop_ban_week_default  # noqa: E402
from src.user_week.ideology_buckets import (  # noqa: E402
    build_author_ideology_buckets,
    ideology_bucket_config,
    load_cohort_authors_from_shift,
    load_semantic_orientation_multipliers,
    label_pre_post_weeks,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for author ideology bucket assignment."""
    parser = argparse.ArgumentParser(description="Assign lexical and semantic ideology buckets per author.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default="both",
        choices=["strict", "loose", "both"],
        help="Cohort universe from shift_per_user exports.",
    )
    return parser.parse_args()


def launch_from_config(config: Dict[str, Any]) -> str:
    """Function summary: ISO Monday string for ban anchor from event_window."""
    raw = str(config["event_window"]["launch_day_utc"])
    launch_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if launch_dt.tzinfo is None:
        launch_dt = launch_dt.replace(tzinfo=timezone.utc)
    return launch_iso_week_str(launch_dt)


def write_methods_note(
    path: Path,
    bucket_cfg: Any,
    cohort: str,
    n_authors: int,
    multipliers: Dict[str, float],
    orientation_path: Path,
) -> None:
    """Function summary: document bucket rules for reproducibility."""
    lines = [
        "Author ideology bucket methods",
        "==============================",
        "",
        f"Cohort: {cohort}",
        f"Authors classified: {n_authors}",
        f"Method: {bucket_cfg.method}",
        f"min_pre_words: {bucket_cfg.min_pre_words}",
        f"min_pre_weeks: {bucket_cfg.min_pre_weeks}",
        f"Bucket labels (low/mid/high): {bucket_cfg.bucket_labels}",
        "",
        "Lexical score: word-weighted pre-ban mean of net_ideology_mean.",
        "Semantic score: word-weighted pre-ban mean of sem_axis_ideology_mean",
        "  (multiplied by per-lexicon orientation from ideology_axis_orientation_report).",
        "",
        "Orientation multipliers applied:",
    ]
    for lang, mult in sorted(multipliers.items()):
        lines.append(f"  {lang}: {mult:+.0f}")
    lines.extend(
        [
            f"Orientation report: {orientation_path}",
            "",
            "Tertiles computed separately within assigned_primary_lexicon (it, en, de).",
            "Do not compare raw score levels across languages.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cohort(config: Dict[str, Any], cohort: str) -> None:
    """Function summary: build and write author_ideology_buckets for one cohort label."""
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    panel_path = tables_dir / "user_week_panel.parquet"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Missing {panel_path}; run prepare_user_week_style_panel.py first.")

    bucket_cfg = ideology_bucket_config(config)
    cohort_authors = load_cohort_authors_from_shift(tables_dir, cohort)
    if not cohort_authors:
        raise FileNotFoundError(
            f"No shift_per_user_{cohort}_polarization.csv under {tables_dir}; "
            "run analyze_user_pre_post_shift.py first."
        )

    panel = pd.read_parquet(panel_path)
    panel["author"] = panel["author"].astype(str)
    launch_iso = launch_from_config(config)
    labelled = label_pre_post_weeks(
        panel,
        launch_iso,
        drop_ban_week=user_week_drop_ban_week_default(config),
    )
    assignment = load_assignment(config)
    multipliers = load_semantic_orientation_multipliers(config)

    out = build_author_ideology_buckets(
        labelled,
        assignment,
        bucket_cfg,
        cohort_authors,
        semantic_multipliers=multipliers,
    )
    out_path = tables_dir / f"author_ideology_buckets_{cohort}.csv"
    out.to_csv(out_path, index=False)
    orient_path = Path(config["paths"]["tables_dir"]) / "semantic_axis" / "ideology_axis_orientation_report.csv"
    write_methods_note(
        tables_dir / f"author_ideology_buckets_methods_{cohort}.txt",
        bucket_cfg,
        cohort,
        len(out),
        multipliers,
        orient_path,
    )
    n_agree = int(out["buckets_agree"].sum()) if "buckets_agree" in out.columns else 0
    n_class = int(
        ((out["lexical_bucket"] != "unclassified") & (out["semantic_bucket"] != "unclassified")).sum()
    )
    print(
        f"[assign_author_ideology_buckets] cohort={cohort} rows={len(out)} classified={n_class} "
        f"exact_agree={n_agree} path={out_path}",
        flush=True,
    )


def main() -> None:
    """Function summary: assign buckets for requested cohort(s)."""
    args = parse_args()
    config = load_config(args.config)
    cohorts: List[str] = ["strict", "loose"] if args.cohort == "both" else [args.cohort]
    for cohort in cohorts:
        run_cohort(config, cohort)
    print("[assign_author_ideology_buckets] done", flush=True)


if __name__ == "__main__":
    main()
