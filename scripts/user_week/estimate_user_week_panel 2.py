"""
Script summary:
Author×week panel regressions on the Italy user-week panel (entity FE; clustered by author).

Functionality:
- Loads user_week_panel.parquet, applies the same strict/loose cohort gates as analyze_user_pre_post_shift.
- Estimates y ~ post | author FE for headline lexical and semantic outcomes.
- Writes event-study coefficients (rel_week dummies, ref week -1) for headline ideology outcomes.

How to apply/run:
  .venv/bin/python scripts/user_week/estimate_user_week_panel.py \\
    --config config/italy_polarization_setup.yaml --cohort both
  .venv/bin/python scripts/user_week/estimate_user_week_panel.py \\
    --config config/italy_polarization_setup.yaml --cohort strict --outcomes headline
"""

from __future__ import annotations

import argparse
import importlib.util
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

from src.config_utils import load_config, user_week_drop_ban_week_default  # noqa: E402
from src.user_week.cohorts import default_cohort_thresholds  # noqa: E402
from src.user_week.estimate import (  # noqa: E402
    estimate_user_week_entity_only,
    estimate_user_week_event_study,
)
from src.user_week.panel_prep import (  # noqa: E402
    feature_track,
    launch_iso_week_from_config,
    outcome_panel_column,
    prepare_regression_sample,
    resolve_outcome_list,
)

EVENT_STUDY_FEATURES = ("net_ideology", "sem_axis_ideology")


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for user-week panel regression exports."""
    parser = argparse.ArgumentParser(description="Estimate author×week panel regressions (Italy).")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default="both",
        choices=["both", "strict", "loose"],
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default="headline",
        choices=["headline", "all"],
        help="Outcome set: headline six features or all config default_features.",
    )
    return parser.parse_args()


def run_cohort(config: Dict[str, Any], cohort_label: str, outcomes_mode: str) -> None:
    """Function summary: regression tables for one cohort label.

    Parameters:
    - config: study YAML.
    - cohort_label: strict or loose.
    - outcomes_mode: headline or all.
    """
    tables_dir = Path(config["paths"]["tables_dir"]) / "user_week"
    panel_path = tables_dir / "user_week_panel.parquet"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Missing {panel_path}; run prepare_user_week_style_panel.py first.")

    thresholds = next(t for t in default_cohort_thresholds() if t.label == cohort_label)
    launch_iso = launch_iso_week_from_config(config)
    drop_ban = user_week_drop_ban_week_default(config)
    panel = pd.read_parquet(panel_path)
    panel["author"] = panel["author"].astype(str)
    sample = prepare_regression_sample(
        panel,
        thresholds,
        launch_iso,
        drop_ban,
    )
    outcomes = resolve_outcome_list(outcomes_mode, config)

    reg_rows: List[Dict[str, Any]] = []
    for feat in outcomes:
        y_col = outcome_panel_column(feat, sample.columns)
        if y_col not in sample.columns:
            reg_rows.append(
                {
                    "cohort": cohort_label,
                    "feature": feat,
                    "track": feature_track(feat),
                    "y_col": y_col,
                    "beta_post": float("nan"),
                    "se_post": float("nan"),
                    "n_obs": 0,
                    "n_authors": 0,
                    "estimation_note": "missing_outcome",
                }
            )
            continue
        res = estimate_user_week_entity_only(sample, y_col)
        reg_rows.append(
            {
                "cohort": cohort_label,
                "feature": feat,
                "track": feature_track(feat),
                "y_col": y_col,
                "beta_post": res["beta"],
                "se_post": res["se"],
                "n_obs": res["n_obs"],
                "n_authors": res["n_clusters"],
                "estimation_note": res["estimation_note"],
            }
        )

    reg_path = tables_dir / f"regression_summary_{cohort_label}.csv"
    pd.DataFrame(reg_rows).to_csv(reg_path, index=False)

    for feat in EVENT_STUDY_FEATURES:
        if feat not in outcomes:
            continue
        y_col = outcome_panel_column(feat, sample.columns)
        if y_col not in sample.columns:
            continue
        es_rows = estimate_user_week_event_study(sample, y_col)
        if not es_rows:
            continue
        es_df = pd.DataFrame(es_rows)
        es_df.insert(0, "cohort", cohort_label)
        es_df.insert(1, "feature", feat)
        es_df.insert(2, "y_col", y_col)
        es_path = tables_dir / f"event_study_{cohort_label}_{feat}.csv"
        es_df.to_csv(es_path, index=False)

    print(
        f"[estimate_user_week_panel] cohort={cohort_label} n_obs={len(sample)} "
        f"n_authors={sample['author'].nunique()} wrote {reg_path.name}",
        flush=True,
    )


def main() -> None:
    """Function summary: run regression exports for requested cohort(s)."""
    args = parse_args()
    config = load_config(args.config)
    cohorts = ["strict", "loose"] if args.cohort == "both" else [args.cohort]
    for label in cohorts:
        run_cohort(config, label, args.outcomes)
    print("[estimate_user_week_panel] done", flush=True)


if __name__ == "__main__":
    main()
