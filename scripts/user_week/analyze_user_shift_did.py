"""
Script summary:
Estimate Italy vs control difference in within-person AI-style composite shifts.

Functionality:
- Reads `shift_per_user_strict_style.csv` from analyze_user_pre_post_shift.
- Treatment: author modal topic is Italian (`it_political` or `it_others`) from top_topic_pre/post.
- Outcome: `delta_weekly_ai_style_composite_user_week` (revised composite from config).
- Estimator: WLS `delta ~ IT` with weights = pre_words_total_good + post_words_total_good.
- Writes shift_did_it_vs_control.csv with OLS/WLS coefficients and HC1 robust SE.

How to apply/run:
  .venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/user_week/analyze_user_shift_did.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


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

from src.config_utils import load_config, tables_subdir, user_week_composites  # noqa: E402

ITALY_TOPICS = frozenset({"it_political", "it_others"})
COMPOSITE_DELTA_COL = "delta_weekly_ai_style_composite_user_week"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments.

    Returns:
    - Parsed namespace with --config and --cohort.
    """
    parser = argparse.ArgumentParser(
        description="IT vs control difference in within-person AI-style composite shifts."
    )
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--cohort", default="strict", choices=["strict", "loose"])
    return parser.parse_args()


def _author_it_flag(row: pd.Series) -> int:
    """Function summary: IT=1 if pre or post modal topic is an Italian family.

    Parameters:
    - row: shift_per_user row with top_topic_pre and top_topic_post.

    Returns:
    - 1 if Italian topic family, else 0.
    """
    pre = str(row.get("top_topic_pre", ""))
    post = str(row.get("top_topic_post", ""))
    return int(pre in ITALY_TOPICS or post in ITALY_TOPICS)


def _fit_shift_did(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Function summary: WLS and OLS of composite delta on IT indicator.

    Parameters:
    - df: per-user rows with delta, IT, and word weights.

    Returns:
    - Tuple of (coefficient summary DataFrame, metadata dict).
    """
    work = df.dropna(subset=[COMPOSITE_DELTA_COL, "IT", "weight"]).copy()
    work = work[work["weight"] > 0]
    if work.empty or work["IT"].nunique() < 2:
        return pd.DataFrame(), {"estimation_note": "degenerate", "n_users": len(work)}

    y = work[COMPOSITE_DELTA_COL].astype(float)
    x = sm.add_constant(work["IT"].astype(float))
    w = work["weight"].astype(float)

    ols = sm.OLS(y, x).fit(cov_type="HC1")
    wls = sm.WLS(y, x, weights=w).fit(cov_type="HC1")

    rows = []
    for model_name, res in [("ols", ols), ("wls", wls)]:
        beta = float(res.params.get("IT", np.nan))
        se = float(res.bse.get("IT", np.nan))
        rows.append(
            {
                "model": model_name,
                "term": "IT",
                "beta": beta,
                "se": se,
                "t": float(res.tvalues.get("IT", np.nan)),
                "p": float(res.pvalues.get("IT", np.nan)),
                "ci95_low": beta - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": beta + 1.96 * se if np.isfinite(se) else np.nan,
                "n_users": int(len(work)),
                "n_it": int(work["IT"].sum()),
                "n_control": int((1 - work["IT"]).sum()),
                "mean_delta_it": float(work.loc[work["IT"] == 1, COMPOSITE_DELTA_COL].mean()),
                "mean_delta_control": float(work.loc[work["IT"] == 0, COMPOSITE_DELTA_COL].mean()),
                "estimation_note": "ok",
            }
        )
    meta = {
        "estimation_note": "ok",
        "n_users": int(len(work)),
        "outcome": COMPOSITE_DELTA_COL,
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    """Function summary: load shift CSV, estimate IT vs control delta contrast, write summary."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    composites = user_week_composites(config)
    style_block = next(
        (c for c in composites if c.get("name") == "ai_style_composite_user_week"),
        composites[1] if len(composites) > 1 else {"name": "ai_style_composite_user_week"},
    )
    style_name = str(style_block.get("name", "ai_style_composite_user_week"))
    if style_name != "ai_style_composite_user_week":
        print(
            f"[analyze_user_shift_did] note: style composite name={style_name} "
            f"(delta col still {COMPOSITE_DELTA_COL})",
            flush=True,
        )

    tables_dir = tables_subdir(config, "user_week")
    in_path = tables_dir / f"shift_per_user_{args.cohort}_style.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"Missing {in_path}; run analyze_user_pre_post_shift.py first.")

    df = pd.read_csv(in_path)
    if COMPOSITE_DELTA_COL not in df.columns:
        raise ValueError(f"{in_path} lacks {COMPOSITE_DELTA_COL}")

    df = df[df["audit_category"].astype(str) == "panel"].copy()
    df["IT"] = df.apply(_author_it_flag, axis=1)
    df["weight"] = (
        pd.to_numeric(df.get("pre_words_total_good"), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("post_words_total_good"), errors="coerce").fillna(0.0)
    )

    summary, meta = _fit_shift_did(df)
    out_path = tables_dir / "shift_did_it_vs_control.csv"
    if summary.empty:
        pd.DataFrame([{"estimation_note": meta.get("estimation_note", "degenerate")}]).to_csv(
            out_path, index=False
        )
        print(f"[analyze_user_shift_did] degenerate n={meta.get('n_users', 0)} wrote {out_path}", flush=True)
        return

    summary["cohort"] = args.cohort
    summary["composite"] = style_name
    summary.to_csv(out_path, index=False)
    print(f"[analyze_user_shift_did] wrote {out_path}", flush=True)
    wls_row = summary[summary["model"] == "wls"].iloc[0]
    print(
        f"[analyze_user_shift_did] WLS IT beta={wls_row['beta']:.4f} "
        f"se={wls_row['se']:.4f} p={wls_row['p']:.4f} "
        f"n_it={int(wls_row['n_it'])} n_control={int(wls_row['n_control'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
