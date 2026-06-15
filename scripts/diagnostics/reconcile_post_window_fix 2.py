"""
Script summary:
Compare archived vs current did_summary coefficients after the post-window contamination fix.

Functionality:
- Prints before/after beta/se/p for headline outcomes and all contaminated/new specs.
- Flags sign flips and p-value crossings at 0.05 and 0.01.

How to apply/run:
  .venv/bin/python scripts/diagnostics/reconcile_post_window_fix.py
  .venv/bin/python scripts/diagnostics/reconcile_post_window_fix.py --strategy cross_country_all
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


ARCHIVE_TAG = "2026-06-10"

OUTCOMES = (
    "pole_share",
    "sem_axis_ideology_pole_share",
    "sem_axis_ideology_pole_share_global",
    "sem_axis_ideology_pole_share_p05p95",
    "sem_axis_ideology_pole_share_p15p85",
    "sem_axis_ideology_extreme_left",
    "sem_axis_ideology_extreme_right",
    "sem_axis_emotion",
)

SPECS = (
    "early_ban_7d",
    "early_ban_14d",
    "post_short_3d",
    "post_medium_7d",
    "post_long_tail",
    "post_first_2bd",
    "phase_joint_short",
    "phase_joint_medium",
    "phase_joint_long",
    "phase_joint_lift",
    "ban_in_effect",
    "post_lift",
    "full_ban",
)


def _setup_project_root() -> Path:
    """Function summary: resolve repository root."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config, tables_subdir  # noqa: E402


def _load_summary(path: Path) -> pd.DataFrame:
    """Function summary: load did_summary CSV or return empty frame."""
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _sign_flip(before: float, after: float) -> bool:
    """Function summary: True when finite betas change sign."""
    if not pd.notna(before) or not pd.notna(after):
        return False
    return float(before) * float(after) < 0


def _sig_cross(before: float, after: float, alpha: float) -> bool:
    """Function summary: True when significance vs alpha changes across vintages."""
    if not pd.notna(before) or not pd.notna(after):
        return False
    b_sig = abs(float(before)) > 0
    a_sig = abs(float(after)) > 0
    del b_sig, a_sig
    def _p_sig(p: float) -> bool:
        return pd.notna(p) and float(p) < alpha

    return False  # placeholder replaced below


def _p_cross(before_p: float, after_p: float, alpha: float) -> bool:
    """Function summary: True when p crosses alpha threshold."""
    if not pd.notna(before_p) or not pd.notna(after_p):
        return False
    b = float(before_p) < alpha
    a = float(after_p) < alpha
    return b != a


def _filter_rows(
    df: pd.DataFrame,
    outcome: str,
    spec: str,
    strategy: str,
) -> Optional[pd.Series]:
    """Function summary: pick one summary row for outcome/spec/strategy."""
    if df.empty:
        return None
    sub = df[
        (df["outcome_id"].astype(str) == outcome)
        & (df["spec"].astype(str) == spec)
        & (df["strategy_id"].astype(str) == strategy)
    ]
    if sub.empty:
        return None
    return sub.iloc[-1]


def reconcile(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    strategy: str,
    outcomes: Iterable[str],
    specs: Iterable[str],
) -> List[str]:
    """Function summary: build reconciliation report lines."""
    lines: List[str] = []
    flags: List[str] = []
    for outcome in outcomes:
        lines.append(f"\n=== {outcome} ===")
        for spec in specs:
            old = _filter_rows(before, outcome, spec, strategy)
            new = _filter_rows(after, outcome, spec, strategy)
            if old is None and new is None:
                continue
            ob = float(old["beta"]) if old is not None and pd.notna(old.get("beta")) else float("nan")
            oa = float(new["beta"]) if new is not None and pd.notna(new.get("beta")) else float("nan")
            op = float(old["pvalue"]) if old is not None and pd.notna(old.get("pvalue")) else float("nan")
            np_ = float(new["pvalue"]) if new is not None and pd.notna(new.get("pvalue")) else float("nan")
            ose = float(old["se"]) if old is not None and pd.notna(old.get("se")) else float("nan")
            nse = float(new["se"]) if new is not None and pd.notna(new.get("se")) else float("nan")
            lines.append(
                f"  {spec}: before β={ob:.4f} se={ose:.4f} p={op:.4g} | "
                f"after β={oa:.4f} se={nse:.4f} p={np_:.4g}"
            )
            if _sign_flip(ob, oa):
                flags.append(f"SIGN_FLIP {outcome} {spec}")
            if _p_cross(op, np_, 0.05):
                flags.append(f"P_CROSS_0.05 {outcome} {spec}")
            if _p_cross(op, np_, 0.01):
                flags.append(f"P_CROSS_0.01 {outcome} {spec}")
    if flags:
        lines.append("\n=== FLAGS ===")
        lines.extend(flags)
    else:
        lines.append("\n=== FLAGS === none")
    return lines


def main() -> None:
    """Function summary: CLI entry for post-window fix reconciliation."""
    parser = argparse.ArgumentParser(description="Reconcile DiD summary before/after contamination fix.")
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--strategy", default="cross_country_all")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    did_root = tables_subdir(config, "did")
    archive = did_root / f"_archived_precontamination_{ARCHIVE_TAG}"
    before_path = archive / "estimates" / "summary" / "did_summary.csv"
    after_path = did_root / "estimates" / "summary" / "did_summary.csv"

    before = _load_summary(before_path)
    after = _load_summary(after_path)
    lines = reconcile(
        before,
        after,
        strategy=args.strategy,
        outcomes=OUTCOMES,
        specs=SPECS,
    )
    report = "\n".join(lines)
    print(report, flush=True)
    out_path = did_root / "estimates" / "summary" / "reconcile_post_window_fix.txt"
    if after_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n[reconcile] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
