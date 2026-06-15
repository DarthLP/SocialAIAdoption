"""
Script summary:
Diagnose the language/subreddit/3d event-study panel for the metadata/collinearity bug.

Functionality:
- Weights: n_comments sum/min/NaN/zero/clip-floor counts at 1d vs 3d.
- Metadata: asserts topic_family (and IT/is_control) survive 3d outcome binning.
- Post-filter treat mix for cross_country_all: must contain IT=0 and IT=1 rows.
- Design rank: mono-treat time_id count and TWFE design condition number.
- Reference bin: rel_period -1 n_comments mass.
- Live estimate_event_study on sem_axis_ideology_extreme_left/right + pole_share
  with |gamma| gates (0.12 for bounded tail shares).
- Same guards on the subreddit×universe_slice 3d panel (language_universe/in_out_slice):
  metadata carried through binning, both arms present, zero mono-treat time bins
  for each in/out strategy.
- Exits 1 when any gate fails; use before regenerating language/subreddit/3d or
  language_universe/in_out_slice/3d CSVs.

How to apply/run:
  .venv/bin/python scripts/diagnostics/diagnose_subreddit_3d_panel.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
from src.did.estimate import estimate_event_study  # noqa: E402
from src.did.outcomes import (  # noqa: E402
    SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL,
    SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL,
)
from src.did.panels import (  # noqa: E402
    load_subreddit_event_study_panel,
    load_subreddit_slice_event_study_panel,
)
from src.did.specs import (  # noqa: E402
    EVENT_WINDOW_DAYS_BY_BIN,
    StrategySpec,
    event_study_language_universe_slice_strategies,
    filter_strategy_sample,
)

TAIL_SHARE_GATE = 0.12
META_REQUIRED = ("topic_family", "IT")
META_OPTIONAL = ("primary_lexicon", "is_control")
WEIGHT_CLIP_FLOOR = 1e-9


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for the subreddit/3d panel diagnostic."""
    parser = argparse.ArgumentParser(description="Diagnose language/subreddit 3d ES panel.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _weights_report(panel: pd.DataFrame, label: str) -> None:
    """Function summary: print n_comments sum/min/NaN/zero/clip-floor for one panel."""
    w = pd.to_numeric(panel.get("n_comments"), errors="coerce")
    print(
        f"  [{label}] n_comments: sum={w.sum():,.0f} min={w.min():.3g} "
        f"NaN={int(w.isna().sum())} zero={int((w == 0).sum())} "
        f"at_clip_floor={int((w.abs() <= WEIGHT_CLIP_FLOOR).sum())}"
    )


def _design_condition_number(sample: pd.DataFrame, rel_col: str) -> float:
    """Function summary: cond number of [treat×rel dummies, entity, time] design."""
    try:
        work = sample.copy()
        work["treat"] = work["treat"].astype(float)
        parts: List[np.ndarray] = []
        for k in sorted(work[rel_col].unique()):
            if int(k) == -1:
                continue
            parts.append(((work[rel_col] == k) * work["treat"]).astype(float).to_numpy().reshape(-1, 1))
        ent = pd.get_dummies(work["subreddit"].astype(str), drop_first=True)
        tim = pd.get_dummies(work["time_id"].astype(str), drop_first=True)
        x = np.column_stack(parts + [ent.to_numpy(dtype=float), tim.to_numpy(dtype=float)])
        return float(np.linalg.cond(x))
    except Exception:
        return float("inf")


def _gate(label: str, ok: bool, detail: str, failures: List[str]) -> None:
    """Function summary: print one PASS/FAIL line and record failures."""
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def run(config: Dict[str, Any]) -> int:
    """Function summary: run all checks; return count of failed gates."""
    failures: List[str] = []

    print("== 1. Weights (n_comments) ==")
    p1 = load_subreddit_event_study_panel(config, 1)
    p3 = load_subreddit_event_study_panel(config, 3)
    _weights_report(p1, "subreddit/1d")
    _weights_report(p3, "subreddit/3d")

    print("== 2. Metadata on 3d panel ==")
    for col in META_REQUIRED:
        _gate(f"3d has {col}", col in p3.columns, f"present={col in p3.columns}", failures)
    for col in META_OPTIONAL:
        note = "present" if col in p3.columns else "absent (ok if absent on 1d too)"
        print(f"  [info] 3d {col}: {note} (1d: {'present' if col in p1.columns else 'absent'})")

    print("== 3. Post-filter treat mix (cross_country_all, 3d) ==")
    window = EVENT_WINDOW_DAYS_BY_BIN[3]
    strat = StrategySpec("cross_country_all")
    sample = filter_strategy_sample(p3, strat, window_days=window)
    it_counts = sample["IT"].astype(float).round().astype(int).value_counts().to_dict()
    n_it0, n_it1 = int(it_counts.get(0, 0)), int(it_counts.get(1, 0))
    _gate(
        "both arms present",
        n_it0 > 0 and n_it1 > 0,
        f"rows IT=0: {n_it0}, IT=1: {n_it1}",
        failures,
    )

    print("== 4. Design rank ==")
    mono = sample.groupby("time_id")["treat"].nunique()
    n_mono = int((mono <= 1).sum())
    _gate("mono-treat time_ids", n_mono == 0, f"{n_mono}/{len(mono)} time bins mono-treat", failures)
    singleton = sample.groupby("subreddit").size()
    print(f"  [info] singleton-bin entities: {int((singleton <= 1).sum())}")
    cond = _design_condition_number(sample, "rel_period")
    _gate("design cond number", cond < 1e8, f"cond={cond:.3g}", failures)

    print("== 5. Reference bin mass ==")
    ref = sample[sample["rel_period"].astype(int) == -1]
    ref_mass = float(pd.to_numeric(ref["n_comments"], errors="coerce").sum())
    _gate("rel_period -1 n_comments mass", ref_mass > 0, f"{ref_mass:,.0f}", failures)

    print("== 6. Live event studies (3d, cross_country_all) ==")
    checks: Tuple[Tuple[str, float], ...] = (
        (SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL, TAIL_SHARE_GATE),
        (SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL, TAIL_SHARE_GATE),
        ("pole_share", 1.0),
    )
    for y_col, gate in checks:
        if y_col not in sample.columns:
            _gate(f"ES {y_col}", False, "column missing on 3d panel", failures)
            continue
        summary, es_df = estimate_event_study(
            sample,
            y_col,
            rel_col="rel_period",
            window=window,
            entity_col="subreddit",
            time_col="time_id",
            bin_days=3,
        )
        if es_df.empty:
            _gate(
                f"ES {y_col}",
                False,
                f"empty ES (note={summary.get('estimation_note')})",
                failures,
            )
            continue
        gmax = float(es_df["gamma"].abs().max())
        _gate(f"ES {y_col} |gamma| <= {gate}", gmax <= gate, f"max|gamma|={gmax:.4f}", failures)

    print("== 7. Slice panel (language_universe/in_out_slice, 3d) ==")
    try:
        sl3 = load_subreddit_slice_event_study_panel(config, 3)
    except FileNotFoundError as exc:
        print(f"  [info] slice panel unavailable, skipping: {exc}")
        sl3 = pd.DataFrame()
    if not sl3.empty:
        _gate(
            "slice 3d has topic_family",
            "topic_family" in sl3.columns,
            f"present={'topic_family' in sl3.columns}",
            failures,
        )
        for strat in event_study_language_universe_slice_strategies():
            sl_sample = filter_strategy_sample(sl3, strat, window_days=window)
            sl_it = (
                sl_sample["IT"].astype(float).round().astype(int).value_counts().to_dict()
            )
            s0, s1 = int(sl_it.get(0, 0)), int(sl_it.get(1, 0))
            _gate(
                f"{strat.strategy_id} both arms",
                s0 > 0 and s1 > 0,
                f"rows IT=0: {s0}, IT=1: {s1}",
                failures,
            )
            sl_mono = sl_sample.groupby("time_id")["treat"].nunique()
            n_sl_mono = int((sl_mono <= 1).sum())
            _gate(
                f"{strat.strategy_id} mono-treat time_ids",
                n_sl_mono == 0,
                f"{n_sl_mono}/{len(sl_mono)} time bins mono-treat",
                failures,
            )

    print("== Verdict ==")
    if failures:
        print(f"FAILED {len(failures)} gate(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print(
            "ALL GATES PASSED — safe to regenerate language/subreddit/3d and "
            "language_universe/in_out_slice/3d CSVs."
        )
    return len(failures)


def main() -> None:
    """Function summary: CLI entry; exit 1 when any gate fails."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    n_failed = run(config)
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
