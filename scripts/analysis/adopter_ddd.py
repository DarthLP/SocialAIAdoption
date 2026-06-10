"""
Script summary:
Estimate adopter-flag triple-differences on the comment panel (author + topic_family×date FE).

Functionality:
- Merges adopter_flags.csv onto prepared comment panel; runs static DDD per scheme and post spec.
- Scheme-2 reversion placebo (7c): uses scheme2_firsthalf flag instead of scheme2_styletop.
- Writes did/adopter_ddd/static_{scheme}_{spec}_{outcome}.csv with cell counts.

How to apply/run:
  .venv/bin/python scripts/analysis/prepare_adopter_flags.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_did_comment_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/adopter_ddd.py --config config/italy_polarization_setup.yaml --max-rows 50000
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCHEMES = {
    "scheme1_inactive": "scheme1_inactive",
    "scheme2_styletop": "scheme2_styletop",
    "scheme2_reversion_placebo": "scheme2_firsthalf",
    "scheme3_mention": "scheme3_mention",
    "scheme3_tech": "scheme3_tech",
}
POST_SPECS = ("full_ban", "post_first_2bd", "post_short_3d")
DEFAULT_OUTCOMES = ("style_index_llm", "ai_style_rate", "net_ideology")


def _setup_project_root() -> Path:
    """Function summary: resolve repo root."""
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
from src.did.bucket_estimate import estimate_adopter_ddd_static  # noqa: E402
from src.did.outcomes import OUTCOME_REGISTRY, outcome_spec  # noqa: E402
from src.did.panels import comment_panel_available, load_comment_panel  # noqa: E402
from src.did.paths import did_adopter_ddd_dir  # noqa: E402
from src.did.specs import apply_post_window, rel_day_from_date  # noqa: E402
from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Adopter-flag triple-diff on comment panel.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES))
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    return p.parse_args()


def _annotate_calendar(df: pd.DataFrame, launch: str) -> pd.DataFrame:
    """Function summary: add rel_day, post, IT for DDD sample."""
    out = df.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    out["rel_day"] = rel_day_from_date(out["date_utc"], launch)
    out["post"] = (out["date_utc"] >= launch).astype(int)
    fam = out["topic_family"].astype(str)
    out["IT"] = fam.isin({"it_political", "it_others"}).astype(int)
    return out


def _apply_spec(df: pd.DataFrame, spec: str, launch: str) -> pd.DataFrame:
    """Function summary: set post indicator for one POST_PHASE spec."""
    work = _annotate_calendar(df, launch)
    if spec == "full_ban":
        return work
    return apply_post_window(work, spec, launch)


def _outcome_col(outcome_id: str) -> str:
    """Function summary: comment-panel column for outcome_id (not subreddit-day means)."""
    for spec in OUTCOME_REGISTRY:
        if spec.outcome_id == outcome_id and spec.panel_kind == "comment":
            return spec.column
    spec = outcome_spec(outcome_id)
    return spec.column if spec is not None else outcome_id


def run_adopter_ddd(
    config: Dict[str, Any],
    *,
    outcome_ids: List[str],
    max_rows: Optional[int] = None,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """Function summary: estimate DDD rows for all scheme×spec×outcome combinations."""
    if not comment_panel_available(config, bin_days=1):
        raise FileNotFoundError("Comment panel missing; run prepare_did_comment_panel.py")
    flags_path = tables_subdir(config, "did") / "adopter_flags.csv"
    if not flags_path.is_file():
        raise FileNotFoundError(f"Missing {flags_path}; run prepare_adopter_flags.py")
    flags = pd.read_csv(flags_path)
    _, _, launch, _ = event_dates_from_config(config)
    panel = load_comment_panel(config, bin_days=1, sample_frac=sample_frac, max_rows=max_rows)
    panel = panel.merge(flags, on="author", how="left", suffixes=("", "_flag"))
    for col in SCHEMES.values():
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)

    rows: List[Dict[str, Any]] = []
    for scheme_id, flag_col in SCHEMES.items():
        if flag_col not in panel.columns:
            continue
        for spec in POST_SPECS:
            work = _apply_spec(panel, spec, launch)
            work["flag"] = work[flag_col].astype(float)
            for oid in outcome_ids:
                ycol = _outcome_col(oid)
                if ycol not in work.columns:
                    rows.append(
                        {
                            "scheme": scheme_id,
                            "spec": spec,
                            "outcome_id": oid,
                            "status": "missing_outcome_column",
                        }
                    )
                    continue
                est = estimate_adopter_ddd_static(work, y_col=ycol, flag_col="flag")
                row = {
                    "scheme": scheme_id,
                    "flag_col": flag_col,
                    "spec": spec,
                    "outcome_id": oid,
                    "column": ycol,
                    "coef_name": "post_IT_flag",
                    **est,
                }
                if est.get("estimation_note") == "ok" and est.get("pvalue") is not None:
                    try:
                        row["significant_5pct"] = int(float(est["pvalue"]) < 0.05)
                    except (TypeError, ValueError):
                        row["significant_5pct"] = 0
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: write adopter DDD CSVs and print scheme-2 placebo headline."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    oids = [x.strip() for x in args.outcomes.split(",") if x.strip()]
    df = run_adopter_ddd(
        config,
        outcome_ids=oids,
        max_rows=args.max_rows,
        sample_frac=args.sample_frac,
    )
    out_dir = did_adopter_ddd_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "adopter_ddd_summary.csv"
    df.to_csv(summary, index=False)
    for (scheme, spec), grp in df.groupby(["scheme", "spec"], observed=True):
        path = out_dir / f"static_{scheme}_{spec}.csv"
        grp.to_csv(path, index=False)
    print(f"[adopter_ddd] wrote {summary} ({len(df)} rows)", flush=True)
    placebo = df[
        (df["scheme"] == "scheme2_reversion_placebo")
        & (df["outcome_id"] == "style_index_llm")
        & (df.get("estimation_note", pd.Series(dtype=str)) == "ok")
    ]
    if not placebo.empty:
        r = placebo.iloc[0]
        p = r.get("pvalue", float("nan"))
        print(
            f"[adopter_ddd] STOP 7c — scheme2 reversion placebo style_index_llm "
            f"beta={r.get('beta', float('nan')):.4g} p={p:.4g} "
            f"(fails if p<0.05 under reversion null)",
            flush=True,
        )


if __name__ == "__main__":
    main()
