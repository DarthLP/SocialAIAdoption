"""
Script summary:
Patch bucket static CSV exports without a full bucket_event_study rerun.

Functionality:
- Adds placebo_note to single-country control variants on existing static_*.csv files.
- Re-combines split_sample paper_eq1 rows (per-split + split_id=-1 combined) when splits exist.
- Optionally re-runs stacked DDD only via bucket_event_study --ddd-only.

How to apply/run:
  .venv/bin/python scripts/diagnostics/patch_bucket_static.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/patch_bucket_static.py --config config/italy_polarization_setup.yaml --rerun-ddd
  .venv/bin/python scripts/diagnostics/patch_bucket_static.py --bin-days 3 --static-only
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from dataclasses import replace  # noqa: E402

from src.config_utils import load_config  # noqa: E402
from src.did.bucket_estimate import combine_split_sample_static  # noqa: E402
from src.did.lean_buckets import bucket_event_study_config, is_placebo_space_eligible_control_variant  # noqa: E402
from src.did.paths import did_bucket_event_study_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for bucket static CSV patch."""
    p = argparse.ArgumentParser(description="Patch bucket static CSV exports.")
    p.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    p.add_argument("--bin-days", type=int, choices=(1, 3), default=None)
    p.add_argument(
        "--static-only",
        action="store_true",
        help="Patch static CSVs only (skip DDD rerun).",
    )
    p.add_argument(
        "--rerun-ddd",
        action="store_true",
        help="After static patch, run bucket_event_study.py --ddd-only --no-figures.",
    )
    return p.parse_args()


def _patch_placebo_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: mark ineligible control variants with explicit placebo_note."""
    if df.empty or "control_variant" not in df.columns:
        return df
    out = df.copy()
    if "placebo_note" not in out.columns:
        out["placebo_note"] = np.nan
    mask = (
        out.get("static_variant", pd.Series(dtype=object)).astype(str) == "paper_eq1"
    ) & ~out["control_variant"].astype(str).map(is_placebo_space_eligible_control_variant)
    out.loc[mask, "placebo_note"] = "not_applicable_single_country_contrast"
    for col in ("p_placebo_space", "perm_p"):
        if col in out.columns:
            out.loc[mask, col] = np.nan
    return out


def _recombine_split_sample_static(rows: List[Dict[str, Any]], scheme: str) -> List[Dict[str, Any]]:
    """Function summary: retain per-split paper_eq1 rows and append split_id=-1 combined rows."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("static_variant") != "paper_eq1":
            continue
        sid = r.get("split_id")
        if sid is not None and int(sid) < 0:
            continue
        key = f"{r.get('control_variant')}|{r.get('bucket')}"
        grouped.setdefault(key, []).append(r)
    keep = [r for r in rows if not (r.get("static_variant") == "paper_eq1" and r.get("split_id") == -1)]
    extra: List[Dict[str, Any]] = []
    for key, grp in grouped.items():
        if len(grp) > 1:
            c = combine_split_sample_static(grp)
            cv, bucket = key.split("|", 1)
            c.update({"scheme": scheme, "control_variant": cv, "bucket": bucket, "split_id": -1})
            extra.append(c)
    return keep + extra


def patch_static_csv(path: Path, scheme: str) -> None:
    """Function summary: patch one static_{scheme}.csv in place."""
    if not path.is_file():
        print(f"[patch_bucket_static] skip missing {path}", flush=True)
        return
    df = pd.read_csv(path)
    n_before = len(df)
    df = _patch_placebo_notes(df)
    if scheme == "split_sample":
        per_split = df[
            (df.get("static_variant", pd.Series(dtype=object)).astype(str) == "paper_eq1")
            & (pd.to_numeric(df.get("split_id", pd.Series(dtype=float)), errors="coerce") >= 0)
        ]
        if not per_split.empty:
            df = pd.DataFrame(_recombine_split_sample_static(df.to_dict("records"), scheme))
        else:
            print(
                f"[patch_bucket_static] {path.name}: no per-split paper_eq1 rows; "
                "re-run split_sample scheme for full fix",
                flush=True,
            )
    df.to_csv(path, index=False)
    print(f"[patch_bucket_static] patched {path.name} ({n_before} rows)", flush=True)


def main() -> None:
    """Function summary: patch static CSVs and optionally rerun DDD estimation."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    bcfg = bucket_event_study_config(config)
    if args.bin_days is not None:
        bcfg = replace(bcfg, bin_days=int(args.bin_days))
    tables_dir = did_bucket_event_study_dir(config, bin_days=bcfg.bin_days)

    for scheme in bcfg.schemes:
        patch_static_csv(tables_dir / f"static_{scheme}.csv", scheme)

    if args.static_only or not args.rerun_ddd:
        if not args.rerun_ddd:
            print(
                "[patch_bucket_static] static patch done; pass --rerun-ddd to refresh ddd_*.csv",
                flush=True,
            )
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/analysis/bucket_event_study.py"),
        "--config",
        args.config,
        "--ddd-only",
        "--no-figures",
    ]
    if args.bin_days is not None:
        cmd.extend(["--bin-days", str(args.bin_days)])
    print(f"[patch_bucket_static] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
