"""
Script summary:
Build before/after inference comparison table from pre-fix and post-fix did_summary CSVs.

Functionality:
- Joins did_summary_pre_inference.csv with did_summary.csv on outcome×strategy×spec.
- Adds gsynth placebo p from did/estimates/gsynth/inference_*.csv.
- Writes results/tables/<study>/did/inference_before_after.md.

How to apply/run:
  .venv/bin/python scripts/diagnostics/build_inference_before_after.py \\
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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
from src.did.outcomes import HEADLINE_OUTCOMES  # noqa: E402
from src.did.paths import did_gsynth_dir, did_summary_dir, did_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for inference comparison table."""
    parser = argparse.ArgumentParser(description="Build inference before/after markdown.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=3)
    return parser.parse_args()


def _load_gsynth_p(config: dict, outcome_id: str, bin_days: int) -> float:
    """Function summary: read gsynth placebo-space p for one outcome."""
    path = did_gsynth_dir(config) / f"inference_{outcome_id}_{bin_days}d.csv"
    if not path.is_file():
        return float("nan")
    row = pd.read_csv(path).iloc[0]
    return float(row.get("p_gsynth_placebo", row.get("p_placebo_space", float("nan"))))


def main() -> None:
    """Function summary: write inference_before_after.md."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    summary_path, _ = did_summary_paths(config)
    pre_path = did_summary_dir(config) / "did_summary_pre_inference.csv"
    post = pd.read_csv(summary_path)
    pre = pd.read_csv(pre_path) if pre_path.is_file() else None

    keys = ["outcome_id", "strategy_id", "spec"]
    strategies = ("cross_country_all", "within_italy_ddd", "author_it_vs_en")
    rows = []
    for oid in HEADLINE_OUTCOMES:
        for sid in strategies:
            mask = (
                (post["outcome_id"] == oid)
                & (post["strategy_id"] == sid)
                & (post["spec"].astype(str) == "full_ban")
            )
            sub = post[mask]
            if sub.empty:
                continue
            row = sub.iloc[0]
            p_cluster_old = float("nan")
            if pre is not None:
                pm = (
                    (pre["outcome_id"] == oid)
                    & (pre["strategy_id"] == sid)
                    & (pre["spec"].astype(str) == "full_ban")
                )
                if pm.any():
                    p_cluster_old = float(pre.loc[pm, "pvalue"].iloc[0])
            p_cluster_new = float(row.get("pvalue", float("nan")))
            floor = row.get("placebo_p_floor", float("nan"))
            rows.append(
                {
                    "outcome": oid,
                    "strategy": sid,
                    "p_cluster_old": p_cluster_old if pd.notna(p_cluster_old) else p_cluster_new,
                    "p_cluster_new": p_cluster_new,
                    "p_wild_wcr": row.get("wild_p", float("nan")),
                    "p_gsynth_placebo": _load_gsynth_p(config, oid, args.bin_days)
                    if sid in ("cross_country_all",)
                    else float("nan"),
                    "p_placebo_space": row.get(
                        "p_placebo_space",
                        row.get("perm_p", float("nan")),
                    ),
                    "placebo_p_floor": floor,
                }
            )

    out_path = did_summary_dir(config).parent.parent / "inference_before_after.md"
    lines = [
        "# DiD inference before/after",
        "",
        "Cross-country forum-clustered p-values are **descriptive only**. "
        "Headline significance uses placebo-in-space (cross-country) or restricted WCB (within-Italy / author).",
        "",
        "Placebo-in-space p-value floor with 4 control countries: **1/5 = 0.2**.",
        "",
        "| outcome | strategy | (a) p_cluster (pre) | (a) p_cluster (post) | (b) p_wild WCR | "
        "(c) p_gsynth placebo | (d) p_placebo_space | floor |",
        "|---------|----------|---------------------|----------------------|----------------|"
        "---------------------|---------------------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['outcome']} | {r['strategy']} | {r['p_cluster_old']:.4g} | {r['p_cluster_new']:.4g} | "
            f"{r['p_wild_wcr']:.4g} | {r['p_gsynth_placebo']:.4g} | {r['p_placebo_space']:.4g} | "
            f"{r['placebo_p_floor']:.4g} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build_inference_before_after] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
