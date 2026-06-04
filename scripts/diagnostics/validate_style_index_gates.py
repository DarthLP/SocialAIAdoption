"""
Script summary:
Task 5 validation gates for the formula style index (histograms, Spearman, pretrend, review sample).

Functionality:
- Samples comments with style_index_full and ai_style_rate_100w; writes gate artifacts under did/style_index_validation/.
- Pretrend gate uses cross_country_all TWFE on subreddit panel when available.

How to apply/run:
  .venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/compute_style_index_on_shards.py --max-shards 20
  .venv/bin/python scripts/diagnostics/validate_style_index_gates.py --max-shards 10
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


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

from src.config_utils import load_config, resolve_primary_subreddits, tables_subdir  # noqa: E402
from src.did.estimate import estimate_pretrend_f  # noqa: E402
from src.did.outcomes import outcome_spec  # noqa: E402
from src.did.panels import load_subreddit_panel  # noqa: E402
from src.did.specs import HEADLINE_BASE_STRATEGIES, filter_strategy_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Style index validation gates.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=30)
    p.add_argument("--review-n", type=int, default=20)
    return p.parse_args()


def _sample_comments(config: Dict[str, Any], max_shards: int) -> pd.DataFrame:
    """Function summary: load comment rows with style index columns from shards."""
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    rows: List[pd.DataFrame] = []
    n = 0
    cols = ["body", "date_utc", "primary_lexicon", "style_index_full", "ai_style_rate_100w", "author", "id"]
    for sub in resolve_primary_subreddits(config):
        for shard in sorted((shard_root / sub).glob("*.parquet")):
            if n >= max_shards:
                break
            try:
                df = pd.read_parquet(shard, columns=cols)
            except Exception:
                continue
            if "style_index_full" not in df.columns:
                continue
            rows.append(df)
            n += 1
        if n >= max_shards:
            break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Function summary: run gates and write validation artifacts."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out_dir = tables_subdir(config, "did") / "style_index_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _sample_comments(config, args.max_shards)
    gate_rows: List[Dict[str, Any]] = []

    if df.empty:
        gate_rows.append({"gate": "data", "status": "fail", "note": "no shards with style_index_full"})
    else:
        si = pd.to_numeric(df["style_index_full"], errors="coerce").dropna()
        ai = pd.to_numeric(df.get("ai_style_rate_100w", pd.Series(dtype=float)), errors="coerce")
        mask = si.notna() & ai.notna()
        rho = float(si[mask].corr(ai[mask], method="spearman")) if mask.sum() >= 30 else float("nan")
        gate_rows.append(
            {
                "gate": "spearman_vs_ai_style_rate_100w",
                "status": "pass" if np.isfinite(rho) and rho > 0.3 else "review",
                "value": rho,
                "n": int(mask.sum()),
            }
        )
        if plt is not None and len(si) >= 50:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(si, bins=50, color="steelblue", edgecolor="white")
            ax.set_title("style_index_full (sampled comments)")
            fig.savefig(out_dir / "hist_style_index_full.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            gate_rows.append({"gate": "histogram", "status": "pass", "path": str(out_dir / "hist_style_index_full.png")})

        review = df.dropna(subset=["style_index_full"]).sample(
            n=min(args.review_n * 2, len(df)), random_state=42
        )
        hi = review.nlargest(args.review_n, "style_index_full")
        lo = review.nsmallest(args.review_n, "style_index_full")
        pd.concat([hi.assign(review_bucket="high"), lo.assign(review_bucket="low")]).to_csv(
            out_dir / "review_20plus20.csv", index=False
        )
        gate_rows.append({"gate": "review_csv", "status": "pass", "path": str(out_dir / "review_20plus20.csv")})

    try:
        panel = load_subreddit_panel(config)
        strat = next(s for s in HEADLINE_BASE_STRATEGIES if s.strategy_id == "cross_country_all")
        work = filter_strategy_sample(panel, strat)
        if "entity_id" not in work.columns:
            work["entity_id"] = work["subreddit"].astype(str)
        if "time_id" not in work.columns:
            work["time_id"] = work["date_utc"].astype(str)
        oc = outcome_spec("style_index_full")
        if oc is None:
            gate_rows.append({"gate": "pretrend_F", "status": "skip", "note": "unknown outcome style_index_full"})
            oc = None
        elif oc.column not in work.columns:
            gate_rows.append({"gate": "pretrend_F", "status": "skip", "note": f"missing {oc.column} on panel"})
        else:
            fp, _note = estimate_pretrend_f(work, oc.column)
            gate_rows.append(
                {
                    "gate": "pretrend_F_style_index_full",
                    "status": "pass" if np.isfinite(fp) and fp > 0.05 else "review",
                    "pretrend_F_p": fp,
                }
            )
    except FileNotFoundError as exc:
        gate_rows.append({"gate": "pretrend_F", "status": "skip", "note": str(exc)})

    summary = pd.DataFrame(gate_rows)
    summary.to_csv(out_dir / "gates_summary.csv", index=False)
    print(f"[validate_style_index_gates] wrote {out_dir / 'gates_summary.csv'}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
