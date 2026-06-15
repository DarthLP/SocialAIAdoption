"""
Script summary:
Build subreddit-day Q&A substitution panels from cleaned enriched shards (question volume + question-mark rate).

Functionality:
- Loads Italian forums and DE/EU/UK hub controls from cleaned_monthly_chunks.
- Computes cheap per-comment question proxies from body text (no full feature re-run).
- Aggregates to 1-day and 3-day panels with DiD calendar and treatment flags.
- Zero-fills every roster subreddit to the full event-window day grid (implicit zeros for inactive days).
- Writes qa_volume_panel_1d.csv and qa_volume_panel_3d.csv under qa_substitution tables.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_qa_volume_panel.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List

import pandas as pd

READ_COLUMNS = [
    "author",
    "subreddit",
    "date_utc",
    "body",
    "n_words",
    "topic_family",
    "topic",
]

BIN_DAYS = 3


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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import (  # noqa: E402
    load_config,
    qa_advice_subreddit_set,
    qa_substitution_panel_subreddits,
    subreddit_family_map,
    tables_subdir,
)
from src.qa_substitution import (  # noqa: E402
    add_did_calendar_columns,
    add_treatment_flags,
    aggregate_panel_bins,
    aggregate_subreddit_day,
    annotate_comment_questions,
    reindex_full_grid,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build Q&A substitution volume panels.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=BIN_DAYS)
    return parser.parse_args()


def load_comment_frame(shard_root: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load comment rows for panel subreddits from monthly Parquet shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks directory.
    - subreddits: subreddit names to include.

    Returns:
    - Combined comment dataframe.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        for shard in sorted(shard_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(shard)
            except Exception:
                continue
            cols = [c for c in READ_COLUMNS if c in df.columns]
            if not cols:
                continue
            chunk = df[cols].copy()
            chunk["subreddit"] = sub
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """Function summary: write Q&A substitution 1d and 3d panels."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    start, end_excl, launch, lift = event_dates_from_config(config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    out_dir = tables_subdir(config, "qa_substitution")
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = qa_substitution_panel_subreddits(config)
    family_map = subreddit_family_map(config)
    qa_set = qa_advice_subreddit_set(config)

    df = load_comment_frame(shard_root, subs)
    if df.empty:
        print("[prepare_qa_volume_panel] no parquet data found", flush=True)
        return

    df = df[(df["date_utc"].astype(str) >= start) & (df["date_utc"].astype(str) < end_excl)].copy()
    if "topic_family" not in df.columns or df["topic_family"].isna().any():
        df["topic_family"] = df["subreddit"].map(family_map).fillna(df.get("topic_family"))

    df = annotate_comment_questions(df)
    panel_1d = aggregate_subreddit_day(df)
    panel_1d = reindex_full_grid(panel_1d, subs, start, end_excl, family_map)
    panel_1d = add_did_calendar_columns(panel_1d, launch, lift, end_excl, bin_days=args.bin_days)
    panel_1d = add_treatment_flags(panel_1d, qa_set)

    out_1d = out_dir / "qa_volume_panel_1d.csv"
    panel_1d.to_csv(out_1d, index=False)
    print(f"[prepare_qa_volume_panel] {out_1d.name} rows={len(panel_1d)}", flush=True)

    panel_3d = aggregate_panel_bins(panel_1d, bin_days=args.bin_days)
    panel_3d["date_utc"] = panel_3d["period_start"].astype(str)
    panel_3d = add_did_calendar_columns(panel_3d, launch, lift, end_excl, bin_days=args.bin_days)
    panel_3d = add_treatment_flags(panel_3d, qa_set)
    out_3d = out_dir / "qa_volume_panel_3d.csv"
    panel_3d.to_csv(out_3d, index=False)
    print(f"[prepare_qa_volume_panel] {out_3d.name} rows={len(panel_3d)}", flush=True)

    roster = pd.DataFrame(
        {
            "subreddit": subs,
            "topic_family": [family_map.get(s, "") for s in subs],
            "qa": [int(s in qa_set) for s in subs],
            "IT": [int(family_map.get(s, "") in {"it_political", "it_others"}) for s in subs],
            "is_hub": [int(family_map.get(s, "") in {"de", "eu", "uk"}) for s in subs],
        }
    )
    roster.to_csv(out_dir / "qa_substitution_subreddit_roster.csv", index=False)
    print(f"[prepare_qa_volume_panel] wrote to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
