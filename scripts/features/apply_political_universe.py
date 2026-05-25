"""
Script summary:
Apply comment-level political universe flags to enriched monthly Parquet shards.

Functionality:
- Loads Mar–Apr 2023 shards per subreddit, groups by link_id on the concatenated frame.
- Writes in_political_universe_* and comment_in_political_universe columns back per month.

How to apply/run:
  .venv/bin/python scripts/features/apply_political_universe.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/apply_political_universe.py --config config/italy_polarization_setup.yaml --subreddit Italia
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

EVENT_MONTHS = ("2023-03", "2023-04")
LOAD_COLUMNS = [
    "id",
    "parent_id",
    "link_id",
    "date_utc",
    "political_weighted_points",
    "n_words",
]


def _setup_project_root(caller_file: Path) -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    for parent in caller_file.resolve().parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller_file)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root(Path(__file__))

from src.config_utils import (  # noqa: E402
    load_config,
    load_political_universe_config,
    load_screening_config,
    load_screening_pooled,
    resolve_primary_subreddits,
    screening_by_subreddit,
    shard_dir_is_enriched,
    should_skip_screened_subreddit,
    subreddit_screening_action,
)
from src.political_filter import ALL_UNIVERSE_BOOL_COLUMNS, apply_all_modes  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Apply political universe flags to enriched shards.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default="")
    parser.add_argument("--include-excluded", action="store_true")
    return parser.parse_args()


def _read_shard(path: Path) -> pd.DataFrame | None:
    """Function summary: read one Parquet shard if present and non-empty."""
    if not path.is_file() or path.stat().st_size < 8:
        return None
    try:
        return pd.read_parquet(path, columns=LOAD_COLUMNS)
    except Exception:
        return None


def _month_key_from_shard(path: Path, df: pd.DataFrame) -> str:
    """Function summary: derive YYYY-MM bucket for splitting rows back to shards."""
    stem = path.stem
    if len(stem) == 7 and stem[4] == "-":
        return stem
    if "date_utc" in df.columns and not df.empty:
        sample = str(df["date_utc"].iloc[0])[:7]
        if len(sample) == 7:
            return sample
    return stem


def process_subreddit(
    subreddit: str,
    shard_dir: Path,
    pu_cfg: Dict[str, Any],
    screening: Dict[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Function summary: score and write universe columns (split by concat chunk order)."""
    shard_paths = [shard_dir / f"{m}.parquet" for m in EVENT_MONTHS]
    parts: List[Tuple[str, Path, pd.DataFrame]] = []
    for path in shard_paths:
        df = _read_shard(path)
        if df is None or df.empty:
            continue
        month = _month_key_from_shard(path, df)
        parts.append((month, path, df))

    if not parts:
        return 0, {}

    combined = pd.concat([p[2] for p in parts], ignore_index=True)
    missing = {"political_weighted_points", "id", "parent_id", "link_id"} - set(combined.columns)
    if missing:
        raise ValueError(
            f"subreddit={subreddit} missing columns {sorted(missing)}; run enrich_cleaned_chunks.py first"
        )

    scored, stats = apply_all_modes(combined, pu_cfg, screening)
    stats["subreddit"] = subreddit
    total_written = 0
    offset = 0
    for month, path, orig in parts:
        n = len(orig)
        chunk_scored = scored.iloc[offset : offset + n]
        offset += n
        existing = pd.read_parquet(path)
        drop_cols = [c for c in ALL_UNIVERSE_BOOL_COLUMNS if c in existing.columns]
        base = existing.drop(columns=drop_cols, errors="ignore")
        for col in ALL_UNIVERSE_BOOL_COLUMNS:
            if col in chunk_scored.columns:
                base[col] = chunk_scored[col].astype(bool).values
        base.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        total_written += n

    return total_written, stats


def main() -> None:
    """Function summary: apply political universe to all primary subreddits."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    pu_cfg = load_political_universe_config(config)
    screening = load_screening_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    screening_by_sub = screening_by_subreddit(load_screening_pooled(tables_dir))

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    grand_total = 0
    all_stats: List[Dict[str, float]] = []

    for subreddit in subs:
        action = subreddit_screening_action(screening_by_sub, subreddit)
        if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
            print(f"[apply_political_universe] skip excluded subreddit={subreddit}", flush=True)
            continue
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        if not shard_dir_is_enriched(shard_dir):
            print(
                f"[apply_political_universe] skip subreddit={subreddit}: "
                "shards not enriched; run enrich_cleaned_chunks.py first",
                flush=True,
            )
            continue
        n, st = process_subreddit(subreddit, shard_dir, pu_cfg, screening)
        grand_total += n
        if st:
            all_stats.append(st)
        print(
            f"[apply_political_universe] subreddit={subreddit} rows={n} "
            f"universe_share={st.get('political_universe_share', 0):.4f} "
            f"orphan_share={st.get('orphan_share', 0):.4f}",
            flush=True,
        )

    if all_stats:
        out_dir = tables_dir / "political_coverage"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_stats).to_csv(out_dir / "apply_political_universe_stats.csv", index=False)

    print(f"[apply_political_universe] done total_rows={grand_total}", flush=True)


if __name__ == "__main__":
    main()
