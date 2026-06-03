"""
Script summary:
Build comment-level and author×day DiD panels from enriched monthly Parquet shards.

Functionality:
- Streams cleaned_monthly_chunks with column projection (low RAM).
- Default sample: comment_in_political_universe and Mar–Apr event window.
- Writes partitioned Parquet under did/panels/comment/did_comment_panel_1d/.
- Writes author×day weighted means to did_author_day_panel_1d.csv for PanelOLS robustness.
- Optional 3-day launch-aligned bins via --bin-days 3.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_comment_panel.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_did_comment_panel.py --max-shards 2
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ITALY_FAMILIES = frozenset({"it_political", "it_others"})
UNIVERSE_SLICE_IN = "in_political_tree"
UNIVERSE_SLICE_OUT = "out_political_tree"

COMMENT_COLUMNS: Tuple[str, ...] = (
    "id",
    "author",
    "subreddit",
    "date_utc",
    "n_words",
    "topic",
    "topic_family",
    "primary_lexicon",
    "comment_in_political_universe",
    "net_ideology",
    "left_hits",
    "right_hits",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "has_sem_axis",
    "ai_style_rate_100w",
    "em_dash_count",
    "sentence_count_comment",
)

WEIGHTED_OUTCOME_COLS: Tuple[str, ...] = (
    "net_ideology",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "ai_style_rate_100w",
)


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
from src.config_utils import load_config, resolve_primary_subreddits  # noqa: E402
from src.did.paths import did_panels_dir  # noqa: E402
from src.did.specs import ITALY_FAMILIES as SPEC_ITALY_FAMILIES, rel_day_from_date  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for comment DiD panel build."""
    parser = argparse.ArgumentParser(description="Prepare comment-level DiD panels.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--all-comments",
        action="store_true",
        help="Include comments outside political universe (default: political tree only).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit parquet shards per subreddit (smoke tests).",
    )
    parser.add_argument(
        "--bin-days",
        type=int,
        default=1,
        choices=(1, 3),
        help="Calendar bin width: 1=date_utc; 3=launch-aligned 3-day blocks.",
    )
    return parser.parse_args()


def _iter_shard_paths(
    interim_dir: Path, subs: List[str], max_shards: Optional[int]
) -> List[Path]:
    """Function summary: list enriched parquet paths for primary subreddits."""
    paths: List[Path] = []
    for sub in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards:
            shards = shards[: max_shards]
        paths.extend(shards)
    return paths


def _read_shard(path: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Function summary: read projected columns from one parquet shard."""
    try:
        import pyarrow.parquet as pq  # noqa: WPS433

        schema = pq.read_schema(path)
        avail = [c for c in columns if c in schema.names]
        if not avail:
            return None
        return pd.read_parquet(path, columns=avail)
    except Exception:
        try:
            df = pd.read_parquet(path)
            cols = [c for c in columns if c in df.columns]
            return df[cols] if cols else None
        except Exception:
            return None


def _universe_slice_label(in_pol: bool) -> str:
    """Function summary: map comment_in_political_universe to slice label."""
    return UNIVERSE_SLICE_IN if in_pol else UNIVERSE_SLICE_OUT


def _annotate_comments(df: pd.DataFrame, launch: str, end_excl: str) -> pd.DataFrame:
    """Function summary: add DiD calendar, IT, and universe_slice on comment rows."""
    out = df.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    out["rel_day"] = rel_day_from_date(out["date_utc"], launch)
    out["post"] = (out["date_utc"] >= launch).astype(int)
    if "topic_family" in out.columns:
        fam = out["topic_family"].astype(str)
        out["IT"] = fam.isin(SPEC_ITALY_FAMILIES).astype(int)
    else:
        out["IT"] = 0
    if "comment_in_political_universe" in out.columns:
        out["universe_slice"] = out["comment_in_political_universe"].astype(bool).map(
            _universe_slice_label
        )
    else:
        out["universe_slice"] = UNIVERSE_SLICE_OUT
    out["author"] = out["author"].astype(str)
    out["time_id"] = out["date_utc"].astype(str)
    return out
    return out


def _apply_3d_bins(df: pd.DataFrame, launch: str, bin_days: int) -> pd.DataFrame:
    """Function summary: map rel_day to launch-aligned period_start for 3-day bins."""
    if bin_days <= 1:
        return df
    out = df.copy()
    out["rel_period"] = (out["rel_day"] // bin_days).astype(int)
    launch_dt = pd.Timestamp(launch)
    out["period_start"] = (
        launch_dt + pd.to_timedelta(out["rel_period"] * bin_days, unit="D")
    ).dt.strftime("%Y-%m-%d")
    out["time_id"] = out["period_start"].astype(str)
    return out


def _weighted_mean_group(
    grp: pd.DataFrame, value_col: str, weight_col: str = "n_words"
) -> float:
    """Function summary: n_words-weighted mean of value_col within a group."""
    if value_col not in grp.columns:
        return float("nan")
    w = pd.to_numeric(grp[weight_col], errors="coerce").fillna(0)
    v = pd.to_numeric(grp[value_col], errors="coerce")
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float(v.mean()) if v.notna().any() else float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def build_author_day_panel(comment_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Function summary: aggregate comment rows to author×day weighted means.

    Parameters:
    - comment_frames: list of annotated comment DataFrames.

    Returns:
    - Author×day panel with entity_id, time_id, outcomes, n_comments.
    """
    if not comment_frames:
        return pd.DataFrame()
    df = pd.concat(comment_frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame()
    keys = ["author", "time_id"]
    rows: List[Dict[str, Any]] = []
    for key, grp in df.groupby(keys, observed=True):
        author, time_id = key if isinstance(key, tuple) else (key, key)
        row: Dict[str, Any] = {
            "author": author,
            "time_id": time_id,
            "date_utc": grp["date_utc"].iloc[0],
            "entity_id": str(author),
            "n_comments": len(grp),
            "rel_day": int(grp["rel_day"].iloc[0]),
            "post": int(grp["post"].iloc[0]),
        }
        if "topic_family" in grp.columns:
            row["topic_family"] = grp["topic_family"].iloc[0]
            row["IT"] = int(grp["IT"].iloc[0])
        if "primary_lexicon" in grp.columns:
            row["primary_lexicon"] = grp["primary_lexicon"].iloc[0]
        if "universe_slice" in grp.columns:
            row["universe_slice"] = grp["universe_slice"].iloc[0]
        wcol = "n_words" if "n_words" in grp.columns else None
        for col in WEIGHTED_OUTCOME_COLS:
            if col in grp.columns:
                if wcol:
                    row[col] = _weighted_mean_group(grp, col, wcol)
                else:
                    row[col] = float(pd.to_numeric(grp[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_comment_panels(
    config: Dict[str, Any],
    political_only: bool = True,
    max_shards: Optional[int] = None,
    bin_days: int = 1,
) -> Tuple[Path, Path, int]:
    """Function summary: stream shards and write comment + author-day DiD panels.

    Parameters:
    - config: loaded YAML.
    - political_only: when True, keep comment_in_political_universe rows only.
    - max_shards: optional per-subreddit shard cap.
    - bin_days: 1 or 3 for time_id binning.

    Returns:
    - Tuple (comment_panel_dir, author_day_csv_path, n_comment_rows).
    """
    start, end_excl, launch, _ = event_dates_from_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = resolve_primary_subreddits(config)
    shards = _iter_shard_paths(interim_dir, subs, max_shards)

    comment_dir = did_panels_dir(config, "comment")
    tag = f"{int(bin_days)}d"
    comment_out = comment_dir / f"did_comment_panel_{tag}"
    comment_out.mkdir(parents=True, exist_ok=True)
    author_day_path = comment_dir / f"did_author_day_panel_{tag}.csv"

    month_chunks: Dict[str, List[pd.DataFrame]] = {}
    n_rows = 0

    for i, shard in enumerate(shards, start=1):
        raw = _read_shard(shard, COMMENT_COLUMNS)
        if raw is None or raw.empty:
            continue
        if "date_utc" not in raw.columns or "author" not in raw.columns:
            continue
        raw["date_utc"] = raw["date_utc"].astype(str)
        raw = raw[(raw["date_utc"] >= start) & (raw["date_utc"] < end_excl)]
        if political_only and "comment_in_political_universe" in raw.columns:
            raw = raw[raw["comment_in_political_universe"].astype(bool)]
        if raw.empty:
            continue
        ann = _annotate_comments(raw, launch, end_excl)
        ann = _apply_3d_bins(ann, launch, bin_days)
        n_rows += len(ann)
        month_key = ann["date_utc"].str[:7].iloc[0]
        month_chunks.setdefault(month_key, []).append(ann)
        if i % 25 == 0:
            print(f"[prepare_did_comment_panel] shard {i}/{len(shards)} rows={n_rows}", flush=True)

    all_frames: List[pd.DataFrame] = []
    for month, frames in month_chunks.items():
        part = pd.concat(frames, ignore_index=True)
        part_path = comment_out / f"month={month}.parquet"
        part.to_parquet(part_path, index=False)
        all_frames.append(part)

    auth_day = build_author_day_panel(all_frames)
    if not auth_day.empty and bin_days > 1:
        auth_day = _apply_3d_bins(auth_day, launch, bin_days)
    auth_day.to_csv(author_day_path, index=False)

    print(
        f"[prepare_did_comment_panel] comment rows={n_rows} -> {comment_out} "
        f"author_day rows={len(auth_day)} -> {author_day_path.name}",
        flush=True,
    )
    return comment_out, author_day_path, n_rows


def main() -> None:
    """Function summary: CLI entry for comment DiD panel build."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    build_comment_panels(
        config,
        political_only=not args.all_comments,
        max_shards=args.max_shards,
        bin_days=args.bin_days,
    )


if __name__ == "__main__":
    main()
