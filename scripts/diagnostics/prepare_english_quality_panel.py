"""
Script summary:
Build comment-level panels for within-author English-quality DiD analysis.

Functionality:
- Loads or rebuilds the author roster (italian_bilingual / native_control).
- Streams enriched shards with lang_comment and writing-quality outcomes.
- Annotates post/rel_period, applies cohort gates, and writes partitioned parquet.
- Stores March standardization moments per outcome for downstream estimation.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_english_quality_panel.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_english_quality_panel.py --bin-days 3 --max-shards 2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute repo root Path.
    """
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

from scripts.diagnostics.build_english_quality_roster import (  # noqa: E402
    collect_author_activity as _collect_author_activity,
)
from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import load_config, resolve_primary_subreddits  # noqa: E402
from src.did.english_quality import (  # noqa: E402
    CROSS_LANGUAGE_MIN_STRICT_AUTHORS,
    DEFAULT_BOT_AUTHORS,
    HEADLINE_OUTCOMES,
    PANEL_COLUMNS,
    POLARIZATION_OUTCOMES,
    ROSTER_WINDOW_CHOICES,
    annotate_english_quality_comments,
    apply_3d_bins,
    classify_author_roster,
    cohort_authors_for_design,
    cohort_thresholds_by_label,
    english_quality_run_tables_dir,
    filter_cross_language_sample,
    filter_native_control_sample,
    march_standardization_moments_by_lang,
    march_standardization_moments_pooled,
)

# Designs whose cohort sizes are tracked in the audit + cached as panel flags.
PANEL_COHORT_DESIGNS: Tuple[str, ...] = (
    "native_control",
    "cross_language",
    "cross_language_native_it",
    "cross_language_langmix",
)
STANDARDIZATION_OUTCOMES: Tuple[str, ...] = tuple(
    dict.fromkeys((*HEADLINE_OUTCOMES, *POLARIZATION_OUTCOMES))
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for English-quality panel build.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare English-quality comment panels.")
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=3, choices=(1, 3))
    parser.add_argument("--min-comment-words", type=int, default=5)
    parser.add_argument("--cohort", default="strict", choices=("strict", "loose"))
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--reuse-roster", action="store_true")
    parser.add_argument(
        "--roster-window",
        default="pre_ban",
        choices=ROSTER_WINDOW_CHOICES,
        help="Roster classification window (pre_ban default; full for legacy comparison).",
    )
    return parser.parse_args()


def _iter_shard_paths(
    interim_dir: Path, subs: List[str], max_shards: Optional[int]
) -> List[Path]:
    """Function summary: list enriched parquet paths for primary subreddits.

    Parameters:
    - interim_dir: interim data root.
    - subs: subreddit names.
    - max_shards: optional per-subreddit cap.

    Returns:
    - Sorted list of shard paths.
    """
    paths: List[Path] = []
    for sub in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards:
            shards = shards[:max_shards]
        paths.extend(shards)
    return paths


def _read_shard(path: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Function summary: read projected columns from one parquet shard.

    Parameters:
    - path: parquet file path.
    - columns: desired column names.

    Returns:
    - DataFrame or None when unreadable/empty projection.
    """
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


def load_or_build_roster(
    config: Dict[str, Any],
    *,
    reuse_roster: bool,
    max_shards: Optional[int],
    roster_window: str = "pre_ban",
) -> pd.DataFrame:
    """Function summary: load existing roster CSV or rebuild from shards.

    Parameters:
    - config: study YAML.
    - reuse_roster: when True, read author_roster.csv if present.
    - max_shards: optional shard cap for rebuild.
    - roster_window: pre_ban or full classification window.

    Returns:
    - Author roster DataFrame.
    """
    roster_path = english_quality_run_tables_dir(config, roster_window) / "author_roster.csv"
    if reuse_roster and roster_path.is_file():
        return pd.read_csv(roster_path)
    start, end_excl, launch, _ = event_dates_from_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = resolve_primary_subreddits(config)
    author_forums, author_lang = _collect_author_activity(
        interim_dir,
        subs,
        start,
        end_excl,
        launch,
        include_deleted=False,
        include_bots=False,
        max_shards=max_shards,
        roster_window=roster_window,
    )
    return classify_author_roster(author_forums, config, author_lang=author_lang)


def build_english_quality_panels(
    config: Dict[str, Any],
    *,
    bin_days: int = 3,
    min_comment_words: int = 5,
    cohort_label: str = "strict",
    max_shards: Optional[int] = None,
    reuse_roster: bool = False,
    roster_window: str = "pre_ban",
) -> Tuple[Path, int]:
    """Function summary: stream shards and write English-quality comment panel parquet.

    Parameters:
    - config: loaded YAML.
    - bin_days: 1 or 3 for time_id binning.
    - min_comment_words: drop short comments for langid reliability.
    - cohort_label: strict or loose author gates.
    - max_shards: optional per-subreddit shard cap.
    - reuse_roster: load existing roster CSV when available.
    - roster_window: pre_ban or full roster classification window.

    Returns:
    - Tuple (panel output directory, number of comment rows written).
    """
    run_dir = english_quality_run_tables_dir(config, roster_window)
    start, end_excl, launch, _ = event_dates_from_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = resolve_primary_subreddits(config)
    shards = _iter_shard_paths(interim_dir, subs, max_shards)
    roster = load_or_build_roster(
        config,
        reuse_roster=reuse_roster,
        max_shards=max_shards,
        roster_window=roster_window,
    )
    roster.to_csv(run_dir / "author_roster.csv", index=False)

    keep_mask = roster["author_group"].isin(["italian_bilingual", "native_control"])
    if "lang_bilingual" in roster.columns:
        keep_mask = keep_mask | (roster["lang_bilingual"].fillna(0).astype(int) == 1)
    keep_authors = set(roster.loc[keep_mask, "author"].astype(str))
    panel_dir = run_dir / "panel" / f"{int(bin_days)}d"
    panel_dir.mkdir(parents=True, exist_ok=True)

    month_chunks: Dict[str, List[pd.DataFrame]] = {}
    n_rows = 0
    bot_list = set(DEFAULT_BOT_AUTHORS)

    for i, shard in enumerate(shards, start=1):
        raw = _read_shard(shard, PANEL_COLUMNS)
        if raw is None or raw.empty:
            continue
        if "date_utc" not in raw.columns or "author" not in raw.columns:
            continue
        raw["date_utc"] = raw["date_utc"].astype(str)
        raw = raw[(raw["date_utc"] >= start) & (raw["date_utc"] < end_excl)]
        raw["author"] = raw["author"].astype(str)
        raw = raw[~raw["author"].isin(bot_list) & (raw["author"] != "[deleted]")]
        raw = raw[raw["author"].isin(keep_authors)]
        if "n_words" in raw.columns:
            raw = raw[pd.to_numeric(raw["n_words"], errors="coerce").fillna(0) >= min_comment_words]
        if raw.empty:
            continue
        ann = annotate_english_quality_comments(raw, launch, roster)
        ann = apply_3d_bins(ann, launch, bin_days)
        n_rows += len(ann)
        month_key = ann["date_utc"].str[:7].iloc[0]
        month_chunks.setdefault(month_key, []).append(ann)
        if i % 25 == 0:
            print(f"[prepare_english_quality_panel] shard {i}/{len(shards)} rows={n_rows}", flush=True)

    all_frames: List[pd.DataFrame] = []
    for month, frames in month_chunks.items():
        part = pd.concat(frames, ignore_index=True)
        part_path = panel_dir / f"month={month}.parquet"
        part.to_parquet(part_path, index=False)
        all_frames.append(part)

    if not all_frames:
        print("[prepare_english_quality_panel] no rows written", flush=True)
        return panel_dir, 0

    panel = pd.concat(all_frames, ignore_index=True)
    for label in ("strict", "loose"):
        th = cohort_thresholds_by_label(label)
        for design in PANEL_COHORT_DESIGNS:
            authors = cohort_authors_for_design(panel, design, th)
            panel[f"cohort_{design}_{label}"] = panel["author"].isin(authors)
    thresholds = cohort_thresholds_by_label(cohort_label)
    native_authors = cohort_authors_for_design(panel, "native_control", thresholds)
    cross_authors = cohort_authors_for_design(panel, "cross_language", thresholds)

    for month, grp in panel.groupby(panel["date_utc"].str[:7], observed=True):
        part_path = panel_dir / f"month={month}.parquet"
        grp.to_parquet(part_path, index=False)

    moments_rows: List[Dict[str, Any]] = []
    nat = filter_native_control_sample(panel)
    cross = filter_cross_language_sample(panel)
    for outcome in STANDARDIZATION_OUTCOMES:
        if outcome not in panel.columns:
            continue
        pooled = march_standardization_moments_pooled(nat, outcome)
        pooled["design"] = "native_control"
        pooled["outcome"] = outcome
        pooled["standardization"] = "pooled"
        moments_rows.extend(pooled.to_dict("records"))
        by_lang = march_standardization_moments_by_lang(cross, outcome)
        for _, row in by_lang.iterrows():
            moments_rows.append(
                {
                    "design": "cross_language",
                    "outcome": outcome,
                    "standardization": "by_lang",
                    "lang_comment": row["lang_comment"],
                    "mu": row["mu"],
                    "sigma": row["sigma"],
                    "n_comments": row["n_comments"],
                }
            )

    moments_path = run_dir / f"standardization_moments_{cohort_label}.csv"
    pd.DataFrame(moments_rows).to_csv(moments_path, index=False)

    audit_rows = []
    for label in ("strict", "loose"):
        th = cohort_thresholds_by_label(label)
        row: Dict[str, Any] = {
            "roster_window": roster_window,
            "cohort_label": label,
            "n_panel_rows": len(panel),
            "n_italian_bilingual_roster": int(
                (roster["author_group"] == "italian_bilingual").sum()
            ),
            "n_native_control_roster": int((roster["author_group"] == "native_control").sum()),
            "n_lang_bilingual_roster": int(
                roster["lang_bilingual"].sum() if "lang_bilingual" in roster.columns else 0
            ),
        }
        for design in PANEL_COHORT_DESIGNS:
            row[f"n_{design}_authors"] = len(cohort_authors_for_design(panel, design, th))
        audit_rows.append(row)
    cohort_path = run_dir / "cohort_audit.csv"
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(cohort_path, index=False)

    strict_cross = int(audit_df.loc[audit_df["cohort_label"] == "strict", "n_cross_language_authors"].iloc[0])
    print(
        f"[prepare_english_quality_panel] roster_window={roster_window} rows={n_rows} "
        f"native_authors={len(native_authors)} cross_authors={len(cross_authors)} "
        f"strict_cross_language={strict_cross} -> {panel_dir}",
        flush=True,
    )
    if strict_cross < CROSS_LANGUAGE_MIN_STRICT_AUTHORS:
        print(
            f"\nSTOP: strict cross_language cohort={strict_cross} "
            f"< {CROSS_LANGUAGE_MIN_STRICT_AUTHORS} under roster_window={roster_window}. "
            "Design too thin once classification is pre-ban only; skipping estimation.",
            flush=True,
        )
        print(f"Cohort audit written to {cohort_path}", flush=True)
        sys.exit(0)
    return panel_dir, n_rows


def main() -> None:
    """Function summary: CLI entry for English-quality panel build."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    build_english_quality_panels(
        config,
        bin_days=args.bin_days,
        min_comment_words=args.min_comment_words,
        cohort_label=args.cohort,
        max_shards=args.max_shards,
        reuse_roster=args.reuse_roster,
        roster_window=args.roster_window,
    )


if __name__ == "__main__":
    main()
