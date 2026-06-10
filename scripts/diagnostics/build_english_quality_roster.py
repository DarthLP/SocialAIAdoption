"""
Script summary:
Build the author roster for within-author English-quality DiD analysis.

Functionality:
- Scans enriched monthly Parquet shards under cleaned_monthly_chunks/.
- Maps each author to the set of subreddits they commented in (Mar–Apr event window)
  and tallies per-author EN/IT comment counts split at the ban launch.
- Classifies authors as italian_bilingual (Italian + English-control forums),
  native_control (English-control only), or other, and attaches a native-language
  proxy (italian_share_pre, dominant_pre_lang) and a forum-agnostic lang_bilingual flag.
- Writes author_roster.csv and a summary CSV under did/english_quality/.

How to apply/run:
  .venv/bin/python scripts/diagnostics/build_english_quality_roster.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/build_english_quality_roster.py --max-shards 2
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import load_config, resolve_primary_subreddits  # noqa: E402
from src.did.english_quality import (  # noqa: E402
    DEFAULT_BOT_AUTHORS,
    ROSTER_WINDOW_CHOICES,
    classify_author_roster,
    english_quality_run_tables_dir,
    roster_summary,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for roster build.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Build English-quality author roster.")
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--include-deleted", action="store_true")
    parser.add_argument("--include-bots", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--roster-window",
        default="pre_ban",
        choices=ROSTER_WINDOW_CHOICES,
        help="pre_ban: classify forum membership and lang_bilingual from pre-launch comments only; "
        "full: use the entire Mar–Apr event window (legacy comparison).",
    )
    return parser.parse_args()


def _empty_lang_stats() -> Dict[str, int]:
    """Function summary: zero-initialised per-author language tally."""
    return {"pre_en": 0, "pre_it": 0, "pre_total": 0, "tot_en": 0, "tot_it": 0}


def collect_author_activity(
    interim_dir: Path,
    subs: list[str],
    start: str,
    end_excl: str,
    launch: str,
    *,
    include_deleted: bool,
    include_bots: bool,
    max_shards: int | None,
    roster_window: str = "pre_ban",
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    """Function summary: scan shards for author forums and EN/IT language tallies.

    Parameters:
    - interim_dir: data/interim/<study>/ root.
    - subs: primary subreddit list.
    - start: event window start (YYYY-MM-DD).
    - end_excl: event window end exclusive.
    - launch: ban launch day (rows with date_utc < launch are pre-ban).
    - include_deleted: keep [deleted] pseudo-user.
    - include_bots: keep bot accounts.
    - max_shards: optional per-subreddit shard cap.
    - roster_window: pre_ban restricts forum membership and tot_en/tot_it to
      comments with date_utc < launch; full uses the entire event window.

    Returns:
    - Tuple (author_forums, author_lang) where author_forums maps author -> set of
      subreddits and author_lang maps author -> {pre_en, pre_it, pre_total,
      tot_en, tot_it} comment counts.
    """
    author_forums: Dict[str, Set[str]] = defaultdict(set)
    author_lang: Dict[str, Dict[str, int]] = defaultdict(_empty_lang_stats)
    bot_list = set() if include_bots else set(DEFAULT_BOT_AUTHORS)
    for sub in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards:
            shards = shards[:max_shards]
        for shard in shards:
            try:
                frame = pd.read_parquet(shard, columns=["author", "date_utc", "lang_comment"])
            except Exception:
                try:
                    frame = pd.read_parquet(shard, columns=["author", "date_utc"])
                    frame["lang_comment"] = ""
                except Exception:
                    continue
            if frame.empty:
                continue
            frame["date_utc"] = frame["date_utc"].astype(str)
            frame = frame[(frame["date_utc"] >= start) & (frame["date_utc"] < end_excl)]
            frame["author"] = frame["author"].astype("string")
            frame = frame[frame["author"].notna() & (frame["author"] != "")]
            if not include_deleted:
                frame = frame[frame["author"] != "[deleted]"]
            if bot_list:
                frame = frame[~frame["author"].isin(bot_list)]
            if frame.empty:
                continue
            frame["author"] = frame["author"].astype(str)
            is_pre = frame["date_utc"] < launch
            if roster_window == "pre_ban":
                classify_mask = is_pre
            else:
                classify_mask = pd.Series(True, index=frame.index)
            classify = frame.loc[classify_mask]
            for author in classify["author"].unique().tolist():
                author_forums[author].add(sub)
            lang = frame.get("lang_comment", pd.Series("", index=frame.index)).astype(str).str.lower()
            is_en = lang == "en"
            is_it = lang == "it"
            tmp = pd.DataFrame(
                {
                    "author": frame["author"].values,
                    "pre_total": is_pre.astype(int).values,
                    "pre_en": (is_pre & is_en).astype(int).values,
                    "pre_it": (is_pre & is_it).astype(int).values,
                    "tot_en": (classify_mask & is_en).astype(int).values,
                    "tot_it": (classify_mask & is_it).astype(int).values,
                }
            )
            agg = tmp.groupby("author")[["pre_total", "pre_en", "pre_it", "tot_en", "tot_it"]].sum()
            for author, row in agg.iterrows():
                stats = author_lang[str(author)]
                stats["pre_total"] += int(row["pre_total"])
                stats["pre_en"] += int(row["pre_en"])
                stats["pre_it"] += int(row["pre_it"])
                stats["tot_en"] += int(row["tot_en"])
                stats["tot_it"] += int(row["tot_it"])
    return dict(author_forums), dict(author_lang)


def collect_author_forums(
    interim_dir: Path,
    subs: list[str],
    start: str,
    end_excl: str,
    *,
    include_deleted: bool,
    include_bots: bool,
    max_shards: int | None,
    roster_window: str = "pre_ban",
) -> Dict[str, Set[str]]:
    """Function summary: map author -> subreddits (forum-only, no language tally).

    Backward-compatible wrapper around collect_author_activity that returns only
    the forum membership map; uses end_excl as the pre/post split (irrelevant here).

    Parameters:
    - interim_dir: data/interim/<study>/ root.
    - subs: primary subreddit list.
    - start: event window start (YYYY-MM-DD).
    - end_excl: event window end exclusive.
    - include_deleted: keep [deleted] pseudo-user.
    - include_bots: keep bot accounts.
    - max_shards: optional per-subreddit shard cap.
    - roster_window: pre_ban or full (see collect_author_activity).

    Returns:
    - Dict mapping author id -> set of subreddit names.
    """
    author_forums, _ = collect_author_activity(
        interim_dir,
        subs,
        start,
        end_excl,
        end_excl,
        include_deleted=include_deleted,
        include_bots=include_bots,
        max_shards=max_shards,
        roster_window=roster_window,
    )
    return author_forums


def main() -> None:
    """Function summary: build roster CSV and print group counts."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    start, end_excl, launch, _ = event_dates_from_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    subs = resolve_primary_subreddits(config)

    print(
        f"Scanning author forum membership + language tallies "
        f"({start} .. {end_excl}, roster_window={args.roster_window}) ...",
        flush=True,
    )
    author_forums, author_lang = collect_author_activity(
        interim_dir,
        subs,
        start,
        end_excl,
        launch,
        include_deleted=args.include_deleted,
        include_bots=args.include_bots,
        max_shards=args.max_shards,
        roster_window=args.roster_window,
    )
    roster = classify_author_roster(author_forums, config, author_lang=author_lang)
    counts = roster_summary(roster)

    out_dir = english_quality_run_tables_dir(config, args.roster_window)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster_path = out_dir / "author_roster.csv"
    summary_path = out_dir / "author_roster_summary.csv"
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {"roster_window": args.roster_window, "author_group": k, "n_authors": v}
            for k, v in sorted(counts.items())
        ]
    ).to_csv(summary_path, index=False)

    n_bilingual = counts.get("italian_bilingual", 0)
    n_lang_bilingual = int(roster["lang_bilingual"].sum()) if "lang_bilingual" in roster.columns else 0
    print(f"\n=== English-quality author roster (roster_window={args.roster_window}) ===")
    for group, n in sorted(counts.items()):
        print(f"  {group}: {n:,}")
    print(f"\nitalian_bilingual count: {n_bilingual:,} (target ~1,884)")
    print(f"lang_bilingual (forum-agnostic EN+IT writers): {n_lang_bilingual:,}")
    print(f"Wrote {roster_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
