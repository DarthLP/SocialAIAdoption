"""
Script summary:
This script measures how many users post on the same UTC day in two or more
different project forums. It complements `scripts/user_overlap_across_forums.py`
(which only checks whether users appear in multiple forums anywhere in the
window) by enforcing temporal co-activity: author X is only counted if, on at
least one specific UTC date, X commented in at least two distinct subreddits.

Functionality:
- Recursively scans `data/raw/political_forums/daily_chunks/<subreddit>/*.ndjson`.
- For each comment, builds a mapping (author, utc_date) -> set of subreddits.
- Optionally excludes removed accounts (`[deleted]`) and known bot accounts.
- Reports:
    * authors who ever had same-day activity in >=2 forums (count and share),
    * number of (author, day) events that span >=2 forums,
    * distribution of max same-day forum count per author,
    * pairwise forum co-activity counts (shared same-day (user, day) events).
- Writes three CSV summaries under `results/tables/user_overlap/`:
    * `user_same_day_cross_forum_summary.csv`
    * `user_same_day_cross_forum_distribution.csv`
    * `user_same_day_cross_forum_pairwise.csv`

How to run:
- `.venv/bin/python scripts/user_same_day_cross_forum.py --config config/political_forums_setup.yaml`
- Add `--include-deleted` to count `[deleted]` as a pseudo-user.
- Add `--include-bots` to keep known bot accounts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config

DEFAULT_BOT_AUTHORS = {
    "AutoModerator",
    "PoliticsModeratorBot",
    "politicsmoderatorbot",
    "ModeratorOfPolitics",
    "SnapshillBot",
    "WikiTextBot",
    "RemindMeBot",
    "sneakpeekbot",
    "converter-bot",
}
OVERLAP_TABLES_SUBDIR = "user_overlap"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for config path and inclusion toggles."""
    parser = argparse.ArgumentParser(
        description="Count users active in >=2 forums on the same UTC day."
    )
    parser.add_argument(
        "--config",
        default="config/political_forums_setup.yaml",
        help="Path to project YAML config.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include '[deleted]' as a pseudo-user (default: excluded).",
    )
    parser.add_argument(
        "--include-bots",
        action="store_true",
        help="Include known bot accounts (default: excluded).",
    )
    return parser.parse_args()


def iter_ndjson_records(path: Path) -> Iterable[Tuple[str, int]]:
    """Function summary: yield (author, created_utc) tuples from an NDJSON file.

    Parameters:
        path: Path to an NDJSON file where each line is a JSON comment object.

    Yields:
        Tuples of (author, created_utc_seconds_int) for valid records only.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = rec.get("author")
            created = rec.get("created_utc")
            if not isinstance(author, str) or not author:
                continue
            if not isinstance(created, (int, float)):
                continue
            yield author, int(created)


def build_same_day_index(
    daily_chunks_dir: Path,
    include_deleted: bool,
    include_bots: bool,
    bot_list: Set[str],
) -> Dict[Tuple[str, str], Set[str]]:
    """Function summary: build a (author, utc_date) -> set of subreddits index.

    Parameters:
        daily_chunks_dir: Directory containing one subdirectory per subreddit.
        include_deleted: If False, drop `[deleted]` author.
        include_bots: If False, drop authors listed in `bot_list`.
        bot_list: Set of bot usernames to exclude when `include_bots` is False.

    Returns:
        Dict mapping (author, 'YYYY-MM-DD') -> set of subreddits the author
        commented in on that UTC day.
    """
    index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for forum_dir in sorted(p for p in daily_chunks_dir.iterdir() if p.is_dir()):
        forum = forum_dir.name
        files = sorted(forum_dir.glob("*.ndjson"))
        n_records = 0
        for f in files:
            file_date_fallback = f.stem
            for author, created in iter_ndjson_records(f):
                if not include_deleted and author == "[deleted]":
                    continue
                if not include_bots and author in bot_list:
                    continue
                try:
                    day = datetime.fromtimestamp(created, tz=timezone.utc).strftime(
                        "%Y-%m-%d"
                    )
                except (OverflowError, OSError, ValueError):
                    day = file_date_fallback
                index[(author, day)].add(forum)
                n_records += 1
        print(
            f"  {forum}: processed {n_records:,} records from {len(files)} files",
            flush=True,
        )
    return index


def write_summary_table(
    total_unique_authors: int,
    authors_same_day_multi: int,
    same_day_multi_events: int,
    total_author_day_events: int,
    out_path: Path,
) -> None:
    """Function summary: write top-level same-day cross-forum summary stats.

    Parameters:
        total_unique_authors: Unique authors after filtering.
        authors_same_day_multi: Authors with >=1 same-day cross-forum event.
        same_day_multi_events: Count of (author, day) pairs spanning >=2 forums.
        total_author_day_events: Total (author, day) pairs observed.
        out_path: Output CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    share_authors = (
        authors_same_day_multi / total_unique_authors if total_unique_authors else 0.0
    )
    share_events = (
        same_day_multi_events / total_author_day_events
        if total_author_day_events
        else 0.0
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "share"])
        writer.writerow(["unique_authors_total", total_unique_authors, "1.0000"])
        writer.writerow(
            [
                "authors_with_same_day_multi_forum_activity",
                authors_same_day_multi,
                f"{share_authors:.4f}",
            ]
        )
        writer.writerow(
            ["total_author_day_events", total_author_day_events, "1.0000"]
        )
        writer.writerow(
            [
                "author_day_events_spanning_multiple_forums",
                same_day_multi_events,
                f"{share_events:.4f}",
            ]
        )


def write_distribution_table(
    author_max_same_day_forums: Dict[str, int],
    num_forums: int,
    out_path: Path,
) -> None:
    """Function summary: write distribution of each author's max same-day forum count.

    Parameters:
        author_max_same_day_forums: Map author -> largest k such that the author
            posted in k distinct forums on at least one UTC day.
        num_forums: Total number of forums considered.
        out_path: Output CSV path.
    """
    counts = Counter(author_max_same_day_forums.values())
    total_authors = sum(counts.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "max_forums_on_single_day",
                "num_authors",
                "share_of_unique_authors",
            ]
        )
        for k in range(1, num_forums + 1):
            n = counts.get(k, 0)
            share = n / total_authors if total_authors else 0.0
            writer.writerow([k, n, f"{share:.4f}"])


def write_pairwise_table(
    pairwise_author_day_counts: Dict[Tuple[str, str], int],
    pairwise_author_counts: Dict[Tuple[str, str], int],
    forums_sorted: list[str],
    out_path: Path,
) -> None:
    """Function summary: write pairwise same-day co-activity counts between forums.

    Parameters:
        pairwise_author_day_counts: Map (forum_a, forum_b) -> count of
            (author, day) events where the author posted in both forums that day.
        pairwise_author_counts: Map (forum_a, forum_b) -> count of distinct
            authors with at least one such same-day event in both forums.
        forums_sorted: Sorted list of forum names used to produce stable pairs.
        out_path: Output CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "forum_a",
                "forum_b",
                "same_day_author_day_events",
                "distinct_authors_with_same_day_event",
            ]
        )
        for a, b in combinations(forums_sorted, 2):
            key = (a, b)
            writer.writerow(
                [
                    a,
                    b,
                    pairwise_author_day_counts.get(key, 0),
                    pairwise_author_counts.get(key, 0),
                ]
            )


def main() -> None:
    """Function summary: compute same-day cross-forum user activity and write CSVs."""
    args = parse_args()
    config = load_config(args.config)

    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    daily_chunks_dir = raw_dir / "daily_chunks"
    tables_dir = PROJECT_ROOT / config["paths"]["tables_dir"]
    overlap_tables_dir = tables_dir / OVERLAP_TABLES_SUBDIR

    if not daily_chunks_dir.exists():
        raise FileNotFoundError(f"Daily chunks dir not found: {daily_chunks_dir}")

    bot_list = DEFAULT_BOT_AUTHORS if not args.include_bots else set()

    print(f"Scanning {daily_chunks_dir} ...", flush=True)
    index = build_same_day_index(
        daily_chunks_dir=daily_chunks_dir,
        include_deleted=args.include_deleted,
        include_bots=args.include_bots,
        bot_list=bot_list,
    )

    forums_sorted = sorted(
        p.name for p in daily_chunks_dir.iterdir() if p.is_dir()
    )

    unique_authors: Set[str] = set()
    author_max_same_day_forums: Dict[str, int] = defaultdict(int)
    authors_with_same_day_multi: Set[str] = set()
    same_day_multi_events = 0
    total_author_day_events = len(index)

    pairwise_event_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pairwise_author_sets: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for (author, _day), forums_today in index.items():
        unique_authors.add(author)
        k = len(forums_today)
        if k > author_max_same_day_forums[author]:
            author_max_same_day_forums[author] = k
        if k >= 2:
            same_day_multi_events += 1
            authors_with_same_day_multi.add(author)
            forum_list = sorted(forums_today)
            for a, b in combinations(forum_list, 2):
                pairwise_event_counts[(a, b)] += 1
                pairwise_author_sets[(a, b)].add(author)

    pairwise_author_counts = {k: len(v) for k, v in pairwise_author_sets.items()}

    total_unique_authors = len(unique_authors)
    authors_same_day_multi = len(authors_with_same_day_multi)
    share = (
        authors_same_day_multi / total_unique_authors if total_unique_authors else 0.0
    )

    print("\n=== Same-day cross-forum user activity ===")
    print(f"Forums: {forums_sorted}")
    print(f"Unique authors (filtered): {total_unique_authors:,}")
    print(
        f"Authors posting in >=2 forums on the SAME UTC day (at least once): "
        f"{authors_same_day_multi:,} ({share*100:.2f}% of unique authors)"
    )
    print(
        f"(author, day) events spanning >=2 forums: {same_day_multi_events:,} "
        f"out of {total_author_day_events:,} total author-day events"
    )
    dist = Counter(author_max_same_day_forums.values())
    for k in sorted(dist):
        print(f"  max {k} forum(s) on a single day: {dist[k]:,} authors")

    summary_path = overlap_tables_dir / "user_same_day_cross_forum_summary.csv"
    distribution_path = overlap_tables_dir / "user_same_day_cross_forum_distribution.csv"
    pairwise_path = overlap_tables_dir / "user_same_day_cross_forum_pairwise.csv"

    write_summary_table(
        total_unique_authors=total_unique_authors,
        authors_same_day_multi=authors_same_day_multi,
        same_day_multi_events=same_day_multi_events,
        total_author_day_events=total_author_day_events,
        out_path=summary_path,
    )
    write_distribution_table(
        author_max_same_day_forums=author_max_same_day_forums,
        num_forums=len(forums_sorted),
        out_path=distribution_path,
    )
    write_pairwise_table(
        pairwise_author_day_counts=pairwise_event_counts,
        pairwise_author_counts=pairwise_author_counts,
        forums_sorted=forums_sorted,
        out_path=pairwise_path,
    )

    print("\nWrote:")
    print(f"  {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"  {distribution_path.relative_to(PROJECT_ROOT)}")
    print(f"  {pairwise_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
