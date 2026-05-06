"""
Script summary:
This script measures whether the same Reddit users appear across multiple
project forums (subreddits) in the filtered daily-chunk dataset. Because
Reddit's `author` field is a globally-unique username, cross-forum user
matching is an exact string match on `author`.

Functionality:
- Recursively scans `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/*.parquet`.
- Builds a mapping of author -> set of subreddits the author commented in.
- Optionally excludes removed accounts (`[deleted]`) and known bots.
- Reports:
    * total unique authors across all forums,
    * unique authors per forum,
    * how many authors appear in exactly 1, 2, 3, ... forums,
    * per-forum count of authors who also appear in at least one other forum,
    * pairwise forum overlap matrix (intersection sizes + Jaccard).
- Writes three CSV summaries under `results/tables/user_overlap/`:
    * `user_overlap_by_forum.csv`
    * `user_overlap_forum_count_distribution.csv`
    * `user_overlap_pairwise.csv`

How to run:
- `.venv/bin/python scripts/diagnostics/user_overlap_across_forums.py --config config/political_forums_setup.yaml`
- Add `--include-deleted` to count `[deleted]` as a single pseudo-user.
- Add `--include-bots` to keep known bot accounts (AutoModerator etc.).
"""

from __future__ import annotations

import argparse
import csv
import sys
import pandas as pd

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import importlib.util
from typing import Dict, Iterable, Set

def _resolve_project_root() -> Path:
    """Load scripts/_project_root.py and return the repository root Path."""
    _scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod",
        _scripts_dir / "_project_root.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
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
    "FuckThesePeople",  # placeholder, keep off unless confirmed bot
}
OVERLAP_TABLES_SUBDIR = "user_overlap"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for config path and inclusion toggles."""
    parser = argparse.ArgumentParser(
        description="Compute cross-forum user overlap from cleaned daily chunks."
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


def iter_parquet_authors(path: Path) -> Iterable[str]:
    """Function summary: yield each record's `author` field from a monthly Parquet file.

    Parameters:
        path: Path to a Parquet file with cleaned comment rows.

    Yields:
        The value of the `author` key for each record (skips malformed lines).
    """
    frame = pd.read_parquet(path, columns=["author"])
    for author in frame["author"].astype("string").dropna().tolist():
        if isinstance(author, str) and author:
            yield author


def collect_authors_by_forum(
    daily_chunks_dir: Path,
    include_deleted: bool,
    include_bots: bool,
    bot_list: Set[str],
) -> Dict[str, Set[str]]:
    """Function summary: walk all forum subdirectories and collect author sets per forum.

    Parameters:
        daily_chunks_dir: Directory containing one subdirectory per subreddit.
        include_deleted: If False, drop `[deleted]` author.
        include_bots: If False, drop authors listed in `bot_list`.
        bot_list: Set of bot usernames to exclude when `include_bots` is False.

    Returns:
        Dict mapping subreddit name -> set of unique authors observed.
    """
    forums: Dict[str, Set[str]] = {}
    for forum_dir in sorted(p for p in daily_chunks_dir.iterdir() if p.is_dir()):
        authors: Set[str] = set()
        files = sorted(forum_dir.glob("*.parquet"))
        for f in files:
            for author in iter_parquet_authors(f):
                if not include_deleted and author == "[deleted]":
                    continue
                if not include_bots and author in bot_list:
                    continue
                authors.add(author)
        forums[forum_dir.name] = authors
        print(
            f"  {forum_dir.name}: {len(authors):,} unique authors "
            f"from {len(files)} files",
            flush=True,
        )
    return forums


def write_per_forum_table(
    forums: Dict[str, Set[str]],
    author_forum_counts: Dict[str, int],
    out_path: Path,
) -> None:
    """Function summary: write a per-forum summary CSV with overlap counts.

    Parameters:
        forums: Map of subreddit -> set of authors.
        author_forum_counts: Map of author -> number of forums they appear in.
        out_path: Output CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subreddit",
                "unique_authors",
                "authors_also_in_other_forums",
                "share_also_in_other_forums",
            ]
        )
        for forum in sorted(forums):
            authors = forums[forum]
            n = len(authors)
            overlap = sum(1 for a in authors if author_forum_counts.get(a, 0) > 1)
            share = overlap / n if n else 0.0
            writer.writerow([forum, n, overlap, f"{share:.4f}"])


def write_distribution_table(
    author_forum_counts: Dict[str, int],
    num_forums: int,
    out_path: Path,
) -> None:
    """Function summary: write a CSV with how many authors appear in k forums for k=1..N.

    Parameters:
        author_forum_counts: Map of author -> number of forums they appear in.
        num_forums: Total number of forums considered.
        out_path: Output CSV path.
    """
    counts = Counter(author_forum_counts.values())
    total_authors = sum(counts.values())
    multi_forum = sum(v for k, v in counts.items() if k >= 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["forums_used", "num_authors", "share_of_all_authors"])
        for k in range(1, num_forums + 1):
            n = counts.get(k, 0)
            share = n / total_authors if total_authors else 0.0
            writer.writerow([k, n, f"{share:.4f}"])
        writer.writerow([])
        writer.writerow(["total_unique_authors", total_authors, "1.0000"])
        share_multi = multi_forum / total_authors if total_authors else 0.0
        writer.writerow(["authors_in_more_than_one_forum", multi_forum, f"{share_multi:.4f}"])


def write_pairwise_table(forums: Dict[str, Set[str]], out_path: Path) -> None:
    """Function summary: write pairwise intersection sizes and Jaccard similarity for forums.

    Parameters:
        forums: Map of subreddit -> set of authors.
        out_path: Output CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "forum_a",
                "forum_b",
                "authors_a",
                "authors_b",
                "shared_authors",
                "jaccard",
            ]
        )
        for a, b in combinations(sorted(forums), 2):
            sa, sb = forums[a], forums[b]
            inter = len(sa & sb)
            union = len(sa | sb)
            jacc = inter / union if union else 0.0
            writer.writerow([a, b, len(sa), len(sb), inter, f"{jacc:.4f}"])


def main() -> None:
    """Function summary: run cross-forum user overlap analysis and write CSV outputs."""
    args = parse_args()
    config = load_config(args.config)

    interim_dir = PROJECT_ROOT / config["paths"]["interim_dir"]
    daily_chunks_dir = interim_dir / "cleaned_monthly_chunks"
    tables_dir = PROJECT_ROOT / config["paths"]["tables_dir"]
    overlap_tables_dir = tables_dir / OVERLAP_TABLES_SUBDIR

    if not daily_chunks_dir.exists():
        raise FileNotFoundError(f"Daily chunks dir not found: {daily_chunks_dir}")

    bot_list = DEFAULT_BOT_AUTHORS if not args.include_bots else set()

    print(f"Scanning {daily_chunks_dir} ...", flush=True)
    forums = collect_authors_by_forum(
        daily_chunks_dir=daily_chunks_dir,
        include_deleted=args.include_deleted,
        include_bots=args.include_bots,
        bot_list=bot_list,
    )

    author_forum_counts: Dict[str, int] = defaultdict(int)
    for authors in forums.values():
        for a in authors:
            author_forum_counts[a] += 1

    total_unique = len(author_forum_counts)
    multi_forum = sum(1 for c in author_forum_counts.values() if c >= 2)
    share_multi = (multi_forum / total_unique) if total_unique else 0.0

    print("\n=== Cross-forum user overlap ===")
    print(f"Forums scanned: {len(forums)} -> {sorted(forums)}")
    print(f"Exclude [deleted]: {not args.include_deleted}")
    print(f"Exclude bots:     {not args.include_bots}")
    print(f"Total unique authors: {total_unique:,}")
    print(
        f"Authors in >1 forum:  {multi_forum:,} "
        f"({share_multi*100:.2f}% of unique authors)"
    )

    k_counts = Counter(author_forum_counts.values())
    for k in sorted(k_counts):
        print(f"  used {k} forum(s): {k_counts[k]:,}")

    per_forum_path = overlap_tables_dir / "user_overlap_by_forum.csv"
    distribution_path = overlap_tables_dir / "user_overlap_forum_count_distribution.csv"
    pairwise_path = overlap_tables_dir / "user_overlap_pairwise.csv"

    write_per_forum_table(forums, author_forum_counts, per_forum_path)
    write_distribution_table(author_forum_counts, len(forums), distribution_path)
    write_pairwise_table(forums, pairwise_path)

    print("\nWrote:")
    print(f"  {per_forum_path.relative_to(PROJECT_ROOT)}")
    print(f"  {distribution_path.relative_to(PROJECT_ROOT)}")
    print(f"  {pairwise_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
