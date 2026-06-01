"""Tests for prepare_wordfish_authors_v2 single-pass shard scan helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnostics import prepare_wordfish_authors_v2 as wfa2


def _write_shard(path: Path, df: pd.DataFrame) -> None:
    """Write a minimal political-universe shard parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def mini_shard_root(tmp_path: Path) -> Path:
    """Two authors, two months, one subreddit."""
    root = tmp_path / "cleaned_monthly_chunks" / "politicaITA"
    rows = [
        {
            "author": "u_it",
            "subreddit": "politicaITA",
            "date_utc": "2023-03-10",
            "body": "italia politica uno",
            "primary_lexicon": "it",
            "comment_in_political_universe": True,
            "is_deleted_author": False,
            "net_ideology": 0.2,
            "sem_axis_ideology": 0.1,
            "n_words": 10,
        },
        {
            "author": "u_en",
            "subreddit": "politicaITA",
            "date_utc": "2023-03-12",
            "body": "english politics two",
            "primary_lexicon": "en",
            "comment_in_political_universe": True,
            "is_deleted_author": False,
            "net_ideology": -0.3,
            "sem_axis_ideology": -0.2,
            "n_words": 12,
        },
        {
            "author": "u_both",
            "subreddit": "politicaITA",
            "date_utc": "2023-03-15",
            "body": "bilingual comment",
            "primary_lexicon": "it",
            "comment_in_political_universe": True,
            "is_deleted_author": False,
            "net_ideology": 0.0,
            "sem_axis_ideology": 0.0,
            "n_words": 8,
        },
        {
            "author": "u_both",
            "subreddit": "politicaITA",
            "date_utc": "2023-04-02",
            "body": "second language later",
            "primary_lexicon": "en",
            "comment_in_political_universe": True,
            "is_deleted_author": False,
            "net_ideology": 0.1,
            "sem_axis_ideology": 0.1,
            "n_words": 9,
        },
    ]
    _write_shard(root / "2023-03.parquet", pd.DataFrame(rows[:3]))
    _write_shard(root / "2023-04.parquet", pd.DataFrame(rows[3:]))
    return tmp_path / "cleaned_monthly_chunks"


def test_pass1_matches_legacy_scan(mini_shard_root: Path) -> None:
    """pass1_author_languages and scan phase 1 yield the same author lexicon sets."""
    subs = ["politicaITA"]
    start, end_excl = "2023-03-01", "2023-05-01"
    langs_legacy, _ = wfa2.pass1_author_languages(
        mini_shard_root, subs, None, start, end_excl
    )
    scan = wfa2.scan_shards_for_wordfish(
        mini_shard_root,
        subs,
        None,
        start,
        end_excl,
        specs=[{"name": "week7", "time_bin": "week", "weekly_bin_days": 7, "min_doc_tokens": 1}],
        wfa_cfg={"ban_anchor_date": "2023-03-31", "filter_comments_to_assigned_lang": True},
        priority=["it", "de", "en"],
        progress=False,
    )
    assert langs_legacy == scan.author_langs


def test_read_parquet_shard_projected_columns(mini_shard_root: Path) -> None:
    """Column projection returns only requested schema fields."""
    shard = mini_shard_root / "politicaITA" / "2023-03.parquet"
    df = wfa2.read_parquet_shard_projected(shard, ("author", "primary_lexicon"))
    assert df is not None
    assert set(df.columns) <= {"author", "primary_lexicon"}


def test_load_assignment_roundtrip(tmp_path: Path) -> None:
    """load_assignment_from_csv reconstructs author maps."""
    path = tmp_path / "wordfish_authors_assignment.csv"
    pd.DataFrame(
        [
            {
                "author": "u1",
                "assigned_primary_lexicon": "it",
                "lexicons_seen": "en;it",
                "cross_language": 1,
            }
        ]
    ).to_csv(path, index=False)
    langs, assigned, _ = wfa2.load_assignment_from_csv(path)
    assert assigned["u1"] == "it"
    assert langs["u1"] == {"en", "it"}
