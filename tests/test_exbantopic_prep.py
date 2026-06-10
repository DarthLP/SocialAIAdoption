"""Smoke tests for ban-topic exclusion prep helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnostics.prepare_polarization_descriptives import (
    _apply_ban_topic_exclusion,
    _csv_out_name,
    _ensure_ban_topic_column,
)
from src.ban_topic import is_ban_topic_text


def test_csv_out_name_exbantopic_suffix():
    """Parallel descriptives files insert suffix before .csv."""
    assert _csv_out_name("daily_by_subreddit.csv", "_exbantopic") == "daily_by_subreddit_exbantopic.csv"
    assert _csv_out_name("daily_by_subreddit.csv", "") == "daily_by_subreddit.csv"


def test_exclusion_drops_flagged_rows():
    """Exclusion keeps only non-ban-topic comments."""
    df = pd.DataFrame(
        {
            "body": ["normal politics", "ChatGPT ban discussion"],
            "is_ban_topic": [False, True],
            "n_words": [10, 12],
        }
    )
    out = _apply_ban_topic_exclusion(df)
    assert len(out) == 1
    assert "ChatGPT" not in str(out["body"].iloc[0])


def test_ensure_ban_topic_from_body_when_column_missing():
    """Fallback regex on body when shard column absent."""
    df = pd.DataFrame({"body": ["nothing here", "OpenAI privacy ban"]})
    out = _ensure_ban_topic_column(df)
    assert out["is_ban_topic"].tolist() == [False, True]


def test_ensure_ban_topic_no_false_positive_from_nan_or_missing_column():
    """Mixed shards: no True without regex match; screening-excluded forums stay False."""
    flagged = pd.DataFrame(
        {
            "body": ["normal politics", "ChatGPT ban discussion"],
            "is_ban_topic": [False, True],
            "subreddit": ["Italia", "Italia"],
        }
    )
    unflagged = pd.DataFrame(
        {
            "body": ["milano politics", "torino news"],
            "subreddit": ["milano", "torino"],
        }
    )
    nan_shard = pd.DataFrame(
        {
            "body": ["another normal", "OpenAI discussion"],
            "is_ban_topic": [np.nan, np.nan],
            "subreddit": ["bologna", "bologna"],
        }
    )
    combined = pd.concat([flagged, unflagged, nan_shard], ignore_index=True)
    out = _ensure_ban_topic_column(combined)

    for _, row in out.iterrows():
        if row["is_ban_topic"]:
            assert is_ban_topic_text(str(row["body"]))

    screening = out[out["subreddit"].isin(["milano", "torino"])]
    assert not screening["is_ban_topic"].any()
    assert out.loc[out["body"] == "another normal", "is_ban_topic"].iloc[0] is np.False_
