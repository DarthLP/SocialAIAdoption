"""Unit tests for semantic-axis aggregation in user-week panel build."""

from __future__ import annotations

import pandas as pd

from scripts.user_week.prepare_user_week_style_panel import (
    aggregate_shard_to_user_week_subreddit,
    merge_user_week_subreddit_rows,
    select_shard_columns,
)


def test_sem_axis_mean_aggregation_excludes_nan_comments() -> None:
    """Function summary: word-weighted sem_axis mean uses only scored comments in n."""
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "subreddit": ["it", "it", "it"],
            "author": ["u1", "u1", "u1"],
            "created_utc": [1_680_508_800, 1_680_595_200, 1_680_681_600],
            "n_words_comment": [100.0, 100.0, 100.0],
            "has_sem_axis": [1, 1, 0],
            "sem_axis_ideology": [0.2, float("nan"), 0.9],
            "total_word_chars_comment": [400.0, 400.0, 400.0],
            "sentence_count_comment": [5.0, 5.0, 5.0],
        }
    )
    config: dict = {}
    slim = select_shard_columns(frame, config)
    inter = aggregate_shard_to_user_week_subreddit(slim, subreddit="it", config=config)
    panel = merge_user_week_subreddit_rows(inter, {"it": "it_political"}, config)
    assert not panel.empty
    assert panel.loc[0, "sem_axis_ideology_n"] == 2
    assert abs(panel.loc[0, "sem_axis_ideology_mean"] - 0.55) < 1e-9
    assert abs(panel.loc[0, "share_scored"] - 2 / 3) < 1e-9
