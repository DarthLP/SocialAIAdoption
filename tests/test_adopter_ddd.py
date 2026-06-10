"""Tests for adopter-flag triple-diff estimation."""

from __future__ import annotations

import pandas as pd

from src.did.bucket_estimate import estimate_adopter_ddd_static


def _synthetic_comment_panel() -> pd.DataFrame:
    """Function summary: minimal comment panel with DDD variation."""
    rows = []
    for it, post, flag in (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ):
        for i in range(8):
            rows.append(
                {
                    "author": f"a{it}_{post}_{flag}_{i}",
                    "date_utc": "2023-04-01" if post else "2023-03-15",
                    "topic_family": "it_political" if it else "de",
                    "post": float(post),
                    "IT": float(it),
                    "flag": float(flag),
                    "style_index_llm": 0.1 + 0.2 * post + 0.3 * it + 0.5 * post * it * flag,
                    "subreddit": "x",
                    "time_id": "2023-04-01" if post else "2023-03-15",
                }
            )
    return pd.DataFrame(rows)


def test_adopter_ddd_returns_cell_counts() -> None:
    """Function summary: DDD estimator reports IT×post×flag cell sizes."""
    panel = _synthetic_comment_panel()
    res = estimate_adopter_ddd_static(panel, y_col="style_index_llm", flag_col="flag")
    assert res.get("cell_IT1_post1_flag1", 0) > 0
    assert "n_obs" in res
