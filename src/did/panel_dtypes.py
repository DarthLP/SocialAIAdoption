"""
Compact dtypes for comment-level DiD panels (lower RAM, faster groupby/feols).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def compact_comment_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: cast comment panel to memory-efficient dtypes in place.

    Parameters:
    - df: annotated comment panel (author, time_id, outcomes, DiD flags).

    Returns:
    - Same DataFrame with categorical / narrow numeric columns where safe.
    """
    if df.empty:
        return df
    cat_cols: Sequence[str] = (
        "author",
        "subreddit",
        "time_id",
        "topic_family",
        "primary_lexicon",
        "date_utc",
        "id",
    )
    for col in cat_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category")
    int_cols = ("post", "IT", "rel_day", "rel_period")
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int16")
    for col in ("net_ideology", "y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df
