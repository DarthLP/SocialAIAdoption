"""Unit tests for comment-level political universe (tree propagation)."""

from __future__ import annotations

import pandas as pd

from src.political_filter import (  # noqa: E402
    apply_all_modes,
    comment_id_key,
    parent_comment_key,
)

ROOT = __file__


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build minimal comment frame for tree tests."""
    return pd.DataFrame(rows)


def test_parent_id_pandas_na() -> None:
    """parent_comment_key accepts pandas NA without raising."""
    import pandas as pd

    from src.political_filter import apply_all_modes, parent_comment_key

    assert parent_comment_key(pd.NA) is None
    df = _frame(
        [
            {
                "id": "x1",
                "parent_id": pd.NA,
                "link_id": "t3_sub",
                "political_weighted_points": 0,
                "n_words": 5,
            },
        ]
    )
    out, _ = apply_all_modes(
        df,
        {"mode": "tree", "comment_political_min_points": 3, "tree_include_parent": True},
        {"thread_political_min_points": 3, "thread_political_rate_threshold": 0.45},
    )
    assert len(out) == 1


def test_id_normalization() -> None:
    """Bare id and t1_ parent normalize to the same key."""
    assert comment_id_key("jaf53b5") == "jaf53b5"
    assert parent_comment_key("t1_jaf53b5") == "jaf53b5"
    assert parent_comment_key("t3_11ckeba") is None


def test_tree_spec_example_b_political() -> None:
    """B political -> A,B,C,D,E in universe; F,G out (plan A-G tree)."""
    df = _frame(
        [
            {"id": "A", "parent_id": "t3_sub", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 5},
            {"id": "B", "parent_id": "t1_A", "link_id": "t3_sub", "political_weighted_points": 3, "n_words": 5},
            {"id": "C", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
            {"id": "D", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
            {"id": "E", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
            {"id": "F", "parent_id": "t1_A", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
            {"id": "G", "parent_id": "t1_F", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
        ]
    )
    pu_cfg = {
        "mode": "tree",
        "comment_political_min_points": 3,
        "tree_include_parent": True,
        "tree_max_depth": None,
    }
    screening = {"thread_political_min_points": 3, "thread_political_rate_threshold": 0.45}
    out, _ = apply_all_modes(df, pu_cfg, screening)
    tree = set(out.loc[out["in_political_universe_tree"], "id"])
    assert tree == {"A", "B", "C", "D", "E"}


def test_tree_spec_example_c_political() -> None:
    """C political -> B,C in universe; D,E out when B not political."""
    df = _frame(
        [
            {"id": "A", "parent_id": "t3_sub", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 5},
            {"id": "B", "parent_id": "t3_sub", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 5},
            {"id": "C", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 3, "n_words": 3},
            {"id": "D", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
            {"id": "E", "parent_id": "t1_B", "link_id": "t3_sub", "political_weighted_points": 0, "n_words": 3},
        ]
    )
    pu_cfg = {
        "mode": "tree",
        "comment_political_min_points": 3,
        "tree_include_parent": True,
        "tree_max_depth": None,
    }
    screening = {"thread_political_min_points": 3, "thread_political_rate_threshold": 0.45}
    out, _ = apply_all_modes(df, pu_cfg, screening)
    tree = set(out.loc[out["in_political_universe_tree"], "id"])
    assert tree == {"B", "C"}
