"""Registry tests for lexical emotion/cognition DiD outcomes."""

from __future__ import annotations

from src.did.outcomes import OUTCOME_REGISTRY


def test_lexical_emotion_cognition_forum_specs() -> None:
    """Forum subreddit-day specs use *_mean columns from descriptives."""
    by_id = {o.outcome_id: o for o in OUTCOME_REGISTRY if o.family == "lexical"}
    for oid, col in (
        ("emotion_rate", "emotion_rate_100w_mean"),
        ("cognition_rate", "cognition_rate_100w_mean"),
    ):
        assert oid in by_id
        assert by_id[oid].column == col
        assert by_id[oid].panel_kind == "subreddit_day"


def test_lexical_emotion_cognition_comment_specs() -> None:
    """Comment-level specs use raw shard column names."""
    by_id = {o.outcome_id: o for o in OUTCOME_REGISTRY if o.family == "lexical_comment"}
    for oid, col in (
        ("emotion_rate", "emotion_rate_100w"),
        ("cognition_rate", "cognition_rate_100w"),
    ):
        assert oid in by_id
        assert by_id[oid].column == col
        assert by_id[oid].panel_kind == "comment"
