"""Registry tests for extended semantic-axis DiD outcomes."""

from __future__ import annotations

from src.did.outcomes import OUTCOME_REGISTRY
from src.embeddings import EXTENDED_AXIS_NAMES


def test_extended_semantic_subreddit_day_specs() -> None:
    """Extended axes register as semantic_axis family with _mean columns."""
    by_id = {o.outcome_id: o for o in OUTCOME_REGISTRY if o.family == "semantic_axis"}
    for axis in EXTENDED_AXIS_NAMES:
        oid = f"sem_axis_{axis}"
        assert oid in by_id
        assert by_id[oid].column == f"{oid}_mean"
        assert by_id[oid].panel_kind == "subreddit_day"


def test_extended_semantic_comment_specs() -> None:
    """Comment-level extended outcomes use raw shard column names."""
    specs = [o for o in OUTCOME_REGISTRY if o.family == "semantic_axis_comment"]
    by_id = {o.outcome_id: o for o in specs}
    for axis in EXTENDED_AXIS_NAMES:
        oid = f"sem_axis_{axis}"
        assert oid in by_id
        assert by_id[oid].column == oid
        assert by_id[oid].panel_kind == "comment"
