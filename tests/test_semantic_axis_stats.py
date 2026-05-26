"""Unit tests for semantic_axis_stats helpers."""

from __future__ import annotations

from src.semantic_axis_stats import (
    absolute_threshold,
    build_pole_bucket_specs,
    topic_family_primary_lexicon,
)


def test_topic_family_lexicon_map() -> None:
    """Italian families map to it; US/UK/EU map to en."""
    assert topic_family_primary_lexicon("it_political") == "it"
    assert topic_family_primary_lexicon("us") == "en"
    assert topic_family_primary_lexicon("de") == "de"


def test_absolute_threshold_by_lexicon() -> None:
    """Per-lexicon thresholds override global defaults."""
    sem_cfg = {
        "pole_thresholds_by_lexicon": {
            "it": {"aggression": 0.08},
            "en": {"aggression": 0.25},
        },
        "pole_cutoffs": [0.25],
    }
    assert absolute_threshold(sem_cfg, "it", "aggression") == 0.08
    assert absolute_threshold(sem_cfg, "en", "aggression") == 0.25


def test_build_pole_bucket_specs_includes_percentiles() -> None:
    """Percentile specs add above_p90 and below_p10 suffixes."""
    sem_cfg = {"pole_percentiles": [10, 90]}
    specs = build_pole_bucket_specs(sem_cfg)
    suffixes = {s.suffix for s in specs}
    assert "abs" in suffixes
    assert "above_p90" in suffixes
    assert "below_p10" in suffixes
