"""Tests for DiD outcome display labels."""

from __future__ import annotations

from src.did.outcomes import outcome_label


def test_outcome_label_short() -> None:
    """Function summary: short outcome labels differ from raw ids."""
    assert outcome_label("net_ideology", short=True) == "Net ideology"
    assert outcome_label("net_ideology", short=False) == "net ideology"


def test_outcome_label_fallback() -> None:
    """Function summary: unknown outcomes fall back to underscore replacement."""
    assert outcome_label("custom_metric", short=True) == "custom metric"
