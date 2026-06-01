"""Tests for dual-tail semantic ideology event-study plotting helpers."""

from __future__ import annotations

import pandas as pd

from src.did.outputs import tail_shift_interpretation_title


def test_tail_shift_title_leftward_shift() -> None:
    """Function summary: left up and right down yields leftward-shift wording."""
    title = tail_shift_interpretation_title(0.018, 0.006, -0.026, 0.003)
    assert "leftward location shift" in title
    assert "extreme-LEFT" in title
    assert "extreme-RIGHT" in title


def test_tail_shift_title_both_tails_up() -> None:
    """Function summary: both positive betas yields tail-widening wording."""
    title = tail_shift_interpretation_title(0.02, 0.01, 0.03, 0.02)
    assert "tail-widening" in title or "both tails up" in title
