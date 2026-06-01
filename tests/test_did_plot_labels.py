"""Tests for DiD plot label disambiguation by post-window spec."""

from __future__ import annotations

import pandas as pd

from src.did.outputs import disambiguate_with_spec
from src.did.specs import spec_label_parenthetical


def test_spec_label_parenthetical() -> None:
    """Function summary: parenthetical spec labels match post windows."""
    assert spec_label_parenthetical("full_ban") == "(full ban)"
    assert spec_label_parenthetical("early_ban_7d") == "(early ban)"
    assert spec_label_parenthetical("early_ban_14d") == "(14d ban)"
    assert spec_label_parenthetical("post_short_3d") == "(short 0–2d)"
    assert spec_label_parenthetical("post_medium_7d") == "(medium 3–9d)"
    assert spec_label_parenthetical("post_long_tail") == "(long 10d+)"


def test_disambiguate_with_spec_only_on_duplicates() -> None:
    """Function summary: suffix only when base label repeats."""
    base = pd.Series(["Net ideology", "Net ideology", "AI style rate"])
    specs = pd.Series(["full_ban", "early_ban_7d", "full_ban"])
    out = disambiguate_with_spec(base, specs)
    assert out.iloc[0] == "Net ideology (full ban)"
    assert out.iloc[1] == "Net ideology (early ban)"
    assert out.iloc[2] == "AI style rate"


def test_disambiguate_strategy_labels() -> None:
    """Function summary: duplicate strategy short labels get spec suffix."""
    base = pd.Series(
        [
            "IT vs pooled (DE/EU/US/UK)",
            "IT vs pooled (DE/EU/US/UK)",
            "IT political vs controls",
        ]
    )
    specs = pd.Series(["full_ban", "early_ban_7d", "full_ban"])
    out = disambiguate_with_spec(base, specs)
    assert "(full ban)" in out.iloc[0]
    assert "(early ban)" in out.iloc[1]
    assert out.iloc[2] == "IT political vs controls"
