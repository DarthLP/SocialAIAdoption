"""Tests for thesis figure theme helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.plotting.thesis_theme import (
    event_study_ban_boundaries,
    shade_ban_window,
    xlabel_event_study,
    ylabel_italy_bin_coefficient,
)


def test_event_study_ban_boundaries_days() -> None:
    """Function summary: ban boundaries sit at bin edges in day units."""
    onset, lift = event_study_ban_boundaries(bin_days=1, x_scale="days")
    assert onset == -0.5
    assert lift == 28.5


def test_event_study_ban_boundaries_period_3d() -> None:
    """Function summary: 3-day period scale places lift at period 9.5."""
    onset, lift = event_study_ban_boundaries(bin_days=3, x_scale="period")
    assert onset == -0.5
    assert lift == 9.5


def test_standard_labels() -> None:
    """Function summary: axis label strings match thesis spec."""
    assert ylabel_italy_bin_coefficient() == "Italy \u00d7 bin coefficient"
    assert xlabel_event_study(3) == "Days relative to ban onset (3-day bins)"


def test_shade_ban_window_smoke(tmp_path: Path) -> None:
    """Function summary: shade_ban_window runs without error on empty axes."""
    fig, ax = plt.subplots()
    shade_ban_window(ax, mode="event_study", bin_days=3, x_scale="days")
    plt.close(fig)
