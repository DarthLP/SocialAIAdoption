"""Tests for gsynth panel reshape and placebo helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.did.gsynth import _placebo_space_gsynth, _wide_matrix, _augmented_sc_att


def test_wide_matrix_shape() -> None:
    """Function summary: pivot produces units × dates matrix."""
    panel = pd.DataFrame(
        {
            "language_hub": ["it", "de", "it", "de"],
            "period_start": ["2023-03-01", "2023-03-01", "2023-03-02", "2023-03-02"],
            "y": [1.0, 0.5, 1.1, 0.55],
        }
    )
    wide, units, times = _wide_matrix(panel, "y")
    assert "it" in units
    assert "de" in units
    assert len(times) == 2
    assert wide.shape == (2, 2)


def test_augmented_sc_att() -> None:
    """Function summary: augmented SC returns att path and finite avg."""
    rng = np.random.default_rng(0)
    times = [f"2023-03-{i:02d}" for i in range(1, 11)]
    data = {"it": rng.normal(1, 0.1, 10)}
    for c in ("de", "eu", "uk", "us"):
        data[c] = rng.normal(0.5, 0.1, 10)
    wide = pd.DataFrame(data, index=times)
    att_df, avg = _augmented_sc_att(wide, pre_periods=5)
    assert not att_df.empty
    assert np.isfinite(avg)


def test_placebo_space_p_floor() -> None:
    """Function summary: placebo-in-space p respects 1/(n_donors+1) floor."""
    times = [f"2023-03-{i:02d}" for i in range(1, 12)]
    wide = pd.DataFrame(
        {
            "it": np.linspace(1, 2, 11),
            "de": np.linspace(0.5, 1, 11),
            "eu": np.linspace(0.4, 0.9, 11),
            "uk": np.linspace(0.45, 0.95, 11),
            "us": np.linspace(0.42, 0.92, 11),
        },
        index=times,
    )
    _, avg = _augmented_sc_att(wide, pre_periods=5)
    p = _placebo_space_gsynth(wide, avg)
    assert p >= 0.2 - 1e-9
