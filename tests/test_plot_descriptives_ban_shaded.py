"""Tests for ban-window shaded descriptives plot helpers."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pandas as pd


def _load_plot_mod():
    """Function summary: load plot_descriptives_ban_shaded module with Agg backend."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/plot_descriptives_ban_shaded.py"
    )
    spec = importlib.util.spec_from_file_location("plot_ban_shaded_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec.loader.exec_module(mod)
    return mod


def test_control_panels_use_us_political() -> None:
    """Control filter uses canonical US_political panel id from prepared tables."""
    mod = _load_plot_mod()
    assert "US_political" in mod.CONTROL_PANELS
    assert "US" not in mod.CONTROL_PANELS
    assert mod._control_display("US_political") == "US"


def test_outcome_metric_covers_ban_window_outcomes() -> None:
    """Registry and extra lexical columns resolve to metric mappings."""
    mod = _load_plot_mod()
    from src.did.outcomes import BAN_WINDOW_DESCRIPTIVE_OUTCOMES, BAN_WINDOW_LEXICAL_EXTRA_COLUMNS

    for oid in BAN_WINDOW_DESCRIPTIVE_OUTCOMES:
        assert oid in mod.OUTCOME_METRIC, oid
    for oid, col in BAN_WINDOW_LEXICAL_EXTRA_COLUMNS:
        assert mod.OUTCOME_METRIC[oid] == ("descriptives", col)


def test_series_for_outcome_matches_us_political_rows() -> None:
    """Lexical loader keeps US_political rows for control plotting."""
    mod = _load_plot_mod()
    lexical = pd.DataFrame(
        {
            "country_panel": ["Italy_political", "US_political", "Germany"],
            "date_utc": ["2023-03-01", "2023-03-01", "2023-03-01"],
            "ai_style_rate_100w_mean": [1.0, 2.0, 3.0],
        }
    )

    def _fake_lexical(_config):
        return lexical

    mod._daily_lexical = _fake_lexical
    series, source = mod._series_for_outcome({}, "ai_style_rate")
    assert source == "descriptives"
    us = series[series["country_panel"] == "US_political"]
    assert len(us) == 1
    assert float(us["value"].iloc[0]) == 2.0


def test_empty_wordfish_germany_note() -> None:
    """Germany on Wordfish gets the expected no-data annotation."""
    mod = _load_plot_mod()
    note = mod._empty_panel_note("Germany", "wordfish")
    assert "Wordfish" in note
    assert mod._empty_panel_note("Germany", "descriptives") == "No data"


def test_plot_outcome_writes_png_with_empty_control() -> None:
    """Plot helper writes PNG and tolerates missing control series."""
    mod = _load_plot_mod()
    series = pd.DataFrame(
        {
            "country_panel": ["Italy_political", "Italy_others", "US_political"],
            "date_utc": pd.to_datetime(["2023-03-01", "2023-03-01", "2023-03-01"]),
            "value": [1.0, 1.2, 2.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lexical" / "ai_style_rate.png"
        mod._plot_outcome(series, "ai_style_rate", out, rolling_window=1, source="descriptives")
        assert out.is_file()
