"""Unit tests for lexical daily panel binning and country DiD merge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from scripts.diagnostics.descriptives_util import bin_lexical_daily_panel
from src.circumvention import merge_circumvention_by_geo


def _load_did_mod():
    """Function summary: load prepare_did_merged_panels module."""
    path = Path(__file__).resolve().parent.parent / "scripts/diagnostics/prepare_did_merged_panels.py"
    spec = importlib.util.spec_from_file_location("prepare_did_merged_panels_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bin_lexical_7d_weighted_mean_and_partial_bin() -> None:
    """Two daily rows in one launch-aligned week bin aggregate with comment weights."""
    launch = "2023-03-31"
    daily = pd.DataFrame(
        {
            "country_panel": ["Germany", "Germany"],
            "date_utc": ["2023-03-01", "2023-03-02"],
            "n_comments": [100, 300],
            "net_ideology_mean": [0.1, 0.3],
        }
    )
    binned = bin_lexical_daily_panel(daily, ("country_panel",), 7, launch)
    assert len(binned) == 1
    row = binned.iloc[0]
    assert row["period_start"] == "2023-02-24"
    assert row["n_days_in_bin"] == 2
    assert bool(row["is_partial_bin"])
    assert abs(row["net_ideology_mean"] - 0.25) < 1e-9


def test_semantic_did_frame_drops_geo_vpn() -> None:
    """Semantic DiD prep removes geo-matched VPN; keeps Italy broadcast columns."""
    mod = _load_did_mod()
    panel = pd.DataFrame(
        {
            "topic_family": ["de"],
            "period_start": ["2023-03-31"],
            "vpn_interest_it": [22.0],
            "vpn_interest": [69.0],
            "geo": ["DE"],
            "tor_relay_users": [300000.0],
            "sem_axis_ideology_mean": [0.01],
        }
    )
    out = mod._prepare_semantic_did_frame(panel)
    assert "vpn_interest_it" in out.columns
    assert "vpn_interest" not in out.columns
    assert "geo" not in out.columns
    assert out.iloc[0]["circumvention_intensity_spec"] == "it_broadcast"


def test_merge_country_panel_geo_vpn() -> None:
    """Binned lexical country panel picks up geo-matched circumvention."""
    launch = "2023-03-31"
    daily = pd.DataFrame(
        {
            "country_panel": ["Germany"],
            "date_utc": ["2023-03-31"],
            "n_comments": [80],
            "net_ideology_mean": [0.0],
        }
    )
    binned = bin_lexical_daily_panel(daily, ("country_panel",), 1, launch)
    circum = pd.DataFrame(
        {
            "geo": ["DE"],
            "period_start": ["2023-03-31"],
            "vpn_interest": [12.0],
            "post": [1],
            "treated": [0],
        }
    )
    merged = merge_circumvention_by_geo(binned, circum, {"Germany": "DE"})
    assert merged.iloc[0]["vpn_interest"] == 12.0
