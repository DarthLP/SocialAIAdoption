"""Unit tests for merging circumvention onto Reddit panels."""

from __future__ import annotations

import pandas as pd

from src.circumvention import (
    ITALY_INTENSITY_COLS,
    attach_italy_circumvention_columns,
    merge_circumvention_by_geo,
)


def test_merge_circumvention_by_geo_country_panel() -> None:
    """Country panel rows pick up geo-matched VPN/Tor on period_start."""
    panel = pd.DataFrame(
        {
            "country_panel": ["Italy_political", "Germany"],
            "period_start": ["2023-03-31", "2023-03-31"],
            "n_comments": [100, 80],
        }
    )
    circum = pd.DataFrame(
        {
            "geo": ["IT", "DE"],
            "period_start": ["2023-03-31", "2023-03-31"],
            "vpn_interest": [55.0, 12.0],
            "tor_bridge_users": [900.0, 4000.0],
            "post": [1, 1],
            "treated": [1, 0],
        }
    )
    geo_map = {"Italy_political": "IT", "Germany": "DE"}
    merged = merge_circumvention_by_geo(panel, circum, geo_map)
    assert merged.loc[0, "geo"] == "IT"
    assert merged.loc[0, "vpn_interest"] == 55.0
    assert merged.loc[1, "geo"] == "DE"
    assert merged.loc[1, "vpn_interest"] == 12.0


def test_attach_italy_columns_language_panel() -> None:
    """language panel gets IT VPN only for primary_lexicon=it rows."""
    panel = pd.DataFrame(
        {
            "primary_lexicon": ["it", "en"],
            "period_start": ["2023-03-31", "2023-03-31"],
            "post": [1, 1],
            "n_comments": [10, 20],
        }
    )
    italy = pd.DataFrame(
        {
            "period_start": ["2023-03-31"],
            "vpn_interest_it": [40.0],
            "tor_bridge_users_it": [500.0],
        }
    )
    out = attach_italy_circumvention_columns(panel, italy, panel_level="language")
    assert out.loc[0, "vpn_interest_it"] == 40.0
    assert pd.isna(out.loc[1, "vpn_interest_it"])


def test_italy_intensity_cols_whitelist() -> None:
    """Italy broadcast column names are stable for estimator configs."""
    assert "vpn_interest_it" in ITALY_INTENSITY_COLS
    assert "vpn_interest" not in ITALY_INTENSITY_COLS
