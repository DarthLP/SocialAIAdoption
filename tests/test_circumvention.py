"""Unit tests for circumvention loader and panel builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.diagnostics.descriptives_util import assign_period_start
from src.circumvention import (
    build_circumvention_geo_panel,
    enrich_daily_with_transforms,
    italy_circumvention_by_period,
    load_circumvention_daily,
)
from src.config_utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_circumvention_daily_geos_and_treated() -> None:
    """Combined loader returns uppercase geo and treated flag for IT."""
    config = load_config(PROJECT_ROOT / "config/italy_polarization_setup.yaml")
    daily = load_circumvention_daily(
        PROJECT_ROOT, config, start="2023-03-01", end_exclusive="2023-04-15"
    )
    assert not daily.empty
    assert "vpn_interest" in daily.columns
    assert set(daily["geo"].unique()) >= {"IT", "DE"}
    it_rows = daily[daily["geo"] == "IT"]
    assert (it_rows["treated"] == 1).all()
    de_rows = daily[daily["geo"] == "DE"]
    assert (de_rows["treated"] == 0).all()


def test_build_panel_post_and_period_bins() -> None:
    """Geo panel sets post from period_start vs launch and respects 1d calendar bins."""
    config = load_config(PROJECT_ROOT / "config/italy_polarization_setup.yaml")
    daily = load_circumvention_daily(
        PROJECT_ROOT, config, start="2023-03-28", end_exclusive="2023-04-05"
    )
    daily = enrich_daily_with_transforms(daily)
    panel = build_circumvention_geo_panel(
        daily, "2023-03-31", 1, assign_period_start=assign_period_start
    )
    assert not panel.empty
    assert "post" in panel.columns
    pre = panel[panel["period_start"] < "2023-03-31"]
    post = panel[panel["period_start"] >= "2023-03-31"]
    if not pre.empty:
        assert (pre["post"] == 0).all()
    if not post.empty:
        assert (post["post"] == 1).all()


def test_italy_circumvention_by_period_suffix() -> None:
    """Italy lookup renames VPN/Tor columns with _it suffix."""
    df = pd.DataFrame(
        {
            "geo": ["IT", "DE"],
            "period_start": ["2023-03-31", "2023-03-31"],
            "vpn_interest": [50.0, 10.0],
            "tor_bridge_users": [100.0, 200.0],
        }
    )
    it = italy_circumvention_by_period(df)
    assert list(it.columns) == ["period_start", "vpn_interest_it", "tor_bridge_users_it"]
    assert len(it) == 1
    assert it.iloc[0]["vpn_interest_it"] == 50.0
