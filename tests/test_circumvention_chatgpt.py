"""Tests for ChatGPT Google Trends merge in circumvention daily loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.circumvention import enrich_daily_with_transforms, load_circumvention_daily


def test_load_circumvention_daily_includes_chatgpt(tmp_path: Path) -> None:
    """Function summary: chatgpt_interest merges without altering vpn_interest."""
    raw = tmp_path / "circ"
    raw.mkdir(parents=True, exist_ok=True)
    gt = raw / "google_trends_vpn_by_country.csv"
    cg = raw / "google_trends_chatgpt_by_country.csv"
    gt.write_text(
        "date,geo,vpn_interest\n2023-03-01,IT,10\n2023-03-01,DE,20\n",
        encoding="utf-8",
    )
    cg.write_text(
        "date,geo,chatgpt_interest\n2023-03-01,IT,30\n2023-03-01,DE,40\n",
        encoding="utf-8",
    )
    config = {
        "circumvention": {
            "raw_dir": str(raw),
            "google_trends_combined": "google_trends_vpn_by_country.csv",
            "google_trends_chatgpt_combined": "google_trends_chatgpt_by_country.csv",
        }
    }
    daily = load_circumvention_daily(tmp_path, config).sort_values("geo").reset_index(drop=True)
    assert list(daily["vpn_interest"]) == [20.0, 10.0]
    assert list(daily["chatgpt_interest"]) == [40.0, 30.0]
    out = enrich_daily_with_transforms(daily)
    assert "chatgpt_interest_z" in out.columns
    assert "vpn_interest_z" in out.columns
