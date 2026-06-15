"""Tests for ChatGPT mention ban-window plot helpers."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def _load_plot_mod():
    """Function summary: load plot_chatgpt_mentions_ban_shaded module with Agg backend."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/plot_chatgpt_mentions_ban_shaded.py"
    )
    spec = importlib.util.spec_from_file_location("plot_chatgpt_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec.loader.exec_module(mod)
    return mod


def test_word_weighted_pool_not_rate_average() -> None:
    """Pooled Italy rate uses summed hits/words, not mean of family rates."""
    mod = _load_plot_mod()
    daily = pd.DataFrame(
        {
            "topic_family": ["it_political", "it_others"],
            "date_utc": ["2023-03-01", "2023-03-01"],
            "n_comments": [10, 10],
            "chatgpt_hits": [10, 270],
            "n_words": [100.0, 900.0],
            "chatgpt_mention_rate_100w": [10.0, 30.0],
        }
    )
    pooled = mod.word_weighted_pool_daily(daily)
    it = pooled[pooled["pool_group"] == "Italy"]
    assert len(it) == 1
    assert float(it["value"].iloc[0]) == pytest.approx(28.0)
    assert float(it["value"].iloc[0]) != pytest.approx(20.0)


def test_sanity_check_passes_on_real_csv() -> None:
    """Production daily table reproduces ban-window benchmark rates."""
    mod = _load_plot_mod()
    csv_path = (
        Path(__file__).resolve().parent.parent
        / "results/tables/italy_polarization/descriptives/daily_chatgpt_mentions_by_topic_family.csv"
    )
    if not csv_path.is_file():
        pytest.skip("daily_chatgpt_mentions_by_topic_family.csv not present")
    daily = pd.read_csv(csv_path)
    pooled = mod.word_weighted_pool_daily(daily)
    mod.sanity_check_pooled_rates(pooled)


def test_sanity_check_fails_on_bad_data() -> None:
    """Sanity gate exits when Italy pre-ban rate is far from benchmark."""
    mod = _load_plot_mod()
    rows = []
    for day in pd.date_range("2023-03-01", "2023-03-30", freq="D"):
        rows.append(
            {
                "pool_group": "Italy",
                "date_utc": day,
                "chatgpt_hits": 1,
                "n_words": 10000.0,
                "value": 0.01,
            }
        )
    for day in pd.date_range("2023-03-31", "2023-04-06", freq="D"):
        rows.append(
            {
                "pool_group": "Italy",
                "date_utc": day,
                "chatgpt_hits": 180,
                "n_words": 10000.0,
                "value": 1.8,
            }
        )
    for day in pd.date_range("2023-03-01", "2023-04-06", freq="D"):
        rows.append(
            {
                "pool_group": "Controls",
                "date_utc": day,
                "chatgpt_hits": 15,
                "n_words": 100000.0,
                "value": 0.015,
            }
        )
    pooled = pd.DataFrame(rows)
    pooled["date_utc"] = pd.to_datetime(pooled["date_utc"])
    with pytest.raises(SystemExit):
        mod.sanity_check_pooled_rates(pooled)


def test_plot_ban_window_pooled_writes_png() -> None:
    """Pooled plot helper writes a PNG file."""
    mod = _load_plot_mod()
    pooled = pd.DataFrame(
        {
            "pool_group": ["Italy", "Controls", "Italy", "Controls"],
            "date_utc": pd.to_datetime(["2023-03-01", "2023-03-01", "2023-03-02", "2023-03-02"]),
            "chatgpt_hits": [10, 5, 12, 6],
            "n_words": [1000.0, 2000.0, 1000.0, 2000.0],
            "value": [1.0, 0.25, 1.2, 0.3],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "chatgpt_mention_rate_100w_pooled.png"
        mod.plot_ban_window_pooled(pooled, out, smoothing_days=1)
        assert out.is_file()


def test_control_panel_band_min_max() -> None:
    """Control band uses per-panel min and max after smoothing."""
    mod = _load_plot_mod()
    series = pd.DataFrame(
        {
            "country_panel": ["Germany", "UK", "Germany", "UK"],
            "date_utc": pd.to_datetime(["2023-03-01", "2023-03-01", "2023-03-02", "2023-03-02"]),
            "value": [0.1, 0.3, 0.2, 0.4],
        }
    )
    band = mod.control_panel_band_daily(series, smoothing_days=1)
    row = band[band["date_utc"] == pd.Timestamp("2023-03-01")].iloc[0]
    assert float(row["ctrl_min"]) == pytest.approx(0.1)
    assert float(row["ctrl_max"]) == pytest.approx(0.3)


def test_plot_ban_window_pooled_with_control_range_writes_png() -> None:
    """Pooled plot with control range writes a separate PNG file."""
    mod = _load_plot_mod()
    pooled = pd.DataFrame(
        {
            "pool_group": ["Italy", "Controls", "Italy", "Controls"],
            "date_utc": pd.to_datetime(["2023-03-01", "2023-03-01", "2023-03-02", "2023-03-02"]),
            "chatgpt_hits": [10, 5, 12, 6],
            "n_words": [1000.0, 2000.0, 1000.0, 2000.0],
            "value": [1.0, 0.25, 1.2, 0.3],
        }
    )
    series = pd.DataFrame(
        {
            "country_panel": ["Germany", "UK", "Germany", "UK"],
            "date_utc": pd.to_datetime(["2023-03-01", "2023-03-01", "2023-03-02", "2023-03-02"]),
            "value": [0.1, 0.3, 0.2, 0.4],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "chatgpt_mention_rate_100w_pooled_range.png"
        mod.plot_ban_window_pooled_with_control_range(pooled, series, out, smoothing_days=1)
        assert out.is_file()
