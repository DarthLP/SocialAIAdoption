"""Unit tests for readability formulas in style_index.py."""

from __future__ import annotations

import math

import pandas as pd

from src.style_index import (
    compute_index,
    fit_preperiod_stats,
    readability_amstad,
    readability_flesch_en,
    readability_gulpease,
    ttr_first_n_tokens,
)


def test_readability_gulpease_hand_example() -> None:
    """Function summary: Gulpease matches hand calculation for short Italian-like sample."""
    # 10 words, 50 chars, 2 sentences -> 300*2/10 - 10*50/10 + 89 = 60 - 50 + 89 = 99
    got = readability_gulpease(50, 10, 2)
    assert abs(got - 99.0) < 1e-6


def test_readability_amstad_hand_example() -> None:
    """Function summary: Amstad matches hand calculation."""
    # 180 - 50/10 - 58.5*(2/10) = 180 - 5 - 11.7 = 163.3
    got = readability_amstad(50, 10, 2)
    assert abs(got - 163.3) < 1e-6


def test_readability_flesch_hand_example() -> None:
    """Function summary: Flesch RE matches hand calculation."""
    # 206.835 - 1.015*(10/2) - 84.6*(12/10) = 206.835 - 5.075 - 101.52
    got = readability_flesch_en(10, 2, 12)
    expected = 206.835 - 1.015 * 5.0 - 84.6 * 1.2
    assert abs(got - expected) < 1e-4


def test_fit_preperiod_stats_persists_clip_bounds() -> None:
    """Function summary: pre-period fit stores clip_lo/hi per feature."""
    rng = pd.Series([float(i) for i in range(40)])
    rows = []
    for day in pd.date_range("2023-03-01", "2023-03-30", freq="D"):
        rows.append(
            {
                "date_utc": day.strftime("%Y-%m-%d"),
                "lang": "it",
                "log_len": float(rng.iloc[len(rows) % len(rng)]),
                "avg_words_per_sentence": 10.0,
                "sentence_length_variance": 2.0,
                "em_dash_rate_100w": 0.1,
                "semicolon_colon_rate_100w": 0.2,
                "hedging_phrase_rate_100w": 0.3,
                "ai_style_rate_100w": 0.4,
                "exclamation_rate_100w": 0.05,
                "caps_word_share": 0.02,
            }
        )
    stats = fit_preperiod_stats(pd.DataFrame(rows))
    meta = stats["languages"]["it"]["log_len"]
    assert "clip_lo" in meta and "clip_hi" in meta
    full, _red = compute_index(rows[0], stats, "it")
    assert full is None or isinstance(full, float)


def test_ttr_50w_requires_50_tokens() -> None:
    """Function summary: ttr_50w is NaN below 50 tokens."""
    short = "one two three four five"
    assert math.isnan(ttr_first_n_tokens(short, 50))
    long = " ".join([f"w{i}" for i in range(55)])
    assert 0.0 < ttr_first_n_tokens(long, 50) <= 1.0
