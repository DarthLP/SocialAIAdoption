"""Unit tests for style_index_llm and leave-one-out ablations."""

from __future__ import annotations

import math

import pandas as pd

from src.style_index import STATS_VERSION, compute_style_index_llm, fit_preperiod_stats
from src.style_index_llm import (
    BUNDLE_LLM,
    LLM_CANDIDATES,
    PRIMARY_COL,
    ablation_column_name,
    compute_llm_index,
    renormalize_weights,
)


def _llm_stats() -> dict:
    """Function summary: minimal stats with theory_base primary."""
    rows = []
    for i in range(50):
        rows.append(
            {
                "date_utc": "2023-03-15",
                "lang": "it",
                "ai_style_rate_100w": float(i) * 0.1,
                "exclamation_rate_100w": 0.02 + float(i) * 0.01,
                "caps_word_share": 0.005 + float(i) * 0.001,
                "hedging_phrase_rate_100w": 0.1 + float(i) * 0.02,
                "avg_words_per_sentence": 10.0 + float(i) * 0.1,
                "sentence_length_variance": 1.0 + float(i) * 0.05,
                "em_dash_rate_100w": 0.05 + float(i) * 0.01,
                "semicolon_colon_rate_100w": 0.02 + float(i) * 0.005,
            }
        )
    stats = fit_preperiod_stats(pd.DataFrame(rows))
    cand = LLM_CANDIDATES["theory_base"]
    stats["primary_candidate"] = "theory_base"
    stats["languages"]["it"]["bundles"] = {
        BUNDLE_LLM: {
            "candidate_id": "theory_base",
            "weights": dict(cand["weights"]),
            "interactions": [],
        },
    }
    return stats


def test_renormalize_drops_ai_and_interactions() -> None:
    """Function summary: dropping ai_rate also drops interaction weight keys."""
    w = renormalize_weights(
        {"ai_style_rate_100w": 0.5, "em_dash_x_ai_rate": 0.2, "hedging_phrase_rate_100w": 0.3},
        drop=("ai_style_rate_100w",),
    )
    assert "ai_style_rate_100w" not in w
    assert "em_dash_x_ai_rate" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_style_index_llm_ablation_columns_exist() -> None:
    """Function summary: compute_style_index_llm returns primary and leave-one-out columns."""
    stats = _llm_stats()
    feats = {
        "n_words": 40.0,
        "ai_style_rate_100w": 1.0,
        "hedging_phrase_rate_100w": 0.3,
        "avg_words_per_sentence": 12.0,
        "sentence_length_variance": 1.5,
        "exclamation_rate_100w": 0.05,
        "caps_word_share": 0.01,
        "em_dash_rate_100w": 0.2,
        "semicolon_colon_rate_100w": 0.1,
    }
    out = compute_style_index_llm(feats, stats, "it")
    assert PRIMARY_COL in out
    assert ablation_column_name("ai_style_rate_100w") in out
    assert math.isfinite(out[PRIMARY_COL])


def test_ablation_no_ai_differs_from_primary() -> None:
    """Function summary: removing ai_rate lowers index when ai signal is high."""
    stats = _llm_stats()
    lang_stats = stats["languages"]["it"]
    w = LLM_CANDIDATES["theory_base"]["weights"]
    feats = {
        "n_words": 40.0,
        "ai_style_rate_100w": 4.0,
        "hedging_phrase_rate_100w": 0.1,
        "avg_words_per_sentence": 12.0,
        "sentence_length_variance": 1.5,
        "exclamation_rate_100w": 0.05,
        "caps_word_share": 0.01,
        "em_dash_rate_100w": 0.0,
        "semicolon_colon_rate_100w": 0.0,
    }
    full = compute_llm_index(feats, lang_stats, w, drop=())
    no_ai = compute_llm_index(feats, lang_stats, w, drop=("ai_style_rate_100w",))
    assert math.isfinite(full) and math.isfinite(no_ai)
    assert full > no_ai


def test_fit_stats_version() -> None:
    """Function summary: fit writes llm version tag."""
    stats = fit_preperiod_stats(
        pd.DataFrame(
            {
                "date_utc": ["2023-03-15"] * 40,
                "lang": ["it"] * 40,
                "ai_style_rate_100w": list(range(40)),
            }
        ),
    )
    assert stats["version"] == STATS_VERSION
