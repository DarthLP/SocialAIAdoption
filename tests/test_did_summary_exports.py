"""Tests for DiD summary CSV/txt export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.did.outputs import dedupe_summary_rows, write_summary_exports


def test_write_summary_exports_creates_theme_files(tmp_path: Path) -> None:
    """Function summary: exports by_outcome and by_theme aggression bundle."""
    summary_dir = tmp_path / "summary"
    df = pd.DataFrame(
        [
            {
                "outcome_id": "aggression_rate",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": 0.01,
                "se": 0.005,
                "pvalue": 0.04,
                "n_obs": 100,
                "n_clusters": 10,
                "estimation_note": "ok",
                "sign_only_cross_country": 0,
            },
            {
                "outcome_id": "sem_axis_aggression",
                "outcome_family": "semantic_axis",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": -0.02,
                "se": 0.01,
                "pvalue": 0.1,
                "n_obs": 80,
                "n_clusters": 8,
                "estimation_note": "ok",
                "sign_only_cross_country": 0,
            },
        ]
    )
    write_summary_exports(df, summary_dir, "2023-03-31")
    assert (summary_dir / "did_summary.csv").is_file()
    assert (summary_dir / "by_theme" / "aggression.csv").is_file()
    assert (summary_dir / "by_theme" / "aggression.txt").is_file()
    assert (summary_dir / "by_outcome" / "aggression_rate.txt").is_file()
    txt = (summary_dir / "by_theme" / "aggression.txt").read_text(encoding="utf-8")
    assert "aggression_rate" in txt
    assert "2023-03-31" in txt


def test_dedupe_summary_rows_nan_vs_empty_weights() -> None:
    """Function summary: NaN and empty weights dedupe as the same key."""
    base = {
        "outcome_id": "sem_axis_emotion",
        "outcome_family": "semantic_axis",
        "strategy_id": "cross_country_all",
        "spec": "early_ban_7d",
        "beta": -0.008738,
        "se": 0.003121,
        "pvalue": 0.005108,
        "n_obs": 100,
        "n_clusters": 117,
        "estimation_note": "ok",
        "sign_only_cross_country": 0,
    }
    df = pd.DataFrame(
        [
            {**base, "weights": float("nan")},
            {**base, "weights": ""},
        ]
    )
    out = dedupe_summary_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["weights"] == ""


def test_write_summary_exports_no_duplicate_by_outcome_keys(tmp_path: Path) -> None:
    """Function summary: by_outcome export has no duplicate strategy/spec rows after dedupe."""
    summary_dir = tmp_path / "summary"
    base = {
        "outcome_id": "sem_axis_emotion",
        "outcome_family": "semantic_axis",
        "strategy_id": "cross_country_vs_de",
        "spec": "early_ban_7d",
        "beta": -0.00309,
        "se": 0.001468,
        "pvalue": 0.035265,
        "n_obs": 100,
        "n_clusters": 110,
        "estimation_note": "ok",
        "sign_only_cross_country": 0,
    }
    df = pd.DataFrame(
        [
            {**base, "weights": float("nan")},
            {**base, "weights": ""},
        ]
    )
    write_summary_exports(df, summary_dir, "2023-03-31")
    out = pd.read_csv(summary_dir / "by_outcome" / "sem_axis_emotion.csv")
    eb = out[out["spec"] == "early_ban_7d"]
    assert len(eb) == len(eb.drop_duplicates(subset=["strategy_id", "spec"]))
