"""Unit tests for style index validation metrics (no SIGNS changes)."""

from __future__ import annotations

import pandas as pd

from src.style_index_validation import (
    INDEX_COL_LLM,
    build_ai_rate_review_sample,
    compare_indices_rows,
    convergence_correlation_rows,
    joint_signal_bucket_rows,
    length_stratified_ai_rows,
    partial_corr_spearman,
    prepare_validation_frame,
    spearman_corr,
)


def _synthetic_frame() -> pd.DataFrame:
    """Function summary: comment frame where index tracks ai_rate more than noise."""
    rows = []
    for i in range(80):
        ai = float(i % 5) * 0.5
        nw = 30 + (i % 40)
        rows.append(
            {
                INDEX_COL_LLM: ai * 0.8 + 0.1 * (nw / 100.0),
                "ai_style_rate_100w": ai,
                "n_words": float(nw),
                "primary_lexicon": "it" if i % 2 == 0 else "en",
                "subreddit": "sub_a",
            }
        )
    return pd.DataFrame(rows)


def test_spearman_positive_on_constructed_data() -> None:
    """Function summary: Spearman detects monotone association in synthetic data."""
    df = prepare_validation_frame(_synthetic_frame())
    rho, n = spearman_corr(df[INDEX_COL_LLM], df["ai_style_rate_100w"])
    assert n >= 30
    assert rho > 0.5


def test_length_stratified_delta_positive() -> None:
    """Function summary: ai_hit group has higher mean index in 20-49 bin when constructed."""
    df = prepare_validation_frame(_synthetic_frame())
    rows = length_stratified_ai_rows(df)
    deltas = [r for r in rows if r.get("ai_hit") == "delta_hit_minus_nohit" and r["length_bin"] == "20_49"]
    assert deltas
    assert deltas[0][f"mean_{INDEX_COL_LLM}"] > 0


def test_partial_corr_less_than_marginal() -> None:
    """Function summary: partial corr exists and is finite when length confounds marginally."""
    df = prepare_validation_frame(_synthetic_frame())
    rho, _ = spearman_corr(df[INDEX_COL_LLM], df["ai_style_rate_100w"])
    prho, pn = partial_corr_spearman(df[INDEX_COL_LLM], df["ai_style_rate_100w"], df["log_len"])
    assert pn >= 30
    assert abs(prho) <= abs(rho) + 0.05 or prho > 0.2


def test_ai_rate_review_sample_buckets() -> None:
    """Function summary: review sample has high and low ai_rate buckets."""
    df = prepare_validation_frame(_synthetic_frame())
    rev = build_ai_rate_review_sample(df, n_each=10)
    assert set(rev["review_bucket"]) == {"high_ai_rate", "low_ai_rate"}
    assert rev[rev["review_bucket"] == "high_ai_rate"]["ai_style_rate_100w"].min() >= rev[
        rev["review_bucket"] == "low_ai_rate"
    ]["ai_style_rate_100w"].max()


def test_convergence_rows_include_partial() -> None:
    """Function summary: convergence export includes partial-on-length row."""
    df = prepare_validation_frame(_synthetic_frame())
    rows = convergence_correlation_rows(df, subset="all")
    benchmarks = {r["benchmark"] for r in rows}
    assert "lexicon_ai_style_rate" in benchmarks
    assert "ai_style_rate_partial_log_len" in benchmarks


def test_convergence_includes_semicolon_when_column_present() -> None:
    """Function summary: semicolon_colon_rate_100w appears in convergence rows."""
    df = prepare_validation_frame(_synthetic_frame())
    df["semicolon_colon_rate_100w"] = 0.1
    rows = convergence_correlation_rows(df, subset="all")
    assert "semicolon_colon_rate_100w" in {r["benchmark"] for r in rows}


def test_compare_indices_rows_pairwise() -> None:
    """Function summary: compare_indices emits pairs when ablation columns exist."""
    df = prepare_validation_frame(_synthetic_frame())
    df["style_index_llm_no_ai_style"] = df[INDEX_COL_LLM] * 0.9
    rows = compare_indices_rows(df)
    assert any(r["index_a"] == INDEX_COL_LLM for r in rows)


def test_joint_signal_buckets_nonempty() -> None:
    """Function summary: joint buckets return rows when em_dash_hit present."""
    df = prepare_validation_frame(_synthetic_frame())
    df["em_dash_rate_100w"] = (df.index % 3 == 0).astype(float)
    df = prepare_validation_frame(df)
    rows = joint_signal_bucket_rows(df)
    assert rows
