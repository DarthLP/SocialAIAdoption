"""Tests for bucket event-study labeling (lean_buckets)."""

from __future__ import annotations

import pandas as pd

from src.did.lean_buckets import (
    assert_net_ideology_sign,
    assert_holdout_windows_disjoint,
    assign_lean_buckets,
    bucket_event_study_config,
    estimation_window_mask,
    labeling_window_mask,
    split_march_halves,
    BucketEventStudyConfig,
)


def _minimal_bcfg() -> BucketEventStudyConfig:
    """Function summary: small BucketEventStudyConfig for unit tests."""
    return bucket_event_study_config(
        {
            "did": {
                "bucket_event_study": {
                    "holdout_2wk": {
                        "label_start": "2023-03-01",
                        "label_end": "2023-03-14",
                        "estimate_start": "2023-03-15",
                    },
                }
            },
            "event_window": {
                "start_utc": "2023-03-01T00:00:00Z",
                "end_utc_exclusive": "2023-05-01T00:00:00Z",
                "launch_day_utc": "2023-03-31T00:00:00Z",
            },
            "plot_reference_dates_utc": ["2023-03-31T00:00:00Z", "2023-04-28T23:59:59Z"],
        }
    )


def test_rel_period_and_post() -> None:
    """Function summary: rel_period from rel_day // bin_days; post at launch."""
    df = pd.DataFrame(
        {
            "date_utc": ["2023-03-30", "2023-03-31"],
            "rel_day": [-1, 0],
            "post": [0, 1],
        }
    )
    df["rel_period"] = (df["rel_day"] // 3).astype(int)
    assert df["rel_period"].tolist() == [-1, 0]
    assert df["post"].tolist() == [0, 1]


def test_it_post_construction() -> None:
    """Function summary: post:IT equals post times IT."""
    df = pd.DataFrame({"post": [0, 1, 1], "IT": [1, 0, 1]})
    df["post_IT"] = df["post"] * df["IT"]
    assert df["post_IT"].tolist() == [0, 0, 1]


def test_split_halves_disjoint() -> None:
    """Function summary: March odd/even split has no shared comment ids."""
    march = pd.DataFrame(
        {
            "id": [str(i) for i in range(10)],
            "author": ["a"] * 10,
            "date_utc": ["2023-03-15"] * 10,
        }
    )
    half_a, half_b = split_march_halves(march, "odd_even", 42, 0)
    assert not set(half_a).intersection(set(half_b))


def test_holdout_nonoverlap() -> None:
    """Function summary: holdout label end strictly before estimate start."""
    bcfg = _minimal_bcfg()
    assert_holdout_windows_disjoint(bcfg)


def test_asymmetric_lexical_buckets() -> None:
    """Function summary: no L/R hits -> neutral; positive lean_mean with hits -> liberal."""
    feats = pd.DataFrame(
        {
            "author": ["no_hits", "liberal", "conservative"],
            "lean_mean": [0.5, 0.3, -0.2],
            "n_label_comments": [5, 5, 5],
            "primary_lexicon": ["it", "it", "it"],
            "left_hits_sum": [0.0, 3.0, 0.0],
            "right_hits_sum": [0.0, 0.0, 2.0],
        }
    )
    bcfg = bucket_event_study_config(
        {
            "did": {"bucket_event_study": {"bucket_method": "asymmetric_lexical"}},
            "event_window": {
                "start_utc": "2023-03-01T00:00:00Z",
                "end_utc_exclusive": "2023-05-01T00:00:00Z",
                "launch_day_utc": "2023-03-31T00:00:00Z",
            },
            "plot_reference_dates_utc": ["2023-03-31T00:00:00Z"],
        }
    )
    buckets = assign_lean_buckets(feats, bcfg)
    assert buckets["no_hits"] == "neutral"
    assert buckets["liberal"] == "liberal_leaning"
    assert buckets["conservative"] == "conservative_leaning"


def test_tertile_removed_raises() -> None:
    """Function summary: tertile_within_language is no longer supported."""
    feats = pd.DataFrame(
        {
            "author": ["a"],
            "lean_mean": [0.1],
            "n_label_comments": [5],
            "primary_lexicon": ["it"],
        }
    )
    bcfg = bucket_event_study_config(
        {
            "did": {"bucket_event_study": {"bucket_method": "tertile_within_language"}},
            "event_window": {
                "start_utc": "2023-03-01T00:00:00Z",
                "end_utc_exclusive": "2023-05-01T00:00:00Z",
                "launch_day_utc": "2023-03-31T00:00:00Z",
            },
            "plot_reference_dates_utc": ["2023-03-31T00:00:00Z"],
        }
    )
    try:
        assign_lean_buckets(feats, bcfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "tertile_within_language" in str(exc)


def test_sign_assertion() -> None:
    """Function summary: net_ideology higher when left_hits exceed right_hits."""
    df = pd.DataFrame(
        {
            "net_ideology": [0.5, -0.5, 0.2],
            "left_hits": [3, 0, 2],
            "right_hits": [0, 3, 1],
        }
    )
    assert_net_ideology_sign(df)


def test_labeling_vs_estimation_windows() -> None:
    """Function summary: holdout estimation mask starts after label end."""
    bcfg = _minimal_bcfg()
    config = {
        "event_window": {
            "start_utc": "2023-03-01T00:00:00Z",
            "end_utc_exclusive": "2023-05-01T00:00:00Z",
            "launch_day_utc": "2023-03-31T00:00:00Z",
        },
        "plot_reference_dates_utc": ["2023-03-31T00:00:00Z", "2023-04-28T23:59:59Z"],
    }
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "date_utc": ["2023-03-10", "2023-03-20"],
            "author": ["a", "a"],
        }
    )
    lab = labeling_window_mask(df, "holdout_2wk", bcfg)
    est = estimation_window_mask(df, "holdout_2wk", bcfg, config)
    assert lab.iloc[0] and not lab.iloc[1]
    assert not est.iloc[0] and est.iloc[1]
