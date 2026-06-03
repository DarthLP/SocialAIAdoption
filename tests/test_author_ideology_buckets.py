"""Tests for lexical vs semantic author ideology bucket assignment and agreement."""

from __future__ import annotations

import pandas as pd

from src.user_week.ideology_buckets import (
    SEMANTICALLY_UNSCORED,
    assign_lexical_buckets,
    assign_semantic_mag_band_buckets,
    assign_semantic_tail_buckets,
    assign_tertile_buckets,
    cohens_kappa,
    collect_preban_author_features,
    filter_agreement_sample,
    ideology_bucket_config,
    load_semantic_orientation_multipliers,
    pct_exact_match,
    preban_author_scores,
    semantic_orientation_flip,
)


LABELS = ("conservative_leaning", "neutral", "liberal_leaning")


def test_preban_author_scores_weighted_mean() -> None:
    """Function summary: pre-ban scores use word-weighted weekly means."""
    panel = pd.DataFrame(
        {
            "author": ["u1", "u1"],
            "period": ["pre", "pre"],
            "n_words": [100.0, 300.0],
            "net_ideology_mean": [0.0, 1.0],
            "sem_axis_ideology_mean": [0.1, 0.5],
            "assigned_primary_lexicon": ["it", "it"],
        }
    )
    lookup = {("it", "ideology", 10): -1.0, ("it", "ideology", 90): 1.0}
    scores = collect_preban_author_features(panel, ["u1"], 100, 1, None, lookup)
    assert len(scores) == 1
    assert abs(scores.loc[0, "lexical_score"] - 0.75) < 1e-9
    assert abs(scores.loc[0, "semantic_score"] - 0.4) < 1e-9


def test_lexical_zero_hits_neutral() -> None:
    """Function summary: no L/R hits yields neutral lexical bucket."""
    df = pd.DataFrame(
        {
            "left_hits_pre": [0.0],
            "right_hits_pre": [0.0],
            "lexical_score": [0.8],
        }
    )
    out = assign_lexical_buckets(df, LABELS)
    assert out.iloc[0] == "neutral"


def test_lexical_hits_positive_score_liberal() -> None:
    """Function summary: L/R hits and positive score yield liberal bucket."""
    df = pd.DataFrame(
        {
            "left_hits_pre": [2.0],
            "right_hits_pre": [0.0],
            "lexical_score": [0.3],
        }
    )
    out = assign_lexical_buckets(df, LABELS)
    assert out.iloc[0] == "liberal_leaning"


def test_semantic_tail_neutral_no_extreme_weeks() -> None:
    """Function summary: no tail weeks yields neutral semantic bucket."""
    df = pd.DataFrame(
        {
            "share_scored": [0.9],
            "n_sem_left_tail_weeks": [0],
            "n_sem_right_tail_weeks": [0],
            "semantic_score": [0.2],
        }
    )
    out = assign_semantic_tail_buckets(df, LABELS, min_share_scored=0.5)
    assert out.iloc[0] == "neutral"


def test_semantic_tail_left_dominates_liberal() -> None:
    """Function summary: more left-tail weeks than right yields liberal bucket."""
    df = pd.DataFrame(
        {
            "share_scored": [0.9],
            "n_sem_left_tail_weeks": [2],
            "n_sem_right_tail_weeks": [0],
            "semantic_score": [0.1],
        }
    )
    out = assign_semantic_tail_buckets(df, LABELS, min_share_scored=0.5)
    assert out.iloc[0] == "liberal_leaning"


def test_semantic_low_coverage_unscored() -> None:
    """Function summary: low share_scored yields semantically_unscored."""
    df = pd.DataFrame(
        {
            "share_scored": [0.2],
            "n_sem_left_tail_weeks": [1],
            "n_sem_right_tail_weeks": [0],
            "semantic_score": [0.5],
        }
    )
    out = assign_semantic_tail_buckets(df, LABELS, min_share_scored=0.5)
    assert out.iloc[0] == SEMANTICALLY_UNSCORED


def test_semantic_mag_band_p25_neutral() -> None:
    """Function summary: scores inside p25 |score| band are neutral."""
    df = pd.DataFrame(
        {
            "assigned_primary_lexicon": ["it"] * 5,
            "share_scored": [1.0] * 5,
            "semantic_score": [0.01, 0.02, 0.03, 0.5, -0.6],
        }
    )
    out = assign_semantic_mag_band_buckets(df, LABELS, min_share_scored=0.5)
    assert out.iloc[0] == "neutral"
    assert out.iloc[3] == "liberal_leaning"
    assert out.iloc[4] == "conservative_leaning"


def test_filter_agreement_excludes_unscored() -> None:
    """Function summary: agreement sample drops semantically_unscored rows."""
    df = pd.DataFrame(
        {
            "lexical_bucket": ["neutral", "liberal_leaning"],
            "semantic_bucket": [SEMANTICALLY_UNSCORED, "liberal_leaning"],
        }
    )
    sub = filter_agreement_sample(df, list(LABELS))
    assert len(sub) == 1


def test_tertiles_per_lexicon() -> None:
    """Function summary: tertile assignment yields three buckets per language."""
    df = pd.DataFrame(
        {
            "assigned_primary_lexicon": ["it"] * 9,
            "lexical_score": [float(i) for i in range(9)],
        }
    )
    out = assign_tertile_buckets(df, "assigned_primary_lexicon", "lexical_score", LABELS, "lexical_bucket")
    assert set(out["lexical_bucket"]) == set(LABELS)


def test_semantic_orientation_flip_sign() -> None:
    """Function summary: negative orientation multiplier flips semantic sign."""
    mult = {"it": -1.0}
    assert semantic_orientation_flip("it", mult) == -1.0
    assert semantic_orientation_flip("en", mult) == 1.0


def test_cohens_kappa_perfect_agreement() -> None:
    """Function summary: identical buckets yield kappa 1 and 100% exact match."""
    labels = list(LABELS)
    y = pd.Series(labels)
    assert pct_exact_match(y, y) == 1.0
    assert cohens_kappa(y, y, labels) == 1.0


def test_ideology_bucket_config_from_yaml_shape() -> None:
    """Function summary: config parser returns asymmetric_v2 defaults when block empty."""
    cfg = ideology_bucket_config({})
    assert cfg.min_pre_words == 400
    assert cfg.method == "asymmetric_v2"
    assert cfg.min_share_scored == 0.5
    assert cfg.tail_percentile_low == 25
    assert cfg.tail_percentile_high == 75
    assert len(cfg.bucket_labels) == 3


def test_load_orientation_defaults_without_file() -> None:
    """Function summary: missing orientation report returns +1 for all langs."""
    mult = load_semantic_orientation_multipliers({"paths": {"tables_dir": "/nonexistent"}})
    assert mult["it"] == 1.0


def test_collect_preban_tail_week_counts() -> None:
    """Function summary: tail-week counters use oriented scores vs p10/p90."""
    panel = pd.DataFrame(
        {
            "author": ["u1", "u1", "u1"],
            "period": ["pre", "pre", "pre"],
            "n_words": [100.0, 100.0, 100.0],
            "net_ideology_mean": [0.0, 0.0, 0.0],
            "sem_axis_ideology_mean": [-2.0, 0.0, 2.0],
            "sem_axis_coverage_mean": [1.0, 1.0, 1.0],
            "left_hits": [0, 0, 0],
            "right_hits": [0, 0, 0],
            "assigned_primary_lexicon": ["it", "it", "it"],
        }
    )
    lookup = {("it", "ideology", 25): -0.5, ("it", "ideology", 75): 0.5}
    feat = collect_preban_author_features(
        panel, ["u1"], 100, 1, {"it": 1.0}, lookup, tail_percentile_low=25, tail_percentile_high=75
    )
    assert feat.loc[0, "n_sem_left_tail_weeks"] == 1
    assert feat.loc[0, "n_sem_right_tail_weeks"] == 1
