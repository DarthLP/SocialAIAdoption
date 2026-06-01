"""Tests for lexical vs semantic author ideology bucket assignment and agreement."""

from __future__ import annotations

import pandas as pd

from src.user_week.ideology_buckets import (
    assign_tertile_buckets,
    cohens_kappa,
    ideology_bucket_config,
    load_semantic_orientation_multipliers,
    pct_exact_match,
    preban_author_scores,
    semantic_orientation_flip,
)


def test_preban_author_scores_weighted_mean() -> None:
    """Function summary: pre-ban scores use word-weighted weekly means."""
    panel = pd.DataFrame(
        {
            "author": ["u1", "u1"],
            "period": ["pre", "pre"],
            "n_words": [100.0, 300.0],
            "net_ideology_mean": [0.0, 1.0],
            "sem_axis_ideology_mean": [0.1, 0.5],
        }
    )
    scores = preban_author_scores(panel, ["u1"], min_pre_words=100, min_pre_weeks=1)
    assert len(scores) == 1
    assert abs(scores.loc[0, "lexical_score"] - 0.75) < 1e-9
    assert abs(scores.loc[0, "semantic_score"] - 0.4) < 1e-9


def test_tertiles_per_lexicon() -> None:
    """Function summary: tertile assignment yields three buckets per language."""
    labels = ("conservative_leaning", "neutral", "liberal_leaning")
    df = pd.DataFrame(
        {
            "assigned_primary_lexicon": ["it"] * 9,
            "lexical_score": [float(i) for i in range(9)],
        }
    )
    out = assign_tertile_buckets(df, "assigned_primary_lexicon", "lexical_score", labels, "lexical_bucket")
    assert set(out["lexical_bucket"]) == set(labels)


def test_semantic_orientation_flip_sign() -> None:
    """Function summary: negative orientation multiplier flips semantic sign."""
    mult = {"it": -1.0}
    assert semantic_orientation_flip("it", mult) == -1.0
    assert semantic_orientation_flip("en", mult) == 1.0


def test_cohens_kappa_perfect_agreement() -> None:
    """Function summary: identical buckets yield kappa 1 and 100% exact match."""
    labels = ["conservative_leaning", "neutral", "liberal_leaning"]
    y = pd.Series(labels)
    assert pct_exact_match(y, y) == 1.0
    assert cohens_kappa(y, y, labels) == 1.0


def test_ideology_bucket_config_from_yaml_shape() -> None:
    """Function summary: config parser returns expected defaults when block empty."""
    cfg = ideology_bucket_config({})
    assert cfg.min_pre_words == 400
    assert len(cfg.bucket_labels) == 3


def test_load_orientation_defaults_without_file() -> None:
    """Function summary: missing orientation report returns +1 for all langs."""
    mult = load_semantic_orientation_multipliers({"paths": {"tables_dir": "/nonexistent"}})
    assert mult["it"] == 1.0
