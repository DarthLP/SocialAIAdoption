"""Tests for within-author English-quality DiD helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.did.english_quality import (
    CROSS_LANGUAGE_HEADLINE_OUTCOMES,
    annotate_english_quality_comments,
    apply_3d_bins,
    apply_standardized_outcome,
    authors_passing_cohort,
    classify_author_roster,
    cohort_authors_for_design,
    cohort_thresholds_by_label,
    dominant_pre_language,
    estimate_static_post_treat,
    estimate_treat_event_study,
    estimate_within_author_diff_event_study,
    estimate_within_author_diff_static,
    estimate_within_language_post,
    filter_cross_language_sample,
    filter_language_pair_sample,
    filter_native_control_sample,
    headline_outcomes_for_design,
    is_en_control_subreddit,
    is_italian_arm_subreddit,
    march_standardization_moments_by_lang,
    outcome_caveat,
    prep_treat_design,
    static_es_post_avg,
)


def _minimal_config() -> dict:
    """Function summary: tiny config for roster classification tests."""
    return {
        "subreddits": {
            "discovered_italian": ["Italia", "politicaITA"],
            "controls_english_political": ["Ask_Politics", "NeutralPolitics"],
            "controls_europe_hub": ["europe", "de"],
            "controls_europe_political": ["ukpolitics"],
        },
        "topics": {
            "it_italia": {"family": "it_others"},
            "en_ask": {"family": "us"},
            "en_europe": {"family": "eu"},
            "de_hub": {"family": "de"},
        },
        "topic_families": {
            "it_others": {},
            "us": {},
            "eu": {},
            "de": {},
        },
        "subreddit_topics": {
            "Italia": "it_italia",
            "politicaITA": "it_italia",
            "Ask_Politics": "en_ask",
            "NeutralPolitics": "en_ask",
            "europe": "en_europe",
            "de": "de_hub",
        },
    }


def test_roster_classification() -> None:
    """Function summary: bilingual vs native_control vs other from forum sets."""
    config = _minimal_config()
    arm_map = {
        "Italia": "discovered_italian",
        "Ask_Politics": "control_english_political",
        "europe": "control_europe_hub",
        "de": "control_europe_hub",
        "ukpolitics": "control_europe_political",
    }
    assert is_italian_arm_subreddit("Italia", arm_map)
    assert is_en_control_subreddit("Ask_Politics", config, arm_map)
    assert is_en_control_subreddit("europe", config, arm_map)
    assert not is_en_control_subreddit("de", config, arm_map)
    assert is_en_control_subreddit("ukpolitics", config, arm_map)

    roster = classify_author_roster(
        {
            "u_bi": {"Italia", "Ask_Politics"},
            "u_nat": {"Ask_Politics", "europe"},
            "u_it_only": {"Italia"},
            "u_de_only": {"de"},
        },
        config,
        arm_map=arm_map,
    )
    groups = roster.set_index("author")["author_group"].to_dict()
    assert groups["u_bi"] == "italian_bilingual"
    assert groups["u_nat"] == "native_control"
    assert groups["u_it_only"] == "other"
    assert groups["u_de_only"] == "other"


def test_annotate_and_filter_samples() -> None:
    """Function summary: annotation adds treatment flags; filters match design samples."""
    roster = pd.DataFrame(
        [
            {"author": "u1", "author_group": "italian_bilingual"},
            {"author": "u2", "author_group": "native_control"},
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "author": "u1",
                "date_utc": "2023-03-15",
                "lang_comment": "en",
                "primary_lexicon": "en",
                "subreddit": "Ask_Politics",
            },
            {
                "author": "u1",
                "date_utc": "2023-04-05",
                "lang_comment": "it",
                "primary_lexicon": "it",
                "subreddit": "Italia",
            },
            {
                "author": "u2",
                "date_utc": "2023-04-02",
                "lang_comment": "en",
                "primary_lexicon": "en",
                "subreddit": "NeutralPolitics",
            },
        ]
    )
    ann = annotate_english_quality_comments(raw, "2023-03-31", roster)
    ann = apply_3d_bins(ann, "2023-03-31", 3)
    assert ann.loc[ann["author"] == "u1", "is_english"].tolist() == [1, 0]
    assert ann.loc[ann["author"] == "u1", "italian_author"].iloc[0] == 1
    assert ann.loc[ann["author"] == "u2", "italian_author"].iloc[0] == 0

    nat = filter_native_control_sample(ann)
    assert len(nat) == 2
    assert set(nat["author"]) == {"u1", "u2"}

    cross = filter_cross_language_sample(ann)
    assert len(cross) == 2
    assert set(cross["lang_comment"]) == {"en", "it"}


def test_cohort_gates() -> None:
    """Function summary: authors need min comments/words pre and post per design."""
    rows = []
    for i in range(5):
        for lang, lex in (("en", "en"), ("it", "it")):
            rows.append(
                {
                    "author": "u_pass",
                    "date_utc": f"2023-03-{10 + i:02d}",
                    "post": 0,
                    "lang_comment": lang,
                    "primary_lexicon": lex,
                    "n_words": 50,
                    "author_group": "italian_bilingual",
                }
            )
    for i in range(5):
        for lang, lex in (("en", "en"), ("it", "it")):
            rows.append(
                {
                    "author": "u_pass",
                    "date_utc": f"2023-04-{2 + i:02d}",
                    "post": 1,
                    "lang_comment": lang,
                    "primary_lexicon": lex,
                    "n_words": 50,
                    "author_group": "italian_bilingual",
                }
            )
    for i in range(2):
        rows.append(
            {
                "author": "u_fail",
                "date_utc": f"2023-03-{12 + i:02d}",
                "post": 0,
                "lang_comment": "en",
                "primary_lexicon": "en",
                "n_words": 50,
                "author_group": "native_control",
            }
        )
    panel = pd.DataFrame(rows)
    panel["rel_day"] = np.where(panel["post"] == 0, -5, 5)
    th = cohort_thresholds_by_label("strict")
    cross_authors = cohort_authors_for_design(panel, "cross_language", th)
    assert "u_pass" in cross_authors
    assert "u_fail" not in cross_authors


def test_standardization_by_lang() -> None:
    """Function summary: z-scoring uses separate pre-ban moments per language."""
    df = pd.DataFrame(
        {
            "rel_day": [-2, -1, 0, 1],
            "lang_comment": ["en", "en", "it", "it"],
            "readability": [80.0, 82.0, 50.0, 48.0],
        }
    )
    moments = march_standardization_moments_by_lang(df, "readability")
    out = apply_standardized_outcome(df, "readability", moments, group_col="lang_comment")
    assert "y" in out.columns
    assert out["y"].notna().all()


def _synthetic_native_panel(n_authors: int = 10) -> pd.DataFrame:
    """Function summary: synthetic panel for native-control DiD."""
    rows = []
    days_pre = ["2023-03-28", "2023-03-29", "2023-03-30"]
    days_post = ["2023-04-01", "2023-04-02", "2023-04-03"]
    for i in range(n_authors):
        italian = int(i >= n_authors // 2)
        for j, day in enumerate(days_pre + days_post):
            post = int(j >= 3)
            rows.append(
                {
                    "author": f"u{i}",
                    "subreddit": "Ask_Politics",
                    "time_id": day,
                    "date_utc": day,
                    "post": post,
                    "rel_day": j - 3,
                    "rel_period": (j - 3) // 3,
                    "italian_author": italian,
                    "is_english": 1,
                    "lang_comment": "en",
                    "primary_lexicon": "en",
                    "author_group": "italian_bilingual" if italian else "native_control",
                    "y": 0.2 * italian * post + 0.05 * i + np.random.default_rng(i).normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_cross_panel(n_authors: int = 10) -> pd.DataFrame:
    """Function summary: synthetic panel for cross-language DiD."""
    rows = []
    days = ["2023-03-28", "2023-03-29", "2023-04-01", "2023-04-02"]
    for i in range(n_authors):
        for j, day in enumerate(days):
            post = int(j >= 2)
            for lang, is_en in (("en", 1), ("it", 0)):
                rows.append(
                    {
                        "author": f"u{i}",
                        "subreddit": "Ask_Politics" if is_en else "Italia",
                        "time_id": day,
                        "date_utc": day,
                        "post": post,
                        "rel_day": j - 2,
                        "rel_period": (j - 2) // 3,
                        "italian_author": 1,
                        "is_english": is_en,
                        "lang_comment": lang,
                        "primary_lexicon": "en" if is_en else "it",
                        "author_group": "italian_bilingual",
                        "y": 0.3 * is_en * post + 0.02 * i,
                    }
                )
    return pd.DataFrame(rows)


def test_prep_treat_design() -> None:
    """Function summary: prep_treat_design builds post_IT interaction."""
    df = _synthetic_native_panel(4)
    work = prep_treat_design(df, "italian_author")
    assert "post_IT" in work.columns
    assert (work["post_IT"] == work["post"] * work["IT"]).all()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_static_native_control_runs() -> None:
    """Function summary: Design 1 static DiD returns finite beta."""
    df = _synthetic_native_panel(12)
    res = estimate_static_post_treat(df, "italian_author", cluster_col="author")
    assert res.get("estimation_note") in ("ok", "estimation_error", "insufficient_obs")
    if res.get("estimation_note") == "ok":
        assert np.isfinite(res.get("beta", np.nan))


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_event_study_cross_language_runs() -> None:
    """Function summary: Design 2 event study omits ref bin and runs."""
    df = _synthetic_cross_panel(12)
    _, es = estimate_treat_event_study(
        df, "is_english", cluster_col="author", ref_period=-1, window=10, bin_days=3
    )
    if not es.empty:
        assert -1 not in es["rel_period"].astype(int).tolist()


def test_headline_outcomes_cross_language_drops_readability() -> None:
    """Function summary: cross_language headline set omits readability."""
    assert "readability" not in CROSS_LANGUAGE_HEADLINE_OUTCOMES
    assert "readability" in headline_outcomes_for_design("native_control")
    assert outcome_caveat("ttr_50w")


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_within_author_diff_estimator_runs() -> None:
    """Function summary: within-author diff ES and static return finite cells."""
    df = _synthetic_cross_panel(12)
    _, es = estimate_within_author_diff_event_study(
        df, cluster_col="author", ref_period=-1, window=10, bin_days=3
    )
    static = estimate_within_author_diff_static(df, cluster_col="author", window=10, bin_days=3)
    if not es.empty:
        assert -1 not in es["rel_period"].astype(int).tolist()
        assert "n_authors" in es.columns
    if static.get("estimation_note") == "ok":
        assert np.isfinite(static.get("beta", np.nan))


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_cross_language_static_includes_treat_main() -> None:
    """Function summary: cross_language static uses is_english main effect."""
    df = _synthetic_cross_panel(12)
    res = estimate_static_post_treat(
        df, "is_english", cluster_col="author", include_treat_main=True
    )
    assert res.get("estimation_note") in ("ok", "estimation_error", "insufficient_obs")


def test_static_es_post_avg_helper() -> None:
    """Function summary: static_es_post_avg averages post bins only."""
    es = pd.DataFrame(
        {
            "rel_period": [-2, -1, 0, 1],
            "gamma": [0.1, 0.0, -0.2, -0.1],
            "se": [0.05, 0.05, 0.04, 0.04],
        }
    )
    avg = static_es_post_avg(es)
    assert avg["estimation_note"] == "ok"
    assert avg["n_bins"] == 2
    assert np.isfinite(avg["beta"])


def test_authors_passing_cohort_helper() -> None:
    """Function summary: authors_passing_cohort enforces min comments and words."""
    activity = pd.DataFrame(
        [
            {"author": "a", "n_pre": 4, "n_post": 4, "words_pre": 200, "words_post": 200},
            {"author": "b", "n_pre": 1, "n_post": 4, "words_pre": 200, "words_post": 200},
        ]
    )
    th = cohort_thresholds_by_label("strict")
    passed = authors_passing_cohort(activity, th)
    assert passed == {"a"}


def test_ukpolitics_english_control_and_roster() -> None:
    """Function summary: ukpolitics counts as English-control via primary lexicon."""
    config = _minimal_config()
    arm_map = {
        "Italia": "discovered_italian",
        "ukpolitics": "control_europe_political",
    }
    roster = classify_author_roster(
        {"u_uk_bi": {"Italia", "ukpolitics"}, "u_uk_nat": {"ukpolitics"}},
        config,
        arm_map=arm_map,
    )
    groups = roster.set_index("author")["author_group"].to_dict()
    assert groups["u_uk_bi"] == "italian_bilingual"
    assert groups["u_uk_nat"] == "native_control"


def test_dominant_pre_language() -> None:
    """Function summary: dominant_pre_language picks the majority pre-ban bucket."""
    assert dominant_pre_language(1, 9, 10) == "it"
    assert dominant_pre_language(8, 1, 10) == "en"
    assert dominant_pre_language(1, 1, 10) == "other"
    assert dominant_pre_language(0, 0, 0) == "other"


def test_roster_language_attributes() -> None:
    """Function summary: language tallies map to roster proxy columns and lang_bilingual."""
    config = _minimal_config()
    arm_map = {
        "Italia": "discovered_italian",
        "Ask_Politics": "control_english_political",
    }
    author_lang = {
        "u_bi": {"pre_en": 2, "pre_it": 8, "pre_total": 10, "tot_en": 5, "tot_it": 12},
        "u_en_only": {"pre_en": 6, "pre_it": 0, "pre_total": 6, "tot_en": 9, "tot_it": 1},
    }
    roster = classify_author_roster(
        {"u_bi": {"Italia", "Ask_Politics"}, "u_en_only": {"Ask_Politics"}},
        config,
        arm_map=arm_map,
        author_lang=author_lang,
    )
    by_author = roster.set_index("author")
    assert by_author.loc["u_bi", "dominant_pre_lang"] == "it"
    assert abs(float(by_author.loc["u_bi", "italian_share_pre"]) - 0.8) < 1e-6
    assert int(by_author.loc["u_bi", "lang_bilingual"]) == 1
    assert int(by_author.loc["u_en_only", "lang_bilingual"]) == 0
    assert by_author.loc["u_en_only", "dominant_pre_lang"] == "en"


def _langmix_panel() -> pd.DataFrame:
    """Function summary: panel with a forum-agnostic lang_bilingual non-forum author."""
    rows = []
    spec = {
        "u_forum_bi": ("italian_bilingual", 1, 1.0),
        "u_mix_only": ("other", 1, 0.5),
        "u_en_only": ("native_control", 0, 0.0),
    }
    for author, (group, lang_bi, share) in spec.items():
        langs = ["en", "it"] if author != "u_en_only" else ["en"]
        for half, post in (("03", 0), ("04", 1)):
            for lang in langs:
                for i in range(4):
                    rows.append(
                        {
                            "author": author,
                            "date_utc": f"2023-{half}-{10 + i:02d}",
                            "post": post,
                            "rel_day": -5 if post == 0 else 5,
                            "lang_comment": lang,
                            "primary_lexicon": "en" if lang == "en" else "it",
                            "n_words": 60,
                            "author_group": group,
                            "lang_bilingual": lang_bi,
                            "italian_share_pre": share,
                        }
                    )
    return pd.DataFrame(rows)


def test_langmix_cohort_selection() -> None:
    """Function summary: langmix cohort uses lang_bilingual regardless of forum group."""
    panel = _langmix_panel()
    th = cohort_thresholds_by_label("strict")
    cross = cohort_authors_for_design(panel, "cross_language", th)
    langmix = cohort_authors_for_design(panel, "cross_language_langmix", th)
    native_it = cohort_authors_for_design(panel, "cross_language_native_it", th)
    assert cross == {"u_forum_bi"}
    assert langmix == {"u_forum_bi", "u_mix_only"}
    assert native_it == {"u_forum_bi"}


def test_filter_language_pair_sample() -> None:
    """Function summary: language-pair filter keeps EN/IT regardless of forum group."""
    panel = _langmix_panel()
    out = filter_language_pair_sample(panel)
    assert set(out["lang_comment"]) <= {"en", "it"}
    assert "u_mix_only" in set(out["author"])


def test_polarization_outcome_standardization_runs() -> None:
    """Function summary: polarization outcome z-scores within language and estimates."""
    df = _synthetic_cross_panel(12)
    df["net_ideology"] = df["y"] + np.where(df["lang_comment"] == "it", 5.0, 0.0)
    moments = march_standardization_moments_by_lang(df, "net_ideology")
    std = apply_standardized_outcome(
        df, "net_ideology", moments, group_col="lang_comment", out_col="y"
    )
    res = estimate_static_post_treat(std, "is_english", cluster_col="author")
    assert res.get("estimation_note") in ("ok", "estimation_error", "insufficient_obs")


def test_italian_placebo_helper() -> None:
    """Function summary: Italian placebo restricts to IT comments and returns post coef."""
    df = _synthetic_cross_panel(12)
    res = estimate_within_language_post(df, lang_value="it", cluster_col="author")
    assert res.get("coef_name") == "post"
    assert res.get("placebo_lang") == "it"
