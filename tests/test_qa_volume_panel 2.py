"""Tests for Q&A substitution question proxies, aggregation, and DiD flags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_utils import (
    is_qa_advice_subreddit,
    load_config,
    non_qa_italian_control_subreddits,
    qa_advice_subreddit_list,
)
from src.qa_substitution import (
    add_did_calendar_columns,
    add_treatment_flags,
    aggregate_subreddit_day,
    annotate_comment_questions,
    ban_phase,
    phase_contrast_table,
    reindex_full_grid,
    score_comment_question,
)


def test_score_comment_question_counts() -> None:
    """Function summary: question mark counting and is_question flag."""
    assert score_comment_question("a?b?") == (1, 2)
    assert score_comment_question("no questions here") == (0, 0)
    assert score_comment_question("") == (0, 0)
    assert score_comment_question(None) == (0, 0)


def test_aggregate_subreddit_day() -> None:
    """Function summary: subreddit-day aggregation of volume and question metrics."""
    df = pd.DataFrame(
        {
            "subreddit": ["Avvocati", "Avvocati", "Italia"],
            "date_utc": ["2023-03-01", "2023-03-01", "2023-03-01"],
            "author": ["a1", "a2", "a3"],
            "n_words": [10, 20, 5],
            "topic_family": ["it_others", "it_others", "it_others"],
        }
    )
    df = annotate_comment_questions(df.assign(body=["help?", "ok", "what?"]))
    out = aggregate_subreddit_day(df)
    avv = out[out["subreddit"] == "Avvocati"].iloc[0]
    assert int(avv["n_comments"]) == 2
    assert int(avv["n_questions"]) == 1
    assert avv["question_share"] == 0.5
    assert avv["qmark_rate_100w"] == 100.0 / 30.0  # one ? over 30 words in Avvocati day


def test_did_calendar_boundaries() -> None:
    """Function summary: rel_day, post, and phase flags at ban and lift dates."""
    panel = pd.DataFrame({"date_utc": ["2023-03-30", "2023-03-31", "2023-04-27", "2023-04-28"]})
    out = add_did_calendar_columns(panel, "2023-03-31", "2023-04-28", "2023-05-01")
    assert ban_phase("2023-03-30", "2023-03-31", "2023-04-28") == "pre"
    assert ban_phase("2023-03-31", "2023-03-31", "2023-04-28") == "ban"
    assert ban_phase("2023-04-28", "2023-03-31", "2023-04-28") == "post"
    row_pre = out[out["date_utc"] == "2023-03-30"].iloc[0]
    row_ban = out[out["date_utc"] == "2023-03-31"].iloc[0]
    row_post = out[out["date_utc"] == "2023-04-28"].iloc[0]
    assert int(row_pre["post"]) == 0
    assert int(row_ban["post"]) == 1
    assert row_pre["phase"] == "pre"
    assert row_ban["phase"] == "ban"
    assert row_post["phase"] == "post"
    assert int(row_pre["rel_day"]) == -1
    assert int(row_ban["rel_day"]) == 0


def test_treatment_flags() -> None:
    """Function summary: qa and IT indicators from subreddit membership."""
    panel = pd.DataFrame(
        {
            "subreddit": ["Avvocati", "Italia", "de"],
            "topic_family": ["it_others", "it_others", "de"],
            "date_utc": ["2023-03-01"] * 3,
            "post": [0, 0, 0],
        }
    )
    qa_set = frozenset({"Avvocati"})
    out = add_treatment_flags(panel, qa_set)
    assert int(out.loc[out["subreddit"] == "Avvocati", "qa"].iloc[0]) == 1
    assert int(out.loc[out["subreddit"] == "Italia", "qa"].iloc[0]) == 0
    assert int(out.loc[out["subreddit"] == "de", "is_hub"].iloc[0]) == 1


def test_phase_contrast_uses_mean_not_mislabeled_sum() -> None:
    """Function summary: phase table reports subreddit-day means for count outcomes."""
    panel = pd.DataFrame(
        {
            "qa": [1, 1, 1, 0, 0],
            "phase": ["pre", "pre", "ban", "pre", "ban"],
            "n_comments": [10, 20, 5, 100, 50],
            "n_questions": [2, 4, 1, 10, 5],
            "n_authors": [3, 4, 2, 8, 6],
            "question_share": [0.2, 0.2, 0.2, 0.1, 0.1],
            "qmark_rate_100w": [1.0, 1.0, 1.0, 0.5, 0.5],
        }
    )
    out = phase_contrast_table(panel, group_col="qa")
    qa_pre = out[(out["qa"] == 1) & (out["phase"] == "pre")].iloc[0]
    assert qa_pre["n_comments_mean"] == 15.0
    assert qa_pre["n_comments_sum"] == 30.0
    assert qa_pre["n_questions_mean"] == 3.0


def test_reindex_full_grid_zero_fills_missing_days() -> None:
    """Function summary: sparse panel expands to full calendar with zeros and NaN rates."""
    sparse = pd.DataFrame(
        {
            "subreddit": ["Avvocati", "Avvocati"],
            "date_utc": ["2023-03-01", "2023-03-03"],
            "n_comments": [5, 2],
            "n_authors": [3, 2],
            "n_questions": [1, 0],
            "question_share": [0.2, 0.0],
            "qmark_rate_100w": [10.0, 0.0],
            "total_words": [100.0, 50.0],
            "qmark_count": [10, 0],
            "topic_family": ["it_others", "it_others"],
        }
    )
    family_map = {"Avvocati": "it_others", "Italia": "it_others"}
    out = reindex_full_grid(
        sparse,
        subreddits=["Avvocati", "Italia"],
        start="2023-03-01",
        end_excl="2023-03-04",
        family_map=family_map,
    )
    assert len(out) == 6  # 2 subs x 3 days
    zero_day = out[(out["subreddit"] == "Avvocati") & (out["date_utc"] == "2023-03-02")].iloc[0]
    assert int(zero_day["n_comments"]) == 0
    assert pd.isna(zero_day["question_share"])
    assert pd.isna(zero_day["qmark_rate_100w"])
    italia_day = out[(out["subreddit"] == "Italia") & (out["date_utc"] == "2023-03-01")].iloc[0]
    assert int(italia_day["n_comments"]) == 0
    assert italia_day["topic_family"] == "it_others"
    out["post"] = 0
    flagged = add_treatment_flags(out, frozenset({"Avvocati"}))
    assert int(flagged.loc[flagged["subreddit"] == "Avvocati", "qa"].iloc[0]) == 1
    assert int(flagged.loc[flagged["subreddit"] == "Italia", "qa"].iloc[0]) == 0


def test_config_qa_lists() -> None:
    """Function summary: metadata Q&A list and derived non-Q&A controls load from config."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        return
    config = load_config(cfg_path)
    qa = qa_advice_subreddit_list(config)
    assert "DomandeDaReddit" in qa
    assert "Avvocati" in qa
    assert is_qa_advice_subreddit("Avvocati", config)
    assert not is_qa_advice_subreddit("Italia", config)
    controls = non_qa_italian_control_subreddits(config)
    assert "Italia" in controls
    assert "Avvocati" not in controls
