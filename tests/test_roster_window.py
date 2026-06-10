"""Tests for pre-ban vs full roster-window classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

import pandas as pd

from scripts.diagnostics.build_english_quality_roster import collect_author_activity
from src.did.english_quality import (
    LANG_BILINGUAL_MIN_COMMENTS,
    classify_author_roster,
    english_quality_run_tables_dir,
    roster_window_subdir,
)


def _minimal_config() -> dict:
    """Function summary: tiny config for roster classification tests."""
    return {
        "subreddits": {
            "discovered_italian": ["Italia"],
            "controls_english_political": ["Ask_Politics"],
        },
        "topics": {
            "it_italia": {"family": "it_others"},
            "en_ask": {"family": "us"},
        },
        "topic_families": {
            "it_others": {},
            "us": {},
        },
        "subreddit_topics": {
            "Italia": "it_italia",
            "Ask_Politics": "en_ask",
        },
        "paths": {"tables_dir": "results/tables/italy_polarization"},
    }


def _classify_from_frames(
    frames_by_sub: Dict[str, pd.DataFrame],
    *,
    roster_window: str,
    launch: str = "2023-03-31",
) -> pd.DataFrame:
    """Function summary: run roster classification on in-memory shard frames.

    Parameters:
    - frames_by_sub: subreddit -> comment DataFrame.
    - roster_window: pre_ban or full.
    - launch: ban launch day for pre/post split.

    Returns:
    - classify_author_roster output DataFrame.
    """
    author_forums: Dict[str, Set[str]] = defaultdict(set)
    author_lang: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"pre_en": 0, "pre_it": 0, "pre_total": 0, "tot_en": 0, "tot_it": 0}
    )
    start, end_excl = "2023-03-01", "2023-05-01"

    for sub, frame in frames_by_sub.items():
        frame = frame.copy()
        frame["date_utc"] = frame["date_utc"].astype(str)
        frame = frame[(frame["date_utc"] >= start) & (frame["date_utc"] < end_excl)]
        frame["author"] = frame["author"].astype(str)
        if frame.empty:
            continue
        is_pre = frame["date_utc"] < launch
        if roster_window == "pre_ban":
            classify_mask = is_pre
        else:
            classify_mask = pd.Series(True, index=frame.index)
        classify = frame.loc[classify_mask]
        for author in classify["author"].unique().tolist():
            author_forums[author].add(sub)
        lang = frame.get("lang_comment", pd.Series("", index=frame.index)).astype(str).str.lower()
        is_en = lang == "en"
        is_it = lang == "it"
        tmp = pd.DataFrame(
            {
                "author": frame["author"].values,
                "pre_total": is_pre.astype(int).values,
                "pre_en": (is_pre & is_en).astype(int).values,
                "pre_it": (is_pre & is_it).astype(int).values,
                "tot_en": (classify_mask & is_en).astype(int).values,
                "tot_it": (classify_mask & is_it).astype(int).values,
            }
        )
        agg = tmp.groupby("author")[["pre_total", "pre_en", "pre_it", "tot_en", "tot_it"]].sum()
        for author, row in agg.iterrows():
            stats = author_lang[str(author)]
            stats["pre_total"] += int(row["pre_total"])
            stats["pre_en"] += int(row["pre_en"])
            stats["pre_it"] += int(row["pre_it"])
            stats["tot_en"] += int(row["tot_en"])
            stats["tot_it"] += int(row["tot_it"])

    return classify_author_roster(dict(author_forums), _minimal_config(), author_lang=dict(author_lang))


def test_roster_window_subdir() -> None:
    """Function summary: roster_window_subdir returns expected path tags."""
    assert roster_window_subdir("pre_ban") == "roster_window=pre_ban"
    assert roster_window_subdir("full") == "roster_window=full"


def test_run_tables_dir_includes_subdir() -> None:
    """Function summary: english_quality_run_tables_dir nests under roster_window tag."""
    config = _minimal_config()
    path = english_quality_run_tables_dir(config, "pre_ban")
    assert path.name == "roster_window=pre_ban"
    assert path.parent.name == "english_quality"


def test_post_ban_forum_does_not_classify_bilingual_pre_ban() -> None:
    """Function summary: EN-control forum activity only post-ban -> other under pre_ban."""
    frames = {
        "Italia": pd.DataFrame(
            [
                {"author": "u_post_en_forum", "date_utc": "2023-03-15", "lang_comment": "it"},
            ]
        ),
        "Ask_Politics": pd.DataFrame(
            [
                {"author": "u_post_en_forum", "date_utc": "2023-04-05", "lang_comment": "en"},
            ]
        ),
    }
    roster_pre = _classify_from_frames(frames, roster_window="pre_ban")
    roster_full = _classify_from_frames(frames, roster_window="full")
    groups_pre = roster_pre.set_index("author")["author_group"].to_dict()
    groups_full = roster_full.set_index("author")["author_group"].to_dict()
    assert groups_pre["u_post_en_forum"] == "other"
    assert groups_full["u_post_en_forum"] == "italian_bilingual"


def test_post_ban_lang_share_does_not_set_lang_bilingual_pre_ban() -> None:
    """Function summary: lang_bilingual requires pre-ban EN+IT counts under pre_ban window."""
    rows = []
    for date, lang in (("2023-03-10", "en"), ("2023-03-12", "it")):
        rows.append({"author": "u_post_lang", "date_utc": date, "lang_comment": lang})
    for i in range(LANG_BILINGUAL_MIN_COMMENTS):
        rows.append(
            {"author": "u_post_lang", "date_utc": f"2023-04-{2 + i:02d}", "lang_comment": "en"}
        )
        rows.append(
            {"author": "u_post_lang", "date_utc": f"2023-04-{3 + i:02d}", "lang_comment": "it"}
        )
    frames = {"Italia": pd.DataFrame(rows)}
    roster_pre = _classify_from_frames(frames, roster_window="pre_ban")
    roster_full = _classify_from_frames(frames, roster_window="full")
    by_pre = roster_pre.set_index("author")
    by_full = roster_full.set_index("author")
    assert int(by_pre.loc["u_post_lang", "lang_bilingual"]) == 0
    assert int(by_full.loc["u_post_lang", "lang_bilingual"]) == 1


def test_collect_author_activity_accepts_roster_window_kwarg() -> None:
    """Function summary: collect_author_activity signature includes roster_window."""
    import inspect

    sig = inspect.signature(collect_author_activity)
    assert "roster_window" in sig.parameters
    assert sig.parameters["roster_window"].default == "pre_ban"
