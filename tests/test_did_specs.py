"""Tests for DiD calendar and treatment specs."""

from __future__ import annotations

import pandas as pd

from src.did.specs import (
    CONTROL_FAMILIES,
    CONTROL_LEXICONS,
    ITALY_FAMILIES,
    StrategySpec,
    activate_post_phases_from_config,
    apply_post_window,
    default_strategies,
    event_study_overlay_strategies,
    filter_strategy_sample,
    headline_base_strategies,
    is_author_strategy,
    is_entity_fe_only_strategy,
    italy_only_strategies,
    lexicon_for_control_family,
    load_post_phases,
    post_phase_strategies,
    rel_day_from_date,
    reset_post_phase_bounds,
    spec_label_short,
    strategy_label,
)


def test_rel_day_launch() -> None:
    """Function summary: rel_day is 0 on ban date."""
    dates = pd.Series(["2023-03-30", "2023-03-31", "2023-04-01"])
    rel = rel_day_from_date(dates, "2023-03-31")
    assert list(rel) == [-1, 0, 1]


def test_early_ban_post_window() -> None:
    """Function summary: early-ban post is 1 only for rel_day 0..6."""
    df = pd.DataFrame({"rel_day": [-1, 0, 6, 7, 14], "post": 1})
    out = apply_post_window(df, "early_ban_7d", "")
    assert list(out["post"]) == [0, 1, 1, 0, 0]


def test_default_strategies_early_ban_vs_control() -> None:
    """Function summary: single-control-family strategies include early_ban_7d clones."""
    specs = default_strategies()
    early_vs = [
        s
        for s in specs
        if s.post_mode == "early_ban_7d" and s.strategy_id.startswith("cross_country_vs_")
    ]
    assert len(early_vs) == len(CONTROL_FAMILIES)
    assert {s.control_family for s in early_vs} == set(CONTROL_FAMILIES)


def test_post_first_2bd_calendar_window() -> None:
    """Function summary: post_first_2bd is 1 only on 2023-04-03 and 2023-04-04."""
    df = pd.DataFrame(
        {
            "date_utc": ["2023-04-02", "2023-04-03", "2023-04-04", "2023-04-05"],
            "rel_day": [2, 3, 4, 5],
            "post": 1,
        }
    )
    out = apply_post_window(df, "post_first_2bd", "2023-03-31")
    assert list(out["post"]) == [0, 1, 1, 0]


def test_post_phase_apply_post_window_defaults() -> None:
    """Function summary: post-phase post indicators match default rel_day windows."""
    activate_post_phases_from_config(None)
    df = pd.DataFrame({"rel_day": [-1, 0, 2, 3, 9, 10, 25], "post": 1})
    assert list(apply_post_window(df.copy(), "post_short_3d", "")["post"]) == [0, 1, 1, 0, 0, 0, 0]
    assert list(apply_post_window(df.copy(), "post_medium_7d", "")["post"]) == [0, 0, 0, 1, 1, 0, 0]
    assert list(apply_post_window(df.copy(), "post_long_tail", "")["post"]) == [0, 0, 0, 0, 0, 1, 1]


def test_load_post_phases_yaml_override_short() -> None:
    """Function summary: did.post_phases.short overrides short-window bounds."""
    cfg = {"did": {"post_phases": {"short": {"rel_day_min": 0, "rel_day_max": 1}}}}
    out = load_post_phases(cfg)
    assert out["post_short_3d"] == (0, 1)
    assert out["post_medium_7d"] == (3, 9)
    activate_post_phases_from_config(cfg)
    try:
        df = pd.DataFrame({"rel_day": [0, 1, 2], "post": 1})
        assert list(apply_post_window(df.copy(), "post_short_3d", "")["post"]) == [1, 1, 0]
    finally:
        reset_post_phase_bounds()


def test_post_phase_strategies_count() -> None:
    """Function summary: post_phase_strategies yields 6 strategies × 4 post phases (incl. post_first_2bd)."""
    from src.did.specs import headline_base_strategies, post_phase_strategies

    assert len(post_phase_strategies()) == 24
    assert len(headline_base_strategies()) == 6


def test_filter_it_political_vs_controls() -> None:
    """Function summary: treated_family it_political keeps IT political + controls."""
    panel = pd.DataFrame(
        {
            "subreddit": ["a", "b", "c"],
            "topic_family": ["it_political", "it_others", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    strat = StrategySpec("x", treated_family="it_political")
    out = filter_strategy_sample(panel, strat)
    assert set(out["subreddit"]) == {"a", "c"}
    assert out.loc[out["subreddit"] == "a", "treat"].iloc[0] == 1
    assert out.loc[out["subreddit"] == "c", "treat"].iloc[0] == 0


def test_filter_treated_topic() -> None:
    """Function summary: treated_topic keeps topic subs + controls."""
    panel = pd.DataFrame(
        {
            "subreddit": ["a", "b", "c"],
            "topic": ["it_pure_political", "it_others", "de"],
            "topic_family": ["it_political", "it_others", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    strat = StrategySpec("t", treated_topic="it_pure_political")
    out = filter_strategy_sample(panel, strat)
    assert set(out["subreddit"]) == {"a", "c"}
    assert out.loc[out["subreddit"] == "a", "treat"].iloc[0] == 1


def test_italy_only_post_filter() -> None:
    """Function summary: italy_only_post keeps IT families only with treat=1."""
    panel = pd.DataFrame(
        {
            "subreddit": ["a", "b", "c"],
            "topic_family": ["it_political", "it_others", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    out = filter_strategy_sample(panel, StrategySpec("italy_only_post"))
    assert set(out["subreddit"]) == {"a", "b"}
    assert (out["treat"] == 1).all()


def test_is_entity_fe_only_strategy() -> None:
    """Function summary: entity-FE-only strategies include italy_only_post and author_it_ban."""
    assert is_entity_fe_only_strategy("italy_only_post")
    assert is_entity_fe_only_strategy("author_it_ban")
    assert not is_entity_fe_only_strategy("cross_country_all")


def test_italy_only_strategies_singleton() -> None:
    """Function summary: italy_only_strategies returns one spec."""
    assert len(italy_only_strategies()) == 1
    assert italy_only_strategies()[0].strategy_id == "italy_only_post"


def test_author_it_ban_treat_constant() -> None:
    """Function summary: author_it_ban sets treat=1 for IT cohort."""
    panel = pd.DataFrame(
        {
            "author": ["u1", "u2"],
            "primary_lexicon": ["it", "it"],
            "rel_day": [0, 1],
            "post": [1, 1],
            "IT": [1, 1],
        }
    )
    strat = StrategySpec("author_it_ban")
    out = filter_strategy_sample(panel, strat)
    assert (out["treat"] == 1).all()


def test_strategy_label_known() -> None:
    """Function summary: strategy_label returns readable text."""
    assert "Italian" in strategy_label("cross_country_all")
    assert "triple" in strategy_label("within_italy_ddd").lower()


def test_strategy_label_short_shorter_than_full() -> None:
    """Function summary: short plot labels are more compact than full labels."""
    full = strategy_label("cross_country_all")
    short = strategy_label("cross_country_all", short=True)
    assert len(short) < len(full)
    assert "IT" in short


def test_strategy_label_short_14d_suffix() -> None:
    """Function summary: _14d strategy ids get compact suffix when short."""
    label = strategy_label("cross_country_all_14d", short=True)
    assert "(14d)" in label


def test_spec_label_short() -> None:
    """Function summary: spec_label_short maps post windows."""
    assert spec_label_short("full_ban") == "full"
    assert spec_label_short("early_ban_7d") == "7d"
    assert spec_label_short("post_short_3d") == "short 0–2d"


def test_italy_families_constant() -> None:
    """Function summary: Italy families include political and others."""
    assert "it_political" in ITALY_FAMILIES
    assert "it_others" in ITALY_FAMILIES


def test_is_author_strategy() -> None:
    """Function summary: author strategy ids detected."""
    assert is_author_strategy("author_it_ban")
    assert not is_author_strategy("cross_country_all")


def test_event_study_overlay_strategies_have_control_family() -> None:
    """Function summary: overlay bundle uses full StrategySpec rows from default_strategies."""
    specs = event_study_overlay_strategies()
    assert len(specs) == 5
    vs_de = next(s for s in specs if s.strategy_id == "cross_country_vs_de")
    assert vs_de.control_family == "de"
    pooled = next(s for s in specs if s.strategy_id == "cross_country_all")
    assert pooled.control_family is None


def test_placebo_in_space_eligible_strategy() -> None:
    """Function summary: placebo-in-space only for pooled multi-control contrasts."""
    from src.did.specs import is_placebo_in_space_eligible_strategy

    assert is_placebo_in_space_eligible_strategy("cross_country_all")
    assert is_placebo_in_space_eligible_strategy("cross_country_it_political")
    assert not is_placebo_in_space_eligible_strategy("cross_country_vs_de")
    assert not is_placebo_in_space_eligible_strategy("cross_country_vs_us")
    assert not is_placebo_in_space_eligible_strategy("within_italy_ddd")


def test_event_study_language_universe_slice_strategies() -> None:
    """Function summary: in/out slice overlay exposes two political-universe strategies."""
    from src.did.specs import (
        EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS,
        event_study_language_universe_slice_strategies,
    )

    specs = event_study_language_universe_slice_strategies()
    assert len(specs) == 2
    assert tuple(s.strategy_id for s in specs) == EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS
    assert specs[0].universe_slice == "in_political_tree"
    assert specs[1].universe_slice == "out_political_tree"


def test_filter_topic_family_overlay_entity_counts() -> None:
    """Function summary: single-control strategies subset entities vs pooled."""
    panel = pd.DataFrame(
        {
            "topic_family": ["it_political", "it_others", "de", "eu", "us", "uk"],
            "entity_id": ["it_political", "it_others", "de", "eu", "us", "uk"],
            "date_utc": ["2023-03-31"] * 6,
            "rel_day": [0] * 6,
            "post": [1] * 6,
            "IT": [1, 1, 0, 0, 0, 0],
        }
    )
    all_ent = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_all"), window_days=14)["entity_id"]
    )
    de_ent = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_vs_de", control_family="de"), window_days=14)[
            "entity_id"
        ]
    )
    assert len(all_ent) == 6
    assert de_ent == {"it_political", "it_others", "de"}


def test_filter_language_panel_hub_entities() -> None:
    """Function summary: hub-split language panels filter eu/us/uk separately."""
    panel = pd.DataFrame(
        {
            "language_hub": ["it", "de", "eu", "us", "uk"],
            "entity_id": ["it", "de", "eu", "us", "uk"],
            "primary_lexicon": ["it", "de", "en", "en", "en"],
            "date_utc": ["2023-03-31"] * 5,
            "rel_day": [0] * 5,
            "post": [1] * 5,
            "IT": [1, 0, 0, 0, 0],
        }
    )
    eu_ent = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_vs_eu", control_family="eu"), window_days=14)[
            "entity_id"
        ]
    )
    us_ent = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_vs_us", control_family="us"), window_days=14)[
            "entity_id"
        ]
    )
    assert eu_ent == {"it", "eu"}
    assert us_ent == {"it", "us"}
    assert eu_ent != us_ent


def test_filter_language_panel_control_lexicon() -> None:
    """Function summary: legacy lexicon panels map eu/us/uk controls to en lexicon."""
    panel = pd.DataFrame(
        {
            "primary_lexicon": ["it", "en", "de"],
            "entity_id": ["it", "en", "de"],
            "date_utc": ["2023-03-31"] * 3,
            "rel_day": [0] * 3,
            "post": [1] * 3,
            "IT": [1, 0, 0],
        }
    )
    assert lexicon_for_control_family("eu") == "en"
    eu_ent = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_vs_eu", control_family="eu"), window_days=14)[
            "entity_id"
        ]
    )
    assert eu_ent == {"it", "en"}
    pooled = set(
        filter_strategy_sample(panel, StrategySpec("cross_country_all"), window_days=14)["entity_id"]
    )
    assert pooled == {"it", "en", "de"}
    assert CONTROL_LEXICONS == frozenset({"de", "en"})
