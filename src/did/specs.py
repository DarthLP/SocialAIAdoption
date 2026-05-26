"""
Treatment indicators, sample filters, and strategy definitions for DiD estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

ITALY_FAMILIES = frozenset({"it_political", "it_others"})
CONTROL_FAMILIES = frozenset({"de", "eu", "us", "uk"})
ITALIAN_TOPICS = ("it_pure_political", "it_political", "it_others")

STRATEGY_LABELS: dict[str, str] = {
    "cross_country_all": "Italian forums vs pooled controls (DE, EU, US, UK)",
    "cross_country_it_political": "Italian political forums vs controls",
    "cross_country_it_others": "Italian non-political forums vs controls",
    "cross_country_political_universe_in": "Italian vs controls — comments in political tree",
    "cross_country_political_universe_out": "Italian vs controls — comments outside political tree",
    "within_italy_ddd": "Within-Italy: ban × political-tree (triple-diff)",
    "cross_country_vs_de": "Italian vs Germany only",
    "cross_country_vs_eu": "Italian vs EU hub only",
    "cross_country_vs_us": "Italian vs US only",
    "cross_country_vs_uk": "Italian vs UK only",
    "cross_country_topic_it_pure_political": "Italian pure-political forums vs controls",
    "cross_country_topic_it_political": "Italian political-topic forums vs controls",
    "cross_country_topic_it_others": "Italian other-topic forums vs controls",
    "author_it_ban": "Italian-writing authors: post-ban shift (IT cohort)",
    "author_it_vs_en": "Italian vs English-writing authors",
    "author_it_vs_de": "Italian vs German-writing authors",
}

PLOT_STRATEGY_GROUPS: dict[str, tuple[str, ...]] = {
    "headline": (
        "cross_country_all",
        "cross_country_it_political",
        "cross_country_it_others",
        "cross_country_political_universe_in",
        "cross_country_political_universe_out",
        "within_italy_ddd",
    ),
    "by_topic": (
        "cross_country_topic_it_pure_political",
        "cross_country_topic_it_political",
        "cross_country_topic_it_others",
    ),
    "early_ban": (
        "cross_country_all",
        "cross_country_it_political",
        "cross_country_it_others",
    ),
    "ddd_only": ("within_italy_ddd",),
    "author_it": ("author_it_ban",),
    "author_cross": ("author_it_vs_en", "author_it_vs_de"),
    "full": (),  # empty = all strategies
}


@dataclass(frozen=True)
class StrategySpec:
    """Function summary: one identification strategy row in did_summary."""

    strategy_id: str
    treat_col: str = "treat"
    description: str = ""
    universe_slice: Optional[str] = None
    treated_family: Optional[str] = None
    treated_topic: Optional[str] = None
    control_family: Optional[str] = None
    post_mode: str = "full_ban"
    placebo: bool = False
    author_only: bool = False


def strategy_label(strategy_id: str) -> str:
    """Function summary: human-readable label for a strategy_id."""
    if strategy_id in STRATEGY_LABELS:
        return STRATEGY_LABELS[strategy_id]
    if strategy_id.endswith("_14d"):
        base = strategy_id[:-4]
        return STRATEGY_LABELS.get(base, base) + " (first 14 ban days)"
    for prefix in ("cross_country_all", "cross_country_it_political", "cross_country_it_others"):
        if strategy_id.startswith(prefix) and strategy_id != prefix:
            return STRATEGY_LABELS.get(prefix, prefix) + " (first 7 ban days)"
    return strategy_id.replace("_", " ")


def is_cross_country_strategy(strategy_id: str) -> bool:
    """Function summary: True if strategy compares Italian forums to controls."""
    return strategy_id.startswith("cross_country")


def is_author_strategy(strategy_id: str) -> bool:
    """Function summary: True if strategy is for author Wordfish panels."""
    return strategy_id.startswith("author_")


def build_treat_post(df: pd.DataFrame, treat_col: str = "treat", post_col: str = "post") -> pd.Series:
    """Function summary: interaction treat × post."""
    return df[treat_col].astype(float) * df[post_col].astype(float)


def apply_post_window(df: pd.DataFrame, mode: str, launch: str) -> pd.DataFrame:
    """Function summary: restrict or redefine post indicator for early-ban specs.

    Parameters:
    - df: panel with rel_day and post.
    - mode: full_ban | early_ban_7d | early_ban_14d.
    - launch: ban date (unused for early; uses rel_day).

    Returns:
    - Copy with post column adjusted.
    """
    out = df.copy()
    if mode == "full_ban":
        return out
    if mode == "early_ban_7d":
        out["post"] = ((out["rel_day"] >= 0) & (out["rel_day"] <= 6)).astype(int)
        return out
    if mode == "early_ban_14d":
        out["post"] = ((out["rel_day"] >= 0) & (out["rel_day"] <= 13)).astype(int)
        return out
    raise ValueError(f"Unknown post_mode: {mode}")


def _filter_author_strategy(work: pd.DataFrame, strategy: StrategySpec) -> pd.DataFrame:
    """Function summary: sample filter for author-panel strategies."""
    if strategy.strategy_id == "author_it_ban":
        work["treat"] = 1
        return work
    if strategy.strategy_id == "author_it_vs_en":
        work = work[work["primary_lexicon"].astype(str).isin(["it", "en"])]
        work["treat"] = (work["primary_lexicon"].astype(str) == "it").astype(int)
        return work
    if strategy.strategy_id == "author_it_vs_de":
        work = work[work["primary_lexicon"].astype(str).isin(["it", "de"])]
        work["treat"] = (work["primary_lexicon"].astype(str) == "it").astype(int)
        return work
    raise ValueError(f"Unknown author strategy: {strategy.strategy_id}")


def filter_strategy_sample(
    df: pd.DataFrame,
    strategy: StrategySpec,
    window_days: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: restrict panel rows for a strategy/subsample.

    Parameters:
    - df: annotated panel.
    - strategy: strategy spec.
    - window_days: if set, keep rel_day in [-window_days, window_days].

    Returns:
    - Filtered copy with treat column.
    """
    work = df.copy()
    if strategy.universe_slice is not None and "universe_slice" in work.columns:
        work = work[work["universe_slice"].astype(str) == strategy.universe_slice]

    if is_author_strategy(strategy.strategy_id):
        work = _filter_author_strategy(work, strategy)
        work = apply_post_window(work, strategy.post_mode, "")
        if window_days is not None:
            work = work[work["rel_day"].between(-window_days, window_days)]
        return work

    if "topic_family" not in work.columns and "IT" in work.columns:
        work["treat"] = work["IT"].astype(int)
        if strategy.control_family:
            work = work[
                (work["treat"] == 1)
                | (work.get("primary_lexicon", pd.Series(dtype=str)).astype(str) == strategy.control_family)
            ]
        elif strategy.treated_family:
            work = work[work["treat"] == 1]
        work = apply_post_window(work, strategy.post_mode, "")
        if window_days is not None:
            work = work[work["rel_day"].between(-window_days, window_days)]
        return work

    if strategy.treated_topic and "topic" in work.columns:
        topic_mask = work["topic"].astype(str) == strategy.treated_topic
        if strategy.control_family:
            fam = work["topic_family"].astype(str)
            work = work[topic_mask | (fam == strategy.control_family)]
            work["treat"] = topic_mask.astype(int)
        else:
            fam = work["topic_family"].astype(str)
            work = work[topic_mask | fam.isin(CONTROL_FAMILIES)]
            work["treat"] = topic_mask.astype(int)
    else:
        fam = work["topic_family"].astype(str)
        if strategy.treated_family:
            treat_mask = fam == strategy.treated_family
            if strategy.control_family:
                work = work[treat_mask | (fam == strategy.control_family)]
                work["treat"] = treat_mask.astype(int)
            else:
                work = work[treat_mask | fam.isin(CONTROL_FAMILIES)]
                work["treat"] = treat_mask.astype(int)
        else:
            work["treat"] = work.get("IT", (fam.isin(ITALY_FAMILIES)).astype(int))
            if strategy.control_family:
                work = work[(work["treat"] == 1) | (fam == strategy.control_family)]
            else:
                work = work[(work["treat"] == 1) | fam.isin(CONTROL_FAMILIES)]

    if window_days is not None:
        work = work[work["rel_day"].between(-window_days, window_days)]
    work = apply_post_window(work, strategy.post_mode, "")
    return work


def default_strategies() -> Sequence[StrategySpec]:
    """Function summary: headline strategy list for subreddit-day panels."""
    base = [
        StrategySpec("cross_country_all", description="IT (all Italian families) vs pooled controls"),
        StrategySpec("cross_country_it_political", treated_family="it_political", description="it_political vs controls"),
        StrategySpec("cross_country_it_others", treated_family="it_others", description="it_others vs controls"),
        StrategySpec(
            "cross_country_political_universe_in",
            universe_slice="in_political_tree",
            description="IT vs controls, in_political_tree only",
        ),
        StrategySpec(
            "cross_country_political_universe_out",
            universe_slice="out_political_tree",
            description="IT vs controls, out_political_tree only",
        ),
        StrategySpec("within_italy_ddd", description="Triple-diff IT×Post×Political (slice panel)"),
    ]
    topic_strats = [
        StrategySpec(
            f"cross_country_topic_{t}",
            treated_topic=t,
            description=f"{t} vs controls",
        )
        for t in ITALIAN_TOPICS
    ]
    early = [
        StrategySpec(
            s.strategy_id,
            treated_family=s.treated_family,
            universe_slice=s.universe_slice,
            description=s.description,
            post_mode="early_ban_7d",
        )
        for s in base[:3]
    ]
    early14 = [
        StrategySpec(
            s.strategy_id + "_14d",
            treated_family=s.treated_family,
            universe_slice=s.universe_slice,
            description=s.description + " (first 14 ban days)",
            post_mode="early_ban_14d",
        )
        for s in base[:3]
    ]
    by_control = [
        StrategySpec(f"cross_country_vs_{c}", control_family=c, description=f"IT vs {c} only")
        for c in sorted(CONTROL_FAMILIES)
    ]
    return tuple(base + topic_strats + list(early) + list(early14) + by_control)


def author_strategies(has_en: bool = False, has_de: bool = False) -> Sequence[StrategySpec]:
    """Function summary: strategies valid for author Wordfish panels."""
    specs = [
        StrategySpec(
            "author_it_ban",
            description="Italian authors: post-ban shift",
            author_only=True,
        ),
    ]
    if has_en:
        specs.append(
            StrategySpec(
                "author_it_vs_en",
                description="Italian vs English authors",
                author_only=True,
            )
        )
    if has_de:
        specs.append(
            StrategySpec(
                "author_it_vs_de",
                description="Italian vs German authors",
                author_only=True,
            )
        )
    return tuple(specs)


def strategies_for_outcome(
    outcome_family: str,
    panel_has_topic_family: bool,
    author_has_multi_lang: bool,
) -> Sequence[StrategySpec]:
    """Function summary: return applicable strategies for an outcome family and panel."""
    if outcome_family in ("wordfish_author", "wordfish_author_v2"):
        langs = author_has_multi_lang
        has_en = langs  # detected upstream per panel
        return author_strategies(has_en=has_en, has_de=has_en)
    if outcome_family in ("wordfish_forum", "wordfish_forum_v2", "lexical", "semantic_axis"):
        return default_strategies()
    return default_strategies()


def rel_day_from_date(
    dates: pd.Series,
    launch: str,
    placebo: bool = False,
    placebo_date: str = "2023-03-16",
) -> pd.Series:
    """Function summary: compute event-study relative day from calendar date."""
    anchor = placebo_date if placebo else launch
    launch_dt = pd.Timestamp(anchor)
    return (pd.to_datetime(dates.astype(str)) - launch_dt).dt.days.astype(int)
