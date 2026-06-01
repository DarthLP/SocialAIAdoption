"""
Treatment indicators, sample filters, and strategy definitions for DiD estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd

ITALY_FAMILIES = frozenset({"it_political", "it_others"})
CONTROL_FAMILIES = frozenset({"de", "eu", "us", "uk"})
ITALIAN_TOPICS = ("it_pure_political", "it_political", "it_others")

# Map forum control-family ids to primary_lexicon codes on language-level panels.
CONTROL_FAMILY_TO_LEXICON: dict[str, str] = {
    "de": "de",
    "eu": "en",
    "us": "en",
    "uk": "en",
}
CONTROL_LEXICONS: frozenset[str] = frozenset(CONTROL_FAMILY_TO_LEXICON.values())

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
    "italy_only_post": "Italian forums only: post-ban shift (national shock)",
}

STRATEGY_LABELS_SHORT: dict[str, str] = {
    "cross_country_all": "IT vs pooled",
    "cross_country_it_political": "IT political vs controls",
    "cross_country_it_others": "IT non-political vs controls",
    "cross_country_political_universe_in": "IT vs ctrl — in tree",
    "cross_country_political_universe_out": "IT vs ctrl — out tree",
    "within_italy_ddd": "Within-IT DDD",
    "cross_country_vs_de": "IT vs DE only",
    "cross_country_vs_eu": "IT vs EU only",
    "cross_country_vs_us": "IT vs US only",
    "cross_country_vs_uk": "IT vs UK only",
    "cross_country_topic_it_pure_political": "IT pure-political vs ctrl",
    "cross_country_topic_it_political": "IT political-topic vs ctrl",
    "cross_country_topic_it_others": "IT other-topic vs ctrl",
    "author_it_ban": "IT authors post-ban",
    "author_it_vs_en": "IT vs EN authors",
    "author_it_vs_de": "IT vs DE authors",
    "italy_only_post": "IT forums post-ban",
}

SPEC_LABELS_SHORT: dict[str, str] = {
    "full_ban": "full",
    "early_ban_7d": "7d",
    "early_ban_14d": "14d",
    "post_short_3d": "short 0–2d",
    "post_medium_7d": "mid 3–9d",
    "post_long_tail": "long 10d+",
}

SPEC_LABELS_PAREN: dict[str, str] = {
    "full_ban": "(full ban)",
    "early_ban_7d": "(early ban)",
    "early_ban_14d": "(14d ban)",
    "post_short_3d": "(short 0–2d)",
    "post_medium_7d": "(medium 3–9d)",
    "post_long_tail": "(long 10d+)",
}

# Calendar post-ban phase spec ids (post_mode) for TWFE; bounds merged from config did.post_phases.
POST_PHASE_MODES: tuple[str, ...] = ("post_short_3d", "post_medium_7d", "post_long_tail")

_DEFAULT_POST_PHASE_BOUNDS: dict[str, tuple[int, Optional[int]]] = {
    "post_short_3d": (0, 2),
    "post_medium_7d": (3, 9),
    "post_long_tail": (10, None),
}

# Merged bounds from load_post_phases / activate_post_phases_from_config; None => use _DEFAULT_POST_PHASE_BOUNDS.
_ACTIVE_POST_PHASE_BOUNDS: Optional[dict[str, tuple[int, Optional[int]]]] = None

_YAML_PHASE_KEY_TO_SPEC: dict[str, str] = {
    "short": "post_short_3d",
    "medium": "post_medium_7d",
    "long": "post_long_tail",
}

EVENT_WINDOW_DAYS_BY_BIN: dict[int, int] = {1: 14, 3: 30}

EVENT_STUDY_OVERLAY_STRATEGY_IDS: tuple[str, ...] = (
    "cross_country_all",
    "cross_country_vs_de",
    "cross_country_vs_eu",
    "cross_country_vs_us",
    "cross_country_vs_uk",
)

# IT vs controls restricted to political-universe slice (in-tree / out-of-tree).
EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS: tuple[str, ...] = (
    "cross_country_political_universe_in",
    "cross_country_political_universe_out",
)


def lexicon_for_control_family(control_family: str) -> str:
    """Function summary: map topic_family control id to primary_lexicon code for language panels."""
    return CONTROL_FAMILY_TO_LEXICON.get(control_family, control_family)


def language_hub_from_lexicon_family(lexicon: str, topic_family: str = "") -> str:
    """Function summary: map subreddit lexicon + family to language-hub entity (it/de/eu/us/uk)."""
    lex = str(lexicon)
    fam = str(topic_family)
    if lex == "it":
        return "it"
    if lex == "de":
        return "de"
    if lex == "en" and fam in CONTROL_FAMILIES:
        return fam
    return lex


def assign_language_hub_series(df: pd.DataFrame) -> pd.Series:
    """Function summary: vectorized language_hub from primary_lexicon and optional topic_family."""
    lex = df["primary_lexicon"].astype(str)
    fam = df["topic_family"].astype(str) if "topic_family" in df.columns else pd.Series("", index=df.index)
    return pd.Series(
        [language_hub_from_lexicon_family(l, f) for l, f in zip(lex, fam, strict=False)],
        index=df.index,
        dtype=str,
    )


def panel_uses_language_hubs(df: pd.DataFrame) -> bool:
    """Function summary: True when panel rows distinguish EU/US/UK hubs (not pooled en)."""
    if "language_hub" in df.columns:
        hubs = set(df["language_hub"].astype(str))
        return bool(hubs & set(CONTROL_FAMILIES))
    ent = set(df.get("entity_id", pd.Series(dtype=str)).astype(str))
    return {"eu", "us", "uk"}.issubset(ent)


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
    "italy_only": ("italy_only_post",),
    "full": (),  # empty = all strategies
    "post_phases": (
        "cross_country_all",
        "cross_country_it_political",
        "cross_country_it_others",
        "cross_country_political_universe_in",
        "cross_country_political_universe_out",
        "within_italy_ddd",
    ),
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


HEADLINE_BASE_STRATEGIES: tuple[StrategySpec, ...] = (
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
)


def headline_base_strategies() -> tuple[StrategySpec, ...]:
    """Function summary: six headline panel strategies cloned for post-phase TWFE specs.

    Returns:
    - Tuple of StrategySpec rows matching PLOT_STRATEGY_GROUPS['headline'].
    """
    return HEADLINE_BASE_STRATEGIES


def load_post_phases(config: Optional[Dict[str, Any]] = None) -> dict[str, tuple[int, Optional[int]]]:
    """Function summary: merge did.post_phases from config with defaults for rel_day windows.

    Parameters:
    - config: project YAML dict; optional did.post_phases.{short,medium,long} with rel_day_min/max.

    Returns:
    - Map post_mode -> (rel_day_min, rel_day_max or None for open-ended long tail).
    """
    out = dict(_DEFAULT_POST_PHASE_BOUNDS)
    if not config:
        return out
    raw = (config.get("did") or {}).get("post_phases") or {}
    for yaml_key, spec_id in _YAML_PHASE_KEY_TO_SPEC.items():
        entry = raw.get(yaml_key)
        if not entry or not isinstance(entry, dict):
            continue
        lo = int(entry["rel_day_min"])
        hi = entry.get("rel_day_max")
        hi_out: Optional[int] = None if hi is None else int(hi)
        out[spec_id] = (lo, hi_out)
    return out


def activate_post_phases_from_config(config: Optional[Dict[str, Any]] = None) -> None:
    """Function summary: set active post-phase rel_day bounds used by apply_post_window.

    Parameters:
    - config: loaded project config; None resets to built-in defaults (lazy via _ACTIVE None).
    """
    global _ACTIVE_POST_PHASE_BOUNDS
    if config is None:
        _ACTIVE_POST_PHASE_BOUNDS = None
        return
    _ACTIVE_POST_PHASE_BOUNDS = load_post_phases(config)


def reset_post_phase_bounds() -> None:
    """Function summary: clear config-driven phase bounds (tests)."""
    activate_post_phases_from_config(None)


def _effective_post_phase_bounds() -> dict[str, tuple[int, Optional[int]]]:
    """Function summary: bounds dict for post_* post_mode values."""
    if _ACTIVE_POST_PHASE_BOUNDS is not None:
        return dict(_ACTIVE_POST_PHASE_BOUNDS)
    return dict(_DEFAULT_POST_PHASE_BOUNDS)


def post_phase_strategies(
    headline_specs: Optional[Sequence[StrategySpec]] = None,
) -> tuple[StrategySpec, ...]:
    """Function summary: clone headline strategies with each post-phase post_mode.

    Parameters:
    - headline_specs: defaults to headline_base_strategies().

    Returns:
    - len(headline_specs) * len(POST_PHASE_MODES) StrategySpec rows.
    """
    base = tuple(headline_specs) if headline_specs is not None else headline_base_strategies()
    out: list[StrategySpec] = []
    for s in base:
        for pm in POST_PHASE_MODES:
            out.append(
                StrategySpec(
                    strategy_id=s.strategy_id,
                    treat_col=s.treat_col,
                    description=s.description,
                    universe_slice=s.universe_slice,
                    treated_family=s.treated_family,
                    treated_topic=s.treated_topic,
                    control_family=s.control_family,
                    post_mode=pm,
                    placebo=s.placebo,
                    author_only=s.author_only,
                )
            )
    return tuple(out)


def strategy_label(strategy_id: str, *, short: bool = False) -> str:
    """Function summary: human-readable label for a strategy_id.

    Parameters:
    - strategy_id: strategy key.
    - short: if True, use compact labels for figure axes.

    Returns:
    - Display label string.
    """
    labels = STRATEGY_LABELS_SHORT if short else STRATEGY_LABELS
    suffix_14 = " (14d)" if short else " (first 14 ban days)"
    suffix_7 = " (7d)" if short else " (first 7 ban days)"
    if strategy_id in labels:
        return labels[strategy_id]
    if strategy_id.endswith("_14d"):
        base = strategy_id[:-4]
        return labels.get(base, base) + suffix_14
    for prefix in ("cross_country_all", "cross_country_it_political", "cross_country_it_others"):
        if strategy_id.startswith(prefix) and strategy_id != prefix:
            return labels.get(prefix, prefix) + suffix_7
    return strategy_id.replace("_", " ")


def spec_label_short(spec: str) -> str:
    """Function summary: compact post-window label for plot legends.

    Parameters:
    - spec: post_mode value (full_ban, early_ban_7d, etc.).

    Returns:
    - Short spec label.
    """
    return SPEC_LABELS_SHORT.get(spec, spec.replace("_", " "))


def spec_label_parenthetical(spec: str) -> str:
    """Function summary: short parenthetical post-window label for plot axes.

    Parameters:
    - spec: post_mode value (full_ban, early_ban_7d, etc.).

    Returns:
    - Label like ``(full ban)`` or ``(early ban)``.
    """
    return SPEC_LABELS_PAREN.get(spec, f"({spec.replace('_', ' ')})")


def is_cross_country_strategy(strategy_id: str) -> bool:
    """Function summary: True if strategy compares Italian forums to controls."""
    return strategy_id.startswith("cross_country")


def is_author_strategy(strategy_id: str) -> bool:
    """Function summary: True if strategy is for author Wordfish panels."""
    return strategy_id.startswith("author_")


ENTITY_FE_ONLY_STRATEGIES: frozenset[str] = frozenset({"author_it_ban", "italy_only_post"})


def is_entity_fe_only_strategy(strategy_id: str) -> bool:
    """Function summary: True when treat is constant and calendar time FE are omitted."""
    return strategy_id in ENTITY_FE_ONLY_STRATEGIES


def _filter_italy_only_post(work: pd.DataFrame) -> pd.DataFrame:
    """Function summary: restrict to Italian forums and set treat=1 for national-shock DiD."""
    out = work.copy()
    if "topic_family" in out.columns:
        out = out[out["topic_family"].astype(str).isin(ITALY_FAMILIES)]
    elif "IT" in out.columns:
        out = out[out["IT"].astype(int) == 1]
    else:
        raise ValueError("italy_only_post requires topic_family or IT column")
    out["treat"] = 1
    return out


def build_treat_post(df: pd.DataFrame, treat_col: str = "treat", post_col: str = "post") -> pd.Series:
    """Function summary: interaction treat × post."""
    return df[treat_col].astype(float) * df[post_col].astype(float)


def apply_post_window(df: pd.DataFrame, mode: str, launch: str) -> pd.DataFrame:
    """Function summary: restrict or redefine post indicator for early-ban and post-phase specs.

    Parameters:
    - df: panel with rel_day and post.
    - mode: full_ban | early_ban_7d | early_ban_14d | post_short_3d | post_medium_7d | post_long_tail.
    - launch: ban date (unused for windowed modes; uses rel_day).

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
    phase_bounds = _effective_post_phase_bounds()
    if mode in phase_bounds:
        lo, hi = phase_bounds[mode]
        if hi is None:
            out["post"] = (out["rel_day"] >= lo).astype(int)
        else:
            out["post"] = ((out["rel_day"] >= lo) & (out["rel_day"] <= hi)).astype(int)
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

    if strategy.strategy_id == "italy_only_post":
        work = _filter_italy_only_post(work)
        work = apply_post_window(work, strategy.post_mode, "")
        if window_days is not None:
            work = work[work["rel_day"].between(-window_days, window_days)]
        return work

    if "topic_family" not in work.columns and "IT" in work.columns:
        work["treat"] = work["IT"].astype(int)
        if panel_uses_language_hubs(work):
            hub = work["language_hub"].astype(str) if "language_hub" in work.columns else work["entity_id"].astype(str)
            if strategy.control_family:
                work = work[(work["treat"] == 1) | (hub == strategy.control_family)]
            elif strategy.treated_family:
                work = work[work["treat"] == 1]
            else:
                work = work[(work["treat"] == 1) | hub.isin(CONTROL_FAMILIES)]
        else:
            lex = work.get("primary_lexicon", pd.Series(dtype=str)).astype(str)
            if strategy.control_family:
                ctrl_lex = lexicon_for_control_family(strategy.control_family)
                work = work[(work["treat"] == 1) | (lex == ctrl_lex)]
            elif strategy.treated_family:
                work = work[work["treat"] == 1]
            else:
                work = work[(work["treat"] == 1) | lex.isin(CONTROL_LEXICONS)]
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
    base = list(HEADLINE_BASE_STRATEGIES)
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


def italy_only_strategies() -> tuple[StrategySpec, ...]:
    """Function summary: Italy-only post-ban strategies (entity FE, no calendar time FE)."""
    return (
        StrategySpec(
            "italy_only_post",
            description="Italian forums: post-ban shift (entity FE only)",
        ),
    )


def subreddit_panel_strategies() -> Sequence[StrategySpec]:
    """Function summary: full strategy list for subreddit-day and comment panels."""
    return tuple(default_strategies()) + post_phase_strategies() + italy_only_strategies()


def event_study_overlay_strategies() -> tuple[StrategySpec, ...]:
    """Function summary: five-strategy bundle for one overlay figure (pooled + all single controls)."""
    by_id = {s.strategy_id: s for s in default_strategies()}
    return tuple(by_id[sid] for sid in EVENT_STUDY_OVERLAY_STRATEGY_IDS if sid in by_id)


def event_study_language_universe_slice_strategies() -> tuple[StrategySpec, ...]:
    """Function summary: in-tree vs out-of-tree IT-vs-controls overlay for slice panels."""
    by_id = {s.strategy_id: s for s in default_strategies()}
    return tuple(by_id[sid] for sid in EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS if sid in by_id)


def event_study_topic_family_it_political_strategy() -> StrategySpec:
    """Function summary: single strategy — Italian political forums vs controls."""
    return StrategySpec(
        "cross_country_it_political",
        treated_family="it_political",
        description="it_political vs controls",
    )


def event_study_topic_family_it_others_strategy() -> StrategySpec:
    """Function summary: single strategy — Italian non-political forums vs controls."""
    return StrategySpec(
        "cross_country_it_others",
        treated_family="it_others",
        description="it_others vs controls",
    )


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
    panel_kind: str = "subreddit_day",
) -> Sequence[StrategySpec]:
    """Function summary: return applicable strategies for an outcome family and panel."""
    if outcome_family in ("wordfish_author", "wordfish_author_v2"):
        langs = author_has_multi_lang
        has_en = langs  # detected upstream per panel
        return author_strategies(has_en=has_en, has_de=has_en)
    if outcome_family == "semantic_axis_author_week" or panel_kind == "author_semantic_week":
        return author_strategies(has_en=author_has_multi_lang, has_de=author_has_multi_lang)
    if panel_kind in ("comment", "author_day") or outcome_family.endswith("_comment") or outcome_family.endswith(
        "_author_day"
    ):
        return subreddit_panel_strategies()
    return subreddit_panel_strategies()


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
