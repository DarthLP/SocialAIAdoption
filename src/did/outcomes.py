"""
Outcome registry for DiD / event-study estimation (prompts 00, 01, 03, 03b, v2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class OutcomeSpec:
    """Function summary: one estimand column and metadata for reporting."""

    outcome_id: str
    column: str
    family: str
    ddd_allowed: bool = True
    tier: Optional[str] = None
    sign_only_cross_country: bool = False
    panel_kind: str = "subreddit_day"


def _wordfish_forum_specs(family: str, prefix: str) -> Tuple[OutcomeSpec, ...]:
    """Function summary: forum Wordfish outcome block for v1 or v2 family."""
    return (
        OutcomeSpec(f"{prefix}extremity", "extremity", family, tier="B", sign_only_cross_country=True),
        OutcomeSpec(f"{prefix}extremity_z", "extremity_z", family, tier="B", sign_only_cross_country=True),
        OutcomeSpec(f"{prefix}change_z", "change_z", family, tier="B", sign_only_cross_country=True),
        OutcomeSpec(f"{prefix}change", "change", family, tier="B", sign_only_cross_country=True),
    )


def _wordfish_author_specs(family: str, prefix: str) -> Tuple[OutcomeSpec, ...]:
    """Function summary: author Wordfish outcome block for v1 or v2 family."""
    return (
        OutcomeSpec(
            f"{prefix}extremity_z",
            "extremity_z",
            family,
            sign_only_cross_country=True,
            panel_kind="author_bin",
        ),
        OutcomeSpec(
            f"{prefix}change_z",
            "change_z",
            family,
            sign_only_cross_country=True,
            panel_kind="author_bin",
        ),
        OutcomeSpec(f"{prefix}extremity", "extremity", family, panel_kind="author_bin"),
        OutcomeSpec(f"{prefix}change", "change", family, panel_kind="author_bin"),
    )


OUTCOME_REGISTRY: Tuple[OutcomeSpec, ...] = (
    # Lexical / descriptives (00)
    OutcomeSpec("net_ideology", "net_ideology_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("extremity", "extremity_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("ambivalence", "ambivalence_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("esteban_ray", "esteban_ray_index", "lexical", ddd_allowed=False),
    OutcomeSpec("bimodality", "bimodality_coefficient", "lexical", ddd_allowed=False),
    OutcomeSpec("pole_share", "pole_share", "lexical", ddd_allowed=False),
    OutcomeSpec("ai_style_rate", "ai_style_rate_100w_mean", "lexical"),
    OutcomeSpec("aggression_rate", "aggression_rate_100w_mean", "lexical"),
    OutcomeSpec("salience_rate", "other_side_salience_rate_100w_mean", "lexical"),
    OutcomeSpec("negative_rate", "negative_rate_100w_mean", "lexical"),
    OutcomeSpec("anger_rate", "anger_rate_100w_mean", "lexical"),
    OutcomeSpec("pair_framing", "pair_framing_net_strict_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("em_dash_rate", "em_dash_rate_100w", "lexical"),
    OutcomeSpec("exclamation_rate", "exclamation_rate_100w_mean", "lexical"),
    OutcomeSpec("sentence_len_var", "sentence_length_variance_mean", "lexical"),
    OutcomeSpec("avg_wps", "avg_words_per_sentence_mean", "lexical"),
    # Semantic axis (01)
    OutcomeSpec("sem_axis_ideology", "sem_axis_ideology_mean", "semantic_axis"),
    OutcomeSpec("sem_axis_ideology_var", "sem_axis_ideology_var", "semantic_axis"),
    OutcomeSpec("sem_axis_emotion", "sem_axis_emotion_mean", "semantic_axis"),
    OutcomeSpec("sem_axis_emotion_var", "sem_axis_emotion_var", "semantic_axis"),
    OutcomeSpec("sem_axis_aggression", "sem_axis_aggression_mean", "semantic_axis"),
    # Forum Wordfish v1 (03)
    *_wordfish_forum_specs("wordfish_forum", "wf_"),
    # Author Wordfish v1 (03b)
    *_wordfish_author_specs("wordfish_author", "wfa_"),
    # Forum Wordfish v2
    *_wordfish_forum_specs("wordfish_forum_v2", "wf2_"),
    # Author Wordfish v2
    *_wordfish_author_specs("wordfish_author_v2", "wfa2_"),
)


FAMILY_FIGURE_DIRS: dict[str, str] = {
    "lexical": "lexical",
    "semantic_axis": "semantic_axis",
    "wordfish_forum": "wordfish_forum",
    "wordfish_author": "wordfish_author",
    "wordfish_forum_v2": "wordfish_forum_v2",
    "wordfish_author_v2": "wordfish_author_v2",
}


def outcomes_for_families(families: List[str]) -> Tuple[OutcomeSpec, ...]:
    """Function summary: filter registry by outcome family ids."""
    fam_set = set(families)
    return tuple(o for o in OUTCOME_REGISTRY if o.family in fam_set)


FIRST_STAGE_OUTCOMES = ("ai_style_rate", "em_dash_rate", "exclamation_rate", "sentence_len_var", "avg_wps")

HEADLINE_OUTCOMES = (
    "net_ideology",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "ai_style_rate",
    "pole_share",
    "wf_extremity_z",
    "wf_change_z",
)

DEFAULT_FAMILIES = (
    "lexical",
    "semantic_axis",
    "wordfish_forum",
    "wordfish_author",
    "wordfish_forum_v2",
    "wordfish_author_v2",
)
