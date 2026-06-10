"""
Outcome registry for DiD / event-study estimation (prompts 00, 01, 03, 03b, v2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.embeddings import EXTENDED_AXIS_NAMES


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
        OutcomeSpec(
            f"{prefix}extremity_within_author",
            "extremity_within_author",
            family,
            sign_only_cross_country=True,
            panel_kind="author_bin",
        ),
        OutcomeSpec(
            f"{prefix}extremity_within_author_z",
            "extremity_within_author_z",
            family,
            sign_only_cross_country=True,
            panel_kind="author_bin",
        ),
    )


def _extended_semantic_axis_specs() -> Tuple[OutcomeSpec, ...]:
    """Function summary: DiD outcomes for extended issue-dimension semantic axes."""
    specs: List[OutcomeSpec] = []
    for axis in EXTENDED_AXIS_NAMES:
        oid = f"sem_axis_{axis}"
        specs.append(OutcomeSpec(oid, f"{oid}_mean", "semantic_axis"))
        specs.append(
            OutcomeSpec(oid, oid, "semantic_axis_comment", panel_kind="comment"),
        )
        specs.append(
            OutcomeSpec(
                oid,
                oid,
                "semantic_axis_author_day",
                ddd_allowed=False,
                panel_kind="author_day",
            ),
        )
        specs.append(
            OutcomeSpec(
                oid,
                f"{oid}_mean",
                "semantic_axis_author_week",
                ddd_allowed=False,
                sign_only_cross_country=True,
                panel_kind="author_semantic_week",
            ),
        )
    return tuple(specs)


OUTCOME_REGISTRY: Tuple[OutcomeSpec, ...] = (
    # Lexical / descriptives (00)
    OutcomeSpec("net_ideology", "net_ideology_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("extremity", "extremity_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("ambivalence", "ambivalence_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("esteban_ray", "esteban_ray_index", "lexical", ddd_allowed=False),
    OutcomeSpec("bimodality", "bimodality_coefficient", "lexical", ddd_allowed=False),
    OutcomeSpec("pole_share", "pole_share", "lexical", ddd_allowed=False),
    OutcomeSpec("left_rate", "left_rate_100w_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("right_rate", "right_rate_100w_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("center_rate", "center_rate_100w_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("ai_style_rate", "ai_style_rate_100w_mean", "lexical"),
    OutcomeSpec("aggression_rate", "aggression_rate_100w_mean", "lexical"),
    OutcomeSpec("salience_rate", "other_side_salience_rate_100w_mean", "lexical"),
    OutcomeSpec("negative_rate", "negative_rate_100w_mean", "lexical"),
    OutcomeSpec("anger_rate", "anger_rate_100w_mean", "lexical"),
    OutcomeSpec("emotion_rate", "emotion_rate_100w_mean", "lexical"),
    OutcomeSpec("cognition_rate", "cognition_rate_100w_mean", "lexical"),
    OutcomeSpec("pair_framing", "pair_framing_net_strict_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("em_dash_rate", "em_dash_rate_100w", "lexical"),
    OutcomeSpec("exclamation_rate", "exclamation_rate_100w_mean", "lexical"),
    OutcomeSpec("sentence_len_var", "sentence_length_variance_mean", "lexical"),
    OutcomeSpec("avg_wps", "avg_words_per_sentence_mean", "lexical"),
    OutcomeSpec("style_index_llm", "style_index_llm_mean", "lexical"),
    OutcomeSpec(
        "style_index_llm_no_ai_style",
        "style_index_llm_no_ai_style_mean",
        "lexical",
    ),
    OutcomeSpec(
        "style_index_llm_no_em_dash",
        "style_index_llm_no_em_dash_mean",
        "lexical",
    ),
    OutcomeSpec(
        "style_index_llm_no_semicolon_colon",
        "style_index_llm_no_semicolon_colon_mean",
        "lexical",
    ),
    OutcomeSpec(
        "style_index_llm_no_hedging_phrase",
        "style_index_llm_no_hedging_phrase_mean",
        "lexical",
    ),
    OutcomeSpec(
        "style_index_llm_no_exclamation",
        "style_index_llm_no_exclamation_mean",
        "lexical",
    ),
    OutcomeSpec("log_len_mean", "log_len_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("share_ge20w", "share_ge20w", "lexical", ddd_allowed=False),
    OutcomeSpec("ttr_50w", "ttr_50w_mean", "lexical", ddd_allowed=False),
    OutcomeSpec("readability", "readability_mean", "lexical", ddd_allowed=False),
    # Subreddit-day quantity (participation margins)
    OutcomeSpec("log_n_comments", "log_n_comments", "quantity", ddd_allowed=False),
    OutcomeSpec("log_n_authors", "log_n_authors", "quantity", ddd_allowed=False),
    # Semantic axis (01)
    OutcomeSpec("sem_axis_ideology", "sem_axis_ideology_mean", "semantic_axis"),
    OutcomeSpec("sem_axis_ideology_var", "sem_axis_ideology_var", "semantic_axis"),
    OutcomeSpec("sem_axis_emotion", "sem_axis_emotion_mean", "semantic_axis"),
    OutcomeSpec("sem_axis_emotion_var", "sem_axis_emotion_var", "semantic_axis"),
    OutcomeSpec("sem_axis_emotion_pruned", "sem_axis_emotion_pruned_mean", "semantic_axis"),
    OutcomeSpec("sem_axis_aggression", "sem_axis_aggression_mean", "semantic_axis"),
    *_extended_semantic_axis_specs(),
    OutcomeSpec(
        "sem_axis_ideology_pole_share",
        "sem_axis_ideology_pole_share",
        "semantic_axis",
        ddd_allowed=False,
    ),
    OutcomeSpec(
        "sem_axis_ideology_esteban_ray",
        "sem_axis_ideology_esteban_ray",
        "semantic_axis",
        ddd_allowed=False,
    ),
    OutcomeSpec(
        "sem_axis_ideology_extreme_left",
        "sem_axis_ideology_share_left_below_p10",
        "semantic_axis",
        ddd_allowed=False,
    ),
    OutcomeSpec(
        "sem_axis_ideology_extreme_right",
        "sem_axis_ideology_share_right_above_p90",
        "semantic_axis",
        ddd_allowed=False,
    ),
    # Comment-level (political universe; pyfixest author + day FE)
    OutcomeSpec("net_ideology", "net_ideology", "lexical_comment", ddd_allowed=False, panel_kind="comment"),
    OutcomeSpec("extremity", "extremity", "lexical_comment", ddd_allowed=False, panel_kind="comment"),
    OutcomeSpec("ai_style_rate", "ai_style_rate_100w", "lexical_comment", panel_kind="comment"),
    OutcomeSpec("style_index_llm", "style_index_llm", "lexical_comment", panel_kind="comment"),
    OutcomeSpec(
        "style_index_llm_no_ai_style",
        "style_index_llm_no_ai_style",
        "lexical_comment",
        panel_kind="comment",
    ),
    OutcomeSpec(
        "style_index_llm_no_em_dash",
        "style_index_llm_no_em_dash",
        "lexical_comment",
        panel_kind="comment",
    ),
    OutcomeSpec(
        "style_index_llm_no_semicolon_colon",
        "style_index_llm_no_semicolon_colon",
        "lexical_comment",
        panel_kind="comment",
    ),
    OutcomeSpec(
        "style_index_llm_no_hedging_phrase",
        "style_index_llm_no_hedging_phrase",
        "lexical_comment",
        panel_kind="comment",
    ),
    OutcomeSpec(
        "style_index_llm_no_exclamation",
        "style_index_llm_no_exclamation",
        "lexical_comment",
        panel_kind="comment",
    ),
    OutcomeSpec("ttr_50w", "ttr_50w", "lexical_comment", panel_kind="comment", ddd_allowed=False),
    OutcomeSpec("readability", "readability", "lexical_comment", panel_kind="comment", ddd_allowed=False),
    OutcomeSpec("emotion_rate", "emotion_rate_100w", "lexical_comment", panel_kind="comment"),
    OutcomeSpec("cognition_rate", "cognition_rate_100w", "lexical_comment", panel_kind="comment"),
    OutcomeSpec("sem_axis_ideology", "sem_axis_ideology", "semantic_axis_comment", panel_kind="comment"),
    OutcomeSpec("sem_axis_emotion", "sem_axis_emotion", "semantic_axis_comment", panel_kind="comment"),
    OutcomeSpec("sem_axis_aggression", "sem_axis_aggression", "semantic_axis_comment", panel_kind="comment"),
    # Author×day robustness (PanelOLS TWFE)
    OutcomeSpec(
        "net_ideology",
        "net_ideology",
        "lexical_author_day",
        ddd_allowed=False,
        panel_kind="author_day",
    ),
    OutcomeSpec(
        "sem_axis_ideology",
        "sem_axis_ideology",
        "semantic_axis_author_day",
        ddd_allowed=False,
        panel_kind="author_day",
    ),
    OutcomeSpec(
        "sem_axis_emotion",
        "sem_axis_emotion",
        "semantic_axis_author_day",
        ddd_allowed=False,
        panel_kind="author_day",
    ),
    # Author×week semantic axis (user-week panel + wordfish assignment)
    OutcomeSpec(
        "sem_axis_ideology",
        "sem_axis_ideology_mean",
        "semantic_axis_author_week",
        ddd_allowed=False,
        sign_only_cross_country=True,
        panel_kind="author_semantic_week",
    ),
    OutcomeSpec(
        "sem_axis_emotion",
        "sem_axis_emotion_mean",
        "semantic_axis_author_week",
        ddd_allowed=False,
        sign_only_cross_country=True,
        panel_kind="author_semantic_week",
    ),
    OutcomeSpec(
        "sem_axis_aggression",
        "sem_axis_aggression_mean",
        "semantic_axis_author_week",
        ddd_allowed=False,
        sign_only_cross_country=True,
        panel_kind="author_semantic_week",
    ),
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
    "quantity": "quantity",
    "lexical": "lexical",
    "lexical_comment": "lexical_comment",
    "lexical_author_day": "lexical_author_day",
    "semantic_axis": "semantic_axis",
    "semantic_axis_comment": "semantic_axis_comment",
    "semantic_axis_author_day": "semantic_axis_author_day",
    "semantic_axis_author_week": "semantic_axis_author_week",
    "wordfish_forum": "wordfish_forum",
    "wordfish_author": "wordfish_author",
    "wordfish_forum_v2": "wordfish_forum_v2",
    "wordfish_author_v2": "wordfish_author_v2",
}


def outcomes_for_families(families: List[str]) -> Tuple[OutcomeSpec, ...]:
    """Function summary: filter registry by outcome family ids."""
    fam_set = set(families)
    return tuple(o for o in OUTCOME_REGISTRY if o.family in fam_set)


def outcome_spec(outcome_id: str) -> Optional[OutcomeSpec]:
    """Function summary: lookup one OutcomeSpec by outcome_id, or None if unknown."""
    for spec in OUTCOME_REGISTRY:
        if spec.outcome_id == outcome_id:
            return spec
    return None


FIRST_STAGE_OUTCOMES = (
    "ai_style_rate",
    "em_dash_rate",
    "exclamation_rate",
    "sentence_len_var",
    "avg_wps",
    "style_index_llm",
    "style_index_llm_no_ai_style",
    "style_index_llm_no_em_dash",
    "style_index_llm_no_semicolon_colon",
    "style_index_llm_no_hedging_phrase",
    "style_index_llm_no_exclamation",
    "ttr_50w",
    "readability",
    "log_len_mean",
    "share_ge20w",
)

HEADLINE_OUTCOMES = (
    "sem_axis_ideology",
    "sem_axis_aggression",
    "ai_style_rate",
    "em_dash_rate",
    "wf_extremity_z",
)

HEADLINE_FOREST_STRATEGIES = (
    "cross_country_all",
    "cross_country_it_political",
    "cross_country_it_others",
)

HEADLINE_EVENT_STUDY_OUTCOMES = HEADLINE_OUTCOMES

# Cross-country ban-window descriptives (plot_descriptives_ban_shaded.py only).
BAN_WINDOW_DESCRIPTIVE_OUTCOMES = (
    "net_ideology",
    "extremity",
    "salience_rate",
    "aggression_rate",
    "esteban_ray",
    "ai_style_rate",
    "style_index_llm",
    "em_dash_rate",
    "exclamation_rate",
    "avg_wps",
    "sentence_len_var",
    "ttr_50w",
    "readability",
    "log_len_mean",
    "share_ge20w",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_economic",
    "sem_axis_cultural",
    "sem_axis_nationalism",
    "sem_axis_anti_establishment",
    "wf_extremity_z",
    "wf_change_z",
)

# Lexical columns in daily_country_panel without outcome_id registry entries.
BAN_WINDOW_LEXICAL_EXTRA_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("semicolon_rate", "semicolon_rate_100w"),
    ("colon_rate", "colon_rate_100w"),
    ("hedging_phrase_rate", "hedging_phrase_rate_100w"),
    ("complexity_index", "complexity_index"),
    ("mean_n_words", "mean_n_words"),
)

# Semantic-axis ideology tails for dual-line event-study figures (p10 / p90).
SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL = "sem_axis_ideology_share_left_below_p10"
SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL = "sem_axis_ideology_share_right_above_p90"

LEXICAL_BY_CONTROL_OUTCOMES = (
    "em_dash_rate",
    "ai_style_rate",
    "exclamation_rate",
    "net_ideology",
)

LEXICAL_BY_CONTROL_STRATEGIES = (
    "cross_country_all",
    "cross_country_vs_de",
    "cross_country_vs_eu",
    "cross_country_vs_uk",
    "cross_country_vs_us",
)

DEFAULT_FAMILIES = (
    "lexical",
    "semantic_axis",
    "wordfish_forum",
    "wordfish_author",
    "wordfish_forum_v2",
    "wordfish_author_v2",
)

WORDFISH_OUTCOME_IDS: Tuple[str, ...] = tuple(
    o.outcome_id for o in OUTCOME_REGISTRY if o.family.startswith("wordfish")
)

LEXICAL_OUTCOME_IDS: Tuple[str, ...] = tuple(
    o.outcome_id for o in OUTCOME_REGISTRY if o.family == "lexical"
)

SUMMARY_THEMES: dict[str, Tuple[str, ...]] = {
    "all": tuple(o.outcome_id for o in OUTCOME_REGISTRY),
    "aggression": ("aggression_rate", "sem_axis_aggression"),
    "ideology": (
        "net_ideology",
        "sem_axis_ideology",
        "sem_axis_ideology_var",
        "sem_axis_ideology_pole_share",
        "sem_axis_ideology_esteban_ray",
        "sem_axis_ideology_extreme_left",
        "sem_axis_ideology_extreme_right",
    ),
    "ideology_poles": (
        "pole_share",
        "left_rate",
        "right_rate",
        "center_rate",
        "sem_axis_ideology_pole_share",
        "sem_axis_ideology_extreme_left",
        "sem_axis_ideology_extreme_right",
    ),
    "emotion": ("sem_axis_emotion", "sem_axis_emotion_var", "emotion_rate", "cognition_rate"),
    "issue_axes": tuple(f"sem_axis_{axis}" for axis in EXTENDED_AXIS_NAMES),
    "ai_style": FIRST_STAGE_OUTCOMES,
    "wordfish": WORDFISH_OUTCOME_IDS,
    "lexical": LEXICAL_OUTCOME_IDS,
}


def outcome_family_map() -> dict[str, str]:
    """Function summary: map outcome_id to family for migration and path resolution.

    Returns:
    - Dict outcome_id -> family.
    """
    return {o.outcome_id: o.family for o in OUTCOME_REGISTRY}


OUTCOME_LABELS_SHORT: dict[str, str] = {
    "net_ideology": "Net ideology",
    "extremity": "Extremity",
    "ambivalence": "Ambivalence",
    "esteban_ray": "Esteban-Ray",
    "bimodality": "Bimodality",
    "pole_share": "Pole share",
    "left_rate": "Left rate",
    "right_rate": "Right rate",
    "center_rate": "Center rate",
    "ai_style_rate": "AI style rate",
    "aggression_rate": "Aggression rate",
    "salience_rate": "Other-side salience",
    "negative_rate": "Negative rate",
    "anger_rate": "Anger rate",
    "emotion_rate": "Emotion rate",
    "cognition_rate": "Cognition rate",
    "pair_framing": "Pair framing",
    "em_dash_rate": "Em-dash rate",
    "exclamation_rate": "Exclamation rate",
    "sentence_len_var": "Sentence len. var.",
    "avg_wps": "Avg words/sent.",
    "mean_n_words": "Mean comment length (words)",
    "log_len_mean": "Log comment length",
    "share_ge20w": "Share comments ≥20w",
    "sem_axis_ideology": "Sem. ideology",
    "sem_axis_ideology_var": "Sem. ideology var.",
    "sem_axis_emotion": "Sem. emotion",
    "sem_axis_emotion_var": "Sem. emotion var.",
    "sem_axis_emotion_pruned": "Sem. emotion (pruned)",
    "sem_axis_aggression": "Sem. aggression",
    "sem_axis_economic": "Sem. economic",
    "sem_axis_cultural": "Sem. cultural",
    "sem_axis_nationalism": "Sem. nationalism",
    "sem_axis_anti_establishment": "Sem. anti-est.",
    "sem_axis_ideology_pole_share": "Sem. pole share",
    "sem_axis_ideology_esteban_ray": "Sem. Esteban-Ray",
    "sem_axis_ideology_extreme_left": "Sem. extreme left",
    "sem_axis_ideology_extreme_right": "Sem. extreme right",
    "wf_extremity": "WF extremity",
    "wf_extremity_z": "WF extremity (z)",
    "wf_change_z": "WF change (z)",
    "wf_change": "WF change",
    "wfa_extremity_z": "WFA extremity (z)",
    "wfa_change_z": "WFA change (z)",
    "wfa_extremity": "WFA extremity",
    "wfa_change": "WFA change",
    "wf2_extremity": "WF2 extremity",
    "wf2_extremity_z": "WF2 extremity (z)",
    "wf2_change_z": "WF2 change (z)",
    "wf2_change": "WF2 change",
    "wfa2_extremity_z": "WFA2 extremity (z)",
    "wfa2_change_z": "WFA2 change (z)",
    "wfa2_extremity": "WFA2 extremity",
    "wfa2_change": "WFA2 change",
    "semicolon_rate": "Semicolon rate",
    "colon_rate": "Colon rate",
    "hedging_phrase_rate": "Hedging phrase rate",
    "complexity_index": "Complexity index",
    "log_n_comments": "Log comments/day",
    "log_n_authors": "Log authors/day",
}


def outcome_label(outcome_id: str, *, short: bool = False) -> str:
    """Function summary: human-readable outcome label for plots and READMEs.

    Parameters:
    - outcome_id: outcome key.
    - short: if True, use compact label for figure axes.

    Returns:
    - Display label string.
    """
    if short and outcome_id in OUTCOME_LABELS_SHORT:
        return OUTCOME_LABELS_SHORT[outcome_id]
    return outcome_id.replace("_", " ")
