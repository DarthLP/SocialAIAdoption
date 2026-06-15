"""
Script summary:
LLM-style composite index (style_index_llm) with frozen-weight leave-one-out ablation robustness.

Functionality:
- Primary index = continuous LLM-likelihood score (weighted signed-z; no cutoff in DiD).
- Leave-one-out ablations = drop one feature and renormalize remaining weights (default).
- Fixed-denominator LOO is opt-in via ablation_renorm=False (legacy).
- Only-* indices = single-feature signed-z (marginal influence vs composite).
- Relative weights may exceed 1.0 (e.g. em_dash=3); only the ratio across features matters.

How to apply/run:
- Imported only; run scripts/diagnostics/validate_style_index_weights.py to pick primary on a sample.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.style_index import (
    MIN_FEATURES_LLM_V3_FULL,
    MIN_FEATURES_LLM_V3_REDUCED,
    MIN_WORDS_FULL_INDEX,
    SIGNS_V3,
    _clip_feature,
    _signed_z,
    _weighted_mean_index,
)

# Base features for all LLM candidates (no log_len / ttr).
LLM_BASE_FEATURES: Tuple[str, ...] = (
    "ai_style_rate_100w",
    "hedging_phrase_rate_100w",
    "avg_words_per_sentence",
    "sentence_length_variance",
    "exclamation_rate_100w",
    "caps_word_share",
    "em_dash_rate_100w",
    "semicolon_colon_rate_100w",
)

LLM_REDUCED_FEATURES: Tuple[str, ...] = ("ai_style_rate_100w",)

# Interaction features (rates product; calibrated on tune sample).
INTERACTION_BUILDERS: Dict[str, Callable[[Mapping[str, float]], float]] = {
    "em_dash_x_ai_rate": lambda f: float(f.get("em_dash_rate_100w", float("nan")))
    * float(f.get("ai_style_rate_100w", float("nan"))),
    "semicolon_x_ai_rate": lambda f: float(f.get("semicolon_colon_rate_100w", float("nan")))
    * float(f.get("ai_style_rate_100w", float("nan"))),
}

# Leave-one-out drops for robustness (primary uses full spec; each drop renormalizes).
ABLATION_DROP_FEATURES: Tuple[str, ...] = (
    "ai_style_rate_100w",
    "em_dash_rate_100w",
    "semicolon_colon_rate_100w",
    "hedging_phrase_rate_100w",
    "exclamation_rate_100w",
    "caps_word_share",
)

THEORY_WEIGHTS_IT: Dict[str, float] = {
    "ai_style_rate_100w": 0.54,
    "exclamation_rate_100w": 0.09,
    "caps_word_share": 0.12,
    "hedging_phrase_rate_100w": 0.11,
    "avg_words_per_sentence": 0.06,
    "sentence_length_variance": 0.04,
    "em_dash_rate_100w": 0.018,
    "semicolon_colon_rate_100w": 0.008,
}

# Small candidate grid (frozen weights; no ρ-tuning to ai_rate).
LLM_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "theory_base": {
        "description": "Theory weights, no interactions",
        "weights": dict(THEORY_WEIGHTS_IT),
        "interactions": [],
    },
    "theory_interact": {
        "description": "Theory + em_dash×ai and semicolon×ai interaction terms",
        "weights": {
            "ai_style_rate_100w": 0.46,
            "exclamation_rate_100w": 0.10,
            "caps_word_share": 0.11,
            "hedging_phrase_rate_100w": 0.09,
            "avg_words_per_sentence": 0.05,
            "sentence_length_variance": 0.04,
            "em_dash_rate_100w": 0.018,
            "semicolon_colon_rate_100w": 0.012,
            "em_dash_x_ai_rate": 0.05,
            "semicolon_x_ai_rate": 0.04,
        },
        "interactions": ["em_dash_x_ai_rate", "semicolon_x_ai_rate"],
    },
    "lexical_heavy": {
        "description": "Higher ai_rate weight (lexicon-forward)",
        "weights": {
            "ai_style_rate_100w": 0.58,
            "exclamation_rate_100w": 0.09,
            "caps_word_share": 0.11,
            "hedging_phrase_rate_100w": 0.09,
            "avg_words_per_sentence": 0.04,
            "sentence_length_variance": 0.03,
            "em_dash_rate_100w": 0.01,
            "semicolon_colon_rate_100w": 0.005,
        },
        "interactions": [],
    },
    "interact_heavy": {
        "description": "Interaction-forward (synergy terms)",
        "weights": {
            "ai_style_rate_100w": 0.42,
            "exclamation_rate_100w": 0.08,
            "caps_word_share": 0.08,
            "hedging_phrase_rate_100w": 0.08,
            "avg_words_per_sentence": 0.05,
            "sentence_length_variance": 0.03,
            "em_dash_rate_100w": 0.018,
            "semicolon_colon_rate_100w": 0.012,
            "em_dash_x_ai_rate": 0.08,
            "semicolon_x_ai_rate": 0.07,
        },
        "interactions": ["em_dash_x_ai_rate", "semicolon_x_ai_rate"],
    },
    "anti_casual": {
        "description": "Hedging & exclamation signed negative (less casual = higher index)",
        "weights": dict(THEORY_WEIGHTS_IT),
        "interactions": [],
        "signs": {
            "ai_style_rate_100w": 1,
            "hedging_phrase_rate_100w": -1,
            "avg_words_per_sentence": 1,
            "sentence_length_variance": -1,
            "exclamation_rate_100w": -1,
            "caps_word_share": -1,
            "em_dash_rate_100w": 1,
            "semicolon_colon_rate_100w": 1,
        },
    },
}

# Single-feature indices for marginal / reverse robustness checks.
ONLY_FEATURES: Tuple[str, ...] = (
    "ai_style_rate_100w",
    "em_dash_rate_100w",
    "semicolon_colon_rate_100w",
    "hedging_phrase_rate_100w",
    "exclamation_rate_100w",
    "caps_word_share",
)

PRIMARY_COL = "style_index_llm"
BUNDLE_LLM = "style_index_llm"


def _short_feature_name(feat: str) -> str:
    """Function summary: shorten feature name for column suffix."""
    return (
        feat.replace("_rate_100w", "")
        .replace("_100w", "")
        .replace("_share", "")
        .replace("_ai_rate", "_ai")
    )


def ablation_column_name(drop_feature: str) -> str:
    """Function summary: shard column name for leave-one-out index."""
    return f"style_index_llm_no_{_short_feature_name(drop_feature)}"


def only_column_name(feature: str) -> str:
    """Function summary: shard column for single-feature signed-z index."""
    return f"style_index_llm_only_{_short_feature_name(feature)}"


def enrich_interaction_features(features: Mapping[str, float], interactions: Sequence[str]) -> Dict[str, float]:
    """Function summary: copy features dict and add interaction columns.

    Parameters:
    - features: base comment features.
    - interactions: names from INTERACTION_BUILDERS.

    Returns:
    - Feature dict including interaction values.
    """
    out = dict(features)
    for name in interactions:
        builder = INTERACTION_BUILDERS.get(name)
        if builder is None:
            continue
        val = builder(features)
        out[name] = float(val) if np.isfinite(val) else float("nan")
    return out


def _total_weight_sum(weights: Mapping[str, float]) -> float:
    """Function summary: sum of positive relative weights (fixed ablation denominator)."""
    return float(sum(max(0.0, float(v)) for v in weights.values()))


def _drop_set(drop: Sequence[str]) -> set[str]:
    """Function summary: expand leave-one-out drop set (ai drop also drops x_ai interactions)."""
    drop_set = set(drop)
    if "ai_style_rate_100w" in drop_set:
        drop_set.update(("em_dash_x_ai_rate", "semicolon_x_ai_rate"))
    return drop_set


def renormalize_weights(
    weights: Mapping[str, float],
    drop: Sequence[str] = (),
) -> Dict[str, float]:
    """Function summary: zero dropped features and renormalize positive relative weights.

    Weights need not be in [0, 1]; values are relative importance (e.g. em_dash=3 vs ai=0.5).

    Parameters:
    - weights: full weight map.
    - drop: feature names to remove.

    Returns:
    - Renormalized weights summing to 1 (empty if all zero).
    """
    drop_set = _drop_set(drop)
    w = {k: float(v) for k, v in weights.items() if k not in drop_set and float(v) > 0}
    s = sum(w.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in w.items()}


def _weighted_mean_fixed_denominator(
    features: Mapping[str, float],
    weights: Mapping[str, float],
    lang_stats: Mapping[str, Any],
    *,
    signs: Mapping[str, int],
    drop: Sequence[str],
    w_total: float,
    min_features: int,
    feat_list: Sequence[str],
) -> float:
    """Function summary: weighted signed-z sum divided by fixed total weight (LOO-safe).

    Leave-one-out zeros dropped features in the numerator but keeps the same
    denominator as the full index, avoiding weight redistribution artifacts.
    """
    if w_total <= 0:
        return float("nan")
    drop_set = _drop_set(drop)
    zs: list[float] = []
    ws: list[float] = []
    for feat in feat_list:
        w = float(weights.get(feat, 0.0))
        if feat in drop_set or w <= 0:
            continue
        raw = float(features.get(feat, float("nan")))
        clipped = _clip_feature(raw, feat, lang_stats)
        z = _signed_z(clipped, feat, lang_stats, signs=signs)
        if np.isfinite(z):
            zs.append(z)
            ws.append(w)
    if len(zs) < min_features:
        return float("nan")
    return float(sum(z * ww for z, ww in zip(zs, ws)) / w_total)


def compute_llm_index(
    features: Mapping[str, float],
    lang_stats: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    interactions: Sequence[str] = (),
    drop: Sequence[str] = (),
    signs: Mapping[str, int] | None = None,
    ablation_renorm: bool = True,
) -> float:
    """Function summary: one LLM composite (primary or ablation).

    Parameters:
    - features: per-comment features (base; interactions added internally).
    - lang_stats: pre-period calibration block for language.
    - weights: feature weights for this candidate/ablation.
    - interactions: interaction feature names to materialize.
    - drop: leave-one-out feature names.
    - signs: signed-z map (default SIGNS_V3).
    - ablation_renorm: if True (default), LOO drops renormalize remaining weights; primary uses row-wise weighted mean. If False, LOO uses fixed full-index denominator (legacy).

    Returns:
    - Composite index or NaN.
    """
    sign_map = dict(signs or SIGNS_V3)
    feat_map = enrich_interaction_features(features, interactions)
    n_words = float(features.get("n_words", 0) or 0)
    feat_keys = list(dict.fromkeys(list(weights.keys()) + list(interactions)))
    use_weights = (
        renormalize_weights(weights, drop=drop)
        if drop and ablation_renorm
        else {k: float(v) for k, v in weights.items() if float(v) > 0}
    )

    if n_words < MIN_WORDS_FULL_INDEX:
        if drop and not ablation_renorm:
            w_total = _total_weight_sum(weights)
            return _weighted_mean_fixed_denominator(
                feat_map,
                weights,
                lang_stats,
                signs=sign_map,
                drop=drop,
                w_total=w_total,
                min_features=MIN_FEATURES_LLM_V3_REDUCED,
                feat_list=LLM_REDUCED_FEATURES,
            )
        red = {k: use_weights[k] for k in LLM_REDUCED_FEATURES if k in use_weights}
        return _weighted_mean_index(
            feat_map, red, lang_stats, signs=sign_map, min_features=MIN_FEATURES_LLM_V3_REDUCED
        )

    if drop and not ablation_renorm:
        w_total = _total_weight_sum(weights)
        return _weighted_mean_fixed_denominator(
            feat_map,
            weights,
            lang_stats,
            signs=sign_map,
            drop=drop,
            w_total=w_total,
            min_features=MIN_FEATURES_LLM_V3_FULL,
            feat_list=feat_keys,
        )

    return _weighted_mean_index(
        feat_map,
        use_weights,
        lang_stats,
        signs=sign_map,
        min_features=MIN_FEATURES_LLM_V3_FULL,
    )


def candidate_spec_from_stats(
    stats: Mapping[str, Any], lang_code: str
) -> Tuple[str, Dict[str, float], List[str], Dict[str, int]]:
    """Function summary: resolve primary candidate id, weights, interactions, signs from v3 JSON.

    Returns:
    - Tuple (candidate_id, weights, interaction_names, signs map).
    """
    lang = stats.get("languages", {}).get(str(lang_code).lower(), {})
    bundles = lang.get("bundles", {}) if isinstance(lang.get("bundles"), Mapping) else {}
    cfg = bundles.get(BUNDLE_LLM, {}) if isinstance(bundles, Mapping) else {}
    if not cfg and isinstance(bundles, Mapping):
        cfg = bundles.get("llm_v3", {})
    cid = str(stats.get("primary_candidate", cfg.get("candidate_id", "theory_base")))
    cand = LLM_CANDIDATES.get(cid, LLM_CANDIDATES["theory_base"])
    w = dict(cfg.get("weights", cand["weights"]))
    interactions = list(cfg.get("interactions", cand.get("interactions", [])))
    signs_raw = cfg.get("signs", cand.get("signs", SIGNS_V3))
    signs = {str(k): int(v) for k, v in dict(signs_raw).items()} if signs_raw else dict(SIGNS_V3)
    return cid, w, interactions, signs


def compute_style_index_llm_columns(
    features: Mapping[str, float],
    stats: Mapping[str, Any],
    lang_code: str,
) -> Dict[str, float]:
    """Function summary: primary style_index_llm plus leave-one-out ablation columns.

    Parameters:
    - features: comment feature dict.
    - stats: calibration JSON with primary_candidate + bundle weights.
    - lang_code: language code.

    Returns:
    - Dict column_name -> index value (includes PRIMARY_COL and ablation columns).
    """
    lang_stats = stats.get("languages", {}).get(str(lang_code).lower(), {})
    _, weights, interactions, sign_map = candidate_spec_from_stats(stats, lang_code)

    out: Dict[str, float] = {}
    out[PRIMARY_COL] = compute_llm_index(
        features,
        lang_stats,
        weights,
        interactions=interactions,
        drop=(),
        signs=sign_map,
    )
    for drop in ABLATION_DROP_FEATURES:
        if drop not in weights and drop not in interactions:
            continue
        col = ablation_column_name(drop)
        out[col] = compute_llm_index(
            features,
            lang_stats,
            weights,
            interactions=interactions,
            drop=(drop,),
            signs=sign_map,
        )
    if interactions:
        for ix in interactions:
            col = ablation_column_name(ix)
            out[col] = compute_llm_index(
                features, lang_stats, weights, interactions=interactions, drop=(ix,)
            )
    for feat in ONLY_FEATURES:
        out[only_column_name(feat)] = compute_llm_index(
            features,
            lang_stats,
            {feat: 1.0},
            interactions=(),
            drop=(),
            signs=sign_map,
        )
    return out


# Back-compat alias.
compute_llm_v3_indices = compute_style_index_llm_columns


def calibration_features_for_candidate(candidate_id: str) -> Tuple[str, ...]:
    """Function summary: all feature names needing pre-period stats for one candidate."""
    cand = LLM_CANDIDATES[candidate_id]
    names = list(LLM_BASE_FEATURES) + list(cand.get("interactions", []))
    return tuple(dict.fromkeys(names))
