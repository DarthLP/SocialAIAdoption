"""
Script summary:
Formula-based LLM-style index (v1) from comment-level stylometric features.

Functionality:
- Readability indices (Gulpease IT, Amstad DE, Flesch RE EN).
- Pre-period winsorized z-scores with persisted clip bounds (Mar 1–30, 2023).
- style_index_full and style_index_reduced (frozen SIGNS v1, 2026-06-04).

How to apply/run:
- Imported by feature and panel scripts; see scripts/diagnostics/fit_style_index_stats.py.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.comment_style import tokenize_words
from src.political_lexicon import political_rate_100w, score_comment_ai_style

PRE_PERIOD_START = "2023-03-01"
PRE_PERIOD_END = "2023-03-30"
STATS_VERSION = "v1_frozen_2026-06-04"

# Frozen signs (v1). formal_register_rate maps to ai_style_rate_100w column.
SIGNS: Dict[str, int] = {
    "log_len": +1,
    "avg_words_per_sentence": +1,
    "sentence_length_variance": -1,
    "em_dash_rate_100w": +1,
    "semicolon_colon_rate_100w": +1,
    "hedging_phrase_rate_100w": +1,
    "ai_style_rate_100w": +1,
    "exclamation_rate_100w": -1,
    "caps_word_share": -1,
}

FULL_INDEX_FEATURES: Tuple[str, ...] = tuple(SIGNS.keys())
REDUCED_INDEX_FEATURES: Tuple[str, ...] = (
    "log_len",
    "em_dash_rate_100w",
    "semicolon_colon_rate_100w",
    "exclamation_rate_100w",
    "caps_word_share",
)
MIN_FEATURES_FOR_INDEX = 3
MIN_WORDS_FULL_INDEX = 20


def count_sentences(text: str) -> int:
    """Function summary: count sentence segments split on .!?"""
    parts = [p for p in re.split(r"[.!?]+", text or "") if p.strip()]
    return max(1, len(parts))


def readability_gulpease(n_chars: int, n_words: int, n_sentences: int) -> float:
    """Function summary: Italian Gulpease readability index."""
    if n_words <= 0 or n_sentences <= 0:
        return float("nan")
    return float(300.0 * n_sentences / n_words - 10.0 * n_chars / n_words + 89.0)


def readability_amstad(n_chars: int, n_words: int, n_sentences: int) -> float:
    """Function summary: German Amstad readability (simplified formula)."""
    if n_words <= 0 or n_sentences <= 0:
        return float("nan")
    return float(180.0 - n_chars / n_words - 58.5 * (n_sentences / n_words))


def readability_flesch_en(n_words: int, n_sentences: int, n_syllables: int) -> float:
    """Function summary: English Flesch Reading Ease with syllable proxy."""
    if n_words <= 0 or n_sentences <= 0:
        return float("nan")
    return float(206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words))


def estimate_syllables_en(word: str) -> int:
    """Function summary: crude English syllable count for one token."""
    w = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_v = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(1, count)


def readability_for_language(lang_code: str, text: str, n_words: int) -> float:
    """Function summary: language-specific readability for one comment."""
    n_chars = len(text or "")
    n_sent = count_sentences(text)
    lang = str(lang_code).lower()
    if lang == "it":
        return readability_gulpease(n_chars, n_words, n_sent)
    if lang == "de":
        return readability_amstad(n_chars, n_words, n_sent)
    if lang == "en":
        syll = sum(estimate_syllables_en(t) for t in tokenize_words(text))
        return readability_flesch_en(n_words, n_sent, syll)
    return float("nan")


def ttr_first_n_tokens(text: str, n: int = 50) -> float:
    """Function summary: type-token ratio on first n word tokens; NaN if fewer than n."""
    tokens = tokenize_words(text)
    if len(tokens) < n:
        return float("nan")
    head = tokens[:n]
    return float(len(set(head)) / float(n))


def comment_feature_dict(
    text: str,
    lang_code: str,
    project_root: Path,
) -> Dict[str, float]:
    """Function summary: per-comment stylometric features for index construction.

    Parameters:
    - text: comment body.
    - lang_code: language code.
    - project_root: repository root.

    Returns:
    - Dict of feature name -> value.
    """
    from src.comment_style import score_comment_style

    ai = score_comment_ai_style(text, lang_code, project_root)
    style = score_comment_style(text, lang_code, project_root, enable_phrase_lexicons=True)
    n_words = int(ai.get("n_words", 0) or 0)
    n_words_f = float(max(n_words, 0))
    semicolon = float(style.get("semicolon_count", 0) or 0)
    colon = float(style.get("colon_count", 0) or 0)
    sc_rate = (
        100.0 * (semicolon + colon) / n_words_f if n_words_f > 0 else float("nan")
    )
    return {
        "n_words": n_words_f,
        "log_len": float(math.log1p(n_words_f)) if n_words_f > 0 else float("nan"),
        "avg_words_per_sentence": float(ai.get("avg_words_per_sentence", float("nan"))),
        "sentence_length_variance": float(ai.get("sentence_length_variance", float("nan"))),
        "em_dash_rate_100w": political_rate_100w(int(ai.get("em_dash_count", 0) or 0), n_words),
        "semicolon_colon_rate_100w": sc_rate,
        "hedging_phrase_rate_100w": political_rate_100w(
            int(style.get("hedging_phrase_hits", 0) or 0), n_words
        ),
        "ai_style_rate_100w": float(ai.get("ai_style_rate_100w", float("nan"))),
        "exclamation_rate_100w": political_rate_100w(
            int(style.get("exclamation_count", 0) or 0), n_words
        ),
        "caps_word_share": float(ai.get("caps_word_share", float("nan"))),
        "ttr_50w": ttr_first_n_tokens(text, 50),
        "readability": readability_for_language(lang_code, text, n_words),
    }


def _winsorize_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Function summary: clip series to [lo, hi]."""
    return s.clip(lower=lo, upper=hi)


def fit_preperiod_stats(comments: pd.DataFrame) -> Dict[str, Any]:
    """Function summary: per-language clip bounds and winsorized mean/SD on pre-period comments.

    Parameters:
    - comments: DataFrame with date_utc, primary_lexicon or lang, and feature columns.

    Returns:
    - JSON-serializable stats dict.
    """
    if comments.empty:
        return {
            "version": STATS_VERSION,
            "pre_period": [PRE_PERIOD_START, PRE_PERIOD_END],
            "languages": {},
        }
    work = comments.copy()
    if "date_utc" not in work.columns:
        raise KeyError("date_utc")
    work["date_utc"] = work["date_utc"].astype(str)
    mask = (work["date_utc"] >= PRE_PERIOD_START) & (work["date_utc"] <= PRE_PERIOD_END)
    work = work[mask]
    if "lang" not in work.columns:
        work["lang"] = work.get("primary_lexicon", pd.Series("", index=work.index)).astype(str)

    out: Dict[str, Any] = {"version": STATS_VERSION, "pre_period": [PRE_PERIOD_START, PRE_PERIOD_END], "languages": {}}
    feature_cols = [c for c in FULL_INDEX_FEATURES if c in work.columns]
    feature_cols += [c for c in ("ttr_50w", "readability") if c in work.columns]

    for lang, grp in work.groupby("lang", observed=True):
        lang_stats: Dict[str, Any] = {}
        for feat in feature_cols:
            s = pd.to_numeric(grp[feat], errors="coerce").dropna()
            if len(s) < 30:
                continue
            lo = float(s.quantile(0.01))
            hi = float(s.quantile(0.99))
            clipped = _winsorize_series(s, lo, hi)
            mu = float(clipped.mean())
            sd = float(clipped.std())
            if not sd or sd <= 0:
                sd = float("nan")
            lang_stats[feat] = {"clip_lo": lo, "clip_hi": hi, "mu": mu, "sigma": sd}
        out["languages"][str(lang)] = lang_stats
    return out


def save_style_index_stats(stats: Mapping[str, Any], path: Path) -> None:
    """Function summary: write stats JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_style_index_stats(path: Path) -> Dict[str, Any]:
    """Function summary: load stats JSON from path."""
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_feature(val: float, feat: str, lang_stats: Mapping[str, Any]) -> float:
    """Function summary: apply stored clip bounds to one feature value."""
    if not np.isfinite(val):
        return float("nan")
    meta = lang_stats.get(feat, {})
    if not meta:
        return val
    lo = meta.get("clip_lo", float("nan"))
    hi = meta.get("clip_hi", float("nan"))
    if np.isfinite(lo) and np.isfinite(hi):
        return float(np.clip(val, lo, hi))
    return float(val)


def _signed_z(val: float, feat: str, lang_stats: Mapping[str, Any]) -> float:
    """Function summary: signed z-score for one feature using pre-period mu/sigma."""
    if not np.isfinite(val):
        return float("nan")
    meta = lang_stats.get(feat, {})
    mu = meta.get("mu", float("nan"))
    sd = meta.get("sigma", float("nan"))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    z = (val - float(mu)) / float(sd)
    sign = int(SIGNS.get(feat, 1))
    return float(sign * z)


def compute_index(
    features: Mapping[str, float],
    stats: Mapping[str, Any],
    lang_code: str,
) -> Tuple[float, float]:
    """Function summary: style_index_full and style_index_reduced from features + stats.

    Parameters:
    - features: per-comment feature dict.
    - stats: output of fit_preperiod_stats / load_style_index_stats.
    - lang_code: language code.

    Returns:
    - Tuple (style_index_full, style_index_reduced); NaN when insufficient features.
    """
    lang_stats = stats.get("languages", {}).get(str(lang_code).lower(), {})
    n_words = float(features.get("n_words", 0) or 0)

    def _mean_index(feat_list: Sequence[str]) -> float:
        zs: list[float] = []
        for feat in feat_list:
            raw = float(features.get(feat, float("nan")))
            clipped = _clip_feature(raw, feat, lang_stats)
            z = _signed_z(clipped, feat, lang_stats)
            if np.isfinite(z):
                zs.append(z)
        if len(zs) < MIN_FEATURES_FOR_INDEX:
            return float("nan")
        return float(np.mean(zs))

    reduced = _mean_index(REDUCED_INDEX_FEATURES)
    if n_words < MIN_WORDS_FULL_INDEX:
        return float("nan"), reduced
    full = _mean_index(FULL_INDEX_FEATURES)
    return full, reduced
