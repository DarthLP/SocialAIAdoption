"""
Script summary:
Load language-specific political and polarization lexicons; score comment text for hits.

Functionality:
- Flat political salience lists (`political_{lang}.txt`).
- Categorized polarization lists (`ideology_{lang}.txt`, etc.) with optional negation masking.
- Derived ideology indices and distributional helpers (Esteban–Ray, bimodality coefficient).

How to apply/run:
- Imported by enrichment and feature scripts under scripts/cleaning/ and scripts/features/.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

TOKEN_PATTERN = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_LEXICON_CACHE: Dict[str, Tuple[List[str], List[Tuple[str, ...]]]] = {}
_CATEGORIZED_CACHE: Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]] = {}

LEXICON_NAMES = ("ideology", "other_side", "aggression", "affect", "issue", "ai_style")
NEGATION_TOKENS_DEFAULT = frozenset(
    {"non", "not", "no", "never", "neither", "nor", "mai", "né", "senza", "without", "kein", "keine", "nicht", "nunca", "jamás"}
)

ISSUE_CATEGORIES = ("eu", "migration", "economy", "culture")
IDEOLOGY_CATEGORIES = ("left", "center", "right")


def lexicon_path(project_root: Path, lang_code: str, lexicon_name: str = "political") -> Path:
    """Function summary: resolve lexicon file path for a language and lexicon family.

    Parameters:
    - project_root: repository root Path.
    - lang_code: `it`, `en`, `de`, or `es`.
    - lexicon_name: `political` or categorized stem (`ideology`, `other_side`, ...).

    Returns:
    - Path to the lexicon text file.
    """
    stem = f"{lexicon_name}_{lang_code.lower()}"
    return project_root / "config" / "lexicons" / f"{stem}.txt"


def load_lexicon_terms(path: Path) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Function summary: load single-token and multi-token terms from a flat lexicon file.

    Parameters:
    - path: lexicon file path.

    Returns:
    - Tuple of (single_token_terms, phrase_token_tuples).
    """
    singles: List[str] = []
    phrases: List[Tuple[str, ...]] = []
    if not path.is_file():
        return singles, phrases
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip().lower()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            singles.append(parts[0])
        else:
            phrases.append(tuple(parts))
    return singles, phrases


def load_categorized_lexicon(path: Path) -> Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]:
    """Function summary: load categorized lexicon entries grouped by category prefix.

    Parameters:
    - path: lexicon file with `category:term` lines.

    Returns:
    - Mapping category -> (single_tokens, phrase_tuples).
    """
    out: Dict[str, Tuple[List[str], List[Tuple[str, ...]]]] = {}
    if not path.is_file():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip().lower()
        if not line or ":" not in line:
            continue
        category, term = line.split(":", 1)
        category = category.strip()
        term = term.strip()
        if not category or not term:
            continue
        singles, phrases = out.get(category, ([], []))
        singles = list(singles)
        phrases = list(phrases)
        parts = term.split()
        if len(parts) == 1:
            singles.append(parts[0])
        else:
            phrases.append(tuple(parts))
        out[category] = (singles, phrases)
    return out


def get_lexicon(project_root: Path, lang_code: str) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Function summary: return cached flat political lexicon terms.

    Parameters:
    - project_root: repository root Path.
    - lang_code: language code.

    Returns:
    - Tuple of (single_token_terms, phrase_token_tuples).
    """
    key = lang_code.lower()
    if key not in _LEXICON_CACHE:
        path = lexicon_path(project_root, key, lexicon_name="political")
        _LEXICON_CACHE[key] = load_lexicon_terms(path)
    return _LEXICON_CACHE[key]


def get_categorized_lexicon(
    project_root: Path, lang_code: str, lexicon_name: str
) -> Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]:
    """Function summary: return cached categorized lexicon for a language and family.

    Parameters:
    - project_root: repository root Path.
    - lang_code: language code.
    - lexicon_name: e.g. `ideology`, `other_side`.

    Returns:
    - Category -> (singles, phrases).
    """
    cache_key = f"{lexicon_name}:{lang_code.lower()}"
    if cache_key not in _CATEGORIZED_CACHE:
        path = lexicon_path(project_root, lang_code, lexicon_name=lexicon_name)
        _CATEGORIZED_CACHE[cache_key] = load_categorized_lexicon(path)
    return _CATEGORIZED_CACHE[cache_key]


def tokenize(text: str) -> List[str]:
    """Function summary: lowercase word tokens for lexicon matching.

    Parameters:
    - text: input string.

    Returns:
    - List of token strings.
    """
    return TOKEN_PATTERN.findall((text or "").lower())


def _count_terms_in_tokens(
    tokens: Sequence[str],
    singles: Sequence[str],
    phrases: Sequence[Tuple[str, ...]],
    negation_window: int = 0,
    negation_tokens: Optional[frozenset[str]] = None,
) -> int:
    """Function summary: count lexicon hits in a token list with optional negation masking.

    Parameters:
    - tokens: token sequence.
    - singles: single-token terms.
    - phrases: multi-token phrases.
    - negation_window: tokens before a hit to scan for negation.
    - negation_tokens: negation word set.

    Returns:
    - Hit count after negation masking.
    """
    n_words = len(tokens)
    if n_words == 0:
        return 0
    neg_set = negation_tokens or NEGATION_TOKENS_DEFAULT
    hits = 0

    def negated_at(index: int) -> bool:
        if negation_window <= 0:
            return False
        start = max(0, index - negation_window)
        return any(tokens[j] in neg_set for j in range(start, index))

    if singles:
        single_set = set(singles)
        for idx, tok in enumerate(tokens):
            if tok in single_set and not negated_at(idx):
                hits += 1
    for phrase in phrases:
        plen = len(phrase)
        if plen < 2:
            continue
        for idx in range(0, n_words - plen + 1):
            if tuple(tokens[idx : idx + plen]) == phrase and not negated_at(idx):
                hits += 1
    return hits


def count_political_hits(text: str, lang_code: str, project_root: Path) -> Tuple[int, int]:
    """Function summary: count political lexicon hits and word tokens in text.

    Parameters:
    - text: comment body.
    - lang_code: lexicon language (`it`, `en`, `de`, `es`).
    - project_root: repository root for lexicon files.

    Returns:
    - Tuple (political_hits, n_words).
    """
    tokens = tokenize(text)
    n_words = len(tokens)
    if n_words == 0:
        return 0, 0
    singles, phrases = get_lexicon(project_root, lang_code)
    hits = _count_terms_in_tokens(tokens, singles, phrases, negation_window=0)
    return hits, n_words


def count_categorized_hits(
    text: str,
    lang_code: str,
    lexicon_name: str,
    project_root: Path,
    negation_window: int = 0,
    categories: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, int], int]:
    """Function summary: count hits per category for a categorized lexicon file.

    Parameters:
    - text: comment body.
    - lang_code: language code.
    - lexicon_name: lexicon family stem.
    - project_root: repository root.
    - negation_window: negation lookback (ideology typically > 0).
    - categories: optional subset of categories to score.

    Returns:
    - Tuple (category -> hit count, n_words).
    """
    tokens = tokenize(text)
    n_words = len(tokens)
    if n_words == 0:
        return {}, 0
    lex = get_categorized_lexicon(project_root, lang_code, lexicon_name)
    use_negation = negation_window > 0 and lexicon_name == "ideology"
    out: Dict[str, int] = {}
    for category, (singles, phrases) in lex.items():
        if categories is not None and category not in categories:
            continue
        out[category] = _count_terms_in_tokens(
            tokens,
            singles,
            phrases,
            negation_window=negation_window if use_negation else 0,
        )
    return out, n_words


def political_rate_100w(hits: int, n_words: int) -> float:
    """Function summary: compute hits per 100 words.

    Parameters:
    - hits: lexicon hit count.
    - n_words: word token count.

    Returns:
    - Rate per 100 words (0.0 if n_words is 0).
    """
    if n_words <= 0:
        return 0.0
    return 100.0 * float(hits) / float(n_words)


def rates_per_100w(hits_by_category: Mapping[str, int], n_words: int) -> Dict[str, float]:
    """Function summary: convert per-category hit counts to per-100-word rates.

    Parameters:
    - hits_by_category: category -> hit count.
    - n_words: word count.

    Returns:
    - category -> rate per 100 words.
    """
    return {cat: political_rate_100w(int(h), n_words) for cat, h in hits_by_category.items()}


def compute_ideology_indices(
    left_hits: int,
    center_hits: int,
    right_hits: int,
    eps: float = 1.0e-6,
) -> Dict[str, float]:
    """Function summary: derive net ideology, extremity, and ambivalence from hit counts.

    Parameters:
    - left_hits: left lexicon hits.
    - center_hits: center hits.
    - right_hits: right hits.
    - eps: numerical stabilizer.

    Returns:
    - Dict with keys net_ideology, extremity, ambivalence (NaN-safe floats).
    """
    left = float(left_hits)
    center = float(center_hits)
    right = float(right_hits)
    lr = left + right
    total = left + center + right
    net = (left - right) / (lr + eps) if lr > 0 else 0.0
    extremity = max(left, right) / (total + eps) if total > 0 else 0.0
    ambivalence = min(left, right) / (lr + eps) if lr > 0 and left > 0 and right > 0 else 0.0
    return {
        "net_ideology": float(net),
        "extremity": float(extremity),
        "ambivalence": float(ambivalence),
    }


def esteban_ray_index(
    left_mass: float,
    center_mass: float,
    right_mass: float,
    alpha: float = 1.6,
) -> float:
    """Function summary: Esteban–Ray polarization index on trinary left/center/right masses.

    Parameters:
    - left_mass: mass share numerator for left (-1).
    - center_mass: mass for center (0).
    - right_mass: mass for right (+1).
    - alpha: distance exponent (default 1.6).

    Returns:
    - Polarization index (0 if total mass is 0).
    """
    masses = [max(0.0, float(left_mass)), max(0.0, float(center_mass)), max(0.0, float(right_mass))]
    total = sum(masses)
    if total <= 0:
        return 0.0
    positions = (-1.0, 0.0, 1.0)
    shares = [m / total for m in masses]
    acc = 0.0
    for i in range(3):
        for j in range(3):
            dist = abs(positions[i] - positions[j]) ** alpha
            acc += shares[i] * shares[j] * dist
    return float(acc)


def bimodality_coefficient(values: Sequence[float]) -> float:
    """Function summary: Sarle bimodality coefficient on a numeric sample (pandas skew/kurtosis).

    Parameters:
    - values: sample of net_ideology or similar.

    Returns:
    - BC value or NaN if insufficient data or zero kurtosis.
    """
    try:
        import pandas as pd
    except ImportError:
        return float("nan")
    series = pd.Series(list(values), dtype="float64").dropna()
    if len(series) < 4:
        return float("nan")
    kurt = float(series.kurtosis())
    if kurt == 0.0 or math.isnan(kurt):
        return float("nan")
    skew = float(series.skew())
    if math.isnan(skew):
        return float("nan")
    return float((skew**2 + 1.0) / kurt)


def score_comment_polarization(
    text: str,
    lang_code: str,
    project_root: Path,
    negation_window: int = 3,
    eps: float = 1.0e-6,
) -> Dict[str, float]:
    """Function summary: compute all polarization lexicon rates and derived indices for one comment.

    Parameters:
    - text: comment body.
    - lang_code: primary lexicon language.
    - project_root: repository root.
    - negation_window: ideology negation lookback tokens.
    - eps: stabilizer for derived indices.

    Returns:
    - Flat dict of hit counts, rates, and derived fields.
    """
    ideology_hits, n_words = count_categorized_hits(
        text, lang_code, "ideology", project_root, negation_window=negation_window
    )
    if n_words == 0:
        return {"n_words": 0}

    result: Dict[str, float] = {"n_words": float(n_words)}
    left_h = int(ideology_hits.get("left", 0))
    center_h = int(ideology_hits.get("center", 0))
    right_h = int(ideology_hits.get("right", 0))
    result["left_hits"] = float(left_h)
    result["center_hits"] = float(center_h)
    result["right_hits"] = float(right_h)
    result["left_rate_100w"] = political_rate_100w(left_h, n_words)
    result["right_rate_100w"] = political_rate_100w(right_h, n_words)
    result["center_rate_100w"] = political_rate_100w(center_h, n_words)
    result.update(compute_ideology_indices(left_h, center_h, right_h, eps=eps))

    other_hits, _ = count_categorized_hits(text, lang_code, "other_side", project_root)
    other_total = sum(other_hits.values())
    result["other_side_salience_hits"] = float(other_total)
    result["other_side_salience_rate_100w"] = political_rate_100w(other_total, n_words)

    agg_hits, _ = count_categorized_hits(text, lang_code, "aggression", project_root)
    agg_total = sum(agg_hits.values())
    result["aggression_hits"] = float(agg_total)
    result["aggression_rate_100w"] = political_rate_100w(agg_total, n_words)

    affect_hits, _ = count_categorized_hits(text, lang_code, "affect", project_root)
    result["negative_rate_100w"] = political_rate_100w(int(affect_hits.get("negative", 0)), n_words)
    result["anger_rate_100w"] = political_rate_100w(int(affect_hits.get("anger", 0)), n_words)

    issue_hits, _ = count_categorized_hits(text, lang_code, "issue", project_root)
    for cat in ISSUE_CATEGORIES:
        h = int(issue_hits.get(cat, 0))
        result[f"issue_{cat}_rate_100w"] = political_rate_100w(h, n_words)

    result["has_left_hit"] = float(left_h > 0)
    result["has_right_hit"] = float(right_h > 0)
    result["has_other_side_hit"] = float(other_total > 0)
    return result


def score_comment_ai_style(
    text: str,
    lang_code: str,
    project_root: Path,
) -> Dict[str, float]:
    """Function summary: compute AI-style lexicon rate for one comment.

    Parameters:
    - text: comment body.
    - lang_code: language code.
    - project_root: repository root.

    Returns:
    - Dict with ai_style hits/rate and lightweight style proxies.
    """
    hits, n_words = count_categorized_hits(text, lang_code, "ai_style", project_root)
    total = sum(hits.values())
    body = text or ""
    n_chars = len(body)
    sentences = [s for s in re.split(r"[.!?]+", body) if s.strip()]
    n_sentences = max(1, len(sentences))
    avg_wps = float(n_words) / float(n_sentences) if n_words > 0 else 0.0
    exclam = body.count("!")
    caps_words = sum(1 for t in tokenize(body) if len(t) > 1 and t.isupper())
    caps_share = float(caps_words) / float(n_words) if n_words > 0 else 0.0
    lengths = [len(tokenize(s)) for s in sentences if s.strip()]
    length_var = 0.0
    if len(lengths) >= 2:
        mean_l = sum(lengths) / len(lengths)
        length_var = sum((x - mean_l) ** 2 for x in lengths) / len(lengths)
    return {
        "n_words": float(n_words),
        "ai_style_hits": float(total),
        "ai_style_rate_100w": political_rate_100w(total, n_words),
        "n_chars": float(n_chars),
        "avg_words_per_sentence": float(avg_wps),
        "exclamation_rate_100w": political_rate_100w(exclam, n_words),
        "caps_word_share": float(caps_share),
        "em_dash_count": float(body.count("\u2014")),
        "sentence_length_variance": float(length_var),
    }
