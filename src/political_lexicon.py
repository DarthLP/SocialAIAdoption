"""
Script summary:
Load graded parallel political salience and polarization lexicons from raw CSV files.

Functionality:
- Political salience: `data/raw/political_lexicon_parallel.csv` (grades 1–3, unique term hits, weighted points).
- Categorized polarization: `data/raw/polarization_lexicon_parallel.csv` with optional negation masking.
- Emotion/cognition: `data/raw/emotion_cognition_parallel.csv`.
- Derived ideology indices and distributional helpers (Esteban–Ray, bimodality coefficient).

How to apply/run:
- Imported by enrichment and feature scripts under scripts/cleaning/ and scripts/features/.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
_GRADED_LEXICON_CACHE: Dict[str, List[Tuple[Tuple[str, ...], int, str]]] = {}
_GRADED_MATCHER_CACHE: Dict[str, "GradedPoliticalMatcher"] = {}
_CATEGORIZED_CACHE: Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]] = {}

PARALLEL_LANG_COLUMNS: Dict[str, Tuple[str, str]] = {
    "it": ("IT", "IT_grade"),
    "en": ("EN (US/UK)", "EN_grade"),
    "de": ("DE", "DE_grade"),
}
GRADE_POINTS = {1: 1, 2: 2, 3: 3}
DEFAULT_PARALLEL_LEXICON_REL = "data/raw/political_lexicon_parallel.csv"
DEFAULT_POLARIZATION_LEXICON_REL = "data/raw/polarization_lexicon_parallel.csv"
DEFAULT_EMOTION_COGNITION_REL = "data/raw/emotion_cognition_parallel.csv"

LEXICON_NAMES = ("ideology", "other_side", "aggression", "affect", "issue", "ai_style")
NEGATION_TOKENS_DEFAULT = frozenset(
    {"non", "not", "no", "never", "neither", "nor", "mai", "né", "senza", "without", "kein", "keine", "nicht"}
)

ISSUE_CATEGORIES = ("eu", "migration", "economy", "culture")
IDEOLOGY_CATEGORIES = ("left", "center", "right")


def default_polarization_lexicon_path(project_root: Path) -> Path:
    """Function summary: resolve default polarization_lexicon_parallel.csv path."""
    return project_root / DEFAULT_POLARIZATION_LEXICON_REL


def default_emotion_cognition_path(project_root: Path) -> Path:
    """Function summary: resolve default emotion_cognition_parallel.csv path."""
    return project_root / DEFAULT_EMOTION_COGNITION_REL


def resolve_polarization_lexicon_path(
    project_root: Path, csv_path: Optional[Path] = None
) -> Path:
    """Function summary: return explicit or default polarization parallel CSV path."""
    if csv_path is not None:
        return csv_path
    return default_polarization_lexicon_path(project_root)


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


def default_parallel_lexicon_path(project_root: Path) -> Path:
    """Function summary: resolve default path to the parallel graded political CSV.

    Parameters:
    - project_root: repository root Path.

    Returns:
    - Path to political_lexicon_parallel.csv under data/raw/.
    """
    return project_root / DEFAULT_PARALLEL_LEXICON_REL


def resolve_parallel_lexicon_path(project_root: Path, csv_path: Optional[Path] = None) -> Path:
    """Function summary: return explicit or default parallel lexicon CSV path.

    Parameters:
    - project_root: repository root Path.
    - csv_path: optional override path.

    Returns:
    - Resolved CSV path.
    """
    if csv_path is not None:
        return csv_path
    return default_parallel_lexicon_path(project_root)


def _norm_term_key(term: str) -> str:
    """Function summary: normalize a lemma for deduplication keys."""
    return " ".join((term or "").strip().lower().split())


def load_parallel_political_lexicon(csv_path: Path, lang_code: str) -> Dict[str, int]:
    """Function summary: load term -> grade (1–3) from parallel CSV with max-grade dedupe.

    Parameters:
    - csv_path: political_lexicon_parallel.csv path.
    - lang_code: it, en, or de.

    Returns:
    - Mapping normalized term key -> grade (duplicate rows keep max grade).
    """
    lang = lang_code.lower()
    if lang not in PARALLEL_LANG_COLUMNS:
        return {}
    term_col, grade_col = PARALLEL_LANG_COLUMNS[lang]
    out: Dict[str, int] = {}
    if not csv_path.is_file():
        return out
    from src.parallel_lexicon import expand_lexicon_variants, split_lexicon_cell

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cell = (row.get(term_col, "") or "").strip()
            if not cell:
                continue
            raw_grade = (row.get(grade_col) or "").strip()
            if not raw_grade.isdigit():
                continue
            grade = int(raw_grade)
            if grade not in GRADE_POINTS:
                continue
            pieces = split_lexicon_cell(cell) if ";" in cell else [cell]
            for piece in pieces:
                for variant in expand_lexicon_variants(piece, lang):
                    key = _norm_term_key(variant)
                    if key:
                        out[key] = max(out.get(key, 0), grade)
    return out


def get_graded_lexicon_entries(
    project_root: Path,
    lang_code: str,
    csv_path: Optional[Path] = None,
) -> List[Tuple[Tuple[str, ...], int, str]]:
    """Function summary: return cached graded entries sorted longest phrase first.

    Parameters:
    - project_root: repository root Path.
    - lang_code: it, en, or de.
    - csv_path: optional CSV override.

    Returns:
    - List of (token_tuple, grade, term_key) for matching.
    """
    path = resolve_parallel_lexicon_path(project_root, csv_path)
    cache_key = f"{path.resolve()}:{lang_code.lower()}"
    if cache_key not in _GRADED_LEXICON_CACHE:
        term_grades = load_parallel_political_lexicon(path, lang_code)
        entries: List[Tuple[Tuple[str, ...], int, str]] = []
        for key, grade in term_grades.items():
            entries.append((tuple(key.split()), grade, key))
        entries.sort(key=lambda e: len(e[0]), reverse=True)
        _GRADED_LEXICON_CACHE[cache_key] = entries
    return _GRADED_LEXICON_CACHE[cache_key]


def get_lexicon(project_root: Path, lang_code: str) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Function summary: return political salience terms from graded parallel lexicon.

    Parameters:
    - project_root: repository root Path.
    - lang_code: language code.

    Returns:
    - Tuple of (single_token_terms, phrase_token_tuples) for audit/listing.
    """
    singles: List[str] = []
    phrases: List[Tuple[str, ...]] = []
    for tokens, _grade, _key in get_graded_lexicon_entries(project_root, lang_code):
        if len(tokens) == 1:
            singles.append(tokens[0])
        else:
            phrases.append(tokens)
    return singles, phrases


def _match_graded_terms_unique(
    tokens: Sequence[str],
    entries: Sequence[Tuple[Tuple[str, ...], int, str]],
) -> Tuple[int, int, int, int]:
    """Function summary: match graded lexicon with unique terms and non-overlapping spans.

    Parameters:
    - tokens: comment token list.
    - entries: graded entries longest-first.

    Returns:
    - Tuple (g1_hits, g2_hits, g3_hits, weighted_points).
    """
    n_words = len(tokens)
    if n_words == 0:
        return 0, 0, 0, 0
    covered: set[int] = set()
    matched_keys: set[str] = set()
    g1 = g2 = g3 = 0
    for phrase, grade, key in entries:
        if key in matched_keys:
            continue
        plen = len(phrase)
        if plen == 0:
            continue
        for idx in range(0, n_words - plen + 1):
            if any(i in covered for i in range(idx, idx + plen)):
                continue
            if tuple(tokens[idx : idx + plen]) != phrase:
                continue
            matched_keys.add(key)
            covered.update(range(idx, idx + plen))
            if grade == 1:
                g1 += 1
            elif grade == 2:
                g2 += 1
            else:
                g3 += 1
            break
    points = g1 * GRADE_POINTS[1] + g2 * GRADE_POINTS[2] + g3 * GRADE_POINTS[3]
    return g1, g2, g3, points


class GradedPoliticalMatcher:
    """Function summary: cached graded lexicon matcher with first-token indexing for phrases."""

    def __init__(self, entries: Sequence[Tuple[Tuple[str, ...], int, str]]):
        """Function summary: build matcher from graded lexicon entries (longest phrase first).

        Parameters:
        - entries: list of (token_tuple, grade, term_key) sorted longest-first.
        """
        self.entries = list(entries)
        by_first: Dict[str, List[Tuple[Tuple[str, ...], int, str]]] = defaultdict(list)
        for phrase, grade, key in self.entries:
            if not phrase:
                continue
            by_first[phrase[0]].append((phrase, grade, key))
        for bucket in by_first.values():
            bucket.sort(key=lambda e: len(e[0]), reverse=True)
        self._by_first = dict(by_first)

    def score_tokens(self, tokens: Sequence[str]) -> Tuple[int, int, int, int]:
        """Function summary: match graded lexicon on tokenized comment text.

        Parameters:
        - tokens: comment token list.

        Returns:
        - Tuple (g1_hits, g2_hits, g3_hits, weighted_points).
        """
        n_words = len(tokens)
        if n_words == 0:
            return 0, 0, 0, 0
        covered: set[int] = set()
        matched_keys: set[str] = set()
        g1 = g2 = g3 = 0
        for idx in range(n_words):
            candidates = self._by_first.get(tokens[idx])
            if not candidates:
                continue
            for phrase, grade, key in candidates:
                if key in matched_keys:
                    continue
                plen = len(phrase)
                if plen == 0 or idx + plen > n_words:
                    continue
                if any(i in covered for i in range(idx, idx + plen)):
                    continue
                if tuple(tokens[idx : idx + plen]) != phrase:
                    continue
                matched_keys.add(key)
                covered.update(range(idx, idx + plen))
                if grade == 1:
                    g1 += 1
                elif grade == 2:
                    g2 += 1
                else:
                    g3 += 1
                break
        points = g1 * GRADE_POINTS[1] + g2 * GRADE_POINTS[2] + g3 * GRADE_POINTS[3]
        return g1, g2, g3, points

    def score_text(self, text: str) -> Tuple[int, int, int, int]:
        """Function summary: tokenize and score political salience on comment body.

        Parameters:
        - text: comment body.

        Returns:
        - Tuple (g1_hits, g2_hits, g3_hits, weighted_points).
        """
        return self.score_tokens(tokenize(text))


def get_graded_matcher(
    project_root: Path,
    lang_code: str,
    csv_path: Optional[Path] = None,
) -> GradedPoliticalMatcher:
    """Function summary: return cached GradedPoliticalMatcher for a language.

    Parameters:
    - project_root: repository root Path.
    - lang_code: it, en, or de.
    - csv_path: optional parallel CSV override.

    Returns:
    - GradedPoliticalMatcher instance.
    """
    path = resolve_parallel_lexicon_path(project_root, csv_path)
    cache_key = f"{path.resolve()}:{lang_code.lower()}"
    if cache_key not in _GRADED_MATCHER_CACHE:
        entries = get_graded_lexicon_entries(project_root, lang_code, csv_path=csv_path)
        _GRADED_MATCHER_CACHE[cache_key] = GradedPoliticalMatcher(entries)
    return _GRADED_MATCHER_CACHE[cache_key]


def score_bodies_political_salience(
    bodies: Sequence[str],
    lang_code: str,
    project_root: Path,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Function summary: batch-score comment bodies with graded unique-term matching.

    Parameters:
    - bodies: sequence of comment body strings.
    - lang_code: it, en, or de.
    - project_root: repository root.
    - csv_path: optional parallel CSV override.

    Returns:
    - DataFrame with political_g1_hits, political_g2_hits, political_g3_hits,
      political_weighted_points, n_words (one row per body).
    """
    matcher = get_graded_matcher(project_root, lang_code, csv_path=csv_path)
    g1_list: List[int] = []
    g2_list: List[int] = []
    g3_list: List[int] = []
    points_list: List[int] = []
    n_words_list: List[int] = []
    for body in bodies:
        tokens = tokenize(body)
        n_words = len(tokens)
        if n_words == 0:
            g1_list.append(0)
            g2_list.append(0)
            g3_list.append(0)
            points_list.append(0)
            n_words_list.append(0)
            continue
        g1, g2, g3, points = matcher.score_tokens(tokens)
        g1_list.append(g1)
        g2_list.append(g2)
        g3_list.append(g3)
        points_list.append(points)
        n_words_list.append(n_words)
    return pd.DataFrame(
        {
            "political_g1_hits": g1_list,
            "political_g2_hits": g2_list,
            "political_g3_hits": g3_list,
            "political_weighted_points": points_list,
            "n_words": n_words_list,
        }
    )


def score_comment_political_salience(
    text: str,
    lang_code: str,
    project_root: Path,
    csv_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Function summary: score political salience with graded unique-term matching.

    Parameters:
    - text: comment body.
    - lang_code: it, en, or de.
    - project_root: repository root.
    - csv_path: optional parallel CSV override.

    Returns:
    - Dict with political_g1_hits, political_g2_hits, political_g3_hits,
      political_weighted_points, n_words.
    """
    matcher = get_graded_matcher(project_root, lang_code, csv_path=csv_path)
    tokens = tokenize(text)
    n_words = len(tokens)
    if n_words == 0:
        return {
            "political_g1_hits": 0,
            "political_g2_hits": 0,
            "political_g3_hits": 0,
            "political_weighted_points": 0,
            "n_words": 0,
        }
    g1, g2, g3, points = matcher.score_tokens(tokens)
    return {
        "political_g1_hits": g1,
        "political_g2_hits": g2,
        "political_g3_hits": g3,
        "political_weighted_points": points,
        "n_words": n_words,
    }


def get_categorized_lexicon(
    project_root: Path,
    lang_code: str,
    lexicon_name: str,
    polarization_csv_path: Optional[Path] = None,
) -> Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]:
    """Function summary: return cached categorized lexicon for a language and family.

    Parameters:
    - project_root: repository root Path.
    - lang_code: language code.
    - lexicon_name: e.g. `ideology`, `other_side`.
    - polarization_csv_path: optional polarization CSV override.

    Returns:
    - Category -> (singles, phrases).
    """
    from src.parallel_lexicon import load_polarization_parallel

    path = resolve_polarization_lexicon_path(project_root, polarization_csv_path)
    cache_key = f"{path.resolve()}:{lang_code.lower()}:{lexicon_name}"
    if cache_key not in _CATEGORIZED_CACHE:
        full = load_polarization_parallel(path, lang_code)
        _CATEGORIZED_CACHE[cache_key] = full.get(lexicon_name, {})
    return _CATEGORIZED_CACHE[cache_key]


def _count_pole_hits(
    tokens: Sequence[str],
    singles: Sequence[str],
    phrases: Sequence[Tuple[str, ...]],
) -> int:
    """Function summary: count hits for one emotion/cognition pole lexicon."""
    return _count_terms_in_tokens(tokens, singles, phrases, negation_window=0)


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


def count_political_hits(
    text: str,
    lang_code: str,
    project_root: Path,
    csv_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Function summary: return weighted political points and word count for one comment.

    Parameters:
    - text: comment body.
    - lang_code: lexicon language (`it`, `en`, `de`).
    - project_root: repository root.
    - csv_path: optional parallel CSV override.

    Returns:
    - Tuple (political_weighted_points, n_words).
    """
    scored = score_comment_political_salience(text, lang_code, project_root, csv_path=csv_path)
    return int(scored["political_weighted_points"]), int(scored["n_words"])


def count_categorized_hits_from_tokens(
    tokens: Sequence[str],
    lang_code: str,
    lexicon_name: str,
    project_root: Path,
    negation_window: int = 0,
    categories: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Function summary: count hits per category using a pre-tokenized comment.

    Parameters:
    - tokens: lowercase word tokens.
    - lang_code: language code.
    - lexicon_name: lexicon family stem.
    - project_root: repository root.
    - negation_window: negation lookback (ideology typically > 0).
    - categories: optional subset of categories to score.

    Returns:
    - category -> hit count (empty dict when tokens empty).
    """
    if len(tokens) == 0:
        return {}
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
    return out


def count_categorized_hits(
    text: str,
    lang_code: str,
    lexicon_name: str,
    project_root: Path,
    negation_window: int = 0,
    categories: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, int], int]:
    """Function summary: count hits per category for a categorized lexicon file.

    Parameters:
    - text: comment body.
    - lang_code: language code.
    - lexicon_name: lexicon family stem.
    - project_root: repository root.
    - negation_window: negation lookback (ideology typically > 0).
    - categories: optional subset of categories to score.
    - tokens: optional pre-tokenized list (avoids re-tokenizing).

    Returns:
    - Tuple (category -> hit count, n_words).
    """
    tok_list = list(tokens) if tokens is not None else tokenize(text)
    n_words = len(tok_list)
    if n_words == 0:
        return {}, 0
    hits = count_categorized_hits_from_tokens(
        tok_list,
        lang_code,
        lexicon_name,
        project_root,
        negation_window=negation_window,
        categories=categories,
    )
    return hits, n_words


def warm_polarization_lexicons(project_root: Path, lang_code: str) -> None:
    """Function summary: preload categorized lexicons for one language into process cache.

    Parameters:
    - project_root: repository root.
    - lang_code: language code.

    Returns:
    - None.
    """
    for name in LEXICON_NAMES:
        get_categorized_lexicon(project_root, lang_code, name)


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
    from src.parallel_lexicon import load_emotion_cognition_parallel
    from src.v4_lexicon import (
        get_pairs_registry,
        score_pair_framing,
        zero_pair_framing_columns,
    )

    tokens = tokenize(text)
    n_words = len(tokens)
    if n_words == 0:
        out = {col: 0.0 for col in (
            "left_hits", "center_hits", "right_hits", "left_rate_100w", "center_rate_100w",
            "right_rate_100w", "net_ideology", "extremity", "ambivalence",
            "other_side_salience_hits", "other_side_salience_rate_100w",
            "aggression_hits", "aggression_rate_100w", "negative_rate_100w", "anger_rate_100w",
            "issue_eu_rate_100w", "issue_migration_rate_100w", "issue_economy_rate_100w",
            "issue_culture_rate_100w", "has_left_hit", "has_right_hit", "has_other_side_hit",
        )}
        out["n_words"] = 0.0
        out.update(zero_pair_framing_columns())
        out["emotion_hits"] = 0.0
        out["emotion_rate_100w"] = 0.0
        out["cognition_hits"] = 0.0
        out["cognition_rate_100w"] = 0.0
        return out

    emo_path = default_emotion_cognition_path(project_root)
    emo_lex = load_emotion_cognition_parallel(emo_path, lang_code)
    emo_s, emo_p = emo_lex.get("emotion", ([], []))
    cog_s, cog_p = emo_lex.get("cognition", ([], []))
    emotion_h = _count_pole_hits(tokens, emo_s, emo_p)
    cognition_h = _count_pole_hits(tokens, cog_s, cog_p)

    ideology_hits = count_categorized_hits_from_tokens(
        tokens, lang_code, "ideology", project_root, negation_window=negation_window
    )
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

    other_hits = count_categorized_hits_from_tokens(tokens, lang_code, "other_side", project_root)
    other_total = sum(other_hits.values())
    result["other_side_salience_hits"] = float(other_total)
    result["other_side_salience_rate_100w"] = political_rate_100w(other_total, n_words)

    agg_hits = count_categorized_hits_from_tokens(tokens, lang_code, "aggression", project_root)
    agg_total = sum(agg_hits.values())
    result["aggression_hits"] = float(agg_total)
    result["aggression_rate_100w"] = political_rate_100w(agg_total, n_words)

    affect_hits = count_categorized_hits_from_tokens(tokens, lang_code, "affect", project_root)
    result["negative_rate_100w"] = political_rate_100w(int(affect_hits.get("negative", 0)), n_words)
    result["anger_rate_100w"] = political_rate_100w(int(affect_hits.get("anger", 0)), n_words)

    issue_hits = count_categorized_hits_from_tokens(tokens, lang_code, "issue", project_root)
    for cat in ISSUE_CATEGORIES:
        h = int(issue_hits.get(cat, 0))
        result[f"issue_{cat}_rate_100w"] = political_rate_100w(h, n_words)

    result["has_left_hit"] = float(left_h > 0)
    result["has_right_hit"] = float(right_h > 0)
    result["has_other_side_hit"] = float(other_total > 0)
    result["emotion_hits"] = float(emotion_h)
    result["emotion_rate_100w"] = political_rate_100w(emotion_h, n_words)
    result["cognition_hits"] = float(cognition_h)
    result["cognition_rate_100w"] = political_rate_100w(cognition_h, n_words)

    if lang_code.lower() == "it":
        pairs = get_pairs_registry(project_root)
        for track in ("strict", "all"):
            result.update(score_pair_framing(text, pairs, track, n_words, eps=eps, tokens=tokens))
    else:
        result.update(zero_pair_framing_columns())
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
