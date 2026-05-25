"""
Script summary:
Load parallel lexicon CSVs (polarization, style phrases, emotion/cognition) and spelling variants.

Functionality:
- split_lexicon_cell splits language cells on semicolons only (never commas).
- expand_lexicon_variants adds DE umlaut/ASCII and IT accent alternates for matching.
- Cached loaders keyed by path mtime and language code.

How to apply/run:
- Imported by src.political_lexicon, src.comment_style, and src.v4_lexicon.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

PARALLEL_LANG_COLUMNS: Dict[str, Tuple[str, str]] = {
    "it": ("IT", "IT_grade"),
    "en": ("EN (US/UK)", "EN_grade"),
    "de": ("DE", "DE_grade"),
}

EMOTION_LANG_COLUMNS: Dict[str, str] = {
    "it": "IT",
    "en": "EN",
    "de": "DE",
}

STYLE_LANG_COLUMNS: Dict[str, str] = {
    "it": "IT",
    "en": "EN",
    "de": "DE",
}

POLARIZATION_LEXICON_NAMES = (
    "ideology",
    "other_side",
    "aggression",
    "affect",
    "issue",
    "ai_style",
)

# German digraph <-> umlaut (apply as whole-token replacements)
_DE_AE = ("ae", "ä")
_DE_OE = ("oe", "ö")
_DE_UE = ("ue", "ü")

# Italian accent stripping pairs (accented -> plain)
_IT_ACCENT_PAIRS = (
    ("à", "a"),
    ("è", "e"),
    ("é", "e"),
    ("ì", "i"),
    ("ò", "o"),
    ("ù", "u"),
)

_POLARIZATION_CACHE: Dict[str, Tuple[float, Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]]]] = {}
_STYLE_CACHE: Dict[str, Tuple[float, Dict[str, Dict[str, Tuple[str, ...]]]]] = {}
_EMOTION_CACHE: Dict[str, Tuple[float, Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]]]] = {}


def split_lexicon_cell(cell: str) -> List[str]:
    """Function summary: split a language cell on semicolons only.

    Parameters:
    - cell: raw CSV language column value.

    Returns:
    - List of stripped lemma strings (empty when cell is blank).
    """
    return [p.strip() for p in (cell or "").split(";") if p.strip()]


def _apply_replacements(word: str, pairs: Sequence[Tuple[str, str]]) -> str:
    """Function summary: apply ordered substring replacements to a token."""
    out = word
    for src, dst in pairs:
        out = out.replace(src, dst)
    return out


def _german_variants(token: str) -> Set[str]:
    """Function summary: add German ASCII/umlaut and eszett alternates for one token."""
    variants = {token}
    if "ae" in token:
        variants.add(_apply_replacements(token, ((_DE_AE[0], _DE_AE[1]),)))
    if "oe" in token:
        variants.add(_apply_replacements(token, ((_DE_OE[0], _DE_OE[1]),)))
    if "ue" in token:
        variants.add(_apply_replacements(token, ((_DE_UE[0], _DE_UE[1]),)))
    if "ss" in token:
        variants.add(token.replace("ss", "ß"))
    for v in list(variants):
        if "ä" in v:
            variants.add(v.replace("ä", "ae"))
        if "ö" in v:
            variants.add(v.replace("ö", "oe"))
        if "ü" in v:
            variants.add(v.replace("ü", "ue"))
        if "ß" in v:
            variants.add(v.replace("ß", "ss"))
    return variants


def _italian_variants(token: str) -> Set[str]:
    """Function summary: add unaccented vowel alternates for Italian tokens."""
    variants = {token}
    for acc, plain in _IT_ACCENT_PAIRS:
        if acc in token:
            variants.add(token.replace(acc, plain))
    for v in list(variants):
        for acc, plain in _IT_ACCENT_PAIRS:
            if plain in v and acc not in v:
                variants.add(v.replace(plain, acc, 1))
    return variants


def expand_lexicon_variants(lemma: str, lang_code: str) -> frozenset[str]:
    """Function summary: return normalized spelling variants for lexicon matching.

    Parameters:
    - lemma: single lemma or phrase (spaces allowed).
    - lang_code: it, en, or de.

    Returns:
    - Frozenset of lowercased variant strings (including per-token expansions for phrases).
    """
    base = " ".join((lemma or "").strip().lower().split())
    if not base:
        return frozenset()
    lang = lang_code.lower()
    tokens = base.split()
    per_token: List[Set[str]] = []
    for tok in tokens:
        opts = {tok}
        if lang == "de":
            opts |= _german_variants(tok)
        if lang in ("it", "de"):
            opts |= _italian_variants(tok)
        per_token.append(opts)
    if len(per_token) == 1:
        return frozenset(per_token[0])
    combined: Set[str] = set()
    for parts in itertools.product(*per_token):
        combined.add(" ".join(parts))
    return frozenset(combined) if combined else frozenset({base})


def _variants_to_singles_phrases(
    lemmas: Sequence[str],
    lang_code: str,
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Function summary: build singles and phrase tuples from lemmas with variant expansion.

    Parameters:
    - lemmas: raw lemma strings from CSV.
    - lang_code: language code.

    Returns:
    - Tuple (single_token_list, phrase_tuple_list) for categorized matching.
    """
    singles: List[str] = []
    phrases: List[Tuple[str, ...]] = []
    seen_s: Set[str] = set()
    seen_p: Set[Tuple[str, ...]] = set()
    for lemma in lemmas:
        for variant in expand_lexicon_variants(lemma, lang_code):
            parts = variant.split()
            if len(parts) == 1:
                if parts[0] not in seen_s:
                    seen_s.add(parts[0])
                    singles.append(parts[0])
            else:
                pt = tuple(parts)
                if pt not in seen_p:
                    seen_p.add(pt)
                    phrases.append(pt)
    return singles, phrases


def _file_mtime(path: Path) -> float:
    """Function summary: return file mtime or 0 when missing."""
    return path.stat().st_mtime if path.is_file() else 0.0


def _lexicon_key_field(row: Dict[str, str]) -> str:
    """Function summary: read lexicon column handling UTF-8 BOM on header."""
    return (row.get("lexicon") or row.get("\ufefflexicon") or "").strip().lower()


def load_polarization_parallel(
    csv_path: Path,
    lang_code: str,
) -> Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]]:
    """Function summary: load categorized polarization lexicons from parallel CSV.

    Parameters:
    - csv_path: polarization_lexicon_parallel.csv path.
    - lang_code: it, en, or de.

    Returns:
    - lexicon_name -> bucket -> (singles, phrases).
    """
    lang = lang_code.lower()
    if lang not in PARALLEL_LANG_COLUMNS:
        return {}
    term_col, _grade_col = PARALLEL_LANG_COLUMNS[lang]
    cache_key = f"{csv_path.resolve()}:{lang}"
    mtime = _file_mtime(csv_path)
    cached = _POLARIZATION_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    out: Dict[str, Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]] = {
        name: {} for name in POLARIZATION_LEXICON_NAMES
    }
    if not csv_path.is_file():
        _POLARIZATION_CACHE[cache_key] = (mtime, out)
        return out
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lex_name = _lexicon_key_field(row)
            if lex_name not in out:
                continue
            bucket = (row.get("bucket") or "").strip().lower()
            if not bucket:
                continue
            cell = row.get(term_col, "") or ""
            pieces = split_lexicon_cell(cell)
            if not pieces:
                continue
            singles_acc, phrases_acc = out[lex_name].get(bucket, ([], []))
            singles_list = list(singles_acc)
            phrases_list = list(phrases_acc)
            s_new, p_new = _variants_to_singles_phrases(pieces, lang)
            singles_list.extend(s_new)
            phrases_list.extend(p_new)
            out[lex_name][bucket] = (singles_list, phrases_list)
    _POLARIZATION_CACHE[cache_key] = (mtime, out)
    return out


def get_polarization_bucket(
    csv_path: Path,
    lang_code: str,
    lexicon_name: str,
    bucket: str,
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Function summary: return singles and phrases for one lexicon bucket.

    Parameters:
    - csv_path: polarization CSV path.
    - lang_code: language code.
    - lexicon_name: e.g. ideology.
    - bucket: category bucket.

    Returns:
    - Tuple (singles, phrases); empty when missing.
    """
    data = load_polarization_parallel(csv_path, lang_code)
    lex = data.get(lexicon_name, {})
    return lex.get(bucket.lower(), ([], []))


def load_style_phrase_parallel(
    csv_path: Path,
    lang_code: str,
    kind: str,
) -> Tuple[str, ...]:
    """Function summary: load lowercased style phrases for one kind and language.

    Parameters:
    - csv_path: style_phrase_parallel.csv path.
    - lang_code: it, en, or de.
    - kind: hedging, signposting, or polite_closer.

    Returns:
    - Tuple of phrase strings with spelling variants expanded for substring match.
    """
    lang = lang_code.lower()
    col = STYLE_LANG_COLUMNS.get(lang)
    if col is None:
        return ()
    cache_key = f"{csv_path.resolve()}:{lang}:{kind.lower()}"
    mtime = _file_mtime(csv_path)
    cached = _STYLE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1].get(kind.lower(), ())
    by_kind: Dict[str, Tuple[str, ...]] = {}
    phrases: List[str] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                lex = (row.get("lexicon") or "").strip().lower()
                if lex != kind.lower():
                    continue
                cell = (row.get(col) or "").strip()
                for piece in split_lexicon_cell(cell) if ";" in cell else ([cell] if cell else []):
                    for variant in expand_lexicon_variants(piece, lang):
                        if variant and variant not in phrases:
                            phrases.append(variant)
    by_kind[kind.lower()] = tuple(phrases)
    _STYLE_CACHE[cache_key] = (mtime, by_kind)
    return by_kind.get(kind.lower(), ())


def load_emotion_cognition_parallel(
    csv_path: Path,
    lang_code: str,
) -> Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]:
    """Function summary: load emotion and cognition lemmas by pole.

    Parameters:
    - csv_path: emotion_cognition_parallel.csv path.
    - lang_code: it, en, or de.

    Returns:
    - pole (emotion|cognition) -> (singles, phrases).
    """
    lang = lang_code.lower()
    col = EMOTION_LANG_COLUMNS.get(lang)
    if col is None:
        return {}
    cache_key = f"{csv_path.resolve()}:{lang}"
    mtime = _file_mtime(csv_path)
    cached = _EMOTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    poles: Dict[str, Tuple[List[str], List[Tuple[str, ...]]]] = {
        "emotion": ([], []),
        "cognition": ([], []),
    }
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pole = (row.get("pole") or "").strip().lower()
                if pole not in poles:
                    continue
                cell = (row.get(col) or "").strip()
                if not cell:
                    continue
                singles_acc, phrases_acc = poles[pole]
                singles_list = list(singles_acc)
                phrases_list = list(phrases_acc)
                s_new, p_new = _variants_to_singles_phrases([cell], lang)
                singles_list.extend(s_new)
                phrases_list.extend(p_new)
                poles[pole] = (singles_list, phrases_list)
    _EMOTION_CACHE[cache_key] = (mtime, poles)
    return poles
