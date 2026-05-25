"""
Script summary:
Language-agnostic Reddit comment style counters (punctuation, markdown, complexity)
and optional per-language phrase lexicon hits for hedging, signposting, and polite closers.

Functionality:
- Shared by Italy in-place feature pass (`compute_comment_style_features.py`) and archived ML merge path.
- No ML detectors or English AI-word lists.

How to apply/run:
- Imported by scripts/features/compute_comment_style_features.py and archive merge tooling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
MARKDOWN_BOLD_PAIR_RE = re.compile(r"\*\*.+?\*\*", re.DOTALL)
MARKDOWN_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)
URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+(?:/[^\s]*)?"
)
TIME_EXPRESSION_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[ap]m)?\b", re.IGNORECASE)
STRAIGHT_QUOTE_RE = re.compile(r"[\"']")

PHRASE_LEXICON_KINDS = ("hedging", "signposting", "polite_closer")

STYLE_COUNT_COLUMNS: Tuple[str, ...] = (
    "exclamation_count",
    "semicolon_count",
    "em_dash_count",
    "em_dash_extended_count",
    "en_dash_count",
    "ascii_double_hyphen_count",
    "colon_count",
    "colon_extended_count",
    "open_paren_count",
    "curly_quote_count",
    "straight_quote_count",
    "quote_all_count",
    "quote_curly_share_num",
    "quote_curly_share_den",
    "url_count",
    "time_expression_count",
    "markdown_bold_pair_count",
    "markdown_heading_line_count",
    "sentence_count_comment",
    "total_word_chars_comment",
    "avg_words_per_sentence_comment",
    "hedging_phrase_hits",
    "signposting_phrase_hits",
    "polite_closer_hits",
)

_PHRASE_CACHE: Dict[Tuple[str, str], Tuple[str, ...]] = {}


def tokenize_words(text: str) -> List[str]:
    """Function summary: tokenize text into lowercase word-like tokens.

    Parameters:
    - text: raw comment body.

    Returns:
    - List of lowercase word tokens.
    """
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def count_phrase_occurrences(text_lc: str, phrase: str) -> int:
    """Function summary: count non-overlapping substring matches for one phrase.

    Parameters:
    - text_lc: lowercased text.
    - phrase: phrase to search.

    Returns:
    - Hit count.
    """
    if not phrase:
        return 0
    return int(text_lc.count(phrase))


def sum_phrase_hits(text_lc: str, phrases: Sequence[str]) -> int:
    """Function summary: sum phrase hit counts over a phrase list.

    Parameters:
    - text_lc: lowercased text.
    - phrases: iterable of phrases.

    Returns:
    - Total hits.
    """
    return int(sum(count_phrase_occurrences(text_lc, p) for p in phrases))


def count_ascii_double_hyphen(text: str) -> int:
    """Function summary: count spaced ASCII double-hyphen ` -- ` spans.

    Parameters:
    - text: raw body.

    Returns:
    - Count.
    """
    return int((text or "").count(" -- "))


def count_curly_quotes(text: str) -> int:
    """Function summary: count Unicode curly quote characters.

    Parameters:
    - text: raw body.

    Returns:
    - Count.
    """
    s = text or ""
    return int(sum(s.count(ch) for ch in "\u201c\u201d\u2018\u2019"))


def count_straight_quotes(text: str) -> int:
    """Function summary: count straight ASCII quote characters.

    Parameters:
    - text: raw body.

    Returns:
    - Count.
    """
    return int(len(STRAIGHT_QUOTE_RE.findall(text or "")))


def count_em_dash_extended(text: str) -> int:
    """Function summary: count em/en dashes plus spaced ASCII hyphen variants.

    Parameters:
    - text: raw body.

    Returns:
    - Extended dash count.
    """
    s = text or ""
    unicode_total = int(s.count("\u2014") + s.count("\u2013"))
    ascii_total = int(s.count(" -- ") + s.count(" --- "))
    return int(unicode_total + ascii_total)


def colon_cleaned_text(text: str) -> str:
    """Function summary: strip URL spans and clock times before colon counting.

    Parameters:
    - text: raw body.

    Returns:
    - Cleaned text.
    """
    s = text or ""
    cleaned = URL_RE.sub(" ", s)
    return TIME_EXPRESSION_RE.sub(" ", cleaned)


def count_colon_strict(text: str) -> int:
    """Function summary: count ASCII colons on URL/time-cleaned text.

    Parameters:
    - text: raw body.

    Returns:
    - Strict colon count.
    """
    return int(colon_cleaned_text(text).count(":"))


def count_colon_extended(text: str) -> int:
    """Function summary: count ASCII and fullwidth colons on cleaned text.

    Parameters:
    - text: raw body.

    Returns:
    - Extended colon count (superset of strict).
    """
    cleaned = colon_cleaned_text(text)
    return int(cleaned.count(":") + cleaned.count("\uff1a"))


def count_urls(text: str) -> int:
    """Function summary: count URL-like spans.

    Parameters:
    - text: raw body.

    Returns:
    - URL count.
    """
    return int(len(URL_RE.findall(text or "")))


def count_time_expressions(text: str) -> int:
    """Function summary: count clock-like time expressions.

    Parameters:
    - text: raw body.

    Returns:
    - Time expression count.
    """
    return int(len(TIME_EXPRESSION_RE.findall(text or "")))


def count_markdown_bold_pairs(text: str) -> int:
    """Function summary: count markdown bold **...** pairs.

    Parameters:
    - text: raw body.

    Returns:
    - Bold pair count.
    """
    return int(len(MARKDOWN_BOLD_PAIR_RE.findall(text or "")))


def count_markdown_heading_lines(text: str) -> int:
    """Function summary: count ATX markdown heading lines.

    Parameters:
    - text: raw body.

    Returns:
    - Heading line count.
    """
    return int(len(MARKDOWN_HEADING_LINE_RE.findall(text or "")))


def avg_words_per_sentence(n_words: int, sentence_count: int) -> float:
    """Function summary: words per sentence when both counts are positive.

    Parameters:
    - n_words: word count.
    - sentence_count: sentence count.

    Returns:
    - Mean words per sentence or NaN.
    """
    if n_words <= 0 or sentence_count <= 0:
        return float("nan")
    return float(n_words) / float(sentence_count)


def compute_complexity_index(
    total_sentences: int, total_words: int, total_word_chars: int, n_comments: int
) -> float:
    """Function summary: lexical/syntactic complexity proxy from daily aggregates.

    Parameters:
    - total_sentences: summed sentence_count_comment.
    - total_words: summed word count.
    - total_word_chars: summed character length in words.
    - n_comments: comment count.

    Returns:
    - Complexity index (0 when empty).
    """
    if n_comments <= 0 or total_words <= 0:
        return 0.0
    mean_sentence_length = float(total_words) / float(max(total_sentences, 1))
    mean_word_length = float(total_word_chars) / float(total_words)
    return 0.5 * mean_sentence_length + 0.5 * mean_word_length


def default_style_phrase_parallel_path(project_root: Path) -> Path:
    """Function summary: resolve default style_phrase_parallel.csv path."""
    return project_root / "data" / "raw" / "style_phrase_parallel.csv"


def load_phrase_lexicon(project_root: Path, lang_code: str, kind: str) -> Tuple[str, ...]:
    """Function summary: load lowercased phrase strings from style_phrase_parallel.csv.

    Parameters:
    - project_root: repository root.
    - lang_code: language code.
    - kind: phrase lexicon kind (hedging, signposting, polite_closer).

    Returns:
    - Tuple of phrases (empty when file missing).
    """
    from src.parallel_lexicon import load_style_phrase_parallel

    key = (lang_code.lower(), kind)
    if key in _PHRASE_CACHE:
        return _PHRASE_CACHE[key]
    path = default_style_phrase_parallel_path(project_root)
    out = load_style_phrase_parallel(path, lang_code, kind)
    _PHRASE_CACHE[key] = out
    return out


def score_comment_style(
    body: str,
    lex_lang: str,
    project_root: Path,
    *,
    enable_phrase_lexicons: bool = True,
    lang_match_filter: bool = False,
    lang_comment: Optional[str] = None,
) -> Dict[str, int | float]:
    """Function summary: compute unprefixed style count columns for one comment.

    Parameters:
    - body: comment text.
    - lex_lang: primary_lexicon language for phrase lists.
    - project_root: repo root for lexicon paths.
    - enable_phrase_lexicons: when False, phrase hit columns are zero.
    - lang_match_filter: zero phrase hits when lang_comment != lex_lang.
    - lang_comment: detected comment language.

    Returns:
    - Dict keyed by STYLE_COUNT_COLUMNS.
    """
    text = body or ""
    text_lc = text.lower()
    words = tokenize_words(text)
    n_words = len(words)
    sentence_count = sum(1 for _ in SENTENCE_SPLIT_RE.finditer(text))
    sentence_count = max(sentence_count, 1 if n_words > 0 else 0)
    total_word_chars = int(sum(len(w) for w in words))
    em_dash_count = int(text.count("\u2014"))
    curly_quote_count = count_curly_quotes(text)
    straight_quote_count = count_straight_quotes(text)
    quote_all_count = int(curly_quote_count + straight_quote_count)

    phrase_ok = enable_phrase_lexicons and (
        not lang_match_filter or (lang_comment or lex_lang) == lex_lang
    )
    if phrase_ok:
        hedging = sum_phrase_hits(text_lc, load_phrase_lexicon(project_root, lex_lang, "hedging"))
        signpost = sum_phrase_hits(text_lc, load_phrase_lexicon(project_root, lex_lang, "signposting"))
        polite = sum_phrase_hits(text_lc, load_phrase_lexicon(project_root, lex_lang, "polite_closer"))
    else:
        hedging = signpost = polite = 0

    return {
        "exclamation_count": int(text.count("!")),
        "semicolon_count": int(text.count(";")),
        "em_dash_count": em_dash_count,
        "em_dash_extended_count": int(count_em_dash_extended(text)),
        "en_dash_count": int(text.count("\u2013")),
        "ascii_double_hyphen_count": int(count_ascii_double_hyphen(text)),
        "colon_count": int(count_colon_strict(text)),
        "colon_extended_count": int(count_colon_extended(text)),
        "open_paren_count": int(text.count("(")),
        "curly_quote_count": curly_quote_count,
        "straight_quote_count": straight_quote_count,
        "quote_all_count": quote_all_count,
        "quote_curly_share_num": int(curly_quote_count),
        "quote_curly_share_den": int(quote_all_count),
        "url_count": int(count_urls(text)),
        "time_expression_count": int(count_time_expressions(text)),
        "markdown_bold_pair_count": int(count_markdown_bold_pairs(text)),
        "markdown_heading_line_count": int(count_markdown_heading_lines(text)),
        "sentence_count_comment": int(sentence_count),
        "total_word_chars_comment": total_word_chars,
        "avg_words_per_sentence_comment": float(avg_words_per_sentence(n_words, sentence_count)),
        "hedging_phrase_hits": int(hedging),
        "signposting_phrase_hits": int(signpost),
        "polite_closer_hits": int(polite),
    }
