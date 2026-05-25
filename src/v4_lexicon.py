"""
Script summary:
Dominant-side assignment and pair-framing scoring for Italian polarization (v4 CSV pairs).

Functionality:
- Maps v4 use columns (yes/some/rarely/no) to a single L/C/R bucket with tie-break rules.
- Loads framing pairs from italian_political_lexicon_v4.csv (section=pairs).
- Scores strict/all pair-framing nets at comment level.

How to apply/run:
- Imported by src/political_lexicon.score_comment_polarization when lang_code is it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.political_lexicon import (
    political_rate_100w,
    tokenize,
)

USE_SCORE_MAP: Dict[str, int] = {"yes": 3, "some": 2, "rarely": 1, "no": 0}
DominantVariant = str  # dominant_v1 | dominant_drop_ties | dominant_downweight_weak

PAIR_TRACKS = ("strict", "all")
PAIR_COLUMN_SUFFIX = {"strict": "_strict", "all": "_all"}


@dataclass(frozen=True)
class PairEntry:
    """Function summary: one v4 framing pair with resolved poles."""

    pair_id: str
    topic: str
    term_a: str
    term_b: str
    pole_a: str
    pole_b: str
    polarized: str
    tokens_a: Tuple[str, ...]
    tokens_b: Tuple[str, ...]


_PAIRS_CACHE: Dict[str, Tuple[float, List["PairEntry"]]] = {}


def normalize_use(value: str) -> str:
    """Function summary: normalize a v4 use-cell string."""
    return (value or "").strip().lower()


def use_score(value: str) -> int:
    """Function summary: map yes/some/rarely/no to numeric score."""
    return USE_SCORE_MAP.get(normalize_use(value), 0)


def dominant_side_from_uses(
    left_use: str,
    center_use: str,
    right_use: str,
    variant: DominantVariant = "dominant_v1",
) -> Tuple[Optional[str], str]:
    """Function summary: assign one ideology bucket from L/C/R use columns.

    Parameters:
    - left_use, center_use, right_use: v4 use cells.
    - variant: dominant_v1, dominant_drop_ties, or dominant_downweight_weak.

    Returns:
    - Tuple (side or None, tie_rule label).
    """
    scores = {
        "left": use_score(left_use),
        "center": use_score(center_use),
        "right": use_score(right_use),
    }
    max_s = max(scores.values())
    if max_s == 0:
        return None, "all_zero"
    winners = [s for s, v in scores.items() if v == max_s]
    if len(winners) == 1:
        return winners[0], "single_max"
    if variant == "dominant_drop_ties":
        return None, "tie_dropped"
    if set(winners) == {"left", "center"}:
        return "left", "left_center_tie"
    if set(winners) == {"right", "center"}:
        return "right", "right_center_tie"
    if set(winners) == {"left", "right"}:
        return None, "left_right_tie"
    if len(winners) == 3:
        return "center", "triple_tie"
    return None, "ambiguous_tie"


def dominant_side_for_role(row: Mapping[str, str], role: str, variant: DominantVariant = "dominant_v1") -> Tuple[Optional[str], str]:
    """Function summary: dominant side for term_a, term_b, or single term row."""
    if role == "term_a":
        return dominant_side_from_uses(
            row.get("term_a_left_use", ""),
            row.get("term_a_center_use", ""),
            row.get("term_a_right_use", ""),
            variant=variant,
        )
    if role == "term_b":
        return dominant_side_from_uses(
            row.get("term_b_left_use", ""),
            row.get("term_b_center_use", ""),
            row.get("term_b_right_use", ""),
            variant=variant,
        )
    return dominant_side_from_uses(
        row.get("left_use", ""),
        row.get("center_use", ""),
        row.get("right_use", ""),
        variant=variant,
    )


def _term_tokens(term: str) -> Tuple[str, ...]:
    """Function summary: token tuple for phrase matching (Unicode-aware)."""
    return tuple(tokenize(term)) if term else ()


def _match_term(tokens: Sequence[str], term_tokens: Tuple[str, ...]) -> bool:
    """Function summary: return whether term_tokens appears in tokens."""
    if not term_tokens:
        return False
    plen = len(term_tokens)
    if plen == 1:
        return term_tokens[0] in tokens
    n = len(tokens)
    for idx in range(0, n - plen + 1):
        if tuple(tokens[idx : idx + plen]) == term_tokens:
            return True
    return False


def pairs_registry_path(project_root: Path) -> Path:
    """Function summary: resolve italian_political_lexicon_v4.csv path for pairs section."""
    return project_root / "data" / "raw" / "italian_political_lexicon_v4.csv"


def load_pairs_from_v4_csv(csv_path: Path) -> List[PairEntry]:
    """Function summary: load PairEntry list from v4 CSV rows with section=pairs.

    Parameters:
    - csv_path: italian_political_lexicon_v4.csv (semicolon-delimited).

    Returns:
    - List of pair entries with dominant L/C/R poles.
    """
    if not csv_path.is_file():
        return []
    out: List[PairEntry] = []
    pair_idx = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if (row.get("section") or "").strip().lower() != "pairs":
                continue
            ta = (row.get("term_a") or "").strip()
            tb = (row.get("term_b") or "").strip()
            if not ta or not tb:
                continue
            topic = (row.get("topic") or "").strip()
            pa, _ = dominant_side_for_role(row, "term_a", variant="dominant_v1")
            pb, _ = dominant_side_for_role(row, "term_b", variant="dominant_v1")
            out.append(
                PairEntry(
                    pair_id=f"{topic}_{pair_idx}",
                    topic=topic,
                    term_a=ta,
                    term_b=tb,
                    pole_a=pa or "ambiguous",
                    pole_b=pb or "ambiguous",
                    polarized=normalize_use(row.get("polarized", "")),
                    tokens_a=_term_tokens(ta),
                    tokens_b=_term_tokens(tb),
                )
            )
            pair_idx += 1
    return out


def get_pairs_registry(project_root: Path) -> List[PairEntry]:
    """Function summary: return cached pair registry from v4 CSV (reload on mtime change).

    Parameters:
    - project_root: repository root.

    Returns:
    - List of PairEntry objects.
    """
    path = pairs_registry_path(project_root)
    key = str(path.resolve())
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    cached = _PAIRS_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    pairs = load_pairs_from_v4_csv(path)
    _PAIRS_CACHE[key] = (mtime, pairs)
    return pairs


def _pair_pole_score(pole: str) -> Optional[int]:
    """Function summary: map pole to +1 left / -1 right; None if not scorable."""
    if pole == "left":
        return 1
    if pole == "right":
        return -1
    return None


def score_pair_framing(
    text: str,
    pairs: Sequence[PairEntry],
    track: str,
    n_words: int,
    eps: float = 1.0e-6,
    tokens: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Function summary: score one comment for pair-framing track strict or all.

    Parameters:
    - text: comment body.
    - pairs: pair registry entries.
    - track: strict (polarized=yes) or all.
    - n_words: word count.
    - eps: stabilizer for rates.

    Returns:
    - Dict with pair_framing_* columns for the track suffix.
    """
    suffix = PAIR_COLUMN_SUFFIX[track]
    zeros = {
        f"pair_framing_net{suffix}": 0.0,
        f"pair_framing_rate_100w{suffix}": 0.0,
        f"pair_active{suffix}": 0.0,
        f"pair_left_only{suffix}": 0.0,
        f"pair_right_only{suffix}": 0.0,
        f"pair_both{suffix}": 0.0,
    }
    if n_words <= 0:
        return zeros
    tok = list(tokens) if tokens is not None else tokenize(text)
    net = 0
    active = left_only = right_only = both = 0
    for p in pairs:
        if track == "strict" and p.polarized != "yes":
            continue
        hit_a = _match_term(tok, p.tokens_a)
        hit_b = _match_term(tok, p.tokens_b)
        if hit_a and hit_b:
            both += 1
            continue
        if not hit_a and not hit_b:
            continue
        active += 1
        pole_hit = p.pole_a if hit_a else p.pole_b
        ps = _pair_pole_score(pole_hit)
        if ps is None:
            continue
        if ps > 0:
            left_only += 1
            net += 1
        else:
            right_only += 1
            net -= 1
    return {
        f"pair_framing_net{suffix}": float(net),
        f"pair_framing_rate_100w{suffix}": political_rate_100w(active, n_words),
        f"pair_active{suffix}": float(active),
        f"pair_left_only{suffix}": float(left_only),
        f"pair_right_only{suffix}": float(right_only),
        f"pair_both{suffix}": float(both),
    }


def zero_pair_framing_columns() -> Dict[str, float]:
    """Function summary: return zeros for pair-framing columns (strict and all tracks)."""
    out: Dict[str, float] = {}
    for track in PAIR_TRACKS:
        out.update(score_pair_framing("", [], track, 0))
    return out


def all_pair_framing_column_names() -> List[str]:
    """Function summary: list comment-level pair-framing column names."""
    return [k for k in zero_pair_framing_columns().keys()]
