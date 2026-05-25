"""
Script summary:
Dominant-side assignment, pair-framing registry, and v4 metadata scoring for Italian polarization.

Functionality:
- Maps v4 use columns (yes/some/rarely/no) to a single L/C/R bucket with tie-break rules.
- Loads pairs_it.json and scores strict/all pair-framing nets.
- Scores stance, valence, polarized, and relevance-weighted contra rates.

How to apply/run:
- Imported by src/political_lexicon.score_comment_polarization when lang_code is it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.political_lexicon import (
    TOKEN_PATTERN,
    _count_terms_in_tokens,
    compute_ideology_indices,
    count_categorized_hits,
    get_categorized_lexicon,
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
    """Function summary: token tuple for phrase matching."""
    return tuple(normalize_use(term).replace("'", "'").split()) if term else ()


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


def load_pairs_registry(path: Path) -> List[PairEntry]:
    """Function summary: load pairs_it.json into PairEntry list.

    Parameters:
    - path: JSON registry path.

    Returns:
    - List of pair entries.
    """
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", data) if isinstance(data, dict) else data
    out: List[PairEntry] = []
    for i, raw in enumerate(pairs):
        if not isinstance(raw, dict):
            continue
        out.append(
            PairEntry(
                pair_id=str(raw.get("pair_id", f"pair_{i}")),
                topic=str(raw.get("topic", "")),
                term_a=str(raw.get("term_a", "")),
                term_b=str(raw.get("term_b", "")),
                pole_a=str(raw.get("pole_a", "ambiguous")),
                pole_b=str(raw.get("pole_b", "ambiguous")),
                polarized=str(raw.get("polarized", "no")).lower(),
                tokens_a=_term_tokens(str(raw.get("term_a", ""))),
                tokens_b=_term_tokens(str(raw.get("term_b", ""))),
            )
        )
    return out


def pairs_registry_path(project_root: Path) -> Path:
    """Function summary: resolve pairs_it.json path."""
    return project_root / "config" / "lexicons" / "pairs_it.json"


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
    tokens = tokenize(text)
    net = 0
    active = left_only = right_only = both = 0
    for p in pairs:
        if track == "strict" and p.polarized != "yes":
            continue
        hit_a = _match_term(tokens, p.tokens_a)
        hit_b = _match_term(tokens, p.tokens_b)
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


def score_v4_metadata(
    text: str,
    project_root: Path,
    n_words: int,
    eps: float = 1.0e-6,
) -> Dict[str, float]:
    """Function summary: stance, valence, polarized, and relevance-weighted contra for IT.

    Parameters:
    - text: comment body.
    - project_root: repo root.
    - n_words: word count.
    - eps: unused stabilizer placeholder.

    Returns:
    - Metadata hit and rate columns.
    """
    _ = eps
    out: Dict[str, float] = {
        "stance_pro_hits": 0.0,
        "stance_contra_hits": 0.0,
        "stance_ambiguous_hits": 0.0,
        "stance_pro_rate_100w": 0.0,
        "stance_contra_rate_100w": 0.0,
        "stance_ambiguous_rate_100w": 0.0,
        "valence_positive_hits": 0.0,
        "valence_negative_hits": 0.0,
        "valence_neutral_hits": 0.0,
        "valence_ambiguous_hits": 0.0,
        "valence_positive_rate_100w": 0.0,
        "valence_negative_rate_100w": 0.0,
        "valence_neutral_rate_100w": 0.0,
        "valence_ambiguous_rate_100w": 0.0,
        "polarized_yes_hits": 0.0,
        "polarized_yes_rate_100w": 0.0,
        "relevance_weighted_contra_rate_100w": 0.0,
        "left_weight_hits": 0.0,
        "center_weight_hits": 0.0,
        "right_weight_hits": 0.0,
        "net_ideology_weighted": 0.0,
    }
    if n_words <= 0:
        return out
    stance_h, _ = count_categorized_hits(text, "it", "stance", project_root)
    valence_h, _ = count_categorized_hits(text, "it", "valence", project_root)
    pol_h, _ = count_categorized_hits(text, "it", "polarized", project_root)
    pro = int(stance_h.get("pro", 0))
    contra = int(stance_h.get("contra", 0))
    stance_amb = int(stance_h.get("ambiguous", 0))
    pos = int(valence_h.get("positive", 0))
    neg = int(valence_h.get("negative", 0))
    neu = int(valence_h.get("neutral", 0))
    val_amb = int(valence_h.get("ambiguous", 0))
    pol_yes = int(pol_h.get("yes", 0))
    meta_path = project_root / "config" / "lexicons" / "term_meta_it.json"
    rel_contra = 0.0
    if meta_path.is_file():
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = raw_meta.get("terms", raw_meta) if isinstance(raw_meta, dict) else {}
        tokens = tokenize(text)
        for term_key, info in meta.items():
            if not isinstance(info, dict):
                continue
            if info.get("stance") != "contra":
                continue
            tt = tuple(term_key.split())
            if not _match_term(tokens, tt):
                continue
            rel_contra += float(info.get("political_relevance", 1))
    out.update(
        {
            "stance_pro_hits": float(pro),
            "stance_contra_hits": float(contra),
            "stance_ambiguous_hits": float(stance_amb),
            "stance_pro_rate_100w": political_rate_100w(pro, n_words),
            "stance_contra_rate_100w": political_rate_100w(contra, n_words),
            "stance_ambiguous_rate_100w": political_rate_100w(stance_amb, n_words),
            "valence_positive_hits": float(pos),
            "valence_negative_hits": float(neg),
            "valence_neutral_hits": float(neu),
            "valence_ambiguous_hits": float(val_amb),
            "valence_positive_rate_100w": political_rate_100w(pos, n_words),
            "valence_negative_rate_100w": political_rate_100w(neg, n_words),
            "valence_neutral_rate_100w": political_rate_100w(neu, n_words),
            "valence_ambiguous_rate_100w": political_rate_100w(val_amb, n_words),
            "polarized_yes_hits": float(pol_yes),
            "polarized_yes_rate_100w": political_rate_100w(pol_yes, n_words),
            "relevance_weighted_contra_rate_100w": political_rate_100w(int(rel_contra), n_words),
        }
    )
    return out


def score_weighted_ideology_hits(
    text: str,
    project_root: Path,
    variant: DominantVariant = "dominant_downweight_weak",
) -> Dict[str, float]:
    """Function summary: weighted L/C/R hits using per-term use scores (exploratory).

    Parameters:
    - text: comment body.
    - project_root: repo root.
    - variant: downweight assigns min(score,2) to dominant bucket per hit.

    Returns:
    - left_weight_hits, center_weight_hits, right_weight_hits, net_ideology_weighted.
    """
    meta_path = project_root / "config" / "lexicons" / "term_meta_it.json"
    zeros = {
        "left_weight_hits": 0.0,
        "center_weight_hits": 0.0,
        "right_weight_hits": 0.0,
        "net_ideology_weighted": 0.0,
    }
    if not meta_path.is_file():
        return zeros
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = raw_meta.get("terms", raw_meta) if isinstance(raw_meta, dict) else {}
    tokens = tokenize(text)
    if not tokens:
        return zeros
    lw = cw = rw = 0.0
    for term_key, info in meta.items():
        if not isinstance(info, dict):
            continue
        side = info.get("dominant_side")
        if not side:
            continue
        tt = tuple(term_key.split())
        if not _match_term(tokens, tt):
            continue
        scores = info.get("use_scores", {})
        w = float(max(scores.get("left", 0), scores.get("center", 0), scores.get("right", 0)))
        if variant == "dominant_downweight_weak":
            w = float(min(w, 2.0))
        if side == "left":
            lw += w
        elif side == "center":
            cw += w
        elif side == "right":
            rw += w
    lr = lw + rw
    net_w = (lw - rw) / (lr + 1.0e-6) if lr > 0 else 0.0
    return {
        "left_weight_hits": lw,
        "center_weight_hits": cw,
        "right_weight_hits": rw,
        "net_ideology_weighted": float(net_w),
    }


def zero_v4_polarization_columns() -> Dict[str, float]:
    """Function summary: return zeros for all IT-only v4 extension columns."""
    out: Dict[str, float] = {}
    for track in PAIR_TRACKS:
        out.update(score_pair_framing("", [], track, 0))
    out.update(score_v4_metadata("", Path("."), 0))
    out.update(score_weighted_ideology_hits("", Path(".")))
    return out


def all_v4_column_names() -> List[str]:
    """Function summary: list comment-level v4 column names (no thread_)."""
    return [k for k in zero_v4_polarization_columns().keys()]
