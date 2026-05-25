"""
Script summary:
fastText-based semantic axis scoring (ideology, emotion, aggression) for enriched comment shards.

Functionality:
- Load language-specific Common Crawl vectors with process-level caching.
- Build unit axes from parallel seed CSVs (ideology, emotion, aggression) and civil neutral lists.
- Aggression axis: positive pole = insult/incivility (higher sem_axis_aggression = more aggressive).
- Mean-token comment vectors, cosine axis scores, per-shard NPZ caches, seed OOV and held-out sanity reports.

How to apply/run:
- Imported by `_enriched_shard_runner` semaxis pass and diagnostics; not run standalone.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.political_lexicon import tokenize

logger = logging.getLogger(__name__)

_LANG_COL = {"it": "IT", "en": "EN", "de": "DE"}
_VECTOR_CACHE: Dict[str, Any] = {}
_AXIS_CACHE: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
_LOAD_LOGGED: set[str] = set()

SEMAXIS_SCORE_KEYS = (
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "has_sem_axis",
)

# Civil/neutral pole for aggression axis (lowercase; match tokenize).
NEUTRAL_POLE_TERMS: Dict[str, Tuple[str, ...]] = {
    "it": (
        "grazie",
        "per favore",
        "gentile",
        "rispetto",
        "cordiale",
        "civile",
        "educato",
        "cortese",
        "dialogo",
        "collaborazione",
        "costruttivo",
        "pacato",
        "ragionevole",
        "composto",
        "rispettoso",
    ),
    "en": (
        "thanks",
        "please",
        "polite",
        "respect",
        "cordial",
        "civil",
        "courteous",
        "dialogue",
        "constructive",
        "calm",
        "reasonable",
        "respectful",
        "civility",
        "courtesy",
        "collaboration",
    ),
    "de": (
        "danke",
        "bitte",
        "höflich",
        "respekt",
        "höflichkeit",
        "zivil",
        "respektvoll",
        "dialog",
        "konstruktiv",
        "ruhig",
        "vernünftig",
        "anstand",
        "freundlich",
        "sachlich",
        "gemeinsam",
    ),
}


def _zero_scores() -> Dict[str, float]:
    """Function summary: empty-row semantic axis scores (mirror polarization)."""
    return {k: 0.0 for k in SEMAXIS_SCORE_KEYS}


def resolve_vector_path(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
) -> Path:
    """Function summary: resolve fastText model path for a language code.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config block.

    Returns:
    - Absolute Path to .bin or .vec model file.

    Raises:
    - FileNotFoundError: if configured file is missing.
    """
    lang = lang_code.lower()
    paths = axes_cfg.get("vector_paths") or {}
    raw = paths.get(lang) or paths.get(lang.upper())
    if not raw:
        raise KeyError(f"semantic_axis.vector_paths missing entry for {lang!r}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise FileNotFoundError(
            f"fastText model not found for {lang!r}: {path}\n"
            "Download cc.{lang}.300.bin.gz from https://fasttext.cc/docs/en/crawl-vectors.html "
            f"and gunzip to {path.parent}/cc.{lang}.300.bin"
        )
    return path


def load_vectors(lang_code: str, project_root: Path, axes_cfg: Mapping[str, Any]) -> Any:
    """Function summary: load and cache KeyedVectors / FastText for one language.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config block.

    Returns:
    - gensim vectors object (FastTextKeyedVectors for .bin).
    """
    lang = lang_code.lower()
    if lang in _VECTOR_CACHE:
        return _VECTOR_CACHE[lang]
    path = resolve_vector_path(lang, project_root, axes_cfg)
    suffix = path.suffix.lower()
    if suffix == ".bin":
        from gensim.models.fasttext import load_facebook_vectors

        kv = load_facebook_vectors(str(path))
    elif suffix == ".vec":
        from gensim.models import KeyedVectors

        kv = KeyedVectors.load_word2vec_format(str(path))
    else:
        raise ValueError(f"Unsupported embedding file type {suffix!r} at {path}")
    _VECTOR_CACHE[lang] = kv
    if lang not in _LOAD_LOGGED:
        logger.info("Loaded fastText vectors lang=%s path=%s", lang, path)
        _LOAD_LOGGED.add(lang)
    return kv


def _token_to_vector(token: str, kv: Any) -> Optional[np.ndarray]:
    """Function summary: lookup one token vector (in-vocab or fastText OOV)."""
    if not token:
        return None
    if token in kv:
        return np.asarray(kv[token], dtype=np.float64)
    if hasattr(kv, "get_vector"):
        try:
            return np.asarray(kv.get_vector(token), dtype=np.float64)
        except KeyError:
            return None
    return None


def _mean_seed_vector(term: str, kv: Any) -> Optional[np.ndarray]:
    """Function summary: average token vectors for a seed (multi-word/hyphenated)."""
    parts = tokenize(term)
    if not parts:
        return None
    vecs = [_token_to_vector(p, kv) for p in parts]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


def build_axis(pole_pos_terms: Sequence[str], pole_neg_terms: Sequence[str], kv: Any) -> np.ndarray:
    """Function summary: unit axis = normalize(mean(pos) - mean(neg)); log OOV seeds.

    Parameters:
    - pole_pos_terms: positive pole seed strings.
    - pole_neg_terms: negative pole seed strings.
    - kv: loaded vectors.

    Returns:
    - Unit numpy vector (dim,) or zero vector if both poles empty.
    """
    pos_vecs: List[np.ndarray] = []
    neg_vecs: List[np.ndarray] = []
    for term in pole_pos_terms:
        v = _mean_seed_vector(term, kv)
        if v is not None:
            pos_vecs.append(v)
        else:
            logger.debug("OOV seed (pos pole): %r", term)
    for term in pole_neg_terms:
        v = _mean_seed_vector(term, kv)
        if v is not None:
            neg_vecs.append(v)
        else:
            logger.debug("OOV seed (neg pole): %r", term)
    if not pos_vecs and not neg_vecs:
        dim = int(getattr(kv, "vector_size", 300))
        return np.zeros(dim, dtype=np.float64)
    pos_mean = np.mean(np.stack(pos_vecs, axis=0), axis=0) if pos_vecs else np.zeros_like(neg_vecs[0])
    neg_mean = np.mean(np.stack(neg_vecs, axis=0), axis=0) if neg_vecs else np.zeros_like(pos_vecs[0])
    diff = pos_mean - neg_mean
    norm = float(np.linalg.norm(diff))
    if norm < 1e-12:
        return diff
    return diff / norm


def comment_vector(tokens: Sequence[str], kv: Any) -> Tuple[Optional[np.ndarray], float]:
    """Function summary: mean in-vocab token vectors and coverage share.

    Parameters:
    - tokens: token list from tokenize().
    - kv: loaded vectors.

    Returns:
    - (mean vector or None if no in-vocab tokens, coverage in [0,1]).
    """
    if not tokens:
        return None, 0.0
    vecs: List[np.ndarray] = []
    for tok in tokens:
        v = _token_to_vector(tok, kv)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None, 0.0
    coverage = len(vecs) / len(tokens)
    return np.mean(np.stack(vecs, axis=0), axis=0), float(coverage)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Function summary: cosine similarity between two vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _read_parallel_pole_terms(csv_path: Path, lang_code: str, pole: str) -> List[str]:
    """Function summary: extract seed terms for one pole from ideology/emotion parallel CSV."""
    col = _LANG_COL.get(lang_code.lower())
    if not col:
        return []
    terms: List[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("pole") or "").strip().lower() != pole.lower():
                continue
            cell = (row.get(col) or "").strip()
            if not cell:
                continue
            for part in re.split(r"[;,]", cell):
                part = part.strip().lower()
                if part:
                    terms.append(part)
    return terms


def _read_aggression_terms(project_root: Path, lang_code: str) -> List[str]:
    """Function summary: load aggression lexicon terms from archive categorized txt."""
    lang = lang_code.lower()
    path = (
        project_root
        / "config"
        / "archive"
        / "lexicons"
        / "categorized"
        / f"aggression_{lang}.txt"
    )
    if not path.is_file():
        return []
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            _, term = line.split(":", 1)
            terms.append(term.strip().lower())
    return terms


def _read_pole_txt(path: Path) -> List[str]:
    """Function summary: one term per line pole list."""
    if not path.is_file():
        return []
    return [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def resolve_aggression_parallel_path(
    project_root: Path,
    axes_cfg: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Function summary: resolve aggression_parallel.csv from config paths or seeds_dir default."""
    if config is not None:
        from src.config_utils import aggression_parallel_path

        return aggression_parallel_path(dict(config), project_root)
    seeds_dir = Path(str(axes_cfg.get("seeds_dir", "data/raw/seeds")))
    if not seeds_dir.is_absolute():
        seeds_dir = project_root / seeds_dir
    return seeds_dir / "aggression_parallel.csv"


def load_seed_poles(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Dict[str, List[str]]:
    """Function summary: resolve ideology/emotion/aggression pole term lists for a language.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config (seeds_dir, optional pole txt overrides).

    Returns:
    - Dict with keys ideology_pos, ideology_neg, emotion_pos, emotion_neg, aggression_pos, aggression_neg.
    """
    lang = lang_code.lower()
    seeds_dir = Path(str(axes_cfg.get("seeds_dir", "data/raw/seeds")))
    if not seeds_dir.is_absolute():
        seeds_dir = project_root / seeds_dir
    poles_dir = seeds_dir / "poles"

    ideology_csv = seeds_dir / "ideology_parallel.csv"
    emotion_csv = seeds_dir / "emotion_cognition_parallel.csv"
    aggression_csv = resolve_aggression_parallel_path(project_root, axes_cfg, config)

    ideology_pos = _read_pole_txt(poles_dir / f"ideology_pos_{lang}.txt")
    ideology_neg = _read_pole_txt(poles_dir / f"ideology_neg_{lang}.txt")
    emotion_pos = _read_pole_txt(poles_dir / f"emotion_pos_{lang}.txt")
    emotion_neg = _read_pole_txt(poles_dir / f"emotion_neg_{lang}.txt")
    aggression_pos = _read_pole_txt(poles_dir / f"aggression_pos_{lang}.txt")
    aggression_neg = _read_pole_txt(poles_dir / f"aggression_neg_{lang}.txt")

    if not ideology_pos and ideology_csv.is_file():
        ideology_pos = _read_parallel_pole_terms(ideology_csv, lang, "right")
    if not ideology_neg and ideology_csv.is_file():
        ideology_neg = _read_parallel_pole_terms(ideology_csv, lang, "left")
    if not emotion_pos and emotion_csv.is_file():
        emotion_pos = _read_parallel_pole_terms(emotion_csv, lang, "emotion")
    if not emotion_neg and emotion_csv.is_file():
        emotion_neg = _read_parallel_pole_terms(emotion_csv, lang, "cognition")
    if not aggression_pos and aggression_csv.is_file():
        aggression_pos = _read_parallel_pole_terms(aggression_csv, lang, "aggression")
    if not aggression_pos:
        aggression_pos = _read_aggression_terms(project_root, lang)
    if not aggression_neg:
        aggression_neg = list(NEUTRAL_POLE_TERMS.get(lang, NEUTRAL_POLE_TERMS["en"]))

    return {
        "ideology_pos": ideology_pos,
        "ideology_neg": ideology_neg,
        "emotion_pos": emotion_pos,
        "emotion_neg": emotion_neg,
        "aggression_pos": aggression_pos,
        "aggression_neg": aggression_neg,
    }


def get_axes_for_language(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Function summary: cached unit axes for ideology, emotion, aggression.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config.

    Returns:
    - Dict mapping axis name to unit vector.
    """
    lang = lang_code.lower()
    cache_key = (lang, str(axes_cfg.get("seeds_dir", "")))
    if cache_key in _AXIS_CACHE:
        return _AXIS_CACHE[cache_key]
    kv = load_vectors(lang, project_root, axes_cfg)
    poles = load_seed_poles(lang, project_root, axes_cfg)
    axes = {
        "ideology": build_axis(poles["ideology_pos"], poles["ideology_neg"], kv),
        "emotion": build_axis(poles["emotion_pos"], poles["emotion_neg"], kv),
        "aggression": build_axis(poles["aggression_pos"], poles["aggression_neg"], kv),
    }
    _AXIS_CACHE[cache_key] = axes
    return axes


def score_vectors_against_axes(
    comment_vecs: Sequence[Optional[np.ndarray]],
    coverages: Sequence[float],
    axes: Mapping[str, np.ndarray],
) -> List[Dict[str, float]]:
    """Function summary: cosine scores for precomputed comment vectors.

    Parameters:
    - comment_vecs: per-row mean vectors (None if unscored).
    - coverages: per-row in-vocab token shares.
    - axes: ideology/emotion/aggression unit vectors.

    Returns:
    - List of score dicts aligned with comment_vecs.
    """
    out: List[Dict[str, float]] = []
    for vec, cov in zip(comment_vecs, coverages, strict=True):
        row = _zero_scores()
        row["sem_axis_coverage"] = float(cov)
        if vec is None:
            out.append(row)
            continue
        row["sem_axis_ideology"] = _cosine(vec, axes["ideology"])
        row["sem_axis_emotion"] = _cosine(vec, axes["emotion"])
        row["sem_axis_aggression"] = _cosine(vec, axes["aggression"])
        row["has_sem_axis"] = 1.0
        out.append(row)
    return out


def score_comment_semantic_axis(
    text: str,
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    """Function summary: semantic axis cosines for one comment body.

    Parameters:
    - text: comment body.
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config block.

    Returns:
    - Dict with SEMAXIS_SCORE_KEYS; zeros and has_sem_axis=0 when empty.
    """
    tokens = tokenize(text)
    if not tokens:
        return _zero_scores()
    kv = load_vectors(lang_code, project_root, axes_cfg)
    vec, cov = comment_vector(tokens, kv)
    axes = get_axes_for_language(lang_code, project_root, axes_cfg)
    return score_vectors_against_axes([vec], [cov], axes)[0]


def shard_embedding_cache_path(
    interim_dir: Path,
    subreddit: str,
    shard_stem: str,
) -> Path:
    """Function summary: NPZ path for cached comment vectors on one monthly shard."""
    return interim_dir / "embeddings" / subreddit / f"{shard_stem}.npz"


def load_shard_vector_cache(
    cache_path: Path,
    comment_ids: Sequence[str],
) -> Tuple[Optional[List[Optional[np.ndarray]]], Optional[List[float]]]:
    """Function summary: load cached vectors if ids match the shard frame.

    Parameters:
    - cache_path: .npz with arrays ids, vectors, coverages.
    - comment_ids: current shard id column in order.

    Returns:
    - (vectors list, coverages list) or (None, None) if cache miss/invalid.
    """
    if not cache_path.is_file():
        return None, None
    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_ids = [str(x) for x in data["ids"].tolist()]
        if cached_ids != [str(x) for x in comment_ids]:
            return None, None
        raw = data["vectors"]
        cov = data["coverages"].astype(float).tolist()
        vecs: List[Optional[np.ndarray]] = []
        for row in raw:
            if row is None or (isinstance(row, float) and np.isnan(row)):
                vecs.append(None)
            else:
                vecs.append(np.asarray(row, dtype=np.float64))
        return vecs, cov
    except Exception:
        return None, None


def save_shard_vector_cache(
    cache_path: Path,
    comment_ids: Sequence[str],
    comment_vecs: Sequence[Optional[np.ndarray]],
    coverages: Sequence[float],
) -> None:
    """Function summary: write id-aligned comment vectors to NPZ cache.

    Parameters:
    - cache_path: output .npz path.
    - comment_ids: comment id strings.
    - comment_vecs: mean vectors or None per row.
    - coverages: in-vocab shares per row.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    obj = np.empty(len(comment_vecs), dtype=object)
    for i, v in enumerate(comment_vecs):
        obj[i] = None if v is None else np.asarray(v, dtype=np.float32)
    np.savez_compressed(
        cache_path,
        ids=np.asarray(list(comment_ids), dtype=object),
        vectors=obj,
        coverages=np.asarray(list(coverages), dtype=np.float32),
    )


def build_comment_vectors_for_texts(
    bodies: Sequence[str],
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """Function summary: batch mean vectors and coverage for many comment bodies.

    Parameters:
    - bodies: comment text strings.
    - lang_code: lexicon language.
    - project_root: repo root.
    - axes_cfg: semantic_axis config.

    Returns:
    - (vectors, coverages) lists aligned with bodies.
    """
    kv = load_vectors(lang_code, project_root, axes_cfg)
    vecs: List[Optional[np.ndarray]] = []
    covs: List[float] = []
    for body in bodies:
        vec, cov = comment_vector(tokenize(body), kv)
        vecs.append(vec)
        covs.append(cov)
    return vecs, covs


# Held-out tokens for axis direction checks (not used as seeds). expected_sign: +1 or -1 on that axis.
HELDOUT_AXIS_CHECKS: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "it": {
        "ideology": [("liberismo", 1), ("redistribuzione", -1)],
        "emotion": [("amore", 1), ("logica", -1)],
        "aggression": [("idiota", 1), ("grazie", -1)],
    },
    "en": {
        "ideology": [("tax-cuts", 1), ("redistribution", -1)],
        "emotion": [("love", 1), ("logic", -1)],
        "aggression": [("idiot", 1), ("thanks", -1)],
    },
    "de": {
        "ideology": [("steuersenkungen", 1), ("umverteilung", -1)],
        "emotion": [("liebe", 1), ("logik", -1)],
        "aggression": [("idiot", 1), ("danke", -1)],
    },
}


def _seed_term_in_vocab(term: str, kv: Any) -> bool:
    """Function summary: True if all tokens of a seed string have vectors in kv."""
    parts = tokenize(term)
    if not parts:
        return False
    return all(_token_to_vector(p, kv) is not None for p in parts)


def seed_coverage_report(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Function summary: per-axis pole seed in-vocab coverage for one language.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config block.
    - config: optional full study YAML for aggression_parallel path.

    Returns:
    - List of dicts suitable for semantic_axis_seed_coverage.csv rows.
    """
    kv = load_vectors(lang_code, project_root, axes_cfg)
    poles = load_seed_poles(lang_code, project_root, axes_cfg, config=config)
    rows: List[Dict[str, Any]] = []
    pole_groups = (
        ("ideology", "pos", poles["ideology_pos"]),
        ("ideology", "neg", poles["ideology_neg"]),
        ("emotion", "pos", poles["emotion_pos"]),
        ("emotion", "neg", poles["emotion_neg"]),
        ("aggression", "pos", poles["aggression_pos"]),
        ("aggression", "neg", poles["aggression_neg"]),
    )
    for axis, pole, terms in pole_groups:
        oov: List[str] = []
        in_vocab = 0
        for term in terms:
            if _seed_term_in_vocab(term, kv):
                in_vocab += 1
            else:
                oov.append(term)
        n = len(terms)
        rows.append(
            {
                "lang": lang_code.lower(),
                "axis": axis,
                "pole": pole,
                "n_seeds": n,
                "n_in_vocab": in_vocab,
                "share_in_vocab": (in_vocab / n) if n else 0.0,
                "oov_terms": ";".join(oov),
            }
        )
    return rows


def seed_oov_summary_by_lang(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    """Function summary: mean OOV share across poles for ideology, emotion, aggression."""
    rows = seed_coverage_report(lang_code, project_root, axes_cfg, config=config)
    out: Dict[str, float] = {}
    for axis in ("ideology", "emotion", "aggression"):
        sub = [r for r in rows if r["axis"] == axis]
        if sub:
            out[f"seed_oov_share_{axis}"] = float(
                np.mean([1.0 - float(r["share_in_vocab"]) for r in sub])
            )
        else:
            out[f"seed_oov_share_{axis}"] = float("nan")
    return out


def held_out_axis_sanity_report(
    lang_code: str,
    project_root: Path,
    axes_cfg: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Function summary: cosine signs for held-out words against each semantic axis.

    Parameters:
    - lang_code: it, en, or de.
    - project_root: repository root.
    - axes_cfg: semantic_axis config.
    - config: optional full study YAML.

    Returns:
    - Rows with lang, axis, token, cosine, expected_sign, pass.
    """
    lang = lang_code.lower()
    checks = HELDOUT_AXIS_CHECKS.get(lang, {})
    if not checks:
        return []
    kv = load_vectors(lang, project_root, axes_cfg)
    axis_vecs = get_axes_for_language(lang, project_root, axes_cfg)
    rows: List[Dict[str, Any]] = []
    for axis_name, tokens in checks.items():
        axis_vec = axis_vecs.get(axis_name)
        if axis_vec is None:
            continue
        for token, expected_sign in tokens:
            vec, _ = comment_vector(tokenize(token), kv)
            cosine = _cosine(vec, axis_vec) if vec is not None else float("nan")
            passed = (
                bool(vec is not None)
                and not np.isnan(cosine)
                and (cosine * float(expected_sign)) > 0
            )
            rows.append(
                {
                    "lang": lang,
                    "axis": axis_name,
                    "token": token,
                    "cosine": cosine,
                    "expected_sign": int(expected_sign),
                    "pass": int(passed),
                }
            )
    return rows


def clear_embedding_caches() -> None:
    """Function summary: reset in-process vector and axis caches (for tests)."""
    _VECTOR_CACHE.clear()
    _AXIS_CACHE.clear()
    _LOAD_LOGGED.clear()
