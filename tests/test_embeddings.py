"""Unit tests for semantic-axis embedding helpers (mock vectors, no fastText download)."""

from __future__ import annotations

import numpy as np
from gensim.models import KeyedVectors

import src.embeddings as emb
from src.embeddings import (
    SEMAXIS_SCORE_KEYS,
    _AXIS_CACHE,
    _VECTOR_CACHE,
    build_axis,
    clear_embedding_caches,
    comment_vector,
    load_seed_poles,
    score_comment_semantic_axis,
    score_vectors_against_axes,
    unload_embeddings_for_language,
)


class _MockKV(KeyedVectors):
    """Minimal KeyedVectors with fixed 3d geometry for tests."""

    def __init__(self) -> None:
        super().__init__(vector_size=3)
        self.add_vector("left", np.array([1.0, 0.0, 0.0]))
        self.add_vector("right", np.array([-1.0, 0.0, 0.0]))
        self.add_vector("love", np.array([0.0, 1.0, 0.0]))
        self.add_vector("logic", np.array([0.0, -1.0, 0.0]))
        self.add_vector("idiot", np.array([0.0, 0.0, 1.0]))
        self.add_vector("thanks", np.array([0.0, 0.0, -1.0]))


def test_build_axis_direction() -> None:
    """Right-left axis points from left cluster toward right cluster."""
    kv = _MockKV()
    axis = build_axis(["right"], ["left"], kv)
    assert axis[0] < 0
    assert abs(axis[1]) < 0.01


def test_comment_vector_and_scores() -> None:
    """Comment toward right pole scores positive on ideology axis."""
    kv = _MockKV()
    axis = build_axis(["right"], ["left"], kv)
    vec, cov = comment_vector(["right", "wing"], kv)
    assert vec is not None
    assert cov > 0
    scores = score_vectors_against_axes(
        [vec],
        [cov],
        {"ideology": axis, "emotion": axis, "aggression": axis},
    )
    assert scores[0]["has_sem_axis"] == 1.0
    assert scores[0]["sem_axis_ideology"] > 0


def test_empty_text_zeros() -> None:
    """Empty body returns zero scores and has_sem_axis=0."""
    clear_embedding_caches()
    out = score_comment_semantic_axis("", "it", __import__("pathlib").Path("."), {})
    assert out["has_sem_axis"] == 0.0
    assert set(out.keys()) == set(SEMAXIS_SCORE_KEYS)


def test_aggression_parallel_has_25_terms() -> None:
    """aggression_parallel.csv yields exactly 25 terms per language."""
    import csv
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    path = root / "data/raw/seeds/aggression_parallel.csv"
    for col in ("IT", "EN", "DE"):
        n = 0
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("pole") or "").lower() != "aggression":
                    continue
                cell = (row.get(col) or "").strip()
                for part in re.split(r"[;,]", cell):
                    if part.strip():
                        n += 1
        assert n == 25, col


def test_unload_embeddings_for_language() -> None:
    """Unload removes one language from vector and axis caches."""
    clear_embedding_caches()
    kv = _MockKV()
    _VECTOR_CACHE["it"] = kv
    _VECTOR_CACHE["en"] = kv
    _AXIS_CACHE[("it", "seeds")] = {"ideology": np.zeros(3)}
    _AXIS_CACHE[("en", "seeds")] = {"ideology": np.zeros(3)}
    unload_embeddings_for_language("it")
    assert "it" not in _VECTOR_CACHE
    assert "en" in _VECTOR_CACHE
    assert not any(k[0] == "it" for k in _AXIS_CACHE)
    assert any(k[0] == "en" for k in _AXIS_CACHE)
    assert emb._ACTIVE_VECTOR_LANG is None
    clear_embedding_caches()


def test_load_seed_poles_it() -> None:
    """Italian ideology seeds load from data/raw/seeds CSVs."""
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    poles = load_seed_poles("it", root, {"seeds_dir": "data/raw/seeds"})
    assert len(poles["ideology_pos"]) >= 10
    assert len(poles["emotion_pos"]) >= 10
    assert any("sovran" in t for t in poles["ideology_pos"])
