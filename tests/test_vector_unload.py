"""Unit tests for exclusive per-process fastText vector caching."""

from __future__ import annotations

import numpy as np

import src.embeddings as emb
from src.embeddings import (
    clear_embedding_caches,
    ensure_exclusive_vector_lang,
    unload_all_vectors,
)


class _FakeKV:
    """Minimal stand-in for gensim vectors."""

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.vector_size = 3


def test_ensure_exclusive_clears_on_language_switch() -> None:
    """Switching lang unloads prior model; only one language remains cached."""
    clear_embedding_caches()
    cfg = {"vector_cache_exclusive": True}
    emb._VECTOR_CACHE["it"] = _FakeKV("it")
    emb._AXIS_CACHE[("it", "seeds")] = {"ideology": np.zeros(3)}
    emb._ACTIVE_VECTOR_LANG = "it"

    ensure_exclusive_vector_lang("en", cfg)
    assert emb._ACTIVE_VECTOR_LANG == "en"
    assert "it" not in emb._VECTOR_CACHE
    assert not any(k[0] == "it" for k in emb._AXIS_CACHE)
    assert "en" not in emb._VECTOR_CACHE

    emb._VECTOR_CACHE["en"] = _FakeKV("en")
    ensure_exclusive_vector_lang("en", cfg)
    assert len(emb._VECTOR_CACHE) == 1
    assert "en" in emb._VECTOR_CACHE

    clear_embedding_caches()


def test_exclusive_disabled_keeps_multiple_langs() -> None:
    """vector_cache_exclusive: false allows stacked language caches."""
    clear_embedding_caches()
    cfg = {"vector_cache_exclusive": False}
    emb._VECTOR_CACHE["it"] = _FakeKV("it")
    emb._VECTOR_CACHE["en"] = _FakeKV("en")
    emb._ACTIVE_VECTOR_LANG = "it"

    ensure_exclusive_vector_lang("en", cfg)
    assert len(emb._VECTOR_CACHE) == 2
    assert "it" in emb._VECTOR_CACHE and "en" in emb._VECTOR_CACHE

    clear_embedding_caches()


def test_unload_all_vectors_resets_active_lang() -> None:
    """unload_all_vectors clears active language tracker."""
    clear_embedding_caches()
    emb._ACTIVE_VECTOR_LANG = "de"
    emb._VECTOR_CACHE["de"] = _FakeKV("de")
    unload_all_vectors()
    assert emb._ACTIVE_VECTOR_LANG is None
    assert not emb._VECTOR_CACHE
