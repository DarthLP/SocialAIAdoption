"""Tests for fastText language-wave helper (mocked loads, no .bin files)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.embeddings import (
    clear_embedding_caches,
    run_language_vector_wave,
    unload_embeddings_for_language,
)


def test_run_language_vector_wave_one_load_per_lang() -> None:
    """Function summary: wave calls load once per language and unloads between."""
    clear_embedding_caches()
    calls: list[str] = []
    kv = MagicMock(name="kv")

    def _cb(lang: str, loaded_kv: object) -> None:
        calls.append(f"work:{lang}")

    with patch("src.embeddings.load_vectors", return_value=kv) as mock_load:
        with patch("src.embeddings.unload_embeddings_for_language") as mock_unload:
            with patch("src.embeddings.ensure_exclusive_vector_lang"):
                run_language_vector_wave(
                    Path("/tmp"),
                    {"vector_cache_exclusive": True},
                    _cb,
                    languages=("it", "en", "de"),
                )

    assert mock_load.call_count == 3
    assert mock_unload.call_count == 3
    assert mock_unload.call_args_list[0][0][0] == "it"
    assert mock_unload.call_args_list[1][0][0] == "en"
    assert mock_unload.call_args_list[2][0][0] == "de"
    assert calls == ["work:it", "work:en", "work:de"]
    clear_embedding_caches()


def test_watch_included_only_if_three_langs_pass() -> None:
    """Function summary: mirror export aggregate rule for Watch inclusion."""
    row_concept = "anti_woke"
    gate_log = [
        {"axis": "cultural", "concept": row_concept, "lang": lang, "in_vocab": 1, "cosine": 0.2,
         "expected_sign": 1, "term": "woke", "included": 0, "reason": ""}
        for lang in ("it", "en", "de")
    ]
    checks = [g for g in gate_log if g["concept"] == row_concept]
    ok = (
        len(checks) == 3
        and all(int(g["in_vocab"]) == 1 for g in checks)
        and all((g["cosine"] * g["expected_sign"]) > 0 for g in checks)
    )
    assert ok
    gate_log[1]["in_vocab"] = 0
    checks_fail = [g for g in gate_log if g["concept"] == row_concept]
    ok_fail = len(checks_fail) == 3 and all(int(g["in_vocab"]) == 1 for g in checks_fail)
    assert not ok_fail
