"""Unit tests for language-wave task grouping in the enriched shard runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    """Function summary: load _enriched_shard_runner module for helper tests."""
    path = Path(__file__).resolve().parent.parent / "scripts/features/_enriched_shard_runner.py"
    spec = importlib.util.spec_from_file_location("_enriched_shard_runner_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task(lex_lang: str, sub: str = "Sub") -> tuple:
    """Function summary: minimal shard task tuple with given lex_lang."""
    return (f"/tmp/{sub}.parquet", sub, lex_lang, "semaxis", "/cfg.yaml", "/root")


def test_group_tasks_into_language_waves_order() -> None:
    """Waves follow it, en, de; no interleaving across waves."""
    mod = _load_runner()
    tasks = [
        _task("en", "A"),
        _task("it", "B"),
        _task("de", "C"),
        _task("it", "D"),
        _task("en", "E"),
    ]
    waves = mod._group_tasks_into_language_waves(tasks, ("it", "en", "de"), "test")
    assert [lang for lang, _ in waves] == ["it", "en", "de"]
    assert len(waves[0][1]) == 2
    assert len(waves[1][1]) == 2
    assert len(waves[2][1]) == 1
    assert mod._task_lex_lang(waves[0][1][0]) == "it"


def test_passes_need_fasttext() -> None:
    """Only semaxis and combined all passes require language waves."""
    mod = _load_runner()
    assert mod._passes_need_fasttext(("semaxis",))
    assert mod._passes_need_fasttext(mod.PASS_ORDER)
    assert not mod._passes_need_fasttext(("polarization",))
    assert not mod._passes_need_fasttext(("ai", "style"))
