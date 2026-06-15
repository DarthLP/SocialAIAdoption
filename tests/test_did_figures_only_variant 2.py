"""Tests for --figures-only variant routing in did_event_study."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_did_event_study_mod():
    """Function summary: load did_event_study module for helper tests."""
    path = Path(__file__).resolve().parent.parent / "scripts/analysis/did_event_study.py"
    spec = importlib.util.spec_from_file_location("did_event_study_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_figures_subdir_name_variants():
    """Function summary: figures subtree matches estimate variant."""
    mod = _load_did_event_study_mod()
    assert mod._figures_subdir_name() == "did"
    assert mod._figures_subdir_name(weighted=True) == "did_weighted"
    assert mod._figures_subdir_name(variant="exbantopic") == "did_exbantopic"
