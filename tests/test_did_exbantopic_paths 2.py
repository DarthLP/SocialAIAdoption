"""Tests for exbantopic variant path resolution in did.paths."""

from __future__ import annotations

from pathlib import Path

from src.did.paths import (
    _estimates_root_name,
    did_estimates_dir,
    did_event_study_path,
    did_summary_paths,
)


def test_estimates_root_name_variants():
    """Baseline, weighted, and exbantopic estimate trees are distinct."""
    assert _estimates_root_name() == "estimates"
    assert _estimates_root_name(weighted=True) == "estimates_weighted"
    assert _estimates_root_name(variant="exbantopic") == "estimates_exbantopic"


def test_exbantopic_paths_under_study_did_root():
    """exbantopic variant resolves under did/ without touching baseline."""
    config = {"paths": {"tables_dir": "results/tables/italy_polarization"}}
    base = did_estimates_dir(config)
    ex = did_estimates_dir(config, variant="exbantopic")
    assert base.name == "estimates"
    assert ex.name == "estimates_exbantopic"
    assert ex != base
    summary_base, _ = did_summary_paths(config)
    summary_ex, _ = did_summary_paths(config, variant="exbantopic")
    assert summary_base.parent.name == "summary"
    assert summary_ex.parent.parent.name == "estimates_exbantopic"
    es = did_event_study_path(config, "semantic_axis", "sem_axis_emotion", variant="exbantopic")
    assert "estimates_exbantopic" in str(es)
    assert es.name == "sem_axis_emotion.csv"
    assert Path(es).parts[-2] == "event_study"
