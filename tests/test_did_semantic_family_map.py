"""Tests that semantic DiD merge retains all topic_family arms."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_did_merge_mod():
    """Function summary: load prepare_did_merged_panels module."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/prepare_did_merged_panels.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_did_merged_panels_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_topic_family_map_includes_us_and_eu() -> None:
    """US and EU semantic families map to daily_country_panel labels."""
    mod = _load_did_merge_mod()
    mapping = mod.TOPIC_FAMILY_TO_COUNTRY_PANEL
    assert mapping["us"] == "US_political"
    assert mapping["eu"] == "EU_hub_en"


def test_semantic_merge_keeps_all_families() -> None:
    """Mapped country_panel must not drop any topic_family from source panel."""
    mod = _load_did_merge_mod()
    families = ["de", "eu", "it_others", "it_political", "uk", "us"]
    panel = pd.DataFrame(
        {
            "topic_family": families,
            "period_start": ["2023-03-01"] * len(families),
            "sem_axis_ideology_mean": [0.0] * len(families),
        }
    )
    panel["country_panel"] = panel["topic_family"].map(mod.TOPIC_FAMILY_TO_COUNTRY_PANEL)
    assert panel["country_panel"].notna().all()
    assert set(panel["topic_family"]) == set(families)
