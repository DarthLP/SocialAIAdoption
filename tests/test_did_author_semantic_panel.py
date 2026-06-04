"""Tests for author×week semantic DiD panel wiring."""

from __future__ import annotations

import pandas as pd

from scripts.diagnostics.prepare_did_author_semantic_week_panel import annotate_author_semantic_week_panel
from scripts.analysis.did_event_study import _panel_for_outcome, _strategies_for_outcome
from src.did.outcomes import OutcomeSpec, outcomes_for_families
from src.did.panels import AnalysisPanels, author_panel_has_multi_lang


def test_annotate_author_semantic_week_adds_did_columns() -> None:
    """Function summary: author semantic panel gets entity_id, treat, rel_day."""
    config = {
        "event_window": {
            "start_utc": "2023-03-01T00:00:00Z",
            "end_utc_exclusive": "2023-05-01T00:00:00Z",
            "launch_day_utc": "2023-03-31T00:00:00Z",
        }
    }
    raw = pd.DataFrame(
        {
            "author": ["a1", "a2"],
            "iso_week_start": ["2023-03-20", "2023-04-10"],
            "primary_lexicon": ["it", "en"],
            "sem_axis_ideology_mean": [0.1, 0.2],
        }
    )
    out = annotate_author_semantic_week_panel(raw, config)
    assert "entity_id" in out.columns
    assert "treat" in out.columns
    assert "rel_day" in out.columns
    assert out.loc[out["author"] == "a1", "treat"].iloc[0] == 1
    assert out.loc[out["author"] == "a2", "treat"].iloc[0] == 0


def test_panel_kind_routes_to_auth_semantic() -> None:
    """Function summary: semantic_axis_author_week outcomes use auth_semantic panel."""
    spec = outcomes_for_families(["semantic_axis_author_week"])[0]
    auth_sem = pd.DataFrame(
        {
            "author": ["x"],
            "sem_axis_ideology_mean": [0.0],
            "primary_lexicon": ["it"],
            "entity_id": ["x"],
            "time_id": ["2023-03-20"],
        }
    )
    panels = AnalysisPanels(
        sub_v1=pd.DataFrame(),
        sub_v2=pd.DataFrame(),
        slice_panel=pd.DataFrame(),
        auth_v1=pd.DataFrame(),
        auth_v2=pd.DataFrame(),
        auth_semantic=auth_sem,
        comment_1d=pd.DataFrame(),
        author_day_1d=pd.DataFrame(),
    )
    panel = _panel_for_outcome(panels, spec)
    assert panel is auth_sem
    multi = pd.DataFrame(
        {
            "primary_lexicon": ["it", "en", "de"],
        }
    )
    has_en, has_de = author_panel_has_multi_lang(multi)
    strategies = _strategies_for_outcome(spec, panel, has_en, has_de)
    assert any(s.strategy_id == "author_it_ban" for s in strategies)


def test_author_semantic_outcome_specs() -> None:
    """Function summary: registry includes author-week semantic outcomes."""
    specs = outcomes_for_families(["semantic_axis_author_week"])
    ids = {s.outcome_id for s in specs}
    assert ids == {
        "sem_axis_ideology",
        "sem_axis_emotion",
        "sem_axis_aggression",
        "sem_axis_economic",
        "sem_axis_cultural",
        "sem_axis_nationalism",
        "sem_axis_anti_establishment",
    }
    for s in specs:
        assert s.panel_kind == "author_semantic_week"
        assert s.ddd_allowed is False
