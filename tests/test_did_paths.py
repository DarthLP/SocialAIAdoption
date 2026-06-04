"""Tests for DiD nested table path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.did.paths import (
    aggregated_event_study_figure_path,
    aggregated_tail_shift_figure_path,
    bucket_event_study_figures_dir,
    did_aggregated_event_study_path,
    did_bucket_event_study_dir,
    did_estimates_dir,
    did_event_study_path,
    did_outcome_table_path,
    did_panels_dir,
    did_root,
    did_summary_dir,
    did_summary_paths,
)
from src.did.outcomes import SUMMARY_THEMES


@pytest.fixture
def config(tmp_path: Path) -> dict:
    """Function summary: minimal config with tables_dir under tmp_path."""
    tables = tmp_path / "tables" / "italy_polarization"
    tables.mkdir(parents=True)
    return {"paths": {"tables_dir": str(tables)}}


def test_did_root_and_panels(config: dict) -> None:
    """Function summary: panel paths live under did/panels/{kind}/."""
    assert did_root(config) == Path(config["paths"]["tables_dir"]) / "did"
    assert did_panels_dir(config, "country") == did_root(config) / "panels" / "country"
    assert did_panels_dir(config, "semantic") == did_root(config) / "panels" / "semantic"
    assert did_panels_dir(config, "subreddit") == did_root(config) / "panels" / "subreddit"


def test_did_estimates_and_summary(config: dict) -> None:
    """Function summary: estimates and summary paths match plan layout."""
    assert did_estimates_dir(config) == did_root(config) / "estimates"
    assert did_summary_dir(config) == did_estimates_dir(config) / "summary"
    master, labeled = did_summary_paths(config)
    assert master.name == "did_summary.csv"
    assert labeled.name == "did_summary_labeled.csv"
    assert master.parent == did_summary_dir(config)


def test_did_outcome_table_paths(config: dict) -> None:
    """Function summary: per-outcome nested CSV paths."""
    coef = did_outcome_table_path(config, "lexical", "coefficients", "aggression_rate")
    assert coef == (
        did_estimates_dir(config) / "lexical" / "coefficients" / "aggression_rate.csv"
    )
    es = did_event_study_path(config, "semantic_axis", "sem_axis_ideology")
    assert es == (
        did_estimates_dir(config)
        / "semantic_axis"
        / "event_study"
        / "sem_axis_ideology.csv"
    )


def test_summary_themes_non_empty() -> None:
    """Function summary: bundled theme registry includes aggression and ideology."""
    assert "aggression_rate" in SUMMARY_THEMES["aggression"]
    assert "net_ideology" in SUMMARY_THEMES["ideology"]
    assert len(SUMMARY_THEMES["all"]) > 0


def test_bucket_event_study_paths_by_bin_days(config: dict) -> None:
    """Function summary: bucket event-study tables/figures split 1d vs 3d."""
    assert did_bucket_event_study_dir(config, 3) == did_root(config) / "bucket_event_study" / "3d"
    assert did_bucket_event_study_dir(config, 1) == did_root(config) / "bucket_event_study" / "1d"
    assert (
        did_bucket_event_study_dir(config, 3, stratification="lexical", outcome="sem_axis_emotion")
        == did_root(config) / "bucket_event_study" / "3d" / "strat_lexical" / "sem_axis_emotion"
    )
    assert (
        did_bucket_event_study_dir(config, 3, stratification="semantic", outcome="sem_axis_emotion")
        == did_root(config) / "bucket_event_study" / "3d" / "strat_semantic" / "sem_axis_emotion"
    )
    assert (
        bucket_event_study_figures_dir(config, 3).name == "3d"
        and bucket_event_study_figures_dir(config, 3).parent.name == "bucket_event_study"
    )


def test_aggregated_event_study_bundle_paths(config: dict, tmp_path: Path) -> None:
    """Function summary: aggregated ES CSV/PNG paths include bundle segment."""
    est = did_aggregated_event_study_path(
        config,
        "lexical",
        "language",
        "hub_pooled",
        3,
        "cross_country_all",
        "aggression_rate",
    )
    assert "event_study/language/hub_pooled/3d/cross_country_all/aggression_rate.csv" in str(
        est
    )
    fig = aggregated_event_study_figure_path(
        tmp_path, "topic_family", "overlay_pooled", 1, "net_ideology"
    )
    assert "topic_family/overlay_pooled/1d/net_ideology.png" in str(fig)
    tail = aggregated_tail_shift_figure_path(
        tmp_path, "language_universe", "in_out_slice", 1, suffix="in_tree"
    )
    assert "language_universe/in_out_slice/1d/sem_axis_ideology_tail_shift_in_tree.png" in str(
        tail
    )
