"""Tests for aggregated DiD event-study bundles, paths, and panel binning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.did.estimate import estimate_event_study
from src.did.event_study_panels import prepare_subreddit_event_study_panel
from src.did.paths import aggregated_event_study_figure_path, did_aggregated_event_study_path
from src.did.specs import (
    EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS,
    StrategySpec,
    event_study_language_universe_slice_strategies,
    filter_strategy_sample,
)
from scripts.analysis.did_aggregated_event_study import (
    AggregatedEsJob,
    _estimate_event_study_bundle,
    _resolve_column,
)


@pytest.fixture
def config(tmp_path: Path) -> dict:
    """Function summary: minimal config with launch-aligned event window."""
    return {
        "paths": {"tables_dir": str(tmp_path / "tables")},
        "event_window": {
            "start_utc": "2023-03-18T00:00:00Z",
            "end_utc_exclusive": "2023-05-01T00:00:00Z",
            "launch_day_utc": "2023-03-31T00:00:00Z",
        },
    }


def test_aggregated_event_study_paths_include_bundle(config: dict, tmp_path: Path) -> None:
    """Function summary: figure and estimate paths nest bundle before bin_days."""
    fig = aggregated_event_study_figure_path(
        tmp_path / "fig", "language", "subreddit", 1, "aggression_rate"
    )
    assert fig == tmp_path / "fig" / "event_study" / "language" / "subreddit" / "1d" / "aggression_rate.png"
    est = did_aggregated_event_study_path(
        config,
        "lexical",
        "language_universe",
        "in_out_slice",
        3,
        "cross_country_political_universe_in",
        "ambivalence",
    )
    assert "language_universe/in_out_slice/3d/" in str(est)
    assert est.name == "ambivalence.csv"


def test_prepare_subreddit_3d_bins_outcomes(config: dict) -> None:
    """Function summary: 3d panel uses period_start time_id and fewer rows than daily."""
    daily = pd.DataFrame(
        {
            "subreddit": ["a"] * 6,
            "date_utc": [
                "2023-03-28",
                "2023-03-29",
                "2023-03-30",
                "2023-03-31",
                "2023-04-01",
                "2023-04-02",
            ],
            "n_comments": [10, 10, 10, 10, 10, 10],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "treat": [0] * 3 + [1] * 3,
            "post": [0] * 3 + [1] * 3,
            "topic_family": ["it_political"] * 6,
            "IT": [1] * 6,
        }
    )
    binned = prepare_subreddit_event_study_panel(daily, config, 3, entity_cols=("subreddit",))
    assert len(binned) < len(daily)
    assert (binned["time_id"] == binned["period_start"]).all()
    assert "rel_period" in binned.columns


def test_hub_pooled_synthetic_estimate_smoke() -> None:
    """Function summary: hub-level panel with ~5 entities yields finite event-study SEs."""
    rows = []
    hubs = ["it", "de", "eu", "us", "uk"]
    anchor = pd.Timestamp("2023-03-26")
    for hub in hubs:
        treat = 1.0 if hub == "it" else 0.0
        for k in range(-5, 6):
            day = (anchor + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "language_hub": hub,
                    "entity_id": hub,
                    "time_id": day,
                    "period_start": day,
                    "rel_day": k,
                    "rel_period": k // 3,
                    "treat": treat,
                    "y": 0.1 * treat + 0.01 * k,
                }
            )
    panel = pd.DataFrame(rows)
    job = AggregatedEsJob("language", "hub_pooled", "aggregated", True, lambda: ())
    _, es = estimate_event_study(
        panel,
        "y",
        rel_col="rel_day",
        window=5,
        entity_col="language_hub",
        time_col="time_id",
    )
    assert not es.empty
    assert float(es["se"].mean()) < float("inf")


def test_slice_in_out_strategies_produce_two_series() -> None:
    """Function summary: in-tree and out-tree filters yield two non-empty ES series."""
    rows = []
    anchor = pd.Timestamp("2023-03-26")
    specs = [
        ("in_political_tree", "it_political", 1),
        ("in_political_tree", "it_others", 1),
        ("in_political_tree", "de", 0),
        ("in_political_tree", "eu", 0),
        ("out_political_tree", "it_political", 1),
        ("out_political_tree", "it_others", 1),
        ("out_political_tree", "us", 0),
        ("out_political_tree", "uk", 0),
    ]
    for idx, (slc, fam, treat) in enumerate(specs):
        sub = f"sub_{idx}"
        for k in range(-5, 6):
            day = (anchor + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "subreddit": sub,
                    "universe_slice": slc,
                    "entity_id": f"{sub}|{slc}",
                    "time_id": day,
                    "rel_day": k,
                    "rel_period": k // 3,
                    "topic_family": fam,
                    "IT": treat,
                    "treat": float(treat),
                    "y": 0.2 + 0.05 * k + 0.01 * treat,
                }
            )
    panel = pd.DataFrame(rows)
    strategies = event_study_language_universe_slice_strategies()
    assert len(strategies) == 2
    job = AggregatedEsJob(
        "language_universe",
        "in_out_slice",
        "slice",
        True,
        event_study_language_universe_slice_strategies,
    )
    series = _estimate_event_study_bundle(
        panel,
        type(
            "OC",
            (),
            {
                "outcome_id": "y",
                "family": "lexical",
                "column": "y",
            },
        )(),
        1,
        "y",
        strategies,
        job=job,
    )
    assert len(series) == 2


def test_language_universe_slice_strategy_ids() -> None:
    """Function summary: slice overlay uses political-universe in/out strategy keys."""
    assert EVENT_STUDY_LANGUAGE_UNIVERSE_SLICE_IDS == (
        "cross_country_political_universe_in",
        "cross_country_political_universe_out",
    )


def test_filter_slice_strategies_keep_universe_column() -> None:
    """Function summary: universe_slice filter restricts rows before estimation."""
    panel = pd.DataFrame(
        {
            "universe_slice": ["in_political_tree", "out_political_tree", "in_political_tree"],
            "topic_family": ["it_political", "it_political", "de"],
            "entity_id": ["a", "b", "c"],
            "rel_day": [0, 0, 0],
            "post": [1, 1, 1],
            "IT": [1, 1, 0],
        }
    )
    strat = StrategySpec(
        "cross_country_political_universe_in",
        universe_slice="in_political_tree",
    )
    out = filter_strategy_sample(panel, strat, window_days=14)
    assert set(out["universe_slice"]) == {"in_political_tree"}


def test_language_subreddit_bundle_inference() -> None:
    """Function summary: language/subreddit bundle estimates on subreddit panel with finite SE."""
    from src.config_utils import load_config
    from src.did.aggregated import AggregatedPanelKey, build_aggregated_panels
    from src.did.outcomes import OUTCOME_REGISTRY
    from src.did.specs import event_study_overlay_strategies

    cfg_path = Path(__file__).resolve().parents[1] / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        return
    config = load_config(cfg_path)
    try:
        from src.did.panels import load_subreddit_event_study_panel

        sub = load_subreddit_event_study_panel(config, 3)
    except FileNotFoundError:
        return
    assert "period_start" in sub.columns or sub["time_id"].nunique() < sub.shape[0] * 0.99
    panel = build_aggregated_panels(config).get(AggregatedPanelKey("language", 3))
    oc = next(o for o in OUTCOME_REGISTRY if o.outcome_id == "ambivalence")
    y_col = _resolve_column(sub, oc)
    if y_col is None:
        return
    job = AggregatedEsJob(
        "language",
        "subreddit",
        "subreddit",
        True,
        event_study_overlay_strategies,
    )
    series = _estimate_event_study_bundle(sub, oc, 3, y_col, event_study_overlay_strategies(), job=job)
    eu = next(s for s in series if "EU" in s.label)
    assert float(eu.es_df["se"].mean()) > 1e-5
