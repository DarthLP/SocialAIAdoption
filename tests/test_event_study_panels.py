"""Tests for 3d event-study panel metadata carry-forward and degeneracy guards."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.did_aggregated_event_study import (
    TAIL_SHARE_MAX_ABS_GAMMA,
    _event_study_series_usable,
)
from src.did.estimate import ES_DEGENERATE_ABS_GAMMA, estimate_event_study
from src.did.event_study_panels import (
    ES_PANEL_META_COLS,
    prepare_subreddit_event_study_panel,
    restore_entity_meta_after_binning,
)
from src.did.specs import (
    StrategySpec,
    event_study_overlay_strategies,
    filter_strategy_sample,
    first_strategy_by_id,
)
from src.plotting.thesis_theme import THESIS_COEF_MARKER


@pytest.fixture
def config() -> Dict:
    """Function summary: minimal config with launch-aligned event window."""
    return {
        "event_window": {
            "start_utc": "2023-03-18T00:00:00Z",
            "end_utc_exclusive": "2023-05-01T00:00:00Z",
            "launch_day_utc": "2023-03-31T00:00:00Z",
        },
    }


def _daily_panel() -> pd.DataFrame:
    """Function summary: synthetic subreddit-day panel with IT + control arms."""
    subs = [
        ("it_a", "it_political", "it", 1, 0, 0, 0),
        ("it_b", "it_others", "it", 1, 0, 0, 0),
        ("de_a", "de", "de", 0, 1, 1, 0),
        ("eu_a", "eu", "en", 0, 1, 0, 1),
    ]
    dates = pd.date_range("2023-03-19", "2023-04-29", freq="D").strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    rows: List[Dict] = []
    for sub, fam, lex, it, is_ctrl, c_de, c_eu in subs:
        for d in dates:
            rows.append(
                {
                    "subreddit": sub,
                    "topic_family": fam,
                    "primary_lexicon": lex,
                    "IT": it,
                    "is_control": is_ctrl,
                    "control_de": c_de,
                    "control_eu": c_eu,
                    "date_utc": d,
                    "n_comments": 50,
                    "y": 0.1 * it + float(rng.normal(scale=0.01)),
                }
            )
    return pd.DataFrame(rows)


def test_3d_binning_retains_entity_metadata(config: Dict) -> None:
    """Function summary: topic_family/primary_lexicon/IT survive 3d outcome binning."""
    panel = prepare_subreddit_event_study_panel(_daily_panel(), config, 3)
    for col in ("topic_family", "primary_lexicon", "IT", "is_control"):
        assert col in panel.columns, f"{col} dropped by 3d binning"
    fam = panel.groupby("subreddit")["topic_family"].first()
    assert fam["it_a"] == "it_political"
    assert fam["de_a"] == "de"


def test_3d_cross_country_all_keeps_both_arms(config: Dict) -> None:
    """Function summary: post-filter 3d sample has IT=0 controls and treat variation per time bin."""
    panel = prepare_subreddit_event_study_panel(_daily_panel(), config, 3)
    sample = filter_strategy_sample(panel, StrategySpec("cross_country_all"), window_days=30)
    it_vals = set(sample["IT"].astype(float).round().astype(int))
    assert it_vals == {0, 1}
    mono = sample.groupby("time_id")["treat"].nunique()
    assert int((mono <= 1).sum()) == 0


def test_filter_fallback_without_lexicon_metadata(config: Dict) -> None:
    """Function summary: missing primary_lexicon falls back to IT/control flags, not IT-only."""
    panel = prepare_subreddit_event_study_panel(_daily_panel(), config, 3)
    bare = panel.drop(columns=["topic_family", "primary_lexicon"])
    pooled = filter_strategy_sample(bare, StrategySpec("cross_country_all"), window_days=30)
    assert set(pooled["IT"].astype(float).round().astype(int)) == {0, 1}
    vs_de = filter_strategy_sample(
        bare,
        StrategySpec("cross_country_vs_de", control_family="de"),
        window_days=30,
    )
    ctrl = vs_de[vs_de["treat"] == 0]
    assert not ctrl.empty
    assert set(ctrl["subreddit"]) == {"de_a"}


def test_restore_entity_meta_noop_when_present() -> None:
    """Function summary: restore is a no-op when metadata already on binned frame."""
    daily = _daily_panel()
    out = restore_entity_meta_after_binning(daily, daily, ("subreddit",))
    assert out is daily
    assert set(ES_PANEL_META_COLS) >= {"topic_family", "primary_lexicon", "IT"}


def test_series_usable_rejects_absurd_gamma() -> None:
    """Function summary: degenerate 1e12-scale gammas rejected; sane unbounded gammas kept."""
    degenerate = pd.DataFrame(
        {"rel_period": [0, 1], "gamma": [1e12, -4e13], "se": [1e12, 2e13]}
    )
    assert not _event_study_series_usable(degenerate)
    sane_unbounded = pd.DataFrame(
        {"rel_period": [0, 1], "gamma": [5.5, 59.6], "se": [1.0, 8.0]}
    )
    assert _event_study_series_usable(sane_unbounded)
    assert not _event_study_series_usable(
        sane_unbounded, max_abs_gamma=TAIL_SHARE_MAX_ABS_GAMMA
    )
    tail = pd.DataFrame({"rel_period": [0, 1], "gamma": [0.05, -0.07], "se": [0.02, 0.02]})
    assert _event_study_series_usable(tail, max_abs_gamma=TAIL_SHARE_MAX_ABS_GAMMA)
    assert ES_DEGENERATE_ABS_GAMMA >= 1e6


def _es_rows() -> pd.DataFrame:
    """Function summary: small estimable two-arm event-study panel."""
    rows: List[Dict] = []
    rng = np.random.default_rng(3)
    for ent, treat in [("it1", 1.0), ("it2", 1.0), ("de1", 0.0), ("de2", 0.0)]:
        for k in range(-3, 4):
            rows.append(
                {
                    "entity_id": ent,
                    "time_id": f"2023-03-{25 + k:02d}",
                    "rel_period": k,
                    "treat": treat,
                    "y": 0.05 * treat * (k >= 0) + float(rng.normal(scale=0.01)),
                }
            )
    return pd.DataFrame(rows)


def test_estimate_event_study_degeneracy_guard_and_rel_day_scaling() -> None:
    """Function summary: max_abs_gamma guard empties ES; rel_day = rel_period * bin_days."""
    summ, es = estimate_event_study(
        _es_rows(), "y", rel_col="rel_period", window=3, bin_days=3
    )
    assert not es.empty
    assert (es["rel_day"] == es["rel_period"] * 3).all()
    summ2, es2 = estimate_event_study(
        _es_rows(), "y", rel_col="rel_period", window=3, bin_days=3, max_abs_gamma=1e-9
    )
    assert es2.empty
    assert summ2["estimation_note"] == "degenerate_collinear"


def test_event_study_strategies_use_full_window_variant() -> None:
    """Function summary: ES bundles pick the full_ban cross_country_all, not the early_ban clone."""
    by_id = first_strategy_by_id()
    assert by_id["cross_country_all"].post_mode == "full_ban"
    overlay = event_study_overlay_strategies()
    assert overlay[0].strategy_id == "cross_country_all"
    assert all(s.post_mode == "full_ban" for s in overlay)


def test_thesis_coef_marker_constant() -> None:
    """Function summary: neutral coefficient marker color is the agreed dark gray."""
    assert THESIS_COEF_MARKER == "#333333"
