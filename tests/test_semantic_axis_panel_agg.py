"""Unit tests for semantic-axis panel aggregation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_prepare_mod():
    """Function summary: load prepare_semantic_axis_descriptives module."""
    path = Path(__file__).resolve().parent.parent / "scripts/diagnostics/prepare_semantic_axis_descriptives.py"
    spec = importlib.util.spec_from_file_location("prepare_semantic_axis_descriptives_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assign_period_launch_aligned_3d() -> None:
    """Launch-aligned 3d bins anchor period_start to launch + k*3 days."""
    from scripts.diagnostics.descriptives_util import assign_period_start

    dates = pd.Series(["2023-03-30", "2023-03-31", "2023-04-02", "2023-04-05"])
    out = assign_period_start(dates, 3, "2023-03-31")
    assert out.tolist() == ["2023-03-28", "2023-03-31", "2023-03-31", "2023-04-03"]


def test_assign_period_calendar_1d() -> None:
    """1d bins use calendar date_utc as period_start."""
    from scripts.diagnostics.descriptives_util import assign_period_start

    dates = pd.Series(["2023-03-15", "2023-04-01"])
    out = assign_period_start(dates, 1, "2023-03-31")
    assert out.tolist() == ["2023-03-15", "2023-04-01"]


def _default_sem_cfg() -> dict:
    """Minimal semantic_axis config for aggregation tests."""
    return {
        "pole_thresholds_by_lexicon": {
            "en": {"ideology": 0.25, "emotion": 0.25, "aggression": 0.25},
        },
        "pole_percentiles": [],
    }


def test_semantic_axis_agg_ideology_left_right_counts() -> None:
    """Ideology buckets use right/left abs labels and report comment/word counts."""
    mod = _load_prepare_mod()
    from src.semantic_axis_stats import build_pole_bucket_specs

    df = pd.DataFrame(
        {
            "primary_lexicon": ["en", "en", "en", "en"],
            "sem_axis_ideology": [0.8, -0.6, 0.1, float("nan")],
            "sem_axis_emotion": [0.0, 0.0, 0.0, 0.0],
            "sem_axis_aggression": [0.0, 0.0, 0.0, 0.0],
            "sem_axis_coverage": [1.0, 1.0, 1.0, 0.0],
            "has_sem_axis": [1.0, 1.0, 1.0, 0.0],
            "net_ideology": [0.0, 0.0, 0.0, 0.0],
            "n_words": [10, 20, 5, 1],
        }
    )
    sem_cfg = _default_sem_cfg()
    specs = build_pole_bucket_specs(sem_cfg)
    out = mod._semantic_axis_agg(df, specs, sem_cfg, {})
    assert out["n_comments"] == 4
    assert out["n_scored"] == 3
    assert out["share_unscored"] == 0.25
    assert out["sem_axis_ideology_n_comments_right_abs"] == 1
    assert out["sem_axis_ideology_n_comments_left_abs"] == 1
    assert out["sem_axis_ideology_n_words_right_abs"] == 10.0
    assert out["sem_axis_ideology_n_words_left_abs"] == 20.0
    assert out["sem_axis_emotion_n_comments_pos_abs"] == 0


def test_build_panel_language_universe_share() -> None:
    """language_universe panel includes share_of_cell_comments within lang×period."""
    mod = _load_prepare_mod()
    df = pd.DataFrame(
        {
            "primary_lexicon": ["it", "it", "it", "it"],
            "universe_slice": [
                mod.UNIVERSE_SLICE_IN,
                mod.UNIVERSE_SLICE_IN,
                mod.UNIVERSE_SLICE_OUT,
                mod.UNIVERSE_SLICE_OUT,
            ],
            "period_start": ["2023-03-15"] * 4,
            "date_utc": ["2023-03-15"] * 4,
            "sem_axis_ideology": [0.5, 0.5, -0.5, -0.5],
            "sem_axis_emotion": [0.0, 0.0, 0.0, 0.0],
            "sem_axis_aggression": [0.0, 0.0, 0.0, 0.0],
            "sem_axis_coverage": [1.0, 1.0, 1.0, 1.0],
            "has_sem_axis": [1.0, 1.0, 1.0, 1.0],
            "net_ideology": [0.0, 0.0, 0.0, 0.0],
            "n_words": [10, 10, 10, 10],
        }
    )
    parent = mod._parent_counts_language(df)
    sem_cfg = _default_sem_cfg()
    from src.semantic_axis_stats import build_pole_bucket_specs

    panel = mod._build_panel(
        df,
        "language_universe",
        ["primary_lexicon", "universe_slice"],
        (),
        1,
        "2023-03-31",
        build_pole_bucket_specs(sem_cfg),
        sem_cfg,
        {},
        parent_counts=parent,
    )
    in_row = panel[panel["universe_slice"] == mod.UNIVERSE_SLICE_IN].iloc[0]
    assert in_row["n_comments"] == 2
    assert in_row["share_of_cell_comments"] == 0.5


def test_panel_columns_exclude_body() -> None:
    """Default panel shard read list omits comment body."""
    mod = _load_prepare_mod()
    assert "body" not in mod.PANEL_COLUMNS
    cols = mod._columns_to_read(include_validation=False, include_examples=False)
    assert "body" not in cols


def test_accumulator_matches_single_pass_agg() -> None:
    """Streaming accumulator matches one-shot _semantic_axis_agg."""
    mod = _load_prepare_mod()
    df = pd.DataFrame(
        {
            "sem_axis_ideology": [0.8, -0.6, 0.1],
            "sem_axis_emotion": [0.0, 0.0, 0.0],
            "sem_axis_aggression": [0.0, 0.0, 0.0],
            "sem_axis_coverage": [1.0, 1.0, 1.0],
            "has_sem_axis": [1.0, 1.0, 1.0],
            "net_ideology": [0.0, 0.0, 0.0],
            "n_words": [10, 20, 5],
        }
    )
    sem_cfg = _default_sem_cfg()
    from src.semantic_axis_stats import build_pole_bucket_specs

    specs = build_pole_bucket_specs(sem_cfg)
    df = df.assign(primary_lexicon="en")
    direct = mod._semantic_axis_agg(df, specs, sem_cfg, {})
    acc = mod._new_agg_accumulator(specs)
    mod._accumulate_group(acc, df, specs, sem_cfg, {})
    merged = mod._finalize_accumulator(acc, specs, bin_days=1)
    for key in ("n_comments", "sem_axis_ideology_n_comments_right_abs", "net_ideology_mean"):
        assert direct[key] == merged[key]


def test_partial_7d_bin_flags() -> None:
    """Launch-aligned 7d bin with two calendar days sets partial-bin flags."""
    mod = _load_prepare_mod()
    from src.semantic_axis_stats import build_pole_bucket_specs

    df = pd.DataFrame(
        {
            "date_utc": ["2023-03-01", "2023-03-02"],
            "sem_axis_ideology": [0.5, -0.5],
            "sem_axis_emotion": [0.0, 0.0],
            "sem_axis_aggression": [0.0, 0.0],
            "sem_axis_coverage": [1.0, 1.0],
            "has_sem_axis": [1.0, 1.0],
            "net_ideology": [0.0, 0.0],
            "n_words": [10, 10],
            "primary_lexicon": ["en", "en"],
        }
    )
    sem_cfg = _default_sem_cfg()
    specs = build_pole_bucket_specs(sem_cfg)
    acc = mod._new_agg_accumulator(specs)
    mod._accumulate_group(acc, df, specs, sem_cfg, {})
    out = mod._finalize_accumulator(acc, specs, bin_days=7)
    assert out["n_days_in_bin"] == 2
    assert bool(out["is_partial_bin"])
