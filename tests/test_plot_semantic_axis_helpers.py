"""Smoke tests for DiD-audit plot helpers (Agg backend, no pixel checks)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pandas as pd


def _load_plot_mod():
    """Function summary: load plot_semantic_axis_descriptives module."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts/diagnostics/plot_semantic_axis_descriptives.py"
    )
    spec = importlib.util.spec_from_file_location("plot_semantic_axis_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec.loader.exec_module(mod)
    return mod


def test_score_cols_cover_all_axes() -> None:
    """Plot module exposes one score column per semantic axis."""
    mod = _load_plot_mod()
    from src.embeddings import ALL_AXIS_NAMES

    assert len(mod.SCORE_COLS) == len(ALL_AXIS_NAMES)


def test_level_out_dir_paths() -> None:
    """Level output paths follow bins_{bd}d/{level}/{chart_type}/ layout."""
    mod = _load_plot_mod()
    root = Path("/tmp/fig_test")
    p = mod._level_out_dir(root, 7, "topic_family", "timeseries")
    assert p == root / "bins_7d" / "topic_family" / "timeseries"
    p3 = mod._level_out_dir(root, 3, "language", "pole_percentiles")
    assert p3 == root / "bins_3d" / "language" / "pole_percentiles"


def test_plot_bin_completeness_and_volume_writes_png() -> None:
    """Bin completeness helper writes an output PNG."""
    mod = _load_plot_mod()
    panel = pd.DataFrame(
        {
            "topic_family": ["de", "de", "it_political"],
            "period_start": ["2023-02-24", "2023-03-03", "2023-03-03"],
            "bin_days": [7, 7, 7],
            "n_days_in_bin": [2, 7, 7],
            "is_partial_bin": [True, False, False],
            "n_comments": [100, 500, 200],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "panel_bin_completeness_and_volume.png"
        mod._plot_bin_completeness_and_volume(panel, "topic_family", out, {}, bin_days=7)
        assert out.is_file()


def test_plot_italy_intensity_writes_png() -> None:
    """Italy circumvention IT helper writes an output PNG."""
    mod = _load_plot_mod()
    panel = pd.DataFrame(
        {
            "period_start": ["2023-03-03", "2023-03-10"],
            "vpn_interest_it": [22.0, 30.0],
            "log1p_tor_bridge_users_it": [6.9, 7.1],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "italy_circumvention_it.png"
        mod._plot_italy_intensity(panel, out, {}, bin_days=7)
        assert out.is_file()


def test_prepare_panel_series_id() -> None:
    """language_universe panels get a composite series_id column."""
    mod = _load_plot_mod()
    panel = pd.DataFrame(
        {
            "primary_lexicon": ["it", "en"],
            "universe_slice": ["in_political_tree", "out_political_tree"],
        }
    )
    out = mod._prepare_panel_for_level(panel, use_series_id=True)
    assert "series_id" in out.columns
    assert "it (in_political_tree)" in out["series_id"].astype(str).tolist()
