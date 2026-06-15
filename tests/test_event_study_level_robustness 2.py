"""Tests for event-study level-leak robustness diagnostic helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_robustness_module():
    """Function summary: import event_study_level_robustness without running main."""
    path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/event_study_level_robustness.py"
    spec = importlib.util.spec_from_file_location("event_study_level_robustness", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_pool() -> pd.DataFrame:
    """Function summary: tiny unbalanced panel with IT level gap (leak-prone)."""
    rows = []
    for author, it, y_base in [("a1", 1.0, 1.0), ("a2", 1.0, 1.0), ("c1", 0.0, 0.0), ("c2", 0.0, 0.0)]:
        for rel in (-2, -1, 0, 1):
            # only a1 has ref bin -1; IT authors higher level
            if author == "a2" and rel == -1:
                continue
            rows.append(
                {
                    "author": author,
                    "IT": it,
                    "rel_period": rel,
                    "rel_day": rel * 3,
                    "y": y_base + 0.01 * rel,
                    "time_id": f"t{rel}",
                }
            )
    return pd.DataFrame(rows)


def test_fd_ref_near_zero_on_synthetic_leak_panel() -> None:
    """Function summary: FD_ref collapses artificial IT level gap to ~0."""
    mod = _load_robustness_module()
    pool = _synthetic_pool()
    meta, es = mod.estimate_first_difference_event_study(
        pool, ref_period=-1, window=3, bin_days=3, baseline="ref"
    )
    assert meta.get("estimation_note") == "ok"
    assert not es.empty
    assert es["gamma"].abs().max() < 0.05


def test_fd_preban_mean_keeps_ref_bin() -> None:
    """Function summary: FD_mean retains rel_period=-1 and uses pre-ban baseline."""
    mod = _load_robustness_module()
    pool = _synthetic_pool()
    meta, es = mod.estimate_first_difference_event_study(
        pool, ref_period=-1, window=3, bin_days=3, baseline="preban_mean"
    )
    assert meta.get("estimation_note") == "ok"
    assert -1 in es["rel_period"].tolist()


def test_author_preban_mean_comment_weighted() -> None:
    """Function summary: pre-ban mean equals simple comment mean over rel<0."""
    mod = _load_robustness_module()
    pool = _synthetic_pool()
    pool["y"] = pool["y"].astype(float)
    preban = mod._author_preban_mean(pool, bin_days=3)
    a1 = preban.loc[preban["author"] == "a1", "preban_mean"].iloc[0]
    expected = pool.loc[(pool["author"] == "a1") & (pool["rel_period"] < 0), "y"].mean()
    assert abs(a1 - expected) < 1e-9
