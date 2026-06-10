"""Tests for pole_share fixed-author event-study estimation and gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.did.bucket_estimate import estimate_panel_it_event_study, manual_panel_it_event_study


def _load_pole_share_mod():
    """Function summary: import pole_share_fixed_authors module for gate helpers."""
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/pole_share_fixed_authors.py"
    spec = importlib.util.spec_from_file_location("pole_share_fixed_authors_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_binned_panel(n_subs: int = 8, n_periods: int = 12) -> pd.DataFrame:
    """Function summary: balanced subreddit×rel_period panel for ES smoke tests."""
    rows = []
    rel_periods = list(range(-5, n_periods - 5))
    for s in range(n_subs):
        treat = int(s >= n_subs // 2)
        for rp in rel_periods:
            rows.append(
                {
                    "subreddit": f"sub_{s}",
                    "time_id": f"2023-03-{10 + rp + 20:02d}",
                    "rel_period": rp,
                    "rel_day": rp * 3,
                    "IT": treat,
                    "pole_share": 0.2 + 0.05 * treat * int(rp >= 0) + 0.01 * s,
                    "n_comments": 10 + s,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_panel_pyfixest_matches_manual_twostep() -> None:
    """Function summary: pyfixest i(rel_period, IT) agrees with manual demean+OLS to 1e-6."""
    panel = _synthetic_binned_panel()
    _, es = estimate_panel_it_event_study(
        panel,
        "pole_share",
        entity_col="subreddit",
        time_col="time_id",
        rel_col="rel_period",
        ref_period=-1,
        window=30,
        cluster_col="subreddit",
        bin_days=3,
    )
    manual = manual_panel_it_event_study(
        panel,
        "pole_share",
        entity_col="subreddit",
        time_col="time_id",
        rel_col="rel_period",
        ref_period=-1,
        window=30,
        cluster_col="subreddit",
        bin_days=3,
    )
    assert not es.empty and not manual.empty
    merged = es.merge(manual, on="rel_period", suffixes=("_py", "_man"))
    for _, row in merged.iterrows():
        assert float(row["gamma_py"]) == pytest.approx(float(row["gamma_man"]), abs=1e-6)


def test_validate_es_gates_rejects_degenerate_gamma() -> None:
    """Function summary: gate (c) refuses absurd coefficient scale."""
    mod = _load_pole_share_mod()
    es_df = pd.DataFrame(
        {
            "rel_period": [-2, 0, 1, 2, 8, 9],
            "gamma": [1e13, 0.0, 0.0, 0.0, 0.05, 0.05],
            "se": [1e13, 0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    manual_df = es_df.copy()
    es_sample = pd.DataFrame(
        {
            "rel_period": [-2, 0, 1, 2, 8, 9],
            "pole_share": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "n_comments": [10, 10, 10, 10, 10, 10],
        }
    )
    with pytest.raises(SystemExit, match="gate \\(c\\) fail"):
        mod._validate_es_gates(
            es_df,
            manual_df,
            static_beta=0.063,
            es_sample=es_sample,
            sample_label="test",
            repair_note="",
        )


def test_validate_es_gates_accepts_well_scaled_es() -> None:
    """Function summary: well-scaled ES passes magnitude and cross-estimator gates."""
    mod = _load_pole_share_mod()
    rel = [-2, 0, 1, 2, 8, 9]
    gamma = [-0.01, 0.005, -0.01, 0.01, 0.05, 0.06]
    es_df = pd.DataFrame({"rel_period": rel, "gamma": gamma, "se": [0.02] * len(rel)})
    manual_df = es_df.copy()
    es_sample = pd.DataFrame(
        {
            "rel_period": rel,
            "pole_share": [0.5] * len(rel),
            "n_comments": [100] * len(rel),
        }
    )
    mod._validate_es_gates(
        es_df,
        manual_df,
        static_beta=0.04,
        es_sample=es_sample,
        sample_label="test",
        repair_note="",
    )
