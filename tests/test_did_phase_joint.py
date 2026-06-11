"""Tests for joint multi-phase TWFE estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.did.estimate import estimate_twfe_phase_joint, run_strategy_phase_joint
from src.did.specs import PHASE_JOINT_SPECS, StrategySpec, filter_strategy_sample


def _synthetic_panel() -> pd.DataFrame:
    """Function summary: IT vs control panel with outcome variation by rel_day."""
    rng = np.random.default_rng(42)
    rows = []
    for ent, treat in [("it1", 1), ("it2", 1), ("de1", 0), ("de2", 0), ("eu1", 0)]:
        for rd in range(-10, 31):
            y = 0.1 * treat * (rd >= 0) + rng.normal(0, 0.01)
            rows.append(
                {
                    "entity_id": ent,
                    "time_id": f"2023-04-{min(28, max(1, rd + 1)):02d}",
                    "topic_family": "it_political" if treat else "de",
                    "IT": treat,
                    "treat": treat,
                    "rel_day": rd,
                    "post": int(rd >= 0),
                    "pole_share": y,
                }
            )
    return pd.DataFrame(rows)


def test_phase_joint_returns_four_specs() -> None:
    """Function summary: joint estimator emits one row per phase_joint_* spec."""
    panel = _synthetic_panel()
    strat = StrategySpec("cross_country_all", post_mode="phase_joint")
    rows = run_strategy_phase_joint(panel, strat, "pole_share")
    assert len(rows) == 4
    assert {r["spec"] for r in rows} == set(PHASE_JOINT_SPECS)
    assert all("design_cond" in r for r in rows)
    assert all("n_postlift_obs" in r for r in rows)


def test_phase_joint_sample_matches_full_ban_n_obs() -> None:
    """Function summary: joint model uses same row count as full_ban sample."""
    panel = _synthetic_panel()
    strat_full = StrategySpec("cross_country_all", post_mode="full_ban")
    strat_joint = StrategySpec("cross_country_all", post_mode="phase_joint")
    full_sample = filter_strategy_sample(panel, strat_full)
    joint_rows = run_strategy_phase_joint(panel, strat_joint, "pole_share")
    direct_rows = estimate_twfe_phase_joint(full_sample, "pole_share")
    assert joint_rows[0]["n_obs"] == direct_rows[0]["n_obs"]
    assert joint_rows[0]["n_obs"] == len(full_sample)


def test_lift_degeneracy_thin_italy_only() -> None:
    """Function summary: phase_joint_lift can mark degenerate_collinear_lift on thin panels."""
    panel = _synthetic_panel()
    panel = panel[panel["topic_family"].astype(str).isin(["it_political"])]
    strat = StrategySpec("italy_only_post", post_mode="phase_joint")
    rows = run_strategy_phase_joint(panel, strat, "pole_share")
    lift = [r for r in rows if r["spec"] == "phase_joint_lift"][0]
    assert lift["estimation_note"] in (
        "degenerate_collinear_lift",
        "ok_entity_fe_only",
        "ok",
        "fully_absorbed",
        "insufficient_obs_or_clusters",
    )
