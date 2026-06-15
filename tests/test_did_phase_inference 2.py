"""Tests for per-phase placebo-in-space on joint phase model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.did.inference import placebo_in_space_phase_joint_all
from src.did.specs import PHASE_JOINT_SPECS, StrategySpec


def _panel_with_phase_signal() -> pd.DataFrame:
    """Function summary: cross-country panel with distinct phase betas."""
    rows = []
    for ent, treat, fam in [
        ("it_a", 1, "it_political"),
        ("it_b", 1, "it_others"),
        ("de_a", 0, "de"),
        ("eu_a", 0, "eu"),
        ("us_a", 0, "us"),
    ]:
        for rd in range(-8, 31):
            phase_boost = 0.0
            if treat and 0 <= rd <= 2:
                phase_boost = 0.05
            elif treat and 3 <= rd <= 9:
                phase_boost = 0.02
            y = 0.5 + phase_boost + 0.001 * rd
            rows.append(
                {
                    "entity_id": ent,
                    "subreddit": ent,
                    "time_id": f"2023-04-{min(28, max(1, rd + 1)):02d}",
                    "topic_family": fam,
                    "language_hub": fam if fam in ("de", "eu", "us") else "it",
                    "IT": int(treat),
                    "treat": int(treat),
                    "rel_day": rd,
                    "post": int(rd >= 0),
                    "sem_axis_emotion": y,
                }
            )
    return pd.DataFrame(rows)


def test_placebo_phase_joint_all_returns_four() -> None:
    """Function summary: placebo_in_space_phase_joint_all returns one result per phase spec."""
    panel = _panel_with_phase_signal()
    strat = StrategySpec("cross_country_all", post_mode="phase_joint")
    results = placebo_in_space_phase_joint_all(panel, strat, "sem_axis_emotion")
    assert set(results.keys()) == set(PHASE_JOINT_SPECS)


def _panel_four_controls(seed: int = 11) -> pd.DataFrame:
    """Function summary: IT + four control families with noise, full lift coverage."""
    rng = np.random.default_rng(seed)
    rows = []
    launch = pd.Timestamp("2023-03-31")
    for ent, treat, fam in [
        ("it_a", 1, "it_political"),
        ("it_b", 1, "it_others"),
        ("de_a", 0, "de"),
        ("eu_a", 0, "eu"),
        ("us_a", 0, "us"),
        ("uk_a", 0, "uk"),
    ]:
        for rd in range(-20, 33):
            boost = 0.05 if (treat and 0 <= rd <= 2) else (0.02 if (treat and 3 <= rd <= 9) else 0.0)
            rows.append(
                {
                    "entity_id": ent,
                    "subreddit": ent,
                    "time_id": (launch + pd.Timedelta(days=rd)).strftime("%Y-%m-%d"),
                    "date_utc": (launch + pd.Timedelta(days=rd)).strftime("%Y-%m-%d"),
                    "topic_family": fam,
                    "IT": int(treat),
                    "treat": int(treat),
                    "rel_day": rd,
                    "post": int(rd >= 0),
                    "y": 0.5 + boost + 0.001 * rd + float(rng.normal(scale=0.01)),
                }
            )
    return pd.DataFrame(rows)


def test_placebo_phase_joint_p_finite_with_four_rotations() -> None:
    """Function summary: each phase gets a finite p from 4 control rotations (floor 0.2).

    Regression test for the treat-wipe bug: run_strategy_phase_joint re-filtered the
    placebo sample, recomputing treat from IT and erasing the rotation, so every
    cross-country phase_joint row carried NaN p_placebo_space.
    """
    panel = _panel_four_controls()
    strat = StrategySpec("cross_country_all", post_mode="phase_joint")
    results = placebo_in_space_phase_joint_all(
        panel, strat, "y", entity_col="subreddit", time_col="time_id"
    )
    assert set(results.keys()) == set(PHASE_JOINT_SPECS)
    for spec in PHASE_JOINT_SPECS:
        pis = results[spec]
        assert pis.n_placebo_draws == 4, spec
        assert pis.p_floor == 0.2, spec
        assert len(pis.placebo_betas) == 4, f"{spec}: rotations lost (treat-wipe regression)"
        assert np.isfinite(pis.p), f"{spec}: p is NaN"
        assert 0.2 <= pis.p <= 1.0, spec
