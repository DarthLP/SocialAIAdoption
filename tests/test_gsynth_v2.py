"""Tests for gsynth v2 demeaning, pre-fit gate, and sign gate logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.did.gsynth import (
    GSYNTH_V2_OUTCOMES,
    GsynthV2Result,
    _attach_gate_columns,
    _augmented_sc_att_v2,
    _demean_by_pre_period_mean,
    _evaluate_pre_fit_gate,
    load_did_summary_beta,
    pole_share_sign_gate_should_abort,
    write_gsynth_v2_outputs,
)


def _synthetic_wide() -> pd.DataFrame:
    """Function summary: build toy country-day wide matrix straddling ban date."""
    times = [f"2023-03-{d:02d}" for d in range(1, 16)] + [f"2023-04-{d:02d}" for d in range(1, 8)]
    rng = np.random.default_rng(42)
    data = {
        "it": 0.5 + rng.normal(0, 0.05, len(times)),
        "de": 0.3 + rng.normal(0, 0.04, len(times)),
        "eu": 0.28 + rng.normal(0, 0.04, len(times)),
        "uk": 0.32 + rng.normal(0, 0.04, len(times)),
        "us": 0.31 + rng.normal(0, 0.04, len(times)),
    }
    return pd.DataFrame(data, index=times)


def test_demean_pre_period_zero_mean() -> None:
    """Function summary: pre-ban demeaned series has ~zero mean per unit."""
    wide = _synthetic_wide()
    launch = "2023-03-31"
    demeaned, _ = _demean_by_pre_period_mean(wide, launch)
    pre = demeaned.loc[demeaned.index.astype(str) < launch[:10]]
    for col in demeaned.columns:
        assert abs(float(pre[col].mean())) < 1e-10


def test_augmented_sc_v2_launch_aligned_post_att() -> None:
    """Function summary: v2 SC uses launch date for post ATT, not positional half."""
    wide = _synthetic_wide()
    launch = "2023-03-31"
    demeaned, _ = _demean_by_pre_period_mean(wide, launch)
    att_df, avg_post = _augmented_sc_att_v2(demeaned, launch)
    post_manual = att_df[att_df["period_start"].astype(str) >= launch[:10]]["att"].mean()
    assert np.isfinite(avg_post)
    assert abs(float(avg_post) - float(post_manual)) < 1e-9
    assert "y_treated" in att_df.columns
    assert "y_synth" in att_df.columns


def test_pre_fit_gate_pass_and_fail() -> None:
    """Function summary: gate passes when pre ATT small vs post; fails when pre RMSE large."""
    launch = "2023-03-31"
    wide = _synthetic_wide()
    demeaned, _ = _demean_by_pre_period_mean(wide, launch)

    pass_att = pd.DataFrame(
        {
            "period_start": demeaned.index.astype(str),
            "att": np.where(
                demeaned.index.astype(str) >= launch[:10],
                0.4,
                0.01,
            ),
        }
    )
    gate_pass = _evaluate_pre_fit_gate(pass_att, demeaned, launch)
    assert gate_pass["pre_fit_ok"] is True
    assert gate_pass["verdict"] == "ok"

    fail_att = pd.DataFrame(
        {
            "period_start": demeaned.index.astype(str),
            "att": np.where(
                demeaned.index.astype(str) >= launch[:10],
                0.1,
                0.5,
            ),
        }
    )
    gate_fail = _evaluate_pre_fit_gate(fail_att, demeaned, launch)
    assert gate_fail["pre_fit_ok"] is False
    assert gate_fail["verdict"] == "failed_pre_fit_do_not_cite"


def test_attach_gate_columns_on_every_row() -> None:
    """Function summary: ATT export repeats pre_fit_ok and verdict on all rows."""
    att = pd.DataFrame({"period_start": ["2023-03-01"], "att": [0.0]})
    gate = {
        "pre_fit_ok": False,
        "verdict": "failed_pre_fit_do_not_cite",
        "mean_pre_att": 0.1,
        "mean_post_att": 0.2,
        "pre_rmse": 0.15,
        "italy_pre_sd": 0.05,
    }
    out = _attach_gate_columns(att, gate)
    assert all(out["pre_fit_ok"] == False)  # noqa: E712
    assert (out["verdict"] == "failed_pre_fit_do_not_cite").all()


def test_load_did_summary_beta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Function summary: reads beta from fixture did_summary.csv."""
    summary = tmp_path / "did_summary.csv"
    pd.DataFrame(
        [
            {
                "outcome_id": "pole_share",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": 0.062304,
            }
        ]
    ).to_csv(summary, index=False)
    config = {"paths": {"tables_dir": str(tmp_path)}}

    from src.did import gsynth as gsynth_mod

    def _fake_summary_paths(cfg, **kwargs):
        return summary, summary

    monkeypatch.setattr(gsynth_mod, "did_summary_paths", _fake_summary_paths)
    beta = load_did_summary_beta(config, "pole_share", "cross_country_all", "full_ban")
    assert abs(beta - 0.062304) < 1e-9


def test_pole_share_sign_gate_scoped_abort() -> None:
    """Function summary: abort only when pre_fit_ok and sign mismatch."""
    assert pole_share_sign_gate_should_abort(True, 0.05, 0.062304) is False
    assert pole_share_sign_gate_should_abort(True, -0.05, 0.062304) is True
    assert pole_share_sign_gate_should_abort(False, -0.05, 0.062304) is False


def test_write_gsynth_v2_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Function summary: write_gsynth_v2_outputs creates att and inference CSVs."""
    from src.did import gsynth as gsynth_mod

    estimates = tmp_path / "estimates" / "gsynth_v2"
    monkeypatch.setattr(gsynth_mod, "did_gsynth_v2_dir", lambda cfg: estimates)
    monkeypatch.setattr(
        gsynth_mod,
        "did_gsynth_v2_att_path",
        lambda cfg, oid, bd: estimates / f"att_{oid}_{bd}d.csv",
    )
    monkeypatch.setattr(
        gsynth_mod,
        "did_gsynth_v2_inference_path",
        lambda cfg, oid, bd: estimates / f"inference_{oid}_{bd}d.csv",
    )

    att = pd.DataFrame({"period_start": ["2023-03-01"], "att": [0.0], "y_treated": [0.1], "y_synth": [0.1]})
    gate = {
        "pre_fit_ok": True,
        "verdict": "ok",
        "mean_pre_att": 0.0,
        "mean_post_att": 0.1,
        "pre_rmse": 0.01,
        "italy_pre_sd": 0.05,
    }
    result = GsynthV2Result(
        att=att,
        inference={"outcome_id": "pole_share", "placebo_p_floor": 0.2},
        backend="augmented_sc",
        gate=gate,
    )
    write_gsynth_v2_outputs({}, result, "pole_share", 3, sign_meta={"did_summary_beta": 0.062304})
    att_out = pd.read_csv(estimates / "att_pole_share_3d.csv")
    assert "pre_fit_ok" in att_out.columns
    assert bool(att_out["pre_fit_ok"].iloc[0]) is True
    inf_out = pd.read_csv(estimates / "inference_pole_share_3d.csv")
    assert float(inf_out["did_summary_beta"].iloc[0]) == pytest.approx(0.062304)


def test_gsynth_v2_outcome_set() -> None:
    """Function summary: fixed v2 outcome list has six entries including pole_share."""
    assert len(GSYNTH_V2_OUTCOMES) == 6
    assert "pole_share" in GSYNTH_V2_OUTCOMES
    assert "ai_style_rate" in GSYNTH_V2_OUTCOMES
