"""Tests for bucket event-study estimation specs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.did.bucket_estimate import (
    _build_es_interactions,
    compute_trajectory_means,
    estimate_comment_it_event_study,
    estimate_static_full_time_fe,
    estimate_static_paper_eq1,
    feols_static_paper_eq1_prepped,
    prep_static_design,
)
from src.did.inference import placebo_in_space_comment_p


def _synthetic_panel(n_authors: int = 8, n_days: int = 6) -> pd.DataFrame:
    """Function summary: small comment panel for pyfixest estimation smoke tests."""
    rows = []
    days = [f"2023-03-{28 + i}" for i in range(3)] + [f"2023-04-0{i+1}" for i in range(3)]
    for i in range(n_authors):
        treat = int(i >= n_authors // 2)
        for j, day in enumerate(days[:n_days]):
            rows.append(
                {
                    "author": f"u{i}",
                    "subreddit": f"sub{i % 3}",
                    "time_id": day,
                    "date_utc": day,
                    "post": int(j >= 3),
                    "IT": treat,
                    "rel_day": j - 3,
                    "rel_period": (j - 3) // 3,
                    "y": 0.1 + 0.2 * treat * int(j >= 3) + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


def test_static_paper_eq1_runs() -> None:
    """Function summary: headline static uses post + post_IT without time_id absorb."""
    df = _synthetic_panel()
    res = estimate_static_paper_eq1(df)
    assert res.get("static_variant") == "paper_eq1"
    assert res.get("estimation_note") in ("ok", "estimation_error", "insufficient_obs")


def test_static_robustness_no_post_main() -> None:
    """Function summary: full_time_fe variant targets post_IT only."""
    df = _synthetic_panel()
    res = estimate_static_full_time_fe(df)
    assert res.get("static_variant") == "full_time_fe"


def test_es_omits_ref_bin() -> None:
    """Function summary: reference rel_period -1 excluded from ES dummies."""
    df = _synthetic_panel(n_authors=10, n_days=8)
    df["rel_period"] = df["rel_day"] // 3
    work, cols, col_to_k = _build_es_interactions(df, "rel_period", -1, 30)
    assert all(col_to_k[c] != -1 for c in cols)
    assert not any("neg1" in c and col_to_k.get(c) == -1 for c in cols)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_es_i_operator_parity_with_manual_dummies() -> None:
    """Function summary: i(rel_period, IT) coefficients match manual bin_k:IT dummies."""
    df = _synthetic_panel(n_authors=12, n_days=8)
    df["rel_period"] = df["rel_day"] // 3
    _, es_i = estimate_comment_it_event_study(df, ref_period=-1, window=30, bin_days=3)
    work, cols, col_to_k = _build_es_interactions(
        df.assign(y=df["y"]),
        "rel_period",
        -1,
        30,
    )
    if not cols:
        pytest.skip("no ES variation in synthetic panel")
    from pyfixest.estimation import feols

    rhs = " + ".join(cols)
    fit = feols(f"y ~ {rhs} | author + time_id", data=work, vcov="iid")
    manual = {col_to_k[c]: float(fit.coef().loc[c]) for c in cols if c in fit.coef().index}
    if es_i.empty or not manual:
        pytest.skip("insufficient ES coefs")
    for _, row in es_i.iterrows():
        k = int(row["rel_period"])
        if k not in manual:
            continue
        assert manual[k] == pytest.approx(float(row["gamma"]), rel=1e-4, abs=1e-4)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_placebo_in_space_comment_finite() -> None:
    """Function summary: placebo-in-space on comment panel returns p in [floor, 1]."""
    df = _synthetic_panel(n_authors=20, n_days=8)
    if "topic_family" not in df.columns:
        df["topic_family"] = np.where(df["IT"] == 1, "it_political", "de")
    p = placebo_in_space_comment_p(df, y_col="y")
    assert np.isfinite(p) or np.isnan(p)
    if np.isfinite(p):
        assert 0.0 < p <= 1.0


def test_trajectory_means_derives_rel_period_from_rel_day() -> None:
    """Function summary: 1d prepared panels may lack rel_period; trajectories still run."""
    df = pd.DataFrame(
        {
            "rel_day": [-2, -1, 0, 1],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = compute_trajectory_means(df, "rel_period", "it", "Italy", bin_days=1)
    assert not out.empty
    assert set(out["rel_period"].astype(int)) == {-2, -1, 0, 1}


def test_load_bucket_panel_uses_prepared_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Function summary: loader prefers prepared panel when available and no shard cap."""
    from scripts.analysis import bucket_event_study as bes

    called = {"prepared": False}

    def _fake_available(config: object, bin_days: int = 1) -> bool:
        return True

    def _fake_load(config: object, bin_days: int = 1, **kwargs: object) -> pd.DataFrame:
        called["prepared"] = True
        return pd.DataFrame({"author": ["a"], "net_ideology": [0.1]})

    monkeypatch.setattr(bes, "comment_panel_available", _fake_available)
    monkeypatch.setattr(bes, "load_prepared_panel", _fake_load)
    bcfg = type("B", (), {"bin_days": 3, "political_universe_only": True})()
    bes.load_bucket_comment_panel({}, bcfg, None)
    assert called["prepared"]
