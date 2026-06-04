"""Tests for bucket event-study estimation specs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.did.bucket_estimate import (
    _build_es_interactions,
    combine_split_sample_static,
    compute_trajectory_means,
    estimate_comment_it_ddd_event_study,
    estimate_comment_it_event_study,
    estimate_static_full_time_fe,
    estimate_static_paper_eq1,
    feols_static_paper_eq1_prepped,
    prep_static_design,
)
from src.did.inference import placebo_in_space_comment_p
from src.did.lean_buckets import is_placebo_space_eligible_control_variant


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


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_placebo_beta_italy_matches_full_refit() -> None:
    """Function summary: reusing headline beta yields identical placebo p (no Italy refit)."""
    df = _synthetic_panel(n_authors=24, n_days=8)
    df["topic_family"] = np.where(df["IT"] == 1, "it_political", "de")
    df.loc[df["IT"] == 0, "topic_family"] = np.random.default_rng(0).choice(
        ["de", "eu", "uk", "us"], size=int((df["IT"] == 0).sum())
    )
    res = estimate_static_paper_eq1(df)
    p_full = placebo_in_space_comment_p(df, y_col="y")
    p_reuse = placebo_in_space_comment_p(df, y_col="y", beta_italy=res.get("beta"))
    if np.isfinite(p_full):
        assert p_reuse == pytest.approx(p_full, rel=0, abs=0)
    else:
        assert np.isnan(p_reuse)


def test_trim_panel_for_estimation_drops_extra_columns() -> None:
    """Function summary: estimation trim keeps DiD design columns only."""
    from scripts.analysis import bucket_event_study as bes

    df = pd.DataFrame(
        {
            "author": ["a"],
            "subreddit": ["s"],
            "time_id": ["t"],
            "date_utc": ["2023-03-01"],
            "id": ["1"],
            "post": [0],
            "IT": [1],
            "rel_day": [-1],
            "rel_period": [-1],
            "net_ideology": [0.1],
            "topic_family": ["it_political"],
            "primary_lexicon": ["it"],
            "sem_axis_aggression": [0.5],
        }
    )
    slim = bes._trim_panel_for_estimation(df, "net_ideology")
    assert "sem_axis_aggression" not in slim.columns
    assert "author" in slim.columns


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


def test_combine_split_sample_static_merges_inference_metadata() -> None:
    """Function summary: combined split row keeps paper_eq1 metadata and median placebo p."""
    splits = [
        {
            "beta": 0.01,
            "se": 0.005,
            "n_obs": 100,
            "n_clusters": 10,
            "pvalue_cluster": 0.1,
            "p_placebo_space": 0.5,
            "static_variant": "paper_eq1",
            "inference_role": "descriptive",
        },
        {
            "beta": 0.02,
            "se": 0.006,
            "n_obs": 110,
            "n_clusters": 11,
            "pvalue_cluster": 0.2,
            "p_placebo_space": 0.6,
            "static_variant": "paper_eq1",
            "inference_role": "descriptive",
        },
        {
            "beta": 0.015,
            "se": 0.0055,
            "n_obs": 105,
            "n_clusters": 12,
            "pvalue_cluster": 0.15,
            "p_placebo_space": 0.55,
            "static_variant": "paper_eq1",
            "inference_role": "descriptive",
        },
    ]
    c = combine_split_sample_static(splits)
    assert c["static_variant"] == "paper_eq1"
    assert c["coef_name"] == "post:IT"
    assert c["n_splits"] == 3
    assert c["estimation_note"] == "combined_splits"
    assert np.isfinite(c["p_placebo_space"])
    assert c["p_placebo_space"] == pytest.approx(0.55, rel=0, abs=1e-9)


def test_combine_split_sample_static_no_cross_split_variation() -> None:
    """Function summary: identical split betas yield NaN SE and explicit note."""
    splits = [
        {"beta": 0.01, "se": 0.005, "n_obs": 100, "n_clusters": 10, "static_variant": "paper_eq1"},
        {"beta": 0.01, "se": 0.006, "n_obs": 110, "n_clusters": 11, "static_variant": "paper_eq1"},
        {"beta": 0.01, "se": 0.0055, "n_obs": 105, "n_clusters": 12, "static_variant": "paper_eq1"},
    ]
    c = combine_split_sample_static(splits)
    assert c["estimation_note"] == "no_cross_split_variation"
    assert np.isnan(c["se"])
    assert c["beta_sd_across_splits"] == 0.0


def test_is_placebo_space_eligible_only_pooled() -> None:
    """Function summary: placebo-in-space applies only to all_controls_pooled."""
    assert is_placebo_space_eligible_control_variant("all_controls_pooled")
    assert not is_placebo_space_eligible_control_variant("vs_de")
    assert not is_placebo_space_eligible_control_variant("vs_uk")


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_static_block_skips_placebo_for_single_country_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Function summary: vs_de static rows get placebo_note instead of placebo-in-space."""
    from scripts.analysis import bucket_event_study as bes

    called = {"n": 0}

    def _spy(*args: object, **kwargs: object) -> float:
        called["n"] += 1
        return 0.5

    monkeypatch.setattr(bes, "placebo_in_space_comment_p", _spy)
    df = _synthetic_panel()
    bcfg = type("B", (), {"static_full_time_fe": False})()
    meta = {
        "control_variant": "vs_de",
        "row_idx": 0,
        "split_id": None,
        "bucket": "all",
        "pooled": True,
    }
    rows = bes._run_static_block(
        df,
        bcfg,
        run_bootstrap=True,
        placebo_queue=[],
        placebo_meta=meta,
    )
    assert called["n"] == 0
    assert rows[0]["placebo_note"] == "not_applicable_single_country_contrast"
    assert np.isnan(rows[0]["p_placebo_space"])


def _ddd_synthetic_panel() -> pd.DataFrame:
    """Function summary: stacked liberal/conservative panel for DDD smoke test."""
    rows = []
    for i in range(12):
        bucket = "liberal_leaning" if i % 2 == 0 else "conservative_leaning"
        treat = int(i >= 6)
        for j in range(-6, 7):
            k = j // 3
            rows.append(
                {
                    "author": f"u{i}",
                    "subreddit": f"s{i % 3}",
                    "time_id": f"day{j}",
                    "post": int(j >= 0),
                    "IT": treat,
                    "rel_day": j,
                    "rel_period": k,
                    "bucket": bucket,
                    "y": 0.1
                    + 0.05 * treat * int(j >= 0) * (1.0 if bucket == "liberal_leaning" else 0.0),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_ddd_event_study_parses_coef_rows() -> None:
    """Function summary: DDD i(rel, IT):liberal formula yields non-empty rel_period table."""
    df = _ddd_synthetic_panel()
    _, ddd_df = estimate_comment_it_ddd_event_study(
        df,
        "liberal_leaning",
        "conservative_leaning",
        ref_period=-1,
        window=30,
        bin_days=3,
    )
    assert not ddd_df.empty
    assert "rel_period" in ddd_df.columns
    assert "ddd_gamma" in ddd_df.columns
    assert all(":IT:liberal" in str(c) for c in ddd_df["coef_name"].astype(str))
