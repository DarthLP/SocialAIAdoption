"""Tests for weighted ai_style_rate diagnosis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis.diagnose_ai_style_weighted import (
    _attenuation_pct,
    _exbantopic_verdict,
    _post_phase_verdict,
    _top_italian_forums,
    build_diagnosis_table,
)


def test_attenuation_pct_positive_baseline() -> None:
    """Function summary: attenuation is (base−variant)/base×100."""
    assert _attenuation_pct(0.0045, 0.003) == pytest.approx(100 * (0.0045 - 0.003) / 0.0045)


def test_exbantopic_verdict_threshold() -> None:
    """Function summary: verdict when attenuation exceeds 30%."""
    assert _exbantopic_verdict(35.0) == "Exbantopic panel attenuates weighted full_ban by >30%"
    assert _exbantopic_verdict(10.0) == ""


def test_post_phase_verdict_long_tail() -> None:
    """Function summary: positive full_ban with largest beta in post_long_tail."""
    phases = [
        {"spec": "post_short_3d", "beta": -0.001},
        {"spec": "post_medium_7d", "beta": 0.0005},
        {"spec": "post_long_tail", "beta": 0.005},
        {"spec": "post_first_2bd", "beta": 0.001},
    ]
    v = _post_phase_verdict(phases, full_ban_beta=0.0045)
    assert "post_long_tail" in v


def test_top_italian_forums_orders_by_n_comments() -> None:
    """Function summary: LOO forum list is data-driven from IT topic_family weights."""
    panel = pd.DataFrame(
        {
            "subreddit": ["italy", "italy", "Italia", "small_it"],
            "topic_family": ["it_political", "it_political", "it_others", "it_others"],
            "n_comments": [100, 50, 80, 5],
        }
    )
    top = _top_italian_forums(panel, n=2)
    assert top[0] == "italy"
    assert top[1] == "Italia"


def test_build_diagnosis_table_schema(monkeypatch) -> None:
    """Function summary: build_diagnosis_table returns expected columns when estimation is mocked."""

    def _fake_estimate(panel, spec, *, entity_col=None):
        beta_map = {
            "full_ban": 0.0045,
            "post_short_3d": -0.001,
            "post_medium_7d": 0.0005,
            "post_long_tail": 0.005,
            "post_first_2bd": 0.001,
        }
        return {
            "beta": beta_map.get(spec, float("nan")),
            "pvalue": 0.05,
            "n_obs": len(panel),
            "estimation_note": "ok",
        }

    def _fake_top_forums(panel, n=3):
        return ["italy", "Italia", "ItaliaPersonalFinance"]

    import scripts.analysis.diagnose_ai_style_weighted as mod

    monkeypatch.setattr(mod, "_estimate", _fake_estimate)
    monkeypatch.setattr(mod, "_top_italian_forums", _fake_top_forums)
    monkeypatch.setattr(
        mod,
        "build_analysis_panels",
        lambda config, families=None, variant=None, **kwargs: type(
            "P",
            (),
            {
                "sub_v1": pd.DataFrame(
                    {
                        "subreddit": ["italy", "de_news"],
                        "ai_style_rate_100w_mean": [0.01, 0.02],
                        "n_comments": [100, 80],
                    }
                )
            },
        )(),
    )
    monkeypatch.setattr(mod, "activate_post_phases_from_config", lambda c: None)

    df = build_diagnosis_table({})
    expected_cols = {
        "check_type",
        "spec",
        "dropped_forum",
        "beta",
        "pvalue",
        "n_obs",
        "baseline_beta",
        "attenuation_pct",
        "verdict",
    }
    assert expected_cols <= set(df.columns)
    assert set(df["check_type"]) >= {"post_phase", "post_phase_summary", "exbantopic", "loo"}
    assert len(df[df["check_type"] == "post_phase"]) == 4
    assert len(df[df["check_type"] == "loo"]) == 3

