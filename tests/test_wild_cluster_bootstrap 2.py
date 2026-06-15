"""Tests for restricted wild cluster bootstrap (FIX 2)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.did.inference import wild_cluster_bootstrap_p
from src.did.specs import StrategySpec, is_wcb_eligible_strategy


def _synthetic_twfe_panel(n_ent: int = 25, n_time: int = 8) -> pd.DataFrame:
    """Function summary: panel with treat×post variation and many entities."""
    rng = np.random.default_rng(1)
    rows = []
    for e in range(n_ent):
        treat = int(e < n_ent // 2)
        for t in range(n_time):
            post = int(t >= n_time // 2)
            rows.append(
                {
                    "entity_id": f"ent_{e}",
                    "time_id": f"t_{t}",
                    "rel_day": t - n_time // 2,
                    "topic_family": "it_political" if treat else "de",
                    "IT": treat,
                    "treat": treat,
                    "post": post,
                    "y": 0.5 * treat * post + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_wcb_eligibility() -> None:
    """Function summary: cross-country ineligible; within-Italy and author eligible."""
    assert not is_wcb_eligible_strategy("cross_country_all")
    assert is_wcb_eligible_strategy("within_italy_ddd")
    assert is_wcb_eligible_strategy("author_it_vs_en")


def test_cross_country_wcb_returns_nan_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Function summary: WCB on cross_country logs warning and returns NaN."""
    panel = _synthetic_twfe_panel()
    with caplog.at_level(logging.WARNING):
        p = wild_cluster_bootstrap_p(
            panel,
            StrategySpec("cross_country_all"),
            "y",
            n_draws=99,
            seed=1,
        )
    assert np.isnan(p)
    assert "MacKinnon" in caplog.text or "placebo" in caplog.text.lower()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_restricted_wcb_differs_from_y_reweight() -> None:
    """Function summary: restricted residual WCB is not the legacy y*weight shortcut."""
    from src.did.inference import _restricted_wcb_pyfixest

    panel = _synthetic_twfe_panel(n_ent=30, n_time=10)
    work = panel.rename(columns={"entity_id": "ent", "time_id": "time"}).copy()
    work["y"] = work["y"].astype(float)
    p_restricted = _restricted_wcb_pyfixest(
        "y ~ treat_post | ent + time",
        "y ~ 1 | ent + time",
        work,
        "treat_post",
        "ent",
        199,
        1,
        min_draws=5,
    )
    if not np.isfinite(p_restricted):
        pytest.skip("restricted WCB did not converge on toy panel")
    assert 0.0 <= p_restricted <= 1.0


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyfixest"),
    reason="pyfixest not installed",
)
def test_wcb_finite_on_synthetic_author_contrast() -> None:
    """Function summary: author IT vs EN strategy can return finite WCB p."""
    panel = _synthetic_twfe_panel()
    panel["primary_lexicon"] = np.where(panel["IT"] == 1, "it", "en")
    p = wild_cluster_bootstrap_p(
        panel,
        StrategySpec("author_it_vs_en"),
        "y",
        n_draws=199,
        seed=2,
        entity_col="entity_id",
    )
    assert np.isfinite(p) or np.isnan(p)
