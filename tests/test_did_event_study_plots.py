"""Tests for event-study plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.did.estimate import estimate_event_study
from src.did.outputs import (
    EventStudySeries,
    _prepare_event_study_plot_df,
    plot_event_study,
    plot_event_study_overlay,
)


def test_prepare_event_study_adds_ref_period() -> None:
    """Function summary: reference event time is appended at zero."""
    es = pd.DataFrame(
        {
            "rel_day": [-2, 0, 1],
            "gamma": [0.1, 0.4, 0.5],
            "se": [0.05, 0.05, 0.05],
        }
    )
    out = _prepare_event_study_plot_df(es, "rel_day", ref_time=-1)
    assert -1 in set(out["event_time"].astype(int))
    ref = out[out["event_time"].astype(int) == -1].iloc[0]
    assert ref["gamma"] == 0.0


def test_plot_event_study_smoke(tmp_path: Path) -> None:
    """Function summary: single-series plot writes a PNG."""
    es = pd.DataFrame(
        {
            "rel_day": [-2, 0, 1],
            "gamma": [0.0, 0.4, 0.5],
            "se": [0.05, 0.05, 0.05],
        }
    )
    out = tmp_path / "es.png"
    plot_event_study(es, "net_ideology", out, rel_col="rel_day")
    assert out.is_file()


def test_event_study_time_fe_fallback_two_entities() -> None:
    """Function summary: few entities use calendar time FE when TWFE is rank-deficient."""
    rows = []
    for ent, treat in [("it", 1.0), ("de", 0.0)]:
        for k in range(-3, 4):
            rows.append(
                {
                    "entity_id": ent,
                    "time_id": f"2023-03-{25 + k}",
                    "rel_day": k,
                    "treat": treat,
                    "y": 0.1 * treat + 0.02 * k,
                }
            )
    summ, es = estimate_event_study(pd.DataFrame(rows), "y", window=3)
    assert summ["estimation_note"] in ("ok", "ok_time_fe_only")
    assert len(es) >= 3


def test_plot_event_study_overlay_smoke(tmp_path: Path) -> None:
    """Function summary: overlay plot writes a PNG with two series."""
    def _mk(beta: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rel_day": [-2, 0, 1],
                "gamma": [0.0, beta, beta + 0.05],
                "se": [0.05, 0.05, 0.05],
            }
        )

    series = [
        EventStudySeries("IT vs pooled", _mk(0.4), rel_col="rel_day"),
        EventStudySeries("IT vs DE", _mk(0.35), rel_col="rel_day"),
    ]
    out = tmp_path / "overlay.png"
    plot_event_study_overlay(series, "net_ideology", out)
    assert out.is_file()
