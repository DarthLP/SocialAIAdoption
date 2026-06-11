"""Smoke tests for post-phase headline DiD coefplots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.did.outputs import plot_coef_post_phases, plot_post_phase_comparison
from src.did.specs import PHASE_JOINT_SPECS


def test_plot_coef_post_phases_writes_png(tmp_path: Path) -> None:
    """Function summary: phased coefplot writes a PNG when summary rows exist."""
    rows: list[dict] = []
    for sid in ("cross_country_all", "cross_country_it_political"):
        for spec in PHASE_JOINT_SPECS:
            rows.append(
                {
                    "outcome_id": "net_ideology",
                    "outcome_family": "lexical",
                    "strategy_id": sid,
                    "spec": spec,
                    "beta": 0.1,
                    "se": 0.05,
                    "pvalue": 0.2,
                }
            )
    summary = pd.DataFrame(rows)
    out = tmp_path / "phases.png"
    plot_coef_post_phases(
        summary,
        "net_ideology",
        out,
        strategies=["cross_country_all", "cross_country_it_political"],
    )
    assert out.is_file()
    assert out.stat().st_size > 100


def test_plot_post_phase_comparison_writes_png(tmp_path: Path) -> None:
    """Function summary: overview-style post-phase bar chart writes PNG."""
    rows = []
    for i, spec in enumerate(PHASE_JOINT_SPECS):
        rows.append(
            {
                "outcome_id": "net_ideology",
                "strategy_id": "cross_country_all",
                "spec": spec,
                "beta": 0.05 * (i + 1),
                "se": 0.02,
                "pvalue": 0.1,
            }
        )
    summary = pd.DataFrame(rows)
    out = tmp_path / "post_phase_bars.png"
    plot_post_phase_comparison(summary, "net_ideology", out)
    assert out.is_file()
    assert out.stat().st_size > 100
