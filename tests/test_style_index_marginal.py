"""Unit tests for marginal (single-feature) LLM index validation."""

from __future__ import annotations

import pandas as pd

from src.style_index_ablation import marginal_influence_rows
from src.style_index_llm import PRIMARY_COL, only_column_name


def test_marginal_rows_include_only_feature_correlation() -> None:
    """Function summary: marginal report links feature-only index to composite."""
    n = 200
    df = pd.DataFrame(
        {
            PRIMARY_COL: [float(i) for i in range(n)],
            "ai_style_rate_100w": [float(i) * 0.1 for i in range(n)],
            "em_dash_rate_100w": [1.0 if i % 5 == 0 else 0.0 for i in range(n)],
            only_column_name("em_dash_rate_100w"): [float(i) * 0.5 for i in range(n)],
            "em_dash_hit": [1 if i % 5 == 0 else 0 for i in range(n)],
            "ai_hit": [1 if i % 2 == 0 else 0 for i in range(n)],
        }
    )
    rows = marginal_influence_rows(df)
    em = [r for r in rows if r["feature"] == "em_dash_rate_100w"]
    assert len(em) == 1
    assert em[0]["rho_only_feature_vs_primary"] > 0.3
