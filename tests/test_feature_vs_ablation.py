"""Unit tests for ρ(feature, index_without_feature) diagnostics."""

from __future__ import annotations

import pandas as pd

from src.style_index_ablation import feature_rate_vs_own_ablation_rows
from src.style_index_llm import PRIMARY_COL, ablation_column_name


def test_own_ablation_lower_rho_than_primary_for_semicolon() -> None:
    """Function summary: removing semicolon from index decouples index from semicolon rate."""
    n = 300
    semi = [float(i % 7) for i in range(n)]
    primary = [semi[i] * 2.0 + float(i) * 0.01 for i in range(n)]
    no_semi = [float(i) * 0.01 for i in range(n)]
    df = pd.DataFrame(
        {
            PRIMARY_COL: primary,
            "semicolon_colon_rate_100w": semi,
            ablation_column_name("semicolon_colon_rate_100w"): no_semi,
        }
    )
    rows = feature_rate_vs_own_ablation_rows(df)
    row = [r for r in rows if r["feature"] == "semicolon_colon_rate_100w"][0]
    assert row["rho_feature_vs_primary"] > row["rho_feature_vs_own_ablation"]
