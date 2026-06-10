"""Tests for scan-wide multiple-testing audit export."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_mod():
    """Function summary: load export_scan_audit module for unit tests."""
    path = Path(__file__).resolve().parent.parent / "scripts/diagnostics/export_scan_audit.py"
    spec = importlib.util.spec_from_file_location("export_scan_audit_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_raw_df() -> pd.DataFrame:
    """Function summary: build synthetic did_summary-like frame for audit tests."""
    return pd.DataFrame(
        [
            {
                "outcome_id": "ai_style_rate",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": 0.05,
                "pvalue": 0.001,
                "estimation_note": "ok",
            },
            {
                "outcome_id": "pole_share",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": 0.06,
                "pvalue": 0.003,
                "estimation_note": "ok",
            },
            {
                "outcome_id": "sem_axis_emotion",
                "outcome_family": "semantic_axis",
                "strategy_id": "cross_country_all",
                "spec": "early_ban_7d",
                "beta": -0.01,
                "pvalue": 0.005,
                "estimation_note": "ok",
            },
            {
                "outcome_id": "sem_axis_emotion",
                "outcome_family": "semantic_axis",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": -0.003,
                "pvalue": 0.20,
                "estimation_note": "ok",
            },
            {
                "outcome_id": "ambivalence",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_vs_de",
                "spec": "full_ban",
                "beta": 0.02,
                "pvalue": 0.00001,
                "estimation_note": "ok",
            },
            {
                "outcome_id": "bad_outcome",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": float("nan"),
                "pvalue": 0.01,
                "estimation_note": "degenerate_collinear",
            },
            {
                "outcome_id": "no_var",
                "outcome_family": "lexical",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": 0.01,
                "pvalue": 0.04,
                "estimation_note": "no_treat_variation",
            },
            {
                "outcome_id": "nan_beta",
                "outcome_family": "semantic_axis",
                "strategy_id": "cross_country_all",
                "spec": "full_ban",
                "beta": float("nan"),
                "pvalue": 0.02,
                "estimation_note": "ok",
            },
        ]
    )


def test_dedupe_summary_rows_keeps_last() -> None:
    """Duplicate outcome×strategy×spec keys collapse to one row."""
    mod = _load_mod()
    raw = _sample_raw_df()
    duped = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert len(duped) == len(raw) + 1
    deduped = mod.dedupe_summary_rows(duped)
    assert len(deduped) == len(raw)


def test_filter_audit_rows_drops_bad_fits() -> None:
    """Filter keeps ok rows with finite beta and pvalue only."""
    mod = _load_mod()
    raw = _sample_raw_df()
    retained = mod.filter_audit_rows(raw)
    assert len(retained) == 5
    assert set(retained["estimation_note"]) == {"ok"}
    assert retained["beta"].notna().all()
    assert retained["pvalue"].notna().all()


def test_bh_qvalues_monotonic_and_family_differs() -> None:
    """Scan-wide q is non-decreasing in sorted p; family BH can differ from scan-wide."""
    mod = _load_mod()
    audit = mod.compute_bh_qvalues(mod.filter_audit_rows(_sample_raw_df()))
    sorted_q = audit.sort_values("pvalue")["q_scanwide"].to_numpy()
    assert np.all(np.diff(sorted_q) >= -1e-12)
    emotion = audit[
        (audit["outcome_id"] == "sem_axis_emotion") & (audit["spec"] == "early_ban_7d")
    ].iloc[0]
    assert emotion["q_family"] != emotion["q_scanwide"]


def test_vintage_signature_includes_marker() -> None:
    """Vintage block reports raw row count and ai_style_rate beta marker."""
    mod = _load_mod()
    raw = _sample_raw_df()
    path = Path(__file__).resolve()
    vintage = mod.input_vintage_signature(raw, path)
    assert vintage["raw_rows"] == len(raw)
    assert vintage["deduped_rows"] == len(raw)
    assert "file_sha256" in vintage
    assert "beta=0.05" in vintage["vintage_marker"]


def test_summary_includes_spotlight_and_low_q_list() -> None:
    """Summary text contains vintage, spotlight rows, and q_scanwide < 0.10 section."""
    mod = _load_mod()
    raw = _sample_raw_df()
    path = Path(__file__).resolve()
    _, summary, _ = mod.export_scan_audit(raw, path)
    assert "Vintage (input did_summary.csv)" in summary
    assert "sem_axis_emotion | cross_country_all | early_ban_7d" in summary
    assert "pole_share | cross_country_all | full_ban" in summary
    assert "Tests with q_scanwide < 0.10" in summary
    assert "ambivalence | cross_country_vs_de" in summary


def test_memo_reproduction_on_controlled_inputs() -> None:
    """Memo check flags match when stats align with memo targets."""
    mod = _load_mod()
    stats = {
        "n_tests_total": 1732,
        "n_distinct_outcome_ids": 62,
        "share_p_lt_0_05": 12.0,
    }
    repro = mod.check_memo_reproduction(stats, 0.163)
    assert repro["n_tests"] is True
    assert repro["n_outcomes"] is True
    assert repro["share_p_lt_0_05"] is True
    assert repro["pole_share_full_ban_q"] is True

    repro_miss = mod.check_memo_reproduction(stats, 0.229)
    assert repro_miss["pole_share_full_ban_q"] is False


def test_audit_table_columns_and_sort() -> None:
    """Output CSV schema uses family/strategy rename and sorts by q_scanwide."""
    mod = _load_mod()
    raw = _sample_raw_df()
    table, _, _ = mod.export_scan_audit(raw, Path(__file__).resolve())
    assert list(table.columns) == [
        "outcome_id",
        "family",
        "strategy",
        "spec",
        "pvalue",
        "q_scanwide",
        "q_family",
    ]
    assert table["q_scanwide"].is_monotonic_increasing
