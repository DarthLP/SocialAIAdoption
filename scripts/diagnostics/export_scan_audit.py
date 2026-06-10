"""
Script summary:
Export scan-wide and within-family Benjamini–Hochberg multiple-testing audit from did_summary.csv.

Functionality:
- Reads baseline estimates/summary/did_summary.csv only (one vintage; no weighted/exbantopic).
- Filters ok TWFE rows, computes scan-wide and family BH q-values, writes scan_audit.csv and
  scan_audit_summary.txt with vintage signature and memo-reproduction check.

How to apply/run:
  .venv/bin/python scripts/diagnostics/export_scan_audit.py --config config/italy_polarization_setup.yaml

Run after did_event_study.py on the rebuilt panel (F10 in cross-plan order: panel → F9 → F7 → F10).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HEADLINE_STRATEGY = "cross_country_all"
VINTAGE_MARKER_OUTCOME = "ai_style_rate"
VINTAGE_MARKER_SPEC = "full_ban"

MEMO_TARGETS = {
    "n_tests": 1732,
    "n_outcomes": 62,
    "share_p_lt_0_05": 12.0,
    "pole_share_full_ban_q": 0.163,
}


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute path to project root.
    """
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config  # noqa: E402
from src.did.paths import did_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for scan audit export.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Export DiD scan-wide multiple-testing audit.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def dedupe_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: collapse duplicate outcome×strategy×spec rows (keep last).

    Parameters:
    - df: raw did_summary frame (may contain duplicate keys from estimation).

    Returns:
    - Deduped copy matching write_summary_rows semantics.
    """
    keys = [c for c in ("outcome_id", "strategy_id", "spec") if c in df.columns]
    if not keys:
        return df.copy()
    return df.drop_duplicates(subset=keys, keep="last").copy()


def filter_audit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: retain ok TWFE rows with finite beta and pvalue for BH.

    Parameters:
    - df: raw did_summary.csv frame.

    Returns:
    - Filtered copy suitable for audit counts and BH.
    """
    work = df.copy()
    ok = work["estimation_note"].astype(str) == "ok"
    finite_beta = pd.to_numeric(work["beta"], errors="coerce").notna()
    finite_p = pd.to_numeric(work["pvalue"], errors="coerce").notna()
    return work.loc[ok & finite_beta & finite_p].copy()


def compute_bh_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add scan-wide and within-family BH q-values.

    Parameters:
    - df: filtered audit rows with finite pvalue column.

    Returns:
    - Copy with q_scanwide and q_family columns.
    """
    out = df.copy()
    pvals = out["pvalue"].to_numpy(dtype=float)
    out["q_scanwide"] = multipletests(pvals, method="fdr_bh")[1]
    out["q_family"] = np.nan
    for _, grp in out.groupby("outcome_family", sort=False):
        qf = multipletests(grp["pvalue"].to_numpy(dtype=float), method="fdr_bh")[1]
        out.loc[grp.index, "q_family"] = qf
    return out


def _lookup_row(
    df: pd.DataFrame,
    *,
    outcome_id: str,
    strategy_id: str,
    spec: str,
) -> pd.Series | None:
    """Function summary: return first matching did_summary row or None."""
    mask = (
        (df["outcome_id"].astype(str) == outcome_id)
        & (df["strategy_id"].astype(str) == strategy_id)
        & (df["spec"].astype(str) == spec)
    )
    sub = df.loc[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def input_vintage_signature(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Function summary: build input-vintage metadata for scan_audit_summary.txt.

    Parameters:
    - df: unfiltered did_summary frame.
    - path: resolved path to did_summary.csv.

    Returns:
    - Dict with path, raw_rows, mtime, sha256, and vintage_marker string.
    """
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    marker_row = _lookup_row(
        df,
        outcome_id=VINTAGE_MARKER_OUTCOME,
        strategy_id=HEADLINE_STRATEGY,
        spec=VINTAGE_MARKER_SPEC,
    )
    if marker_row is None:
        vintage_marker = "(not found)"
    else:
        beta = float(marker_row["beta"])
        vintage_marker = (
            f"{VINTAGE_MARKER_OUTCOME} | {HEADLINE_STRATEGY} | {VINTAGE_MARKER_SPEC} | beta={beta:.6g}"
        )
    deduped = dedupe_summary_rows(df)
    return {
        "path": str(path),
        "raw_rows": int(len(df)),
        "deduped_rows": int(len(deduped)),
        "file_mtime_utc": mtime,
        "file_sha256": digest,
        "vintage_marker": vintage_marker,
    }


def compute_headline_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Function summary: aggregate nominal significance counts from retained rows.

    Parameters:
    - df: filtered audit rows with pvalue column.

    Returns:
    - Dict of n_tests, n_outcomes, p-threshold counts and shares.
    """
    n_tests = int(len(df))
    n_outcomes = int(df["outcome_id"].nunique())
    p_lt_05 = int((df["pvalue"] < 0.05).sum())
    p_lt_01 = int((df["pvalue"] < 0.01).sum())
    share_05 = (100.0 * p_lt_05 / n_tests) if n_tests else 0.0
    share_01 = (100.0 * p_lt_01 / n_tests) if n_tests else 0.0
    return {
        "n_tests_total": n_tests,
        "n_distinct_outcome_ids": n_outcomes,
        "n_p_lt_0_05": p_lt_05,
        "share_p_lt_0_05": share_05,
        "n_p_lt_0_01": p_lt_01,
        "share_p_lt_0_01": share_01,
    }


def check_memo_reproduction(
    stats: dict[str, float | int],
    pole_q: float | None,
    *,
    memo: dict[str, float | int] | None = None,
) -> dict[str, bool]:
    """Function summary: compare audit stats to session-memo targets (informational).

    Parameters:
    - stats: headline stats from compute_headline_stats.
    - pole_q: scan-wide BH q for pole_share full_ban cross_country_all, or None.
    - memo: optional override of MEMO_TARGETS.

    Returns:
    - Dict mapping metric name to whether memo target reproduces.
    """
    targets = memo or MEMO_TARGETS
    share = float(stats["share_p_lt_0_05"])
    pole_ok = False
    if pole_q is not None and np.isfinite(pole_q):
        pole_ok = round(float(pole_q), 3) == round(float(targets["pole_share_full_ban_q"]), 3)
    return {
        "n_tests": int(stats["n_tests_total"]) == int(targets["n_tests"]),
        "n_outcomes": int(stats["n_distinct_outcome_ids"]) == int(targets["n_outcomes"]),
        "share_p_lt_0_05": abs(share - float(targets["share_p_lt_0_05"])) <= 0.15,
        "pole_share_full_ban_q": pole_ok,
    }


def _spotlight_line(audit: pd.DataFrame, outcome_id: str, spec: str) -> str:
    """Function summary: format one spotlight q-value line for summary text."""
    row = _lookup_row(
        audit,
        outcome_id=outcome_id,
        strategy_id=HEADLINE_STRATEGY,
        spec=spec,
    )
    label = f"{outcome_id} | {HEADLINE_STRATEGY} | {spec}"
    if row is None:
        return f"  {label}: (not found)"
    return (
        f"  {label}: p={float(row['pvalue']):.6g}, "
        f"q_scanwide={float(row['q_scanwide']):.6g}, q_family={float(row['q_family']):.6g}"
    )


def build_summary_text(
    audit: pd.DataFrame,
    vintage: dict[str, Any],
    stats: dict[str, float | int],
    memo_repro: dict[str, bool],
    pole_q: float | None,
) -> str:
    """Function summary: assemble scan_audit_summary.txt body.

    Parameters:
    - audit: filtered rows with BH q columns.
    - vintage: input_vintage_signature dict.
    - stats: headline counts.
    - memo_repro: check_memo_reproduction output.
    - pole_q: pole_share full_ban scan-wide q (may be None).

    Returns:
    - Full summary text.
    """
    lines = [
        "DiD scan-wide multiple-testing audit",
        "=" * 40,
        "",
        "Vintage (input did_summary.csv)",
        f"  path: {vintage['path']}",
        f"  raw_rows: {vintage['raw_rows']}",
        f"  deduped_rows: {vintage['deduped_rows']}",
        f"  file_mtime_utc: {vintage['file_mtime_utc']}",
        f"  file_sha256: {vintage['file_sha256']}",
        f"  vintage_marker: {vintage['vintage_marker']}",
        "",
        "Headline counts (retained tests: estimation_note=ok, finite beta & pvalue)",
        f"  n_tests_total: {stats['n_tests_total']}",
        f"  n_distinct_outcome_ids: {stats['n_distinct_outcome_ids']}",
        f"  n_p_lt_0_05: {stats['n_p_lt_0_05']} ({stats['share_p_lt_0_05']:.1f}%)",
        f"  n_p_lt_0_01: {stats['n_p_lt_0_01']} ({stats['share_p_lt_0_01']:.1f}%)",
        "",
        "Spotlight q-values (cross_country_all)",
        _spotlight_line(audit, "sem_axis_emotion", "early_ban_7d"),
        _spotlight_line(audit, "pole_share", "full_ban"),
        "",
        "Tests with q_scanwide < 0.10",
    ]
    low_q = audit[audit["q_scanwide"] < 0.10].sort_values("q_scanwide")
    if low_q.empty:
        lines.append("  (none)")
    else:
        for _, row in low_q.iterrows():
            lines.append(
                f"  {row['outcome_id']} | {row['strategy_id']} | {row['spec']} | "
                f"p={float(row['pvalue']):.6g} | q_scanwide={float(row['q_scanwide']):.6g}"
            )
    lines.extend(
        [
            "",
            "Memo reproduction (session memo; informational only)",
            (
                "  Memo targets: n_tests=1732, n_outcomes=62, share_p_lt_0_05=12.0%, "
                "pole_share_full_ban_q≈0.163"
            ),
            f"  n_tests: {'YES' if memo_repro['n_tests'] else 'NO'} "
            f"(audit={stats['n_tests_total']}, memo=1732)",
            f"  n_outcomes: {'YES' if memo_repro['n_outcomes'] else 'NO'} "
            f"(audit={stats['n_distinct_outcome_ids']}, memo=62)",
            f"  share_p_lt_0_05: {'YES' if memo_repro['share_p_lt_0_05'] else 'NO'} "
            f"(audit={stats['share_p_lt_0_05']:.1f}%, memo=12.0%)",
        ]
    )
    if pole_q is not None and np.isfinite(pole_q):
        lines.append(
            f"  pole_share_full_ban_q: {'YES' if memo_repro['pole_share_full_ban_q'] else 'NO'} "
            f"(audit={pole_q:.3g}, memo≈0.163)"
        )
    else:
        lines.append("  pole_share_full_ban_q: NO (audit row not found)")
    return "\n".join(lines) + "\n"


def build_audit_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Function summary: format scan_audit.csv output columns sorted by q_scanwide.

    Parameters:
    - audit: filtered rows with BH q columns.

    Returns:
    - Output frame with renamed columns.
    """
    out = audit[
        [
            "outcome_id",
            "outcome_family",
            "strategy_id",
            "spec",
            "pvalue",
            "q_scanwide",
            "q_family",
        ]
    ].copy()
    out = out.rename(
        columns={
            "outcome_family": "family",
            "strategy_id": "strategy",
        }
    )
    return out.sort_values("q_scanwide", kind="mergesort").reset_index(drop=True)


def export_scan_audit(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, str, dict[str, bool]]:
    """Function summary: run full audit pipeline on loaded did_summary.

    Parameters:
    - df: raw did_summary.csv frame.
    - path: path to input CSV (for vintage signature).

    Returns:
    - Tuple of (audit_table, summary_text, memo_repro dict).
    """
    vintage = input_vintage_signature(df, path)
    retained = filter_audit_rows(dedupe_summary_rows(df))
    audit = compute_bh_qvalues(retained)
    stats = compute_headline_stats(audit)
    pole_row = _lookup_row(
        audit,
        outcome_id="pole_share",
        strategy_id=HEADLINE_STRATEGY,
        spec="full_ban",
    )
    pole_q = float(pole_row["q_scanwide"]) if pole_row is not None else None
    memo_repro = check_memo_reproduction(stats, pole_q)
    summary = build_summary_text(audit, vintage, stats, memo_repro, pole_q)
    return build_audit_table(audit), summary, memo_repro


def main() -> None:
    """Function summary: load did_summary.csv, write scan audit artifacts, print memo check."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    summary_path, _ = did_summary_paths(config)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; run did_event_study.py first.")
    raw = pd.read_csv(summary_path)
    audit_table, summary_text, memo_repro = export_scan_audit(raw, summary_path)
    out_dir = summary_path.parent
    csv_path = out_dir / "scan_audit.csv"
    txt_path = out_dir / "scan_audit_summary.txt"
    audit_table.to_csv(csv_path, index=False)
    txt_path.write_text(summary_text, encoding="utf-8")
    print(f"[export_scan_audit] wrote {csv_path} ({len(audit_table)} rows)", flush=True)
    print(f"[export_scan_audit] wrote {txt_path}", flush=True)
    print(
        "[export_scan_audit] memo reproduction: "
        + ", ".join(f"{k}={'YES' if v else 'NO'}" for k, v in memo_repro.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
