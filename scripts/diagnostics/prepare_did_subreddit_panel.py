"""
Script summary:
Build subreddit-day DiD panels from polarization descriptives for prompt 04 estimation.

Functionality:
- Reads daily_by_subreddit.csv and daily_by_subreddit_universe_slice.csv.
- Adds rel_day, post, IT/treatment flags, and control-family indicators.
- Writes did_subreddit_panel_1d.csv, did_subreddit_panel_by_universe_slice_1d.csv, and
  did_subreddit_quantity_panel_1d.csv (zero-filled active forums with log1p counts).
  With --exclude-ban-topic, parallel *_exbantopic.csv variants including quantity panel.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ITALY_FAMILIES = frozenset({"it_political", "it_others"})
CONTROL_FAMILIES = frozenset({"de", "eu", "us", "uk"})


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import load_config, tables_subdir  # noqa: E402
from src.did.paths import did_panels_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build subreddit-day DiD panels.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--exclude-ban-topic",
        action="store_true",
        help="Read *_exbantopic descriptives and write did_subreddit_panel_*_exbantopic.csv "
        "(including did_subreddit_quantity_panel_1d_exbantopic.csv).",
    )
    return parser.parse_args()


def _add_did_calendar(df: pd.DataFrame, launch: str, end_excl: str) -> pd.DataFrame:
    """Function summary: add rel_day, post, and ban-window flags from date_utc.

    Parameters:
    - df: panel with date_utc column.
    - launch: ban onset YYYY-MM-DD.
    - end_excl: corpus end (exclusive) YYYY-MM-DD.

    Returns:
    - Copy with calendar DiD columns.
    """
    out = df.copy()
    launch_dt = pd.Timestamp(launch)
    out["date_utc"] = out["date_utc"].astype(str)
    out["rel_day"] = (pd.to_datetime(out["date_utc"]) - launch_dt).dt.days.astype(int)
    out["post"] = (out["date_utc"].astype(str) >= launch).astype(int)
    out["in_corpus"] = (
        (out["date_utc"].astype(str) >= out["date_utc"].min())
        & (out["date_utc"].astype(str) < end_excl)
    ).astype(int)
    ref = (launch_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out["is_ref_day"] = (out["date_utc"] == ref).astype(int)
    return out


def _add_treatment_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add IT and topic_family treatment indicators.

    Parameters:
    - df: panel with topic_family column.

    Returns:
    - Copy with treatment columns.
    """
    out = df.copy()
    fam = out["topic_family"].astype(str)
    out["IT"] = fam.isin(ITALY_FAMILIES).astype(int)
    out["IT_political"] = (fam == "it_political").astype(int)
    out["IT_others"] = (fam == "it_others").astype(int)
    out["is_control"] = fam.isin(CONTROL_FAMILIES).astype(int)
    for c in sorted(CONTROL_FAMILIES):
        out[f"control_{c}"] = (fam == c).astype(int)
    out["political_universe"] = (
        out["universe_slice"].astype(str) == "in_political_tree"
    ).astype(int) if "universe_slice" in out.columns else 0
    return out


def _annotate_subreddit_panel(panel: pd.DataFrame, launch: str, end_excl: str) -> pd.DataFrame:
    """Function summary: full DiD annotation on subreddit-day panel."""
    out = _add_did_calendar(panel, launch, end_excl)
    return _add_treatment_flags(out)


def _calendar_days(start: str, end_exclusive: str) -> List[str]:
    """Function summary: enumerate inclusive-start exclusive-end UTC calendar days.

    Parameters:
    - start: YYYY-MM-DD window start (inclusive).
    - end_exclusive: YYYY-MM-DD window end (exclusive).

    Returns:
    - Sorted list of date strings.
    """
    cur = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end_exclusive, "%Y-%m-%d")
    days: List[str] = []
    while cur < end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def _active_forums_both_months(sub: pd.DataFrame) -> List[str]:
    """Function summary: subreddits with >=1 comment in both March and April 2023.

    Parameters:
    - sub: daily_by_subreddit rows in event window with n_comments.

    Returns:
    - Sorted list of active subreddit names.
    """
    work = sub.copy()
    work["date_utc"] = work["date_utc"].astype(str)
    work["month"] = work["date_utc"].str.slice(0, 7)
    march = set(
        work.loc[(work["month"] == "2023-03") & (work["n_comments"].astype(float) > 0), "subreddit"]
        .astype(str)
        .unique()
    )
    april = set(
        work.loc[(work["month"] == "2023-04") & (work["n_comments"].astype(float) > 0), "subreddit"]
        .astype(str)
        .unique()
    )
    return sorted(march & april)


def build_quantity_panel(
    sub: pd.DataFrame,
    start: str,
    end_excl: str,
    launch: str,
) -> pd.DataFrame:
    """Function summary: zero-filled subreddit-day quantity panel for DiD estimation.

    Parameters:
    - sub: daily_by_subreddit in event window (sparse days only).
    - start: YYYY-MM-DD inclusive.
    - end_excl: YYYY-MM-DD exclusive.
    - launch: ban onset for annotation.

    Returns:
    - Annotated panel with n_comments, n_authors, log_n_comments, log_n_authors.
    """
    if sub.empty:
        return pd.DataFrame()

    active = _active_forums_both_months(sub)
    if not active:
        return pd.DataFrame()

    meta = (
        sub[["subreddit", "topic_family", "topic"]]
        .drop_duplicates("subreddit")
        .set_index("subreddit")
    )
    counts = sub[["subreddit", "date_utc", "n_comments", "n_authors"]].copy()
    counts["date_utc"] = counts["date_utc"].astype(str)

    days = _calendar_days(start, end_excl)
    grid = pd.MultiIndex.from_product([active, days], names=["subreddit", "date_utc"])
    panel = grid.to_frame(index=False)
    panel = panel.merge(counts, on=["subreddit", "date_utc"], how="left")
    panel["n_comments"] = panel["n_comments"].fillna(0).astype(int)
    panel["n_authors"] = panel["n_authors"].fillna(0).astype(int)
    panel["topic_family"] = panel["subreddit"].map(meta["topic_family"])
    panel["topic"] = panel["subreddit"].map(meta.get("topic", pd.Series(dtype=str)))
    panel["log_n_comments"] = np.log1p(panel["n_comments"].astype(float))
    panel["log_n_authors"] = np.log1p(panel["n_authors"].astype(float))

    out = _annotate_subreddit_panel(panel, launch, end_excl)
    out["period_start"] = out["date_utc"]
    out["bin_days"] = 1
    return out


def main() -> None:
    """Function summary: write DiD-ready subreddit panels."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    start, end_excl, launch, _lift = event_dates_from_config(config)
    desc_dir = tables_subdir(config, "descriptives")
    did_dir = did_panels_dir(config, "subreddit")
    did_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_exbantopic" if args.exclude_ban_topic else ""

    sub_path = desc_dir / f"daily_by_subreddit{suffix}.csv"
    if not sub_path.is_file():
        raise FileNotFoundError(
            f"Missing {sub_path}; run prepare_polarization_descriptives.py"
            + (" --exclude-ban-topic" if args.exclude_ban_topic else "")
            + " first."
        )
    sub = pd.read_csv(sub_path)
    if "topic_family" not in sub.columns or sub["topic_family"].isna().all():
        sem_path = tables_subdir(config, "semantic_axis") / f"semantic_axis_panel{suffix}.csv"
        if sem_path.is_file():
            meta = (
                pd.read_csv(sem_path, usecols=["subreddit", "topic_family", "topic"])
                .drop_duplicates("subreddit")
            )
            sub = sub.merge(meta, on="subreddit", how="left", suffixes=("", "_sem"))
            if "topic_family_sem" in sub.columns:
                sub["topic_family"] = sub["topic_family"].fillna(sub["topic_family_sem"])
                sub["topic"] = sub.get("topic", pd.Series(dtype=str)).fillna(
                    sub.get("topic_sem", pd.Series(dtype=str))
                )
                sub = sub.drop(columns=[c for c in sub.columns if c.endswith("_sem")], errors="ignore")
        if "topic_family" not in sub.columns or sub["topic_family"].isna().all():
            raise ValueError("daily_by_subreddit.csv lacks topic_family; re-run descriptives.")
    sub = sub[(sub["date_utc"].astype(str) >= start) & (sub["date_utc"].astype(str) < end_excl)]
    sub_out = _annotate_subreddit_panel(sub, launch, end_excl)
    sub_out["period_start"] = sub_out["date_utc"]
    sub_out["bin_days"] = 1
    out_sub = did_dir / f"did_subreddit_panel_1d{suffix}.csv"
    sub_out.to_csv(out_sub, index=False)
    print(f"[prepare_did_subreddit_panel] {out_sub.name} rows={len(sub_out)}", flush=True)

    slice_path = desc_dir / f"daily_by_subreddit_universe_slice{suffix}.csv"
    if slice_path.is_file():
        sl = pd.read_csv(slice_path)
        sl = sl[(sl["date_utc"].astype(str) >= start) & (sl["date_utc"].astype(str) < end_excl)]
        meta_cols = ["subreddit", "topic_family", "topic"]
        meta = sub_out[[c for c in meta_cols if c in sub_out.columns]].drop_duplicates("subreddit")
        sl = sl.merge(meta, on="subreddit", how="left")
        sl_out = _annotate_subreddit_panel(sl, launch, end_excl)
        sl_out["period_start"] = sl_out["date_utc"]
        sl_out["bin_days"] = 1
        out_sl = did_dir / f"did_subreddit_panel_by_universe_slice_1d{suffix}.csv"
        sl_out.to_csv(out_sl, index=False)
        print(f"[prepare_did_subreddit_panel] {out_sl.name} rows={len(sl_out)}", flush=True)
    else:
        print(
            "[prepare_did_subreddit_panel] skip universe slice "
            "(re-run prepare_polarization_descriptives.py)",
            flush=True,
        )

    qty_out = build_quantity_panel(sub, start, end_excl, launch)
    out_qty = did_dir / f"did_subreddit_quantity_panel_1d{suffix}.csv"
    if qty_out.empty:
        print(
            "[prepare_did_subreddit_panel] skip quantity panel (no active forums in Mar+Apr)",
            flush=True,
        )
    else:
        qty_out.to_csv(out_qty, index=False)
        print(
            f"[prepare_did_subreddit_panel] {out_qty.name} rows={len(qty_out)} "
            f"forums={qty_out['subreddit'].nunique()}",
            flush=True,
        )

    print(f"[prepare_did_subreddit_panel] wrote to {did_dir}", flush=True)


if __name__ == "__main__":
    main()
