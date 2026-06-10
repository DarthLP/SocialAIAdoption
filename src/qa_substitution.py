"""
Script summary:
Core helpers for the Italian Q&A substitution test (question-mark feature, panel aggregation, DiD flags).

Functionality:
- Cheap per-comment question proxies from comment body text.
- Subreddit-day and 3-day panel aggregation.
- Calendar DiD columns (post, phase, rel_day, bin3_id) for within-Italy volume/rate estimation.

How to apply/run:
- Imported by scripts/diagnostics/prepare_qa_volume_panel.py, scripts/analysis/qa_volume_did.py,
  and tests/test_qa_volume_panel.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

ITALY_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})
HUB_TOPIC_FAMILIES = frozenset({"de", "eu", "uk"})


def score_comment_question(body: str | None) -> Tuple[int, int]:
    """Function summary: count question marks and flag comments containing at least one.

    Parameters:
    - body: raw comment body.

    Returns:
    - Tuple (is_question as 0/1, qmark_count).
    """
    text = body or ""
    qmark_count = int(text.count("?"))
    is_question = int(qmark_count > 0)
    return is_question, qmark_count


def annotate_comment_questions(df: pd.DataFrame, body_col: str = "body") -> pd.DataFrame:
    """Function summary: add is_question and qmark_count columns from body text.

    Parameters:
    - df: comment-level frame with a body column.
    - body_col: name of the text column.

    Returns:
    - Copy with is_question and qmark_count int columns.
    """
    out = df.copy()
    if body_col not in out.columns:
        out["is_question"] = 0
        out["qmark_count"] = 0
        return out
    scored = out[body_col].map(lambda b: score_comment_question(b if isinstance(b, str) else ""))
    out["is_question"] = scored.map(lambda t: t[0]).astype(int)
    out["qmark_count"] = scored.map(lambda t: t[1]).astype(int)
    return out


def aggregate_subreddit_day(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: pool comments to subreddit-day volume and question-rate metrics.

    Parameters:
    - df: comment frame with subreddit, date_utc, author, n_words, is_question, qmark_count.

    Returns:
    - Subreddit-day panel with n_comments, n_authors, n_questions, question_share, qmark_rate_100w.
    """
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["subreddit", "date_utc"]
    meta_cols = [c for c in ("topic_family", "topic") if c in df.columns]
    for key, grp in df.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        sub, day = key
        total_words = float(pd.to_numeric(grp["n_words"], errors="coerce").fillna(0).sum())
        n_comments = int(len(grp))
        n_questions = int(grp["is_question"].sum()) if "is_question" in grp.columns else 0
        qmark_total = int(grp["qmark_count"].sum()) if "qmark_count" in grp.columns else 0
        row: Dict[str, Any] = {
            "subreddit": sub,
            "date_utc": str(day),
            "n_comments": n_comments,
            "n_authors": int(grp["author"].nunique()) if "author" in grp.columns else 0,
            "n_questions": n_questions,
            "question_share": float(n_questions / n_comments) if n_comments > 0 else float("nan"),
            "qmark_rate_100w": 100.0 * qmark_total / total_words if total_words > 0 else float("nan"),
            "total_words": total_words,
            "qmark_count": qmark_total,
        }
        for col in meta_cols:
            row[col] = grp[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def reindex_full_grid(
    panel: pd.DataFrame,
    subreddits: List[str],
    start: str,
    end_excl: str,
    family_map: Dict[str, str],
) -> pd.DataFrame:
    """Function summary: expand subreddit-day panel to full calendar grid with zero-filled counts.

    Parameters:
    - panel: sparse subreddit-day panel from aggregate_subreddit_day.
    - subreddits: roster of subreddits to include (full grid per forum).
    - start: corpus start YYYY-MM-DD (inclusive).
    - end_excl: corpus end YYYY-MM-DD (exclusive).
    - family_map: subreddit -> topic_family for metadata on zero-filled rows.

    Returns:
    - Dense panel with one row per subreddit-day; count columns zero-filled, rate columns NaN on zero days.
    """
    if not subreddits:
        return pd.DataFrame()
    dates = pd.date_range(start=start, end=pd.Timestamp(end_excl) - pd.Timedelta(days=1), freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    grid = pd.MultiIndex.from_product([sorted(set(subreddits)), date_strs], names=["subreddit", "date_utc"])
    grid_df = grid.to_frame(index=False)
    grid_df["topic_family"] = grid_df["subreddit"].map(family_map).fillna("")

    if panel.empty:
        out = grid_df.copy()
    else:
        work = panel.copy()
        work["date_utc"] = work["date_utc"].astype(str)
        work["subreddit"] = work["subreddit"].astype(str)
        out = grid_df.merge(work, on=["subreddit", "date_utc"], how="left", suffixes=("", "_obs"))
        if "topic_family_obs" in out.columns:
            out["topic_family"] = out["topic_family"].where(
                out["topic_family"].astype(str).str.len() > 0,
                out["topic_family_obs"],
            )
            out = out.drop(columns=["topic_family_obs"], errors="ignore")

    count_cols = ["n_comments", "n_questions", "n_authors", "qmark_count", "total_words"]
    for col in count_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    has_comments = out["n_comments"] > 0
    if "question_share" not in out.columns:
        out["question_share"] = float("nan")
    if "qmark_rate_100w" not in out.columns:
        out["qmark_rate_100w"] = float("nan")
    out.loc[~has_comments, "question_share"] = float("nan")
    out.loc[~has_comments, "qmark_rate_100w"] = float("nan")
    qs_missing = has_comments & out["question_share"].isna()
    out.loc[qs_missing, "question_share"] = (
        out.loc[qs_missing, "n_questions"] / out.loc[qs_missing, "n_comments"]
    )
    qr_missing = has_comments & out["qmark_rate_100w"].isna()
    tw = out.loc[qr_missing, "total_words"].replace(0, float("nan"))
    out.loc[qr_missing, "qmark_rate_100w"] = 100.0 * out.loc[qr_missing, "qmark_count"] / tw

    out["n_comments"] = out["n_comments"].astype(int)
    out["n_questions"] = out["n_questions"].astype(int)
    out["n_authors"] = out["n_authors"].astype(int)
    out["qmark_count"] = out["qmark_count"].astype(int)
    return out.sort_values(["subreddit", "date_utc"]).reset_index(drop=True)


def ban_phase(date_utc: str, launch: str, lift: str) -> str:
    """Function summary: assign pre/ban/post phase label from calendar date.

    Parameters:
    - date_utc: YYYY-MM-DD string.
    - launch: ban onset date (inclusive for ban phase).
    - lift: first post-lift date (inclusive).

    Returns:
    - Phase label: pre, ban, or post.
    """
    d = str(date_utc)
    if d < launch:
        return "pre"
    if d < lift:
        return "ban"
    return "post"


def add_did_calendar_columns(
    panel: pd.DataFrame,
    launch: str,
    lift: str,
    end_excl: str,
    *,
    bin_days: int = 3,
) -> pd.DataFrame:
    """Function summary: add rel_day, post, phase, time_id, and optional 3-day bin columns.

    Parameters:
    - panel: subreddit-day panel with date_utc.
    - launch: ban onset YYYY-MM-DD.
    - lift: ban lift YYYY-MM-DD (first post day).
    - end_excl: corpus end exclusive YYYY-MM-DD.
    - bin_days: width for rel_period / bin3_id (default 3).

    Returns:
    - Copy with DiD calendar columns.
    """
    out = panel.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    launch_dt = pd.Timestamp(launch)
    out["rel_day"] = (pd.to_datetime(out["date_utc"]) - launch_dt).dt.days.astype(int)
    out["post"] = (out["date_utc"] >= launch).astype(int)
    out["phase"] = out["date_utc"].map(lambda d: ban_phase(str(d), launch, lift))
    out["in_corpus"] = (
        (out["date_utc"] >= out["date_utc"].min()) & (out["date_utc"] < end_excl)
    ).astype(int)
    ref = (launch_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out["is_ref_day"] = (out["date_utc"] == ref).astype(int)
    out["time_id"] = out["date_utc"].astype(str)
    if bin_days > 1:
        out["rel_period"] = (out["rel_day"] // int(bin_days)).astype(int)
        out["bin3_id"] = out["rel_period"]
        out["period_start"] = (
            launch_dt + pd.to_timedelta(out["rel_period"] * int(bin_days), unit="D")
        ).dt.strftime("%Y-%m-%d")
        out["time_id_bin3"] = out["period_start"].astype(str)
    else:
        out["rel_period"] = out["rel_day"]
        out["bin3_id"] = out["rel_period"]
        out["period_start"] = out["date_utc"]
        out["time_id_bin3"] = out["time_id"]
    return out


def add_treatment_flags(
    panel: pd.DataFrame,
    qa_subreddits: frozenset[str] | set[str],
) -> pd.DataFrame:
    """Function summary: add qa, IT, is_hub, and qa_post interaction columns.

    Parameters:
    - panel: subreddit-day panel with subreddit and topic_family.
    - qa_subreddits: treated Q&A/advice subreddit names.

    Returns:
    - Copy with treatment indicator columns.
    """
    out = panel.copy()
    fam = out["topic_family"].astype(str) if "topic_family" in out.columns else pd.Series("", index=out.index)
    out["qa"] = out["subreddit"].astype(str).isin(qa_subreddits).astype(int)
    out["IT"] = fam.isin(ITALY_TOPIC_FAMILIES).astype(int)
    out["is_hub"] = fam.isin(HUB_TOPIC_FAMILIES).astype(int)
    out["qa_post"] = out["qa"].astype(float) * out["post"].astype(float)
    out["IT_post"] = out["IT"].astype(float) * out["post"].astype(float)
    return out


def aggregate_panel_bins(panel_1d: pd.DataFrame, bin_days: int = 3) -> pd.DataFrame:
    """Function summary: aggregate subreddit-day panel to launch-aligned multi-day bins.

    Parameters:
    - panel_1d: annotated 1-day panel with rel_period and period_start.
    - bin_days: bin width in days (must match rel_period definition).

    Returns:
    - Subreddit-bin panel with summed counts and recomputed rates.
    """
    if panel_1d.empty:
        return pd.DataFrame()
    work = panel_1d.copy()
    if "rel_period" not in work.columns:
        raise ValueError("panel_1d must include rel_period; run add_did_calendar_columns first.")
    group_cols = ["subreddit", "rel_period"]
    meta_cols = [c for c in ("topic_family", "topic", "qa", "IT", "is_hub") if c in work.columns]
    rows: List[Dict[str, Any]] = []
    for key, grp in work.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        sub, rel_period = key
        n_comments = int(grp["n_comments"].sum())
        n_questions = int(grp["n_questions"].sum()) if "n_questions" in grp.columns else 0
        total_words = float(grp["total_words"].sum()) if "total_words" in grp.columns else float("nan")
        qmark_total = int(grp["qmark_count"].sum()) if "qmark_count" in grp.columns else 0
        row: Dict[str, Any] = {
            "subreddit": sub,
            "rel_period": int(rel_period),
            "bin3_id": int(rel_period),
            "period_start": str(grp["period_start"].iloc[0]) if "period_start" in grp.columns else "",
            "n_comments": n_comments,
            "n_authors": int(grp["n_authors"].sum()) if "n_authors" in grp.columns else 0,
            "n_questions": n_questions,
            "question_share": float(n_questions / n_comments) if n_comments > 0 else float("nan"),
            "qmark_rate_100w": 100.0 * qmark_total / total_words if total_words > 0 else float("nan"),
            "total_words": total_words,
            "qmark_count": qmark_total,
            "post": int(grp["post"].max()) if "post" in grp.columns else 0,
            "phase": str(grp["phase"].iloc[-1]) if "phase" in grp.columns else "",
        }
        for col in meta_cols:
            row[col] = grp[col].iloc[0]
        if "rel_day" in grp.columns:
            row["rel_day"] = int(grp["rel_day"].min())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "qa" in out.columns and "post" in out.columns:
        out["qa_post"] = out["qa"].astype(float) * out["post"].astype(float)
    if "IT" in out.columns and "post" in out.columns:
        out["IT_post"] = out["IT"].astype(float) * out["post"].astype(float)
    out["time_id"] = out["period_start"].astype(str)
    out["time_id_bin3"] = out["time_id"]
    return out


def phase_contrast_table(panel: pd.DataFrame, group_col: str = "qa") -> pd.DataFrame:
    """Function summary: compute phase means by treatment group for headline descriptives.

    Parameters:
    - panel: annotated panel with phase, group_col, and outcome columns.
    - group_col: qa or IT grouping column.

    Returns:
    - Table with group × phase means for volume and question metrics.
    """
    if panel.empty or group_col not in panel.columns or "phase" not in panel.columns:
        return pd.DataFrame()
    metrics = ["n_comments", "n_questions", "n_authors", "question_share", "qmark_rate_100w"]
    present = [m for m in metrics if m in panel.columns]
    rows: List[Dict[str, Any]] = []
    for (group_val, phase), grp in panel.groupby([group_col, "phase"], sort=True):
        row: Dict[str, Any] = {group_col: group_val, "phase": phase, "n_subreddit_days": len(grp)}
        for metric in present:
            vals = pd.to_numeric(grp[metric], errors="coerce")
            if metric in {"question_share", "qmark_rate_100w"}:
                w = pd.to_numeric(grp.get("n_comments", 1), errors="coerce").fillna(1).clip(lower=1)
                ok = vals.notna()
                row[f"{metric}_mean"] = float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
            elif metric in {"n_comments", "n_questions"}:
                row[f"{metric}_sum"] = float(vals.sum())
                row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            else:
                row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
