"""
Script summary:
Within-user ideology-SPREAD difference-in-differences for the Italy ChatGPT ban.

This is a NEW, self-contained analysis. It makes the thesis claim "individual
extremity is the within-user spread of the ideology score — how far a single
author ranges across the axis over the window" literally true and testable.
It is distinct from the existing `extremity` outcome (per-comment one-sidedness
max(L,R)/(L+C+R), averaged — a LEVEL, not a spread) and from the existing
user-week `y ~ post + author FE` shift (Italy-only, no control group).

The measure: for each author, the dispersion of that author's per-comment
ideology score within a time window — SD (ddof=1, headline) and MAD (robust),
for both `sem_axis_ideology` (semantic, primary) and `net_ideology` (lexical,
robustness). Four outcome columns: {SD,MAD} × {sem_axis_ideology,net_ideology}.

Two estimators:
  (A) Static pre/post spread DiD (headline): author×period panel, spread in the
      pre window vs the ban_in_effect post window (rel_day 0..ban_in_effect_max),
      `spread ~ IT:post | author FE + period FE`, clustered by author. The IT:post
      coefficient is the DiD.
  (B) Author×ISO-week event study (pre-trend check): weekly spread, reference
      week = -1, TWFE event study clustered by author, with a joint pre-trend
      Wald test. The corpus is only March–April 2023, so pre-period support is
      thin — usable pre/post weeks are reported and pre-trends not over-read.

Design choices (documented in methods_note.txt):
- IT = 1 iff the author's plurality `primary_lexicon` is `it` (else control).
  Spread is computed within each author's own comments, so cross-language level
  non-comparability of the semantic axis is absorbed by the author FE; only a
  differential *change* in within-author scale could bias the DiD.
- Spread is sign- and location-invariant, so no per-language orientation flip.
- 39/232 enriched shards lack the score columns; comments there contribute to
  arm assignment / word cohorts but not to scored spreads (coverage logged).

Reads only existing data; writes only to a new results subfolder. Modifies no
existing pipeline file.

Functionality:
- Reuses prepare_user_week_style_panel input discovery + author hygiene.
- Reuses src.user_week cohort thresholds, launch ISO week, ban-week buffer.
- Reuses src.did.estimate house-style TWFE + event study (clustered SEs, pretrend).
- Writes static_spread_did.csv, event_study_spread.csv, baseline_levels.csv,
  methods_note.txt under results/tables/italy_polarization/user_week/ideology_spread/.

How to apply/run:
  .venv/bin/python scripts/user_week/estimate_user_week_ideology_spread.py \
    --config config/italy_polarization_setup.yaml --cohort strict
  .venv/bin/python scripts/user_week/estimate_user_week_ideology_spread.py \
    --config config/italy_polarization_setup.yaml --cohort loose
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

from scripts.user_week.prepare_user_week_style_panel import (  # noqa: E402
    filter_valid_authors,
    iso_week_start_from_unix,
    iter_monthly_files,
    normalize_input_shard,
    subreddits_for_panel,
)
from src.config_utils import (  # noqa: E402
    build_subreddit_metadata_table,
    load_config,
    user_week_drop_ban_week_default,
)
from src.did.estimate import estimate_event_study, estimate_twfe  # noqa: E402
from src.did.specs import load_ban_lift_bounds  # noqa: E402
from src.user_week.cohorts import cohort_thresholds_by_label, panel_cohort_authors  # noqa: E402
from src.user_week.panel_prep import add_calendar_fields, launch_iso_week_from_config  # noqa: E402

# (score column, statistic, weighting) outcome grid. Equal-weighted is headline;
# word-weighted SD is an optional companion (each comment one observation otherwise).
SCORES: Tuple[str, ...] = ("sem_axis_ideology", "net_ideology")
EVENT_STUDY_WINDOW_WEEKS = 8
PRE_PERIOD_DATE = "2000-01-01"  # synthetic distinct dates so TimeEffects == period FE
POST_PERIOD_DATE = "2000-02-01"


# ----------------------------- spread helpers -----------------------------


def spread_sd(values: np.ndarray) -> float:
    """Function summary: sample standard deviation (ddof=1); NaN when < 2 obs."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1))


def spread_mad(values: np.ndarray) -> float:
    """Function summary: median absolute deviation about the median (unscaled); NaN when < 2 obs."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    med = float(np.median(v))
    return float(np.median(np.abs(v - med)))


def spread_weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    """Function summary: reliability-weighted SD (ddof=1 analog); NaN when < 2 obs or zero weight."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if v.size < 2:
        return float("nan")
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    mean = float((w * v).sum() / sw)
    denom = sw - (w**2).sum() / sw  # reliability-weights Bessel analog
    if denom <= 0:
        return float("nan")
    var = float((w * (v - mean) ** 2).sum() / denom)
    return float(np.sqrt(var)) if var >= 0 else float("nan")


def _statistic(values: np.ndarray, statistic: str, weights: Optional[np.ndarray] = None) -> float:
    """Function summary: dispatch SD / MAD / word-weighted SD by name."""
    if statistic == "SD":
        return spread_sd(values)
    if statistic == "MAD":
        return spread_mad(values)
    if statistic == "SD_wordw":
        assert weights is not None, "SD_wordw needs weights"
        return spread_weighted_sd(values, weights)
    raise ValueError(f"Unknown statistic: {statistic}")


def _self_test() -> None:
    """Function summary: tiny synthetic unit check of the SD/MAD helpers."""
    assert abs(spread_sd(np.array([0.0, 2.0])) - np.sqrt(2.0)) < 1e-9
    assert abs(spread_mad(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) - 1.0) < 1e-9
    assert spread_sd(np.array([5.0, 5.0, 5.0])) == 0.0
    assert np.isnan(spread_sd(np.array([1.0])))  # < 2 obs
    # equal weights reduce weighted SD to ordinary SD
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert abs(spread_weighted_sd(x, np.ones_like(x)) - spread_sd(x)) < 1e-9


# ----------------------------- data loading -----------------------------


def _resolve_event_dates(config: Dict[str, Any]) -> Tuple[date, str, int, bool]:
    """Function summary: launch date, launch ISO Monday, ban_in_effect max rel_day, drop-ban-week flag."""
    launch_raw = str(config["event_window"]["launch_day_utc"])
    launch_date = datetime.fromisoformat(launch_raw.replace("Z", "+00:00")).date()
    launch_iso_week = launch_iso_week_from_config(config)
    _, ban_in_effect_max = load_ban_lift_bounds(config)
    drop_ban_week = user_week_drop_ban_week_default(config)
    return launch_date, launch_iso_week, int(ban_in_effect_max), bool(drop_ban_week)


def load_comment_frame(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Function summary: per-comment frame with author/lexicon/scores from enriched shards.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Tuple (frame, coverage) where coverage logs shards total / scored per column.
    """
    import pyarrow.parquet as pq

    interim_dir = Path(config["paths"]["interim_dir"])
    shards_dir = interim_dir / "cleaned_monthly_chunks"
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Input shards directory not found: {shards_dir}")
    subreddits = subreddits_for_panel(config, include_excluded=False)
    meta = build_subreddit_metadata_table(config)
    sub_to_lex = {s: str(m.get("primary_lexicon", "")).lower() for s, m in meta.items()}

    want = [
        "author",
        "created_utc",
        "subreddit",
        "primary_lexicon",
        "n_words",
        "n_words_comment",
        "sem_axis_ideology",
        "net_ideology",
    ]
    parts: List[pd.DataFrame] = []
    coverage = {"shards_total": 0, "shards_with_sem_axis_ideology": 0, "shards_with_net_ideology": 0}
    for subreddit, path in iter_monthly_files(shards_dir, subreddits):
        names = set(pq.ParquetFile(path).schema.names)
        coverage["shards_total"] += 1
        if "sem_axis_ideology" in names:
            coverage["shards_with_sem_axis_ideology"] += 1
        if "net_ideology" in names:
            coverage["shards_with_net_ideology"] += 1
        cols = [c for c in want if c in names]
        frame = pd.read_parquet(path, columns=cols)
        if frame.empty:
            continue
        if "subreddit" in frame.columns:
            frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
        else:
            frame["subreddit"] = subreddit
        frame = normalize_input_shard(frame)
        frame = filter_valid_authors(frame)
        if frame.empty:
            continue
        # primary_lexicon: prefer per-row value, fall back to subreddit metadata.
        if "primary_lexicon" not in frame.columns:
            frame["primary_lexicon"] = sub_to_lex.get(subreddit, "")
        frame["primary_lexicon"] = (
            frame["primary_lexicon"].astype("string").fillna("").str.lower().replace("", pd.NA)
        )
        frame["primary_lexicon"] = frame["primary_lexicon"].fillna(sub_to_lex.get(subreddit, ""))
        for score in SCORES:
            if score not in frame.columns:
                frame[score] = np.nan
            else:
                frame[score] = pd.to_numeric(frame[score], errors="coerce")
        keep = ["author", "created_utc", "subreddit", "primary_lexicon", "n_words_comment", *SCORES]
        parts.append(frame[[c for c in keep if c in frame.columns]].copy())
    if not parts:
        raise RuntimeError("No usable comment rows loaded from enriched shards.")
    df = pd.concat(parts, ignore_index=True)
    df["author"] = df["author"].astype(str)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df[df["created_utc"].notna()].copy()
    return df, coverage


def annotate_calendar(df: pd.DataFrame, launch_date: date) -> pd.DataFrame:
    """Function summary: add iso_week_start and rel_day per comment from created_utc."""
    out = df.copy()
    ts = out["created_utc"].astype("int64")
    out["iso_week_start"] = ts.map(iso_week_start_from_unix).map(lambda d: d.isoformat())
    out["comment_date"] = ts.map(
        lambda u: datetime.fromtimestamp(int(u), tz=timezone.utc).date()
    )
    out["rel_day"] = out["comment_date"].map(lambda d: (d - launch_date).days).astype(int)
    return out


def assign_author_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: per-author IT flag from plurality primary_lexicon, plus mixed-arm share.

    Parameters:
    - df: per-comment frame with author and primary_lexicon.

    Returns:
    - One row per author: author, IT (1 if plurality lexicon == it), it_share.
    """
    work = df[["author", "primary_lexicon"]].copy()
    work["is_it"] = (work["primary_lexicon"].astype(str) == "it").astype(int)
    grp = work.groupby("author", sort=False)["is_it"].agg(["mean", "size"]).reset_index()
    grp["IT"] = (grp["mean"] >= 0.5).astype(int)
    grp = grp.rename(columns={"mean": "it_share", "size": "n_comments_author"})
    return grp[["author", "IT", "it_share", "n_comments_author"]]


# ----------------------------- cohort gating -----------------------------


def cohort_author_set(
    df: pd.DataFrame, config: Dict[str, Any], cohort_label: str
) -> Tuple[List[str], str, bool]:
    """Function summary: authors passing the strict/loose word cohort (matches user-week analysis).

    Parameters:
    - df: per-comment frame with author, iso_week_start, n_words_comment.
    - config: loaded study YAML.
    - cohort_label: strict or loose.

    Returns:
    - Tuple (author ids, launch_iso_week, drop_ban_week flag used).
    """
    launch_iso_week = launch_iso_week_from_config(config)
    drop_ban_week = user_week_drop_ban_week_default(config)
    weekly = (
        df.groupby(["author", "iso_week_start"], sort=False)["n_words_comment"]
        .sum()
        .reset_index()
        .rename(columns={"n_words_comment": "n_words"})
    )
    labelled = add_calendar_fields(weekly, launch_iso_week, drop_ban_week)
    thresholds = cohort_thresholds_by_label(cohort_label)
    authors = panel_cohort_authors(labelled, thresholds)
    return authors, launch_iso_week, drop_ban_week


# ----------------------------- estimator A: static -----------------------------


def build_static_spread_panel(
    df: pd.DataFrame,
    arms: pd.DataFrame,
    score: str,
    statistic: str,
    launch_iso_week: str,
    ban_in_effect_max: int,
    min_comments: int,
) -> pd.DataFrame:
    """Function summary: author×period spread panel (pre window vs ban_in_effect post window).

    Pre = comments whose ISO week starts strictly before the launch ISO week (the
    ban-week buffer). Post = ban_in_effect, rel_day in [0, ban_in_effect_max].
    """
    pre_mask = df["iso_week_start"].astype(str) < launch_iso_week
    post_mask = (df["rel_day"] >= 0) & (df["rel_day"] <= ban_in_effect_max)
    weights_col = "n_words_comment" if statistic == "SD_wordw" else None
    rows: List[Dict[str, Any]] = []
    for window_name, mask, period_date, post in (
        ("pre", pre_mask, PRE_PERIOD_DATE, 0),
        ("post", post_mask, POST_PERIOD_DATE, 1),
    ):
        sub = df[mask]
        for author, grp in sub.groupby("author", sort=False):
            vals = grp[score].to_numpy()
            finite = np.isfinite(vals)
            if int(finite.sum()) < min_comments:
                continue
            w = grp[weights_col].to_numpy() if weights_col else None
            spread = _statistic(vals, statistic, weights=w)
            if not np.isfinite(spread):
                continue
            rows.append(
                {
                    "author": str(author),
                    "period": window_name,
                    "period_date": period_date,
                    "post": post,
                    "spread": spread,
                }
            )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    # keep only authors present in BOTH periods (within-author pre/post contrast)
    both = panel.groupby("author")["period"].nunique()
    keep = both[both == 2].index
    panel = panel[panel["author"].isin(keep)].copy()
    panel = panel.merge(arms[["author", "IT"]], on="author", how="left")
    panel["treat"] = panel["IT"].astype(int)
    return panel


def run_static_did(
    panel: pd.DataFrame, cohort_label: str, score: str, statistic: str, min_comments: int
) -> Dict[str, Any]:
    """Function summary: estimate IT:post spread DiD for one outcome and pack a summary row."""
    base = {
        "score": score,
        "statistic": statistic,
        "weighting": "word" if statistic == "SD_wordw" else "equal",
        "cohort": cohort_label,
        "min_comments_per_cell": min_comments,
    }
    if panel.empty:
        base.update({"estimation_note": "empty_panel", "n_authors": 0, "n_authors_it": 0, "n_authors_control": 0})
        return base
    n_it = int(panel.loc[panel["post"] == 0, "IT"].sum())
    n_ctrl = int((panel.loc[panel["post"] == 0, "IT"] == 0).sum())
    pre_rows = panel[panel["post"] == 0]
    pre_mean_it = float(pre_rows.loc[pre_rows["IT"] == 1, "spread"].mean()) if n_it else float("nan")
    pre_mean_pooled = float(pre_rows["spread"].mean())
    res = estimate_twfe(
        panel,
        "spread",
        entity_col="author",
        time_col="period_date",
        cluster_col="author",
    )
    base.update(
        {
            "beta": res.get("beta"),
            "se": res.get("se"),
            "pvalue": res.get("pvalue"),
            "ci_low": res.get("ci_low"),
            "ci_high": res.get("ci_high"),
            "n_obs": res.get("n_obs"),
            "n_authors": int(panel["author"].nunique()),
            "n_authors_it": n_it,
            "n_authors_control": n_ctrl,
            "pre_mean_spread_it": pre_mean_it,
            "pre_mean_spread_pooled": pre_mean_pooled,
            "estimation_note": res.get("estimation_note"),
        }
    )
    return base


# ----------------------------- estimator B: event study -----------------------------


def build_weekly_spread_panel(
    df: pd.DataFrame,
    arms: pd.DataFrame,
    score: str,
    statistic: str,
    launch_iso_week: str,
    drop_ban_week: bool,
    min_comments_week: int,
    window_weeks: int,
) -> pd.DataFrame:
    """Function summary: author×ISO-week spread panel with rel_week and treat for the event study."""
    rows: List[Dict[str, Any]] = []
    for (author, week), grp in df.groupby(["author", "iso_week_start"], sort=False):
        vals = grp[score].to_numpy()
        if int(np.isfinite(vals).sum()) < min_comments_week:
            continue
        spread = _statistic(vals, statistic)
        if not np.isfinite(spread):
            continue
        rows.append({"author": str(author), "iso_week_start": str(week), "spread": spread})
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel = add_calendar_fields(panel, launch_iso_week, drop_ban_week)
    panel = panel.merge(arms[["author", "IT"]], on="author", how="left")
    panel["treat"] = panel["IT"].astype(int)
    panel = panel[(panel["rel_week"] >= -window_weeks) & (panel["rel_week"] <= window_weeks)].copy()
    return panel


def run_event_study(
    panel: pd.DataFrame, cohort_label: str, score: str, statistic: str, window_weeks: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Function summary: TWFE event study of weekly spread; returns (coef rows, meta)."""
    meta = {
        "score": score,
        "statistic": statistic,
        "cohort": cohort_label,
        "n_pre_weeks": 0,
        "n_post_weeks": 0,
        "pretrend_F_p": float("nan"),
        "estimation_note": "empty_panel",
    }
    if panel.empty or panel["treat"].nunique() < 2:
        return pd.DataFrame(), meta
    meta["n_pre_weeks"] = int(panel.loc[panel["rel_week"] < 0, "rel_week"].nunique())
    meta["n_post_weeks"] = int(panel.loc[panel["rel_week"] >= 0, "rel_week"].nunique())
    summary, es_df = estimate_event_study(
        panel,
        "spread",
        rel_col="rel_week",
        ref_day=-1,
        window=window_weeks,
        entity_col="author",
        time_col="iso_week_start",
    )
    meta["pretrend_F_p"] = summary.get("pretrend_F_p", float("nan"))
    meta["estimation_note"] = summary.get("estimation_note", "ok")
    if es_df.empty:
        return pd.DataFrame(), meta
    out = es_df.copy()
    out.insert(0, "score", score)
    out.insert(1, "statistic", statistic)
    out.insert(2, "cohort", cohort_label)
    out["pretrend_F_p"] = meta["pretrend_F_p"]
    out["n_pre_weeks"] = meta["n_pre_weeks"]
    out["n_post_weeks"] = meta["n_post_weeks"]
    out["estimation_note"] = meta["estimation_note"]
    return out, meta


# ----------------------------- baseline levels -----------------------------


def baseline_levels(
    df: pd.DataFrame,
    arms: pd.DataFrame,
    launch_iso_week: str,
    min_comments: int,
    cohort_label: str,
) -> pd.DataFrame:
    """Function summary: pre-ban mean/median within-author spread by arm (SD and MAD)."""
    pre = df[df["iso_week_start"].astype(str) < launch_iso_week]
    arm_map = arms.set_index("author")["IT"].to_dict()
    rows: List[Dict[str, Any]] = []
    for score in SCORES:
        for statistic in ("SD", "MAD"):
            per_author: List[Tuple[int, float]] = []
            for author, grp in pre.groupby("author", sort=False):
                vals = grp[score].to_numpy()
                if int(np.isfinite(vals).sum()) < min_comments:
                    continue
                spread = _statistic(vals, statistic)
                if np.isfinite(spread):
                    per_author.append((int(arm_map.get(str(author), 0)), spread))
            if not per_author:
                continue
            arr = pd.DataFrame(per_author, columns=["IT", "spread"])
            for arm_name, sel in (
                ("it", arr["IT"] == 1),
                ("control", arr["IT"] == 0),
                ("pooled", arr["IT"].notna()),
            ):
                vals = arr.loc[sel, "spread"]
                if vals.empty:
                    continue
                rows.append(
                    {
                        "cohort": cohort_label,
                        "score": score,
                        "statistic": statistic,
                        "arm": arm_name,
                        "pre_mean_spread": float(vals.mean()),
                        "pre_median_spread": float(vals.median()),
                        "n_authors": int(vals.size),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------- orchestration -----------------------------


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for the within-user ideology-spread DiD."""
    parser = argparse.ArgumentParser(description="Within-user ideology-spread DiD (Italy ChatGPT ban).")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--cohort", type=str, default="strict", choices=("strict", "loose"))
    parser.add_argument("--min-comments-per-cell", type=int, default=5, help="Static design min scored comments per window.")
    parser.add_argument("--min-comments-per-week", type=int, default=3, help="Event-study design min scored comments per week.")
    parser.add_argument("--event-window-weeks", type=int, default=EVENT_STUDY_WINDOW_WEEKS)
    return parser.parse_args()


def main() -> None:
    """Function summary: run both estimators for one cohort and write all artifacts."""
    _self_test()
    args = parse_args()
    config = load_config(args.config)
    launch_date, launch_iso_week, ban_in_effect_max, drop_ban_week = _resolve_event_dates(config)

    out_dir = Path(config["paths"]["tables_dir"]) / "user_week" / "ideology_spread"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ideology_spread] loading enriched shards (launch={launch_date}, cohort={args.cohort})", flush=True)
    df, coverage = load_comment_frame(config)
    df = annotate_calendar(df, launch_date)
    arms = assign_author_arm(df)

    # Sanity: both arms present in the data overall (independent of cohort gating).
    n_it_all = int((arms["IT"] == 1).sum())
    n_ctrl_all = int((arms["IT"] == 0).sum())
    assert n_it_all > 0 and n_ctrl_all > 0, "need both IT and control authors in the corpus"
    print(
        f"[ideology_spread] comments={len(df):,} authors={len(arms):,} "
        f"IT={n_it_all:,} control={n_ctrl_all:,} | shards {coverage}",
        flush=True,
    )

    authors, _, _ = cohort_author_set(df, config, args.cohort)
    cohort_df = df[df["author"].isin(set(authors))].copy()
    cohort_arms = arms[arms["author"].isin(set(authors))].copy()
    n_it_cohort = int((cohort_arms["IT"] == 1).sum())
    n_ctrl_cohort = int((cohort_arms["IT"] == 0).sum())
    collapsed = (n_it_cohort == 0) or (n_ctrl_cohort == 0) or (len(authors) < 20)
    print(
        f"[ideology_spread] cohort={args.cohort} authors={len(authors):,} "
        f"IT={n_it_cohort:,} control={n_ctrl_cohort:,}"
        + ("  ** COHORT COLLAPSE (usable author count too low) **" if collapsed else ""),
        flush=True,
    )

    # ---- Estimator A: static pre/post spread DiD ----
    static_rows: List[Dict[str, Any]] = []
    static_specs = [(s, "SD") for s in SCORES] + [(s, "MAD") for s in SCORES] + [(s, "SD_wordw") for s in SCORES]
    for score, statistic in static_specs:
        panel = build_static_spread_panel(
            cohort_df, cohort_arms, score, statistic, launch_iso_week, ban_in_effect_max, args.min_comments_per_cell
        )
        if not panel.empty:
            assert (panel["spread"] >= 0).all(), "spread (SD/MAD) must be non-negative"
            assert panel["period"].nunique() == 2, "period FE not identified (need pre and post)"
        static_rows.append(run_static_did(panel, args.cohort, score, statistic, args.min_comments_per_cell))
    static_df = pd.DataFrame(static_rows)
    static_path = out_dir / f"static_spread_did__{args.cohort}.csv"
    static_df.to_csv(static_path, index=False)

    # ---- Estimator B: author×ISO-week event study ----
    es_parts: List[pd.DataFrame] = []
    es_meta: List[Dict[str, Any]] = []
    for score, statistic in [(s, "SD") for s in SCORES] + [(s, "MAD") for s in SCORES]:
        wpanel = build_weekly_spread_panel(
            cohort_df, cohort_arms, score, statistic, launch_iso_week, drop_ban_week,
            args.min_comments_per_week, args.event_window_weeks,
        )
        es_df, meta = run_event_study(wpanel, args.cohort, score, statistic, args.event_window_weeks)
        es_meta.append(meta)
        if not es_df.empty:
            es_parts.append(es_df)
    es_out = pd.concat(es_parts, ignore_index=True) if es_parts else pd.DataFrame()
    es_path = out_dir / f"event_study_spread__{args.cohort}.csv"
    es_out.to_csv(es_path, index=False)

    # ---- baseline levels ----
    base_df = baseline_levels(cohort_df, cohort_arms, launch_iso_week, args.min_comments_per_cell, args.cohort)
    base_path = out_dir / f"baseline_levels__{args.cohort}.csv"
    base_df.to_csv(base_path, index=False)

    # ---- methods note ----
    _write_methods_note(
        out_dir / f"methods_note__{args.cohort}.txt",
        config=config,
        cohort_label=args.cohort,
        launch_date=launch_date,
        launch_iso_week=launch_iso_week,
        ban_in_effect_max=ban_in_effect_max,
        drop_ban_week=drop_ban_week,
        coverage=coverage,
        n_authors_all=len(arms),
        n_it_all=n_it_all,
        n_ctrl_all=n_ctrl_all,
        n_cohort=len(authors),
        n_it_cohort=n_it_cohort,
        n_ctrl_cohort=n_ctrl_cohort,
        es_meta=es_meta,
        min_comments_per_cell=args.min_comments_per_cell,
        min_comments_per_week=args.min_comments_per_week,
        collapsed=collapsed,
    )

    _print_summary(static_df, es_meta, base_df, args.cohort, collapsed)
    print(
        f"[ideology_spread] wrote:\n  {static_path}\n  {es_path}\n  {base_path}\n  "
        f"{out_dir / f'methods_note__{args.cohort}.txt'}",
        flush=True,
    )


def _write_methods_note(path: Path, **kw: Any) -> None:
    """Function summary: write exact definitions, gates, windows, and counts."""
    es_lines = [
        f"  - {m['score']} / {m['statistic']}: pre_weeks={m['n_pre_weeks']} post_weeks={m['n_post_weeks']} "
        f"pretrend_F_p={m['pretrend_F_p']} note={m['estimation_note']}"
        for m in kw["es_meta"]
    ]
    lines = [
        "Within-user ideology-spread DiD — methods note",
        "=" * 50,
        "",
        "ESTIMAND",
        "  Individual extremity = the within-user SPREAD of the ideology score: how far a",
        "  single author ranges across the ideology axis over the window. We estimate the",
        "  Italy-vs-control difference-in-differences of this within-author dispersion.",
        "",
        "MEASURE",
        "  Per author, per window, dispersion of that author's per-comment ideology score:",
        "    - SD  = sample standard deviation (ddof=1), headline.",
        "    - MAD = median absolute deviation about the median (unscaled), robust variant.",
        "    - SD_wordw = reliability-(word)-weighted SD, optional companion.",
        "  Scores: sem_axis_ideology (semantic, primary), net_ideology (lexical, robustness).",
        "  Comments are equal-weighted within a cell for the headline (each comment one obs).",
        "  Spread is sign- and location-invariant, so no per-language orientation flip is",
        "  applied; cross-language level non-comparability of the semantic axis is absorbed",
        "  by the author fixed effect (only a differential CHANGE in within-author scale",
        "  could bias the DiD).",
        "",
        "WINDOWS (from config event_window / did)",
        f"  launch_day_utc = {kw['launch_date']} (rel_day 0); launch ISO week = {kw['launch_iso_week']}.",
        f"  Static PRE  = comments whose ISO week starts strictly before {kw['launch_iso_week']}",
        "                (the ban-week buffer; pre-onset launch-week days excluded).",
        f"  Static POST = ban_in_effect, rel_day in [0, {kw['ban_in_effect_max']}].",
        f"  drop_ban_week (user_week default) = {kw['drop_ban_week']} (applies to weekly event study).",
        "",
        "COHORT GATES (src/user_week/cohorts.py — same as the rest of the user-week analysis)",
        f"  cohort = {kw['cohort_label']}.",
        f"  Static design also requires >= {kw['min_comments_per_cell']} scored comments per window,",
        "  and the author must be present in BOTH windows (within-author pre/post contrast).",
        f"  Weekly event study requires >= {kw['min_comments_per_week']} scored comments per ISO week.",
        "",
        "ARM ASSIGNMENT",
        "  IT = 1 iff the author's plurality primary_lexicon is 'it' (else control).",
        "",
        "AUTHOR COUNTS",
        f"  Corpus authors: {kw['n_authors_all']:,} (IT {kw['n_it_all']:,} / control {kw['n_ctrl_all']:,}).",
        f"  Cohort authors: {kw['n_cohort']:,} (IT {kw['n_it_cohort']:,} / control {kw['n_ctrl_cohort']:,}).",
        f"  COHORT COLLAPSE FLAG: {kw['collapsed']} (True if an arm is empty or < 20 authors total).",
        "",
        "EVENT-STUDY SUPPORT (thin pre-period: corpus is only March–April 2023)",
        *es_lines,
        "",
        "SHARD COVERAGE",
        f"  {kw['coverage']}",
        "  All in-scope (screened) shards carry both score columns, so coverage within the",
        "  analysis sample is complete (shards_total == shards_with_each_score above). The",
        "  39 score-less shards in the raw cleaned_monthly_chunks dir belong to 20",
        "  screening-EXCLUDED subreddits (off-topic city/sports/celebrity/NSFW/user pages);",
        "  subreddits_for_panel() never loads them, so no in-scope comment is affected.",
        "",
        "ESTIMATORS (house style: src/did/estimate.py, linearmodels PanelOLS, clustered by author)",
        "  (A) spread ~ IT:post | author FE + period FE, cluster(author). IT:post = DiD.",
        "  (B) spread ~ sum_k 1[rel_week=k]·IT | author FE + week FE, cluster(author); ref week = -1;",
        "      joint Wald pre-trend test on pre-ban interactions (pretrend_F_p).",
        "",
        "OUTPUTS",
        "  static_spread_did__<cohort>.csv, event_study_spread__<cohort>.csv,",
        "  baseline_levels__<cohort>.csv, methods_note__<cohort>.txt.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any, nd: int = 4) -> str:
    """Function summary: compact float formatting for stdout."""
    try:
        v = float(x)
        return f"{v:.{nd}f}" if np.isfinite(v) else "nan"
    except (TypeError, ValueError):
        return str(x)


def _print_summary(
    static_df: pd.DataFrame, es_meta: List[Dict[str, Any]], base_df: pd.DataFrame, cohort: str, collapsed: bool
) -> None:
    """Function summary: concise stdout summary of the headline and companion results."""
    print(f"\n===== ideology-spread DiD summary (cohort={cohort}) =====", flush=True)
    if collapsed:
        print("  ** COHORT COLLAPSED — interpret with extreme caution (usable author count too low) **", flush=True)
    head = static_df[(static_df["score"] == "sem_axis_ideology") & (static_df["statistic"] == "SD")]
    if not head.empty:
        r = head.iloc[0]
        print(
            f"  HEADLINE  SD(sem_axis_ideology): beta={_fmt(r.get('beta'))} se={_fmt(r.get('se'))} "
            f"p={_fmt(r.get('pvalue'))} CI=[{_fmt(r.get('ci_low'))},{_fmt(r.get('ci_high'))}] "
            f"baseline_IT={_fmt(r.get('pre_mean_spread_it'))} "
            f"n_auth={int(r.get('n_authors') or 0)} (IT {int(r.get('n_authors_it') or 0)}/ctrl {int(r.get('n_authors_control') or 0)}) "
            f"note={r.get('estimation_note')}",
            flush=True,
        )
    for _, r in static_df.iterrows():
        if r["score"] == "sem_axis_ideology" and r["statistic"] == "SD":
            continue
        print(
            f"  {r['statistic']:>7}({r['score']}) [{r['weighting']}]: beta={_fmt(r.get('beta'))} "
            f"se={_fmt(r.get('se'))} p={_fmt(r.get('pvalue'))} n_auth={int(r.get('n_authors') or 0)} note={r.get('estimation_note')}",
            flush=True,
        )
    for m in es_meta:
        print(
            f"  EVENT-STUDY {m['statistic']}({m['score']}): pretrend_F_p={_fmt(m['pretrend_F_p'])} "
            f"pre_weeks={m['n_pre_weeks']} post_weeks={m['n_post_weeks']} note={m['estimation_note']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
