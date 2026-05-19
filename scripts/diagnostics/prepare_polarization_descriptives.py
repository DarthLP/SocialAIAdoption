"""
Script summary:
Aggregate polarization and AI-use features into Mar–Apr descriptives tables.

Functionality:
- Daily subreddit/family/country-panel series; ban-phase summaries; author retention and balanced panel.
- Distributional metrics (Esteban–Ray, bimodality coefficient); attrition from cleaning audits.
- AI first-stage diagnostic text for Italy vs control families.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import pandas as pd

READ_COLUMNS = [
    "id",
    "author",
    "subreddit",
    "date_utc",
    "body",
    "n_words",
    "political_rate_100w",
    "thread_is_political",
    "topic_family",
    "topic",
    "primary_lexicon",
    "lang_comment",
    "net_ideology",
    "extremity",
    "other_side_salience_rate_100w",
    "aggression_rate_100w",
    "negative_rate_100w",
    "anger_rate_100w",
    "left_hits",
    "center_hits",
    "right_hits",
    "ai_style_rate_100w",
    "thread_has_both_ideology_sides",
]

COUNTRY_PANEL_FAMILIES = {
    "italian": "Italy",
    "german_hub": "Germany",
    "spanish_hub": "Spain",
    "english_us_political": "US_political",
    "uk_political": "UK_political",
    "uk_hub": "UK_hub",
    "europe_hub_english": "EU_hub_en",
}


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import (  # noqa: E402
    load_config,
    load_polarization_config,
    resolve_primary_subreddits,
    subreddit_family_map,
    utc_ts,
)
from src.political_lexicon import bimodality_coefficient, esteban_ray_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare polarization descriptives tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def event_dates(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Function summary: return start, end_exclusive, launch_date, lift_date strings (YYYY-MM-DD).

    Parameters:
    - config: study YAML.

    Returns:
    - Tuple of date strings for filtering and phase splits.
    """
    ew = config["event_window"]
    start = datetime.fromtimestamp(utc_ts(ew["start_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
    end_excl = datetime.fromtimestamp(utc_ts(ew["end_utc_exclusive"]), tz=timezone.utc).strftime("%Y-%m-%d")
    launch = datetime.fromtimestamp(utc_ts(ew["launch_day_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
    refs = config.get("plot_reference_dates_utc") or []
    lift = "2023-04-29"
    if isinstance(refs, list) and len(refs) >= 2:
        lift = datetime.fromisoformat(str(refs[1]).replace("Z", "+00:00")).strftime("%Y-%m-%d")
    return start, end_excl, launch, lift


def ban_phase(date_utc: str, launch: str, lift: str) -> str:
    """Function summary: assign pre/ban/post label from calendar date.

    Parameters:
    - date_utc: YYYY-MM-DD.
    - launch: ban start date.
    - lift: ban end date (first post day).

    Returns:
    - Phase label.
    """
    if date_utc < launch:
        return "pre"
    if date_utc < lift:
        return "ban"
    return "post"


def load_comment_frame(interim_dir: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load enriched/feature parquet for all subreddits in window columns.

    Parameters:
    - interim_dir: interim root.
    - subreddits: subreddit names.

    Returns:
    - Combined dataframe.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        for shard in sorted(shard_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(shard)
            except Exception:
                continue
            cols = [c for c in READ_COLUMNS if c in df.columns]
            if not cols:
                continue
            chunk = df[cols].copy()
            chunk["subreddit"] = sub
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Function summary: compute weighted mean with zero-weight guard.

    Parameters:
    - series: values.
    - weights: weights.

    Returns:
    - Weighted mean or NaN.
    """
    w = weights.astype(float)
    if w.sum() <= 0:
        return float("nan")
    return float((series.astype(float) * w).sum() / w.sum())


def daily_subreddit_table(df: pd.DataFrame, pol_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: aggregate daily metrics by subreddit.

    Parameters:
    - df: comment frame in event window.
    - pol_cfg: polarization config.

    Returns:
    - Daily subreddit summary.
    """
    if df.empty:
        return pd.DataFrame()
    dip_min = int(pol_cfg.get("dip_min_n", 30))
    er_alpha = float(pol_cfg.get("er_alpha", 1.6))
    rows: List[Dict[str, Any]] = []
    for (sub, day), grp in df.groupby(["subreddit", "date_utc"], sort=True):
        nw = grp["n_words"].astype(float)
        rows.append(
            {
                "subreddit": sub,
                "date_utc": day,
                "n_comments": len(grp),
                "n_authors": grp["author"].nunique(),
                "median_n_words": float(grp["n_words"].median()),
                "political_rate_100w_mean": weighted_mean(grp["political_rate_100w"], nw),
                "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
                "extremity_mean": weighted_mean(grp["extremity"], nw),
                "other_side_salience_rate_100w_mean": weighted_mean(grp["other_side_salience_rate_100w"], nw),
                "aggression_rate_100w_mean": weighted_mean(grp["aggression_rate_100w"], nw),
                "ai_style_rate_100w_mean": weighted_mean(grp.get("ai_style_rate_100w", pd.Series(0.0)), nw),
                "political_thread_share": float(grp["thread_is_political"].astype(float).mean())
                if "thread_is_political" in grp.columns
                else float("nan"),
                "esteban_ray_index": esteban_ray_index(
                    float(grp["left_hits"].sum()),
                    float(grp["center_hits"].sum()),
                    float(grp["right_hits"].sum()),
                    alpha=er_alpha,
                ),
                "bimodality_coefficient": bimodality_coefficient(grp["net_ideology"].tolist())
                if len(grp) >= dip_min
                else float("nan"),
                "coverage_bimodality": float(len(grp) >= dip_min),
            }
        )
    return pd.DataFrame(rows)


def daily_family_table(df: pd.DataFrame, family_map: Dict[str, str]) -> pd.DataFrame:
    """Function summary: pool daily metrics by topic_family with author set unions.

    Parameters:
    - df: comment frame.
    - family_map: subreddit -> family.

    Returns:
    - Family-day aggregates.
    """
    if df.empty or "topic_family" not in df.columns:
        work = df.copy()
        work["topic_family"] = work["subreddit"].map(family_map)
    else:
        work = df
    rows: List[Dict[str, Any]] = []
    for (family, day), grp in work.groupby(["topic_family", "date_utc"], sort=True):
        authors: Set[str] = set(grp["author"].astype(str))
        nw = grp["n_words"].astype(float)
        rows.append(
            {
                "topic_family": family,
                "date_utc": day,
                "n_comments": len(grp),
                "n_authors_union": len(authors),
                "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
                "extremity_mean": weighted_mean(grp["extremity"], nw),
                "other_side_salience_rate_100w_mean": weighted_mean(grp["other_side_salience_rate_100w"], nw),
                "aggression_rate_100w_mean": weighted_mean(grp["aggression_rate_100w"], nw),
                "ai_style_rate_100w_mean": weighted_mean(grp.get("ai_style_rate_100w", pd.Series(0.0)), nw),
            }
        )
    return pd.DataFrame(rows)


def author_retention(df: pd.DataFrame, launch: str, lift: str) -> pd.DataFrame:
    """Function summary: per-subreddit author activity across ban phases.

    Parameters:
    - df: comments with date_utc and author.
    - launch: ban start date.
    - lift: ban end date.

    Returns:
    - Retention summary by subreddit.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["phase"] = work["date_utc"].map(lambda d: ban_phase(str(d), launch, lift))
    rows: List[Dict[str, Any]] = []
    for sub, grp in work.groupby("subreddit"):
        by_phase: DefaultDict[str, Set[str]] = defaultdict(set)
        for phase, authors in grp.groupby("phase")["author"]:
            by_phase[str(phase)] = set(authors.astype(str))
        pre, ban, post = by_phase.get("pre", set()), by_phase.get("ban", set()), by_phase.get("post", set())
        all_three = pre & ban & post
        rows.append(
            {
                "subreddit": sub,
                "n_authors_pre": len(pre),
                "n_authors_ban": len(ban),
                "n_authors_post": len(post),
                "share_authors_all_three_phases": len(all_three) / len(pre | ban | post)
                if (pre | ban | post)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def balanced_panel_daily(df: pd.DataFrame, launch: str, lift: str) -> pd.DataFrame:
    """Function summary: daily metrics restricted to authors active in pre, ban, and post.

    Parameters:
    - df: full comment frame.
    - launch: ban start.
    - lift: ban end.

    Returns:
    - Balanced-panel daily series by topic_family.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["phase"] = work["date_utc"].map(lambda d: ban_phase(str(d), launch, lift))
    author_phases = work.groupby("author")["phase"].apply(set)
    balanced_authors = {a for a, phases in author_phases.items() if {"pre", "ban", "post"}.issubset(phases)}
    sub = work[work["author"].isin(balanced_authors)]
    return daily_family_table(sub, {})


def window_summary(df: pd.DataFrame, launch: str, lift: str) -> pd.DataFrame:
    """Function summary: phase means by topic.

    Parameters:
    - df: comment frame.
    - launch: ban start.
    - lift: ban end.

    Returns:
    - Topic × phase table.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["phase"] = work["date_utc"].map(lambda d: ban_phase(str(d), launch, lift))
    rows: List[Dict[str, Any]] = []
    group_col = "topic" if "topic" in work.columns else "subreddit"
    for (topic, phase), grp in work.groupby([group_col, "phase"]):
        nw = grp["n_words"].astype(float)
        rows.append(
            {
                group_col: topic,
                "phase": phase,
                "n_comments": len(grp),
                "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
                "extremity_mean": weighted_mean(grp["extremity"], nw),
                "other_side_salience_rate_100w_mean": weighted_mean(grp["other_side_salience_rate_100w"], nw),
                "aggression_rate_100w_mean": weighted_mean(grp["aggression_rate_100w"], nw),
            }
        )
    return pd.DataFrame(rows)


def country_panel_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: map topic_family to country panel labels for cross-country plots.

    Parameters:
    - df: comment frame with topic_family.

    Returns:
    - Country-panel daily AI and polarization means.
    """
    if df.empty or "topic_family" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["country_panel"] = work["topic_family"].map(COUNTRY_PANEL_FAMILIES)
    work = work[work["country_panel"].notna()]
    rows: List[Dict[str, Any]] = []
    for (panel, day), grp in work.groupby(["country_panel", "date_utc"]):
        nw = grp["n_words"].astype(float)
        rows.append(
            {
                "country_panel": panel,
                "date_utc": day,
                "n_comments": len(grp),
                "ai_style_rate_100w_mean": weighted_mean(grp.get("ai_style_rate_100w", pd.Series(0.0)), nw),
                "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
                "esteban_ray_index": esteban_ray_index(
                    float(grp["left_hits"].sum()),
                    float(grp["center_hits"].sum()),
                    float(grp["right_hits"].sum()),
                ),
            }
        )
    return pd.DataFrame(rows)


def attrition_table(tables_dir: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: summarize row counts from cleaning audit CSVs when present.

    Parameters:
    - tables_dir: study tables dir.
    - subreddits: primary list.

    Returns:
    - Attrition summary.
    """
    audit_path = tables_dir / "cleaning" / "clean_daily_chunks_audit_by_subreddit.csv"
    if not audit_path.is_file():
        return pd.DataFrame({"note": ["Run clean_daily_chunks for attrition audit"]})
    audit = pd.read_csv(audit_path)
    keep_cols = [c for c in audit.columns if c in ("subreddit", "rows_kept", "rows_read", "rows_dropped")]
    if "subreddit" not in audit.columns:
        return pd.DataFrame()
    sub_set = set(subreddits)
    out = audit[audit["subreddit"].isin(sub_set)][keep_cols].copy()
    return out


def write_ai_diagnostic(country_daily: pd.DataFrame, out_path: Path, launch: str) -> None:
    """Function summary: write short Italy vs control AI first-stage diagnostic note.

    Parameters:
    - country_daily: country panel daily table.
    - out_path: output text path.
    - launch: ban start date string.
    """
    lines = [
        "AI first-stage diagnostic (descriptive only).",
        f"Ban start (UTC date): {launch}",
        "Compare ai_style_rate_100w_mean for Italy vs control panels around ban dates.",
        "",
    ]
    if country_daily.empty or "ai_style_rate_100w_mean" not in country_daily.columns:
        lines.append("No AI feature columns found; run compute_ai_use_features.py first.")
    else:
        it = country_daily[country_daily["country_panel"] == "Italy"]
        if not it.empty:
            pre = it[it["date_utc"] < launch]["ai_style_rate_100w_mean"].mean()
            ban = it[it["date_utc"] >= launch]["ai_style_rate_100w_mean"].mean()
            lines.append(f"Italy mean ai_style pre={pre:.4f} ban/post={ban:.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: write all descriptives tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    pol_cfg = load_polarization_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    out_dir = tables_dir / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end_excl, launch, lift = event_dates(config)
    subs = resolve_primary_subreddits(config)
    family_map = subreddit_family_map(config)

    df = load_comment_frame(interim_dir, subs)
    if df.empty:
        print("[prepare_polarization_descriptives] no parquet data found", flush=True)
        return
    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    if "topic_family" not in df.columns:
        df["topic_family"] = df["subreddit"].map(family_map)

    daily_subreddit_table(df, pol_cfg).to_csv(out_dir / "daily_by_subreddit.csv", index=False)
    daily_family_table(df, family_map).to_csv(out_dir / "daily_by_topic_family.csv", index=False)
    country = country_panel_daily(df)
    country.to_csv(out_dir / "daily_country_panel.csv", index=False)
    author_retention(df, launch, lift).to_csv(out_dir / "author_retention_by_subreddit.csv", index=False)
    balanced_panel_daily(df, launch, lift).to_csv(out_dir / "balanced_panel_daily.csv", index=False)
    window_summary(df, launch, lift).to_csv(out_dir / "window_summary_by_topic.csv", index=False)

    if "thread_is_political" in df.columns:
        pol_threads = df[df["thread_is_political"].astype(bool)]
        daily_family_table(pol_threads, family_map).to_csv(
            out_dir / "stratified_political_threads_daily.csv", index=False
        )

    attrition_table(tables_dir, subs).to_csv(out_dir / "attrition_by_subreddit.csv", index=False)
    write_ai_diagnostic(country, out_dir / "ai_first_stage_diagnostic.txt", launch)
    print(f"[prepare_polarization_descriptives] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
