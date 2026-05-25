"""
Script summary:
Aggregate polarization and AI-use features into Mar–Apr descriptives tables.

Functionality:
- Daily subreddit/family/country-panel series; political-universe slice tables (in-tree vs out-tree).
- Ban-phase summaries; author retention and balanced panel.
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
    "comment_in_political_universe",
    "link_id",
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
    "ai_sentence_length_variance",
    "em_dash_count",
    "exclamation_count",
    "semicolon_count",
    "colon_count",
    "hedging_phrase_hits",
    "sentence_count_comment",
    "total_word_chars_comment",
    "avg_words_per_sentence_comment",
]

REQUIRED_FEATURE_COLUMNS = (
    "net_ideology",
    "other_side_salience_rate_100w",
    "aggression_rate_100w",
    "left_hits",
)

COUNTRY_PANEL_FAMILIES = {
    "it_political": "Italy_political",
    "it_others": "Italy_others",
    "de": "Germany",
    "us": "US_political",
    "uk": "UK",
    "eu": "EU_hub_en",
}

UNIVERSE_SLICE_IN = "in_political_tree"
UNIVERSE_SLICE_OUT = "out_political_tree"
ITALY_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})



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

from src.config_utils import (  # noqa: E402
    load_config,
    load_polarization_config,
    require_dominant_v1_ideology_scoring,
    resolve_primary_subreddits,
    subreddit_family_map,
    subreddit_topic_map,
    utc_ts,
)
from src.comment_style import compute_complexity_index  # noqa: E402
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


def load_comment_frame(shard_root: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load enriched/feature parquet for all subreddits in window columns.

    Parameters:
    - shard_root: directory containing per-subreddit monthly Parquet folders
      (e.g. data/interim/italy_polarization/cleaned_monthly_chunks).
    - subreddits: subreddit names.

    Returns:
    - Combined dataframe.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
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


def validate_feature_columns_present(shard_root: Path, sample_shards: int = 5) -> None:
    """Function summary: fail fast if polarization features were not written to Parquet.

    Parameters:
    - shard_root: path to cleaned_monthly_chunks.
    - sample_shards: number of shards to inspect.

    Raises:
    - SystemExit: when no shard contains required feature columns.
    """
    paths = [p for p in sorted(shard_root.rglob("*.parquet")) if p.stat().st_size >= 8][:sample_shards]
    if not paths:
        raise SystemExit(
            f"No Parquet under {shard_root}. Run clean_daily_chunks.py and enrich_cleaned_chunks.py first."
        )
    ok = 0
    for path in paths:
        if path.stat().st_size < 8:
            continue
        try:
            cols = set(pd.read_parquet(path).columns)
        except Exception:
            continue
        if all(c in cols for c in REQUIRED_FEATURE_COLUMNS):
            ok += 1
    if ok == 0:
        raise SystemExit(
            "Polarization/AI features missing on interim Parquet. After enrich finishes, run:\n"
            "  .venv/bin/python scripts/features/compute_ai_use_features.py "
            "--config config/italy_polarization_setup.yaml\n"
            "  .venv/bin/python scripts/features/compute_polarization_features.py "
            "--config config/italy_polarization_setup.yaml\n"
            "  .venv/bin/python scripts/features/compute_comment_style_features.py "
            "--config config/italy_polarization_setup.yaml\n"
            "Then re-run this script."
        )


def enrich_style_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: unify em-dash and style fields for aggregation.

    Parameters:
    - df: comment-level frame.

    Returns:
    - Copy with _em_dash_count helper column.
    """
    out = df.copy()
    if "em_dash_count" in out.columns:
        out["_em_dash_count"] = out["em_dash_count"].astype(float)
    elif "ai_em_dash_count" in out.columns:
        out["_em_dash_count"] = out["ai_em_dash_count"].astype(float)
    elif "pol_em_dash_count" in out.columns:
        out["_em_dash_count"] = out["pol_em_dash_count"].astype(float)
    else:
        out["_em_dash_count"] = 0.0
    return out


def style_aggregate_fields(grp: pd.DataFrame) -> Dict[str, float]:
    """Function summary: aggregate em-dash and related style metrics for one group.

    Parameters:
    - grp: comment rows sharing a day/family/subreddit.

    Returns:
    - Dict of style metric means/rates for descriptives tables.
    """
    nw = grp["n_words"].astype(float)
    total_words = float(nw.sum())
    out: Dict[str, float] = {}
    if "_em_dash_count" in grp.columns and total_words > 0:
        out["em_dash_rate_100w"] = 100.0 * float(grp["_em_dash_count"].sum()) / total_words
    else:
        out["em_dash_rate_100w"] = 0.0
    if total_words > 0:
        if "semicolon_count" in grp.columns:
            out["semicolon_rate_100w"] = 100.0 * float(grp["semicolon_count"].sum()) / total_words
        if "colon_count" in grp.columns:
            out["colon_rate_100w"] = 100.0 * float(grp["colon_count"].sum()) / total_words
        if "hedging_phrase_hits" in grp.columns:
            out["hedging_phrase_rate_100w"] = 100.0 * float(grp["hedging_phrase_hits"].sum()) / total_words
        if "exclamation_count" in grp.columns:
            out["exclamation_rate_100w_mean"] = 100.0 * float(grp["exclamation_count"].sum()) / total_words
    if "avg_words_per_sentence_comment" in grp.columns:
        out["avg_words_per_sentence_mean"] = weighted_mean(grp["avg_words_per_sentence_comment"], nw)
    elif "ai_avg_words_per_sentence" in grp.columns:
        out["avg_words_per_sentence_mean"] = weighted_mean(grp["ai_avg_words_per_sentence"], nw)
    elif "pol_avg_words_per_sentence" in grp.columns:
        out["avg_words_per_sentence_mean"] = weighted_mean(grp["pol_avg_words_per_sentence"], nw)
    if "ai_sentence_length_variance" in grp.columns:
        out["sentence_length_variance_mean"] = weighted_mean(grp["ai_sentence_length_variance"], nw)
    if (
        "sentence_count_comment" in grp.columns
        and "total_word_chars_comment" in grp.columns
        and total_words > 0
    ):
        out["complexity_index"] = compute_complexity_index(
            int(grp["sentence_count_comment"].sum()),
            int(total_words),
            int(grp["total_word_chars_comment"].sum()),
            int(len(grp)),
        )
    return out


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


def ideology_bucket_aggregate_fields(grp: pd.DataFrame, eps: float = 1.0e-6) -> Dict[str, float]:
    """Function summary: pool ideology hit sums into per-100w rates and pole share.

    Parameters:
    - grp: comment rows for one day and aggregation group.
    - eps: stabilizer for pole_share when ideology hits are zero.

    Returns:
    - Dict with left/center/right_rate_100w_mean and pole_share.
    """
    nw = float(grp["n_words"].astype(float).sum())
    left = float(grp["left_hits"].sum())
    center = float(grp["center_hits"].sum())
    right = float(grp["right_hits"].sum())
    ideology_total = left + center + right
    if nw <= 0:
        rate_nan = float("nan")
        return {
            "left_rate_100w_mean": rate_nan,
            "center_rate_100w_mean": rate_nan,
            "right_rate_100w_mean": rate_nan,
            "pole_share": float("nan"),
        }
    pole_share = float((left + right) / (ideology_total + eps)) if ideology_total > 0 else float("nan")
    return {
        "left_rate_100w_mean": 100.0 * left / nw,
        "center_rate_100w_mean": 100.0 * center / nw,
        "right_rate_100w_mean": 100.0 * right / nw,
        "pole_share": pole_share,
    }


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
                "political_comment_share": float(grp["comment_in_political_universe"].astype(float).mean())
                if "comment_in_political_universe" in grp.columns
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
                **ideology_bucket_aggregate_fields(grp),
                **style_aggregate_fields(grp),
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
                "ai_style_rate_100w_mean": weighted_mean(grp["ai_style_rate_100w"], nw)
                if "ai_style_rate_100w" in grp.columns
                else float("nan"),
                "political_rate_100w_mean": weighted_mean(grp["political_rate_100w"], nw)
                if "political_rate_100w" in grp.columns
                else float("nan"),
                "political_thread_share": float(grp["thread_is_political"].astype(float).mean())
                if "thread_is_political" in grp.columns
                else float("nan"),
                "political_comment_share": float(grp["comment_in_political_universe"].astype(float).mean())
                if "comment_in_political_universe" in grp.columns
                else float("nan"),
                **ideology_bucket_aggregate_fields(grp),
                **style_aggregate_fields(grp),
            }
        )
    return pd.DataFrame(rows)


def daily_topic_table(
    df: pd.DataFrame,
    family_map: Dict[str, str],
    topic_map: Dict[str, str],
    pol_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: pool daily metrics by assigned topic (e.g. it_political).

    Parameters:
    - df: comment frame with topic or subreddit for mapping.
    - family_map: subreddit -> topic_family.
    - topic_map: subreddit -> topic.
    - pol_cfg: polarization config for Esteban–Ray alpha.

    Returns:
    - Topic-day aggregates with topic_family for faceting.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "topic" not in work.columns:
        work["topic"] = work["subreddit"].map(topic_map)
    if "topic_family" not in work.columns:
        work["topic_family"] = work["subreddit"].map(family_map)
    work = work.dropna(subset=["topic"])
    er_alpha = float(pol_cfg.get("er_alpha", 1.6))
    rows: List[Dict[str, Any]] = []
    for (topic, day), grp in work.groupby(["topic", "date_utc"], sort=True):
        authors: Set[str] = set(grp["author"].astype(str))
        nw = grp["n_words"].astype(float)
        family = grp["topic_family"].iloc[0] if "topic_family" in grp.columns else ""
        rows.append(
            {
                "topic": topic,
                "topic_family": family,
                "date_utc": day,
                "n_comments": len(grp),
                "n_authors_union": len(authors),
                "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
                "extremity_mean": weighted_mean(grp["extremity"], nw),
                "other_side_salience_rate_100w_mean": weighted_mean(grp["other_side_salience_rate_100w"], nw),
                "aggression_rate_100w_mean": weighted_mean(grp["aggression_rate_100w"], nw),
                "ai_style_rate_100w_mean": weighted_mean(grp["ai_style_rate_100w"], nw)
                if "ai_style_rate_100w" in grp.columns
                else float("nan"),
                "political_rate_100w_mean": weighted_mean(grp["political_rate_100w"], nw)
                if "political_rate_100w" in grp.columns
                else float("nan"),
                "political_thread_share": float(grp["thread_is_political"].astype(float).mean())
                if "thread_is_political" in grp.columns
                else float("nan"),
                "political_comment_share": float(grp["comment_in_political_universe"].astype(float).mean())
                if "comment_in_political_universe" in grp.columns
                else float("nan"),
                "esteban_ray_index": esteban_ray_index(
                    float(grp["left_hits"].sum()),
                    float(grp["center_hits"].sum()),
                    float(grp["right_hits"].sum()),
                    alpha=er_alpha,
                ),
                **ideology_bucket_aggregate_fields(grp),
                **style_aggregate_fields(grp),
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


def _universe_slice_label(in_universe: bool) -> str:
    """Function summary: map boolean political-universe flag to slice label string.

    Parameters:
    - in_universe: comment_in_political_universe value.

    Returns:
    - Slice id for grouping and CSV output.
    """
    return UNIVERSE_SLICE_IN if in_universe else UNIVERSE_SLICE_OUT


def _slice_metrics_row(grp: pd.DataFrame, pol_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Function summary: compute weighted daily metrics for one universe slice group.

    Parameters:
    - grp: comment rows for one panel-day-slice.
    - pol_cfg: polarization config (Esteban–Ray alpha).

    Returns:
    - Metric fields shared with daily_family_table.
    """
    nw = grp["n_words"].astype(float)
    er_alpha = float(pol_cfg.get("er_alpha", 1.6))
    row: Dict[str, Any] = {
        "n_comments": len(grp),
        "n_authors_union": grp["author"].nunique(),
        "net_ideology_mean": weighted_mean(grp["net_ideology"], nw),
        "extremity_mean": weighted_mean(grp["extremity"], nw),
        "other_side_salience_rate_100w_mean": weighted_mean(grp["other_side_salience_rate_100w"], nw),
        "aggression_rate_100w_mean": weighted_mean(grp["aggression_rate_100w"], nw),
        "ai_style_rate_100w_mean": weighted_mean(grp["ai_style_rate_100w"], nw)
        if "ai_style_rate_100w" in grp.columns
        else float("nan"),
        "political_rate_100w_mean": weighted_mean(grp["political_rate_100w"], nw)
        if "political_rate_100w" in grp.columns
        else float("nan"),
        "political_thread_share": float(grp["thread_is_political"].astype(float).mean())
        if "thread_is_political" in grp.columns
        else float("nan"),
        "esteban_ray_index": esteban_ray_index(
            float(grp["left_hits"].sum()),
            float(grp["center_hits"].sum()),
            float(grp["right_hits"].sum()),
            alpha=er_alpha,
        ),
        **ideology_bucket_aggregate_fields(grp),
        **style_aggregate_fields(grp),
    }
    return row


def daily_metrics_by_slice(
    df: pd.DataFrame,
    group_cols: List[str],
    pol_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: daily aggregates split by political-tree membership.

    Parameters:
    - df: comment frame with comment_in_political_universe.
    - group_cols: extra grouping keys (e.g. country_panel); may be empty for pooled Italy.
    - pol_cfg: polarization config.

    Returns:
    - Daily table with universe_slice and share_of_panel_comments.
    """
    if df.empty or "comment_in_political_universe" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["universe_slice"] = work["comment_in_political_universe"].astype(bool).map(_universe_slice_label)
    group_keys = list(group_cols) + ["date_utc", "universe_slice"]
    panel_keys = list(group_cols) + ["date_utc"]
    totals = (
        work.groupby(panel_keys, sort=True)
        .size()
        .reset_index(name="panel_n_comments")
    )
    rows: List[Dict[str, Any]] = []
    for key_tuple, grp in work.groupby(group_keys, sort=True):
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        key_dict = dict(zip(group_keys, key_tuple))
        nw = grp["n_words"].astype(float)
        row = {
            **{c: key_dict[c] for c in group_cols},
            "date_utc": key_dict["date_utc"],
            "universe_slice": key_dict["universe_slice"],
            **_slice_metrics_row(grp, pol_cfg),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.merge(totals, on=panel_keys, how="left")
    out["share_of_panel_comments"] = out["n_comments"] / out["panel_n_comments"].replace(0, float("nan"))
    out = out.drop(columns=["panel_n_comments"])
    return out


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
                **style_aggregate_fields(grp),
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
    require_dominant_v1_ideology_scoring(config)
    pol_cfg = load_polarization_config(config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    tables_dir = Path(config["paths"]["tables_dir"])
    out_dir = tables_dir / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_feature_columns_present(shard_root)

    start, end_excl, launch, lift = event_dates(config)
    subs = resolve_primary_subreddits(config)
    family_map = subreddit_family_map(config)
    topic_map = subreddit_topic_map(config, include_topic_aliases=False)

    df = load_comment_frame(shard_root, subs)
    if df.empty:
        print("[prepare_polarization_descriptives] no parquet data found", flush=True)
        return
    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    df = enrich_style_helper_columns(df)
    if "topic_family" not in df.columns:
        df["topic_family"] = df["subreddit"].map(family_map)

    daily_subreddit_table(df, pol_cfg).to_csv(out_dir / "daily_by_subreddit.csv", index=False)
    daily_family_table(df, family_map).to_csv(out_dir / "daily_by_topic_family.csv", index=False)
    daily_topic_table(df, family_map, topic_map, pol_cfg).to_csv(out_dir / "daily_by_topic.csv", index=False)
    country = country_panel_daily(df)
    country.to_csv(out_dir / "daily_country_panel.csv", index=False)
    author_retention(df, launch, lift).to_csv(out_dir / "author_retention_by_subreddit.csv", index=False)
    balanced_panel_daily(df, launch, lift).to_csv(out_dir / "balanced_panel_daily.csv", index=False)
    window_summary(df, launch, lift).to_csv(out_dir / "window_summary_by_topic.csv", index=False)

    if "thread_is_political" in df.columns:
        pol_threads = df[df["thread_is_political"].astype(bool)]
        daily_family_table(pol_threads, family_map).to_csv(
            out_dir / "stratified_political_threads_daily_legacy.csv", index=False
        )

    if "comment_in_political_universe" in df.columns:
        pol_comments = df[df["comment_in_political_universe"].astype(bool)]
        daily_family_table(pol_comments, family_map).to_csv(
            out_dir / "stratified_political_comments_daily.csv", index=False
        )

    if pol_cfg.get("restrict_to_political_comments") and "comment_in_political_universe" in df.columns:
        pol_only = df[df["comment_in_political_universe"].astype(bool)]
        daily_subreddit_table(pol_only, pol_cfg).to_csv(
            out_dir / "daily_by_subreddit_political_universe.csv", index=False
        )
        daily_family_table(pol_only, family_map).to_csv(
            out_dir / "daily_by_topic_family_political_universe.csv", index=False
        )
        daily_topic_table(pol_only, family_map, topic_map, pol_cfg).to_csv(
            out_dir / "daily_by_topic_political_universe.csv", index=False
        )

    if "comment_in_political_universe" in df.columns:
        cp_work = df.copy()
        cp_work["country_panel"] = cp_work["topic_family"].map(COUNTRY_PANEL_FAMILIES)
        cp_work = cp_work[cp_work["country_panel"].notna()]
        daily_metrics_by_slice(cp_work, ["country_panel"], pol_cfg).to_csv(
            out_dir / "daily_country_panel_by_universe_slice.csv", index=False
        )
        italy_all = cp_work[cp_work["topic_family"].isin(ITALY_TOPIC_FAMILIES)]
        daily_metrics_by_slice(italy_all, [], pol_cfg).to_csv(
            out_dir / "daily_italy_all_by_universe_slice.csv", index=False
        )
        it_pol = cp_work[cp_work["topic_family"] == "it_political"]
        daily_metrics_by_slice(it_pol, [], pol_cfg).to_csv(
            out_dir / "daily_it_political_by_universe_slice.csv", index=False
        )
        it_oth = cp_work[cp_work["topic_family"] == "it_others"]
        daily_metrics_by_slice(it_oth, [], pol_cfg).to_csv(
            out_dir / "daily_it_others_by_universe_slice.csv", index=False
        )

    attrition_table(tables_dir, subs).to_csv(out_dir / "attrition_by_subreddit.csv", index=False)
    write_ai_diagnostic(country, out_dir / "ai_first_stage_diagnostic.txt", launch)
    print(f"[prepare_polarization_descriptives] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
