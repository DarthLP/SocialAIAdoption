"""
Script summary:
Stage-3 enrichment for cleaned Italy polarization Parquet: taxonomy columns,
language-matched political lexicon scores, thread roll-ups, topic assignment audit,
and screening audit tables.

Functionality:
- Reads screening outputs and adds metadata plus political scores per comment.
- Assigns topics via first-match-wins priority (metadata → controls → config topic lists → graded WW thresholds).
- Political salience from data/raw/political_lexicon_parallel.csv (grades 1–3 → points 1/2/3; thread political if thread points ≥3).
- Writes assignment, political profile, and political mismatch audit CSVs.

How to apply/run:
  .venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml --assign-only
  .venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import langid
except ImportError as exc:
    raise SystemExit("langid is required: pip install langid") from exc


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
    build_subreddit_metadata_table,
    forum_political_thresholds,
    infer_subreddit_topic,
    italian_arms_for_langid,
    load_config,
    load_screening_config,
    load_screening_pooled,
    load_subreddit_metadata,
    parallel_political_lexicon_path,
    resolve_primary_subreddits,
    screening_by_subreddit,
    shard_dir_is_enriched,
    should_skip_screened_subreddit,
    subreddit_arm_map,
    subreddit_screening_action,
    subreddit_family_map,
    subreddit_topic_map,
    topic_family_map,
)
from src.political_lexicon import (  # noqa: E402
    get_graded_matcher,
    political_rate_100w,
    score_bodies_political_salience,
)

POLITICAL_TOPICS = frozenset({"it_political", "it_pure_political", "us", "uk_political"})
ITALIAN_POLITICAL_TOPICS = frozenset({"it_political", "it_pure_political"})
CONFIG_LIST_TOPICS = frozenset({"it_political", "de", "eu", "us", "uk", "uk_political"})
CONTROL_SUBREDDITS_WARN_ZERO_WW = frozenset({"europe", "ukpolitics", "unitedkingdom", "de"})


def read_parquet_shard_safe(shard: Path, columns: List[str] | None = None) -> pd.DataFrame | None:
    """Function summary: read a monthly Parquet shard, skipping corrupt or empty files.

    Parameters:
    - shard: path to a cleaned monthly Parquet file.
    - columns: optional column subset for pandas.read_parquet.

    Returns:
    - DataFrame on success, or None if the file is missing, empty, or unreadable.
    """
    if not shard.is_file() or shard.stat().st_size < 8:
        return None
    try:
        if columns is None:
            return pd.read_parquet(shard)
        return pd.read_parquet(shard, columns=columns)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for enrichment."""
    parser = argparse.ArgumentParser(description="Enrich cleaned monthly Parquet with taxonomy and political scores.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Enrich excluded subreddits too (for audit plots).",
    )
    parser.add_argument(
        "--skip-pipeline-plots",
        action="store_true",
        help="Do not run plot_cleaning_pipeline_trends.py after enrichment (default: run when inputs exist).",
    )
    parser.add_argument(
        "--assign-only",
        action="store_true",
        help="Re-assign topics from existing political_weighted_points columns (no rescoring).",
    )
    parser.add_argument(
        "--skip-langid",
        action="store_true",
        help="Skip per-comment langid.classify during finalize pass.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel shard workers for scoring pass (default: min(8, cpu_count-1); use 1 for deterministic logs).",
    )
    return parser.parse_args()


def default_worker_count() -> int:
    """Function summary: choose default ProcessPool worker count.

    Returns:
    - Worker count at least 1.
    """
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 1))


def run_cleaning_pipeline_plots_if_ready(config_path: str) -> None:
    """Function summary: refresh cleaning_pipeline tables/figures after stage 3 CSVs are written.

    Parameters:
    - config_path: path to study YAML passed to enrich.

    Returns:
    - None. Logs and swallows plot errors so enrichment still completes.
    """
    import importlib.util

    plot_path = PROJECT_ROOT / "scripts" / "diagnostics" / "plot_cleaning_pipeline_trends.py"
    spec = importlib.util.spec_from_file_location("_plot_cleaning_pipeline_mod", plot_path)
    if spec is None or spec.loader is None:
        print("[enrich_cleaned_chunks] skip pipeline plots: could not load plot module", flush=True)
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_cleaning_pipeline_plots = mod.run_cleaning_pipeline_plots

    config = load_config(PROJECT_ROOT / config_path)
    print("[enrich_cleaned_chunks] running cleaning pipeline plots (post-enrich)...", flush=True)
    try:
        run_cleaning_pipeline_plots(config, project_root=PROJECT_ROOT)
    except Exception as exc:
        print(f"[enrich_cleaned_chunks] pipeline plots failed (non-fatal): {exc}", flush=True)


def load_topic_overrides(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Function summary: extract topic override mapping from metadata YAML.

    Parameters:
    - metadata: loaded metadata dict.

    Returns:
    - Subreddit -> topic overrides.
    """
    raw = metadata.get("topic_overrides", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def compute_political_thresholds(
    subreddit_stats: Dict[str, Dict[str, float]],
    screening: Dict[str, Any],
) -> Tuple[float, float, float]:
    """Function summary: soft/pure word-weighted rate thresholds for Italian topic assignment.

    Parameters:
    - subreddit_stats: per-subreddit stats (unused; kept for API stability).
    - screening: screening config.

    Returns:
    - Tuple (soft_threshold, pure_threshold, politicaITA_benchmark_median for audit).
    """
    del subreddit_stats
    soft_tau, pure_tau = forum_political_thresholds(screening)
    benchmark = 0.0
    return soft_tau, pure_tau, benchmark


SCORE_COLS = [
    "political_g1_hits",
    "political_g2_hits",
    "political_g3_hits",
    "political_weighted_points",
    "n_words",
]


def _empty_subreddit_stats(lex_lang: str) -> Dict[str, Any]:
    """Function summary: return zeroed forum-level political stats."""
    return {
        "primary_lexicon": lex_lang,
        "total_weighted_points": 0,
        "total_words": 0,
        "n_comments": 0,
        "n_comments_with_hits": 0,
        "n_shards_read": 0,
        "n_shards_skipped": 0,
        "median_political_rate_100w": 0.0,
        "mean_political_rate_100w": 0.0,
        "word_weighted_political_rate_100w": 0.0,
        "comment_hit_share": 0.0,
    }


def _stats_from_scored_df(scored: pd.DataFrame, lex_lang: str) -> Dict[str, Any]:
    """Function summary: build forum stats fragment from scored comment columns.

    Parameters:
    - scored: dataframe with political_weighted_points and n_words.
    - lex_lang: primary lexicon code.

    Returns:
    - Stats dict fragment.
    """
    if scored.empty:
        return _empty_subreddit_stats(lex_lang)
    points = scored["political_weighted_points"].astype(int)
    n_words = scored["n_words"].astype(int)
    rates = [
        political_rate_100w(int(p), int(w))
        for p, w in zip(points.tolist(), n_words.tolist(), strict=True)
    ]
    total_points = int(points.sum())
    total_words = int(n_words.sum())
    series = pd.Series(rates)
    return {
        "primary_lexicon": lex_lang,
        "total_weighted_points": total_points,
        "total_words": total_words,
        "n_comments": len(scored),
        "n_comments_with_hits": int((points > 0).sum()),
        "n_shards_read": 1,
        "n_shards_skipped": 0,
        "median_political_rate_100w": float(series.median()),
        "mean_political_rate_100w": float(series.mean()),
        "word_weighted_political_rate_100w": political_rate_100w(total_points, total_words),
        "comment_hit_share": float((points > 0).mean()),
        "_rates": rates,
    }


def merge_subreddit_stats(
    acc: Dict[str, Any],
    fragment: Dict[str, Any],
) -> Dict[str, Any]:
    """Function summary: merge a shard-level stats fragment into accumulated forum stats.

    Parameters:
    - acc: accumulated stats (or empty dict).
    - fragment: shard fragment from _stats_from_scored_df.

    Returns:
    - Updated accumulated stats.
    """
    if not acc:
        out = dict(fragment)
        out["_rates"] = list(fragment.get("_rates", []))
        return out
    acc["total_weighted_points"] = int(acc.get("total_weighted_points", 0)) + int(
        fragment.get("total_weighted_points", 0)
    )
    acc["total_words"] = int(acc.get("total_words", 0)) + int(fragment.get("total_words", 0))
    acc["n_comments"] = int(acc.get("n_comments", 0)) + int(fragment.get("n_comments", 0))
    acc["n_comments_with_hits"] = int(acc.get("n_comments_with_hits", 0)) + int(
        fragment.get("n_comments_with_hits", 0)
    )
    acc["n_shards_read"] = int(acc.get("n_shards_read", 0)) + int(fragment.get("n_shards_read", 0))
    acc["n_shards_skipped"] = int(acc.get("n_shards_skipped", 0)) + int(
        fragment.get("n_shards_skipped", 0)
    )
    acc.setdefault("_rates", []).extend(fragment.get("_rates", []))
    rates = acc["_rates"]
    series = pd.Series(rates) if rates else pd.Series([0.0])
    acc["median_political_rate_100w"] = float(series.median())
    acc["mean_political_rate_100w"] = float(series.mean())
    acc["word_weighted_political_rate_100w"] = political_rate_100w(
        int(acc["total_weighted_points"]), int(acc["total_words"])
    )
    n_comments = int(acc["n_comments"])
    acc["comment_hit_share"] = (
        int(acc["n_comments_with_hits"]) / n_comments if n_comments else 0.0
    )
    return acc


def finalize_subreddit_stats(acc: Dict[str, Any]) -> Dict[str, float]:
    """Function summary: drop internal keys and return public stats dict.

    Parameters:
    - acc: accumulated stats possibly with _rates.

    Returns:
    - Stats dict without private keys.
    """
    return {k: v for k, v in acc.items() if not str(k).startswith("_")}


def collect_subreddit_stats_from_enriched(
    interim_dir: Path,
    subreddit: str,
    meta_row: Dict[str, str],
) -> Dict[str, float]:
    """Function summary: aggregate WW political stats from enriched Parquet columns.

    Parameters:
    - interim_dir: interim root.
    - subreddit: subreddit name.
    - meta_row: metadata with primary_lexicon.

    Returns:
    - Dict with hit/word counts and rate metrics.
    """
    lex_lang = meta_row["primary_lexicon"]
    shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    if not shard_dir.exists():
        return _empty_subreddit_stats(lex_lang)
    acc: Dict[str, Any] = {}
    for shard in sorted(shard_dir.glob("*.parquet")):
        df = read_parquet_shard_safe(shard, columns=SCORE_COLS)
        if df is None or df.empty:
            if acc:
                acc["n_shards_skipped"] = int(acc.get("n_shards_skipped", 0)) + 1
            continue
        acc = merge_subreddit_stats(acc, _stats_from_scored_df(df, lex_lang))
    if not acc:
        return _empty_subreddit_stats(lex_lang)
    return finalize_subreddit_stats(acc)


def score_shard_write(
    shard: Path,
    lex_lang: str,
    meta_row: Dict[str, str],
    screening_row: Dict[str, Any],
    parallel_csv: Path,
) -> Dict[str, Any]:
    """Function summary: batch-score one shard and write political columns (no topic/langid).

    Parameters:
    - shard: Parquet path.
    - lex_lang: primary lexicon language.
    - meta_row: arm/forum_type/primary_lexicon.
    - screening_row: screening record.
    - parallel_csv: graded lexicon CSV path.

    Returns:
    - Stats fragment dict for forum aggregation.
    """
    df = read_parquet_shard_safe(shard)
    if df is None or df.empty:
        return _empty_subreddit_stats(lex_lang)
    scored = score_bodies_political_salience(
        df["body"].astype(str).tolist(), lex_lang, PROJECT_ROOT, csv_path=parallel_csv
    )
    fragment = _stats_from_scored_df(scored, lex_lang)
    drop_cols = [
        "political_lexicon_hits",
        "political_g1_hits",
        "political_g2_hits",
        "political_g3_hits",
        "political_weighted_points",
        "n_words",
        "political_rate_100w",
        "thread_id",
        "arm",
        "forum_type",
        "primary_lexicon",
        "volume_band",
        "exclusion_code",
    ]
    out = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    out["political_g1_hits"] = scored["political_g1_hits"].values
    out["political_g2_hits"] = scored["political_g2_hits"].values
    out["political_g3_hits"] = scored["political_g3_hits"].values
    out["political_weighted_points"] = scored["political_weighted_points"].values
    out["political_lexicon_hits"] = scored["political_weighted_points"].values
    out["n_words"] = scored["n_words"].values
    out["political_rate_100w"] = [
        political_rate_100w(int(p), int(w))
        for p, w in zip(out["political_weighted_points"], out["n_words"], strict=True)
    ]
    out["thread_id"] = out["link_id"].astype(str)
    out["arm"] = meta_row["arm"]
    out["forum_type"] = meta_row["forum_type"]
    out["primary_lexicon"] = lex_lang
    out["volume_band"] = str(
        screening_row.get("volume_band", screening_row.get("analysis_tier", "low_volume"))
    )
    out["exclusion_code"] = str(screening_row.get("exclusion_codes", "")) or pd.NA
    out.to_parquet(shard, index=False, engine="pyarrow", compression="snappy")
    return fragment


def assign_topics(
    config: Dict[str, Any],
    subreddit_stats: Dict[str, Dict[str, float]],
    screening: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Function summary: assign topics using first-match-wins priority.

    Parameters:
    - config: study config.
    - subreddit_stats: per-subreddit political aggregates.
    - screening: screening thresholds.
    - metadata: metadata YAML dict.

    Returns:
    - Mapping subreddit -> assignment record.
    """
    overrides = load_topic_overrides(metadata)
    base_map = subreddit_topic_map(config, include_topic_aliases=False)
    soft_tau, pure_tau, benchmark = compute_political_thresholds(subreddit_stats, screening)
    italian_arms = italian_arms_for_langid(config)
    arms = subreddit_arm_map(config)

    assignments: Dict[str, Dict[str, Any]] = {}
    for subreddit in resolve_primary_subreddits(config):
        stats = subreddit_stats.get(subreddit, {})
        ww_rate = float(stats.get("word_weighted_political_rate_100w", 0.0))

        if subreddit in overrides:
            assignments[subreddit] = {
                "topic": overrides[subreddit],
                "assignment_source": "metadata_override",
                "political_soft_threshold": soft_tau,
                "political_pure_threshold": pure_tau,
                "political_threshold": pure_tau,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "word_weighted",
            }
            continue

        arm = arms.get(subreddit, "")
        if arm not in italian_arms:
            topic = base_map.get(subreddit, infer_subreddit_topic(config, subreddit, metadata=metadata))
            assignments[subreddit] = {
                "topic": topic,
                "assignment_source": "config_control",
                "political_soft_threshold": soft_tau,
                "political_pure_threshold": pure_tau,
                "political_threshold": pure_tau,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "word_weighted",
            }
            continue

        if subreddit in base_map and base_map[subreddit] in CONFIG_LIST_TOPICS:
            assignments[subreddit] = {
                "topic": base_map[subreddit],
                "assignment_source": "config_topic_list",
                "political_soft_threshold": soft_tau,
                "political_pure_threshold": pure_tau,
                "political_threshold": pure_tau,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "word_weighted",
            }
            continue

        if ww_rate >= pure_tau:
            topic = "it_pure_political"
            source = "auto_lexicon_pure"
        elif ww_rate >= soft_tau:
            topic = "it_political"
            source = "auto_lexicon_soft"
        else:
            topic = "it_others"
            source = "auto_lexicon"

        assignments[subreddit] = {
            "topic": topic,
            "assignment_source": source,
            "political_soft_threshold": soft_tau,
            "political_pure_threshold": pure_tau,
            "political_threshold": pure_tau,
            "politicaITA_benchmark_median": benchmark,
            "assignment_metric_used": "word_weighted",
        }
    return assignments


def build_political_audit_rows(
    assignments: Dict[str, Dict[str, Any]],
    subreddit_stats: Dict[str, Dict[str, float]],
    sub_to_family: Dict[str, str],
    topic_to_family: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Function summary: build political label vs score mismatch audit rows.

    Parameters:
    - assignments: topic assignments per subreddit.
    - subreddit_stats: political aggregates.
    - sub_to_family: subreddit -> family.
    - topic_to_family: topic -> family.

    Returns:
    - List of audit row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for subreddit, info in assignments.items():
        stats = subreddit_stats.get(subreddit, {})
        topic = info["topic"]
        source = info["assignment_source"]
        soft_tau = float(info.get("political_soft_threshold", info.get("political_threshold", 0.6)))
        pure_tau = float(info.get("political_pure_threshold", info.get("political_threshold", 1.2)))
        median_r = float(stats.get("median_political_rate_100w", 0.0))
        mean_r = float(stats.get("mean_political_rate_100w", 0.0))
        ww_r = float(stats.get("word_weighted_political_rate_100w", 0.0))
        hit_share = float(stats.get("comment_hit_share", 0.0))

        high_score_non_political = topic == "it_others" and ww_r >= soft_tau
        low_score_political = (
            topic == "it_pure_political" and ww_r < pure_tau
        ) or (
            topic == "it_political" and ww_r < soft_tau
        )
        informational_seed_low = False
        special_topic_high = False

        note_parts: List[str] = []
        if high_score_non_political:
            if ww_r >= pure_tau:
                note_parts.append(
                    f"Assigned {topic} via {source} but ww={ww_r:.2f} ≥ pure {pure_tau:.2f} "
                    f"— consider it_pure_political."
                )
            else:
                note_parts.append(
                    f"Assigned {topic} via {source} but ww={ww_r:.2f} ≥ soft {soft_tau:.2f} "
                    f"— consider it_political."
                )
        elif low_score_political:
            note_parts.append(
                f"Labeled {topic} but ww={ww_r:.2f} below applicable cutoff "
                f"(soft={soft_tau:.2f}, pure={pure_tau:.2f})."
            )
        else:
            note_parts.append(f"Topic {topic} ({source}) consistent with lexicon scores.")

        family = topic_to_family.get(topic, sub_to_family.get(subreddit, "it_others"))
        rows.append(
            {
                "subreddit": subreddit,
                "topic": topic,
                "topic_family": family,
                "assignment_source": source,
                "political_soft_threshold": round(soft_tau, 4),
                "political_pure_threshold": round(pure_tau, 4),
                "political_threshold": round(pure_tau, 4),
                "politicaITA_benchmark_median": round(float(info["politicaITA_benchmark_median"]), 4),
                "median_political_rate_100w": round(median_r, 4),
                "mean_political_rate_100w": round(mean_r, 4),
                "word_weighted_political_rate_100w": round(ww_r, 4),
                "comment_hit_share": round(hit_share, 4),
                "high_score_non_political": high_score_non_political,
                "low_score_political": low_score_political,
                "informational_seed_low": informational_seed_low,
                "special_topic_high": special_topic_high,
                "plain_english_note": " ".join(note_parts),
            }
        )
    return rows


def finalize_enrichment_dataframe(
    df: pd.DataFrame,
    meta_row: Dict[str, str],
    topic: str,
    topic_family: str,
    screening_row: Dict[str, Any],
    screening: Dict[str, Any],
    skip_langid: bool,
) -> pd.DataFrame:
    """Function summary: add topic/langid/thread columns to a scored shard (no rescoring).

    Parameters:
    - df: shard with political_weighted_points and n_words.
    - meta_row: arm/forum_type/primary_lexicon.
    - topic: assigned topic.
    - topic_family: topic family name.
    - screening_row: pooled screening record.
    - screening: screening config.
    - skip_langid: if True, omit lang_comment classification.

    Returns:
    - Fully enriched dataframe.
    """
    enrich_cols = [
        "lang_comment",
        "topic",
        "topic_family",
        "analysis_tier",
        "thread_political_weighted_points",
        "thread_political_rate_100w",
        "thread_is_political",
    ]
    out = df.drop(columns=[c for c in enrich_cols if c in df.columns], errors="ignore").copy()
    if "thread_id" not in out.columns:
        out["thread_id"] = out["link_id"].astype(str)
    out["arm"] = meta_row["arm"]
    out["forum_type"] = meta_row["forum_type"]
    out["primary_lexicon"] = meta_row["primary_lexicon"]
    out["topic"] = topic
    out["topic_family"] = topic_family
    out["volume_band"] = str(
        screening_row.get("volume_band", screening_row.get("analysis_tier", "low_volume"))
    )
    out["exclusion_code"] = str(screening_row.get("exclusion_codes", "")) or pd.NA
    if skip_langid:
        out["lang_comment"] = pd.NA
    else:
        out["lang_comment"] = [langid.classify(str(b))[0] for b in out["body"].astype(str)]

    min_points = int(screening.get("thread_political_min_points", 3))
    thread_stats = (
        out.groupby("thread_id", as_index=False)
        .agg(
            thread_political_weighted_points=("political_weighted_points", "sum"),
            thread_n_words=("n_words", "sum"),
        )
    )
    thread_stats["thread_political_rate_100w"] = thread_stats.apply(
        lambda r: political_rate_100w(
            int(r["thread_political_weighted_points"]), int(r["thread_n_words"])
        ),
        axis=1,
    )
    thread_stats["thread_is_political"] = (
        thread_stats["thread_political_weighted_points"] >= min_points
    )
    out = out.merge(
        thread_stats[
            [
                "thread_id",
                "thread_political_weighted_points",
                "thread_political_rate_100w",
                "thread_is_political",
            ]
        ],
        on="thread_id",
        how="left",
    )
    return out


def warn_zero_ww_control_forums(
    subreddit_stats: Dict[str, Dict[str, Any]],
    interim_dir: Path,
) -> None:
    """Function summary: log warnings when control forums have shards but zero word-weighted political rate.

    Parameters:
    - subreddit_stats: per-subreddit aggregates from collect_subreddit_political_stats.
    - interim_dir: interim data root.

    Returns:
    - None.
    """
    for subreddit in CONTROL_SUBREDDITS_WARN_ZERO_WW:
        stats = subreddit_stats.get(subreddit, {})
        ww = float(stats.get("word_weighted_political_rate_100w", 0.0))
        n_comments = int(stats.get("n_comments", 0))
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        if n_comments > 0 and ww <= 0.0:
            print(
                f"[enrich_cleaned_chunks] WARN zero_ww subreddit={subreddit} "
                f"lexicon={stats.get('primary_lexicon', '')} "
                f"n_comments={n_comments} shards_read={stats.get('n_shards_read', 0)} "
                f"shards_skipped={stats.get('n_shards_skipped', 0)}",
                flush=True,
            )


def thread_political_share_from_enriched(enriched: pd.DataFrame) -> List[bool]:
    """Function summary: per-thread political flags from one enriched month dataframe.

    Parameters:
    - enriched: enriched month with thread_id and thread_is_political.

    Returns:
    - List of bool per unique thread.
    """
    if "thread_is_political" not in enriched.columns:
        return []
    per_thread = enriched.drop_duplicates(subset=["thread_id"])
    return per_thread["thread_is_political"].astype(bool).tolist()


def _score_shard_worker(
    shard_str: str,
    subreddit: str,
    lex_lang: str,
    arm: str,
    forum_type: str,
    volume_band: str,
    exclusion_codes: str,
    parallel_csv_str: str,
) -> Tuple[str, Dict[str, Any]]:
    """Function summary: process-pool worker to score one Parquet shard.

    Parameters:
    - shard_str: absolute shard path string.
    - subreddit: subreddit name.
    - lex_lang: lexicon language code.
    - arm: arm label.
    - forum_type: forum type label.
    - volume_band: volume band string.
    - exclusion_codes: exclusion codes string.
    - parallel_csv_str: graded lexicon CSV path.

    Returns:
    - Tuple (subreddit, stats fragment).
    """
    screening_row = {
        "volume_band": volume_band,
        "exclusion_codes": exclusion_codes,
    }
    meta_row = {"arm": arm, "forum_type": forum_type, "primary_lexicon": lex_lang}
    fragment = score_shard_write(
        Path(shard_str),
        lex_lang,
        meta_row,
        screening_row,
        Path(parallel_csv_str),
    )
    return subreddit, fragment


def main() -> None:
    """Function summary: enrich all cleaned Parquet shards and write assignment tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    screening = load_screening_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    screening_df = load_screening_pooled(tables_dir)
    screening_by_sub = screening_by_subreddit(screening_df)

    metadata = load_subreddit_metadata(config, project_root=PROJECT_ROOT)
    meta_table = build_subreddit_metadata_table(config, project_root=PROJECT_ROOT)
    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    parallel_csv = parallel_political_lexicon_path(config, project_root=PROJECT_ROOT)
    workers = args.workers if args.workers is not None else default_worker_count()

    subreddits = resolve_primary_subreddits(config)
    subreddit_stats: Dict[str, Dict[str, float]] = {}

    if args.assign_only:
        print(
            f"[enrich_cleaned_chunks] phase=aggregate_from_enriched subreddits={len(subreddits)}",
            flush=True,
        )
        for idx, subreddit in enumerate(subreddits, start=1):
            action = subreddit_screening_action(screening_by_sub, subreddit)
            if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
                continue
            subreddit_stats[subreddit] = collect_subreddit_stats_from_enriched(
                interim_dir, subreddit, meta_table[subreddit]
            )
            st = subreddit_stats[subreddit]
            print(
                f"[enrich_cleaned_chunks] aggregate_done {idx}/{len(subreddits)} "
                f"subreddit={subreddit} ww={st.get('word_weighted_political_rate_100w', 0):.4f}",
                flush=True,
            )
    else:
        for lang in ("it", "en", "de"):
            get_graded_matcher(PROJECT_ROOT, lang, csv_path=parallel_csv)
        score_tasks: List[Tuple[str, str, str, str, str, str, str, str]] = []
        for subreddit in subreddits:
            action = subreddit_screening_action(screening_by_sub, subreddit)
            if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
                continue
            shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
            if not shard_dir.exists():
                continue
            meta_row = meta_table[subreddit]
            screen_row = screening_by_sub.get(subreddit, {})
            for shard in sorted(shard_dir.glob("*.parquet")):
                score_tasks.append(
                    (
                        str(shard.resolve()),
                        subreddit,
                        meta_row["primary_lexicon"],
                        meta_row["arm"],
                        meta_row["forum_type"],
                        str(
                            screen_row.get(
                                "volume_band", screen_row.get("analysis_tier", "low_volume")
                            )
                        ),
                        str(screen_row.get("exclusion_codes", "")),
                        str(parallel_csv.resolve()),
                    )
                )
        print(
            f"[enrich_cleaned_chunks] phase=score_shards tasks={len(score_tasks)} workers={workers}",
            flush=True,
        )
        if workers <= 1:
            for task in score_tasks:
                sub, fragment = _score_shard_worker(*task)
                subreddit_stats[sub] = merge_subreddit_stats(
                    subreddit_stats.get(sub, {}), fragment
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_score_shard_worker, *t) for t in score_tasks]
                for fut in as_completed(futures):
                    sub, fragment = fut.result()
                    subreddit_stats[sub] = merge_subreddit_stats(
                        subreddit_stats.get(sub, {}), fragment
                    )
        for sub, acc in list(subreddit_stats.items()):
            subreddit_stats[sub] = finalize_subreddit_stats(acc)
            st = subreddit_stats[sub]
            print(
                f"[enrich_cleaned_chunks] score_done subreddit={sub} "
                f"ww={st.get('word_weighted_political_rate_100w', 0):.4f} "
                f"n_comments={st.get('n_comments', 0)}",
                flush=True,
            )

    warn_zero_ww_control_forums(subreddit_stats, interim_dir)

    print("[enrich_cleaned_chunks] phase=assign_topics", flush=True)
    assignments = assign_topics(config, subreddit_stats, screening, metadata)
    topic_to_family = topic_family_map(config, include_family_aliases=False)
    thread_political_by_sub: Dict[str, List[bool]] = defaultdict(list)

    print("[enrich_cleaned_chunks] phase=finalize_shards", flush=True)
    for idx, subreddit in enumerate(subreddits, start=1):
        action = subreddit_screening_action(screening_by_sub, subreddit)
        if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
            continue
        screen_row = screening_by_sub.get(subreddit, {})
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.exists():
            continue
        meta_row = meta_table[subreddit]
        topic = assignments.get(subreddit, {}).get(
            "topic", infer_subreddit_topic(config, subreddit, metadata=metadata)
        )
        if isinstance(topic, dict):
            topic = topic.get("topic", "it_others")
        family = topic_to_family.get(str(topic), sub_to_family.get(subreddit, "it_others"))
        shards = sorted(shard_dir.glob("*.parquet"))
        print(
            f"[enrich_cleaned_chunks] finalize_start {idx}/{len(subreddits)} "
            f"subreddit={subreddit} topic={topic} shards={len(shards)}",
            flush=True,
        )
        for shard in shards:
            df = read_parquet_shard_safe(shard)
            if df is None or df.empty:
                continue
            if args.assign_only and "political_weighted_points" not in df.columns:
                print(
                    f"[enrich_cleaned_chunks] WARN missing_score_cols subreddit={subreddit} "
                    f"shard={shard.name}; run without --assign-only",
                    flush=True,
                )
                continue
            enriched = finalize_enrichment_dataframe(
                df=df,
                meta_row=meta_row,
                topic=str(topic),
                topic_family=family,
                screening_row=screen_row,
                screening=screening,
                skip_langid=args.skip_langid,
            )
            enriched.to_parquet(shard, index=False, engine="pyarrow", compression="snappy")
            thread_political_by_sub[subreddit].extend(thread_political_share_from_enriched(enriched))
        print(f"[enrich_cleaned_chunks] finalize_done subreddit={subreddit}", flush=True)

    print("[enrich_cleaned_chunks] phase=write_tables", flush=True)
    assignment_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    for subreddit, info in assignments.items():
        topic = str(info["topic"])
        family = topic_to_family.get(topic, sub_to_family.get(subreddit, "it_others"))
        stats = subreddit_stats.get(subreddit, {})
        screen_row = screening_by_sub.get(subreddit, {})
        flags = thread_political_by_sub.get(subreddit, [])
        thread_share = (sum(flags) / len(flags)) if flags else 0.0
        assignment_rows.append(
            {
                "subreddit": subreddit,
                "topic": topic,
                "topic_family": family,
                "assignment_source": info["assignment_source"],
                "political_soft_threshold": round(float(info["political_soft_threshold"]), 4),
                "political_pure_threshold": round(float(info["political_pure_threshold"]), 4),
                "political_threshold": round(float(info["political_threshold"]), 4),
                "politicaITA_benchmark_median": round(float(info["politicaITA_benchmark_median"]), 4),
                "assignment_metric_used": info["assignment_metric_used"],
                "median_political_rate_100w": round(
                    float(stats.get("median_political_rate_100w", 0.0)), 4
                ),
            }
        )
        profile_rows.append(
            {
                "subreddit": subreddit,
                "topic": topic,
                "topic_family": family,
                "primary_lexicon": str(stats.get("primary_lexicon", meta_table.get(subreddit, {}).get("primary_lexicon", ""))),
                "assignment_source": info["assignment_source"],
                "political_soft_threshold": round(float(info["political_soft_threshold"]), 4),
                "political_pure_threshold": round(float(info["political_pure_threshold"]), 4),
                "median_political_rate_100w": round(
                    float(stats.get("median_political_rate_100w", 0.0)), 4
                ),
                "mean_political_rate_100w": round(
                    float(stats.get("mean_political_rate_100w", 0.0)), 4
                ),
                "word_weighted_political_rate_100w": round(
                    float(stats.get("word_weighted_political_rate_100w", 0.0)), 4
                ),
                "comment_hit_share": round(float(stats.get("comment_hit_share", 0.0)), 4),
                "thread_political_share": round(thread_share, 4),
                "n_comments_stats": int(stats.get("n_comments", 0)),
                "volume_band": screen_row.get("volume_band", screen_row.get("analysis_tier", "")),
                "action": screen_row.get("action", ""),
            }
        )

    audit_rows = build_political_audit_rows(
        assignments, subreddit_stats, sub_to_family, topic_to_family
    )
    mismatch_rows = [
        r
        for r in audit_rows
        if (r["high_score_non_political"] or (r["low_score_political"] and not r["informational_seed_low"]))
    ]

    out_dir = tables_dir / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(assignment_rows).to_csv(out_dir / "subreddit_topic_assignment.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(out_dir / "subreddit_forum_political_profile.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "subreddit_topic_political_audit.csv", index=False)
    if mismatch_rows:
        pd.DataFrame(mismatch_rows).to_csv(
            out_dir / "subreddit_topic_political_mismatches.csv", index=False
        )

    print("[enrich_cleaned_chunks] done", flush=True)

    if not args.skip_pipeline_plots:
        run_cleaning_pipeline_plots_if_ready(args.config)


if __name__ == "__main__":
    main()
