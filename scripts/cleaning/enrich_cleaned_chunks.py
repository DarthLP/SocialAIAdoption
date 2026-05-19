"""
Script summary:
Stage-3 enrichment for cleaned Italy polarization Parquet: taxonomy columns,
language-matched political lexicon scores, thread roll-ups, topic assignment audit,
and optional deprecated family-month aggregates.

Functionality:
- Reads screening outputs and adds metadata plus political scores per comment.
- Assigns topics via first-match-wins priority (metadata → controls → config lists → lexicon).
- Writes assignment, political profile, and political mismatch audit CSVs.

How to apply/run:
  .venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

try:
    import langid
except ImportError as exc:
    raise SystemExit("langid is required: pip install langid") from exc


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
    build_subreddit_metadata_table,
    infer_subreddit_topic,
    load_config,
    load_screening_config,
    load_subreddit_metadata,
    resolve_primary_subreddits,
    subreddit_family_map,
    subreddit_topic_map,
    topic_family_map,
)
from src.political_lexicon import count_political_hits, political_rate_100w  # noqa: E402

POLITICAL_TOPICS = frozenset({"it_political", "en_us_political", "uk_political"})
SPECIAL_TOPICS = frozenset({"it_meme_humor", "it_creator_celebrity", "it_nsfw_sensitivity"})
CONFIG_LIST_TOPICS = frozenset(
    {"it_political", "it_meme_humor", "it_creator_celebrity", "it_nsfw_sensitivity"}
)


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
        "--write-by-family",
        action="store_true",
        help="DEPRECATED: also write cleaned_monthly_by_family/ aggregates. Prefer groupby on shards.",
    )
    parser.add_argument(
        "--prune-family-copies",
        action="store_true",
        help="Remove cleaned_monthly_by_family/ under interim_dir if present.",
    )
    return parser.parse_args()


def load_screening_pooled(tables_dir: Path) -> pd.DataFrame:
    """Function summary: load pooled screening table.

    Parameters:
    - tables_dir: study tables directory.

    Returns:
    - Screening pooled dataframe.
    """
    path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run screen_subreddits.py first: missing {path}")
    return pd.read_csv(path)


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


def compute_political_threshold(
    subreddit_stats: Dict[str, Dict[str, float]],
    screening: Dict[str, Any],
) -> Tuple[float, float]:
    """Function summary: politicaITA-calibrated threshold for auto_lexicon assignment.

    Parameters:
    - subreddit_stats: per-subreddit stats with median_political_rate_100w.
    - screening: screening config.

    Returns:
    - Tuple (threshold, politicaITA_benchmark_median).
    """
    benchmark = float(subreddit_stats.get("politicaITA", {}).get("median_political_rate_100w", 0.0))
    multiplier = float(screening.get("forum_political_rate_multiplier_vs_politicaita", 0.25))
    threshold = max(0.5, benchmark * multiplier) if benchmark > 0 else 0.5
    return threshold, benchmark


def collect_subreddit_political_stats(
    interim_dir: Path,
    subreddit: str,
    meta_row: Dict[str, str],
) -> Dict[str, float]:
    """Function summary: aggregate political lexicon stats for one subreddit.

    Parameters:
    - interim_dir: interim root.
    - subreddit: subreddit name.
    - meta_row: metadata with primary_lexicon.

    Returns:
    - Dict with hit/word counts and rate metrics.
    """
    shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    if not shard_dir.exists():
        return {
            "total_hits": 0,
            "total_words": 0,
            "n_comments": 0,
            "n_comments_with_hits": 0,
            "median_political_rate_100w": 0.0,
            "mean_political_rate_100w": 0.0,
            "word_weighted_political_rate_100w": 0.0,
            "comment_hit_share": 0.0,
        }

    lex_lang = meta_row["primary_lexicon"]
    rates: List[float] = []
    total_hits = 0
    total_words = 0
    n_with_hits = 0

    for shard in sorted(shard_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["body"])
        for body in df["body"].astype(str).tolist():
            hits, nw = count_political_hits(body, lex_lang, PROJECT_ROOT)
            total_hits += hits
            total_words += nw
            rate = political_rate_100w(hits, nw)
            rates.append(rate)
            if hits > 0:
                n_with_hits += 1

    n_comments = len(rates)
    if n_comments == 0:
        return {
            "total_hits": 0,
            "total_words": 0,
            "n_comments": 0,
            "n_comments_with_hits": 0,
            "median_political_rate_100w": 0.0,
            "mean_political_rate_100w": 0.0,
            "word_weighted_political_rate_100w": 0.0,
            "comment_hit_share": 0.0,
        }

    series = pd.Series(rates)
    return {
        "total_hits": total_hits,
        "total_words": total_words,
        "n_comments": n_comments,
        "n_comments_with_hits": n_with_hits,
        "median_political_rate_100w": float(series.median()),
        "mean_political_rate_100w": float(series.mean()),
        "word_weighted_political_rate_100w": political_rate_100w(total_hits, total_words),
        "comment_hit_share": n_with_hits / n_comments,
    }


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
    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    threshold, benchmark = compute_political_threshold(subreddit_stats, screening)

    meta_lists = {
        "it_nsfw_sensitivity": set(metadata.get("nsfw_subreddits", []) or []),
        "it_meme_humor": set(metadata.get("meme_humor_subreddits", []) or []),
        "it_creator_celebrity": set(metadata.get("creator_celebrity_subreddits", []) or []),
    }

    assignments: Dict[str, Dict[str, Any]] = {}
    for subreddit in resolve_primary_subreddits(config):
        stats = subreddit_stats.get(subreddit, {})
        median_rate = float(stats.get("median_political_rate_100w", 0.0))

        if subreddit in overrides:
            assignments[subreddit] = {
                "topic": overrides[subreddit],
                "assignment_source": "metadata_override",
                "political_threshold": threshold,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "median",
            }
            continue

        family = sub_to_family.get(subreddit, "italian")
        if family != "italian":
            topic = base_map.get(subreddit, infer_subreddit_topic(config, subreddit, metadata=metadata))
            assignments[subreddit] = {
                "topic": topic,
                "assignment_source": "config_control",
                "political_threshold": threshold,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "median",
            }
            continue

        if subreddit in base_map and base_map[subreddit] in CONFIG_LIST_TOPICS:
            assignments[subreddit] = {
                "topic": base_map[subreddit],
                "assignment_source": "config_topic_list",
                "political_threshold": threshold,
                "politicaITA_benchmark_median": benchmark,
                "assignment_metric_used": "median",
            }
            continue

        if subreddit in meta_lists["it_nsfw_sensitivity"]:
            topic = "it_nsfw_sensitivity"
            source = "metadata_special_list"
        elif subreddit in meta_lists["it_meme_humor"]:
            topic = "it_meme_humor"
            source = "metadata_special_list"
        elif subreddit in meta_lists["it_creator_celebrity"]:
            topic = "it_creator_celebrity"
            source = "metadata_special_list"
        elif median_rate >= threshold:
            topic = "it_political"
            source = "auto_lexicon"
        else:
            topic = "it_general"
            source = "auto_lexicon"

        assignments[subreddit] = {
            "topic": topic,
            "assignment_source": source,
            "political_threshold": threshold,
            "politicaITA_benchmark_median": benchmark,
            "assignment_metric_used": "median",
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
        threshold = float(info["political_threshold"])
        median_r = float(stats.get("median_political_rate_100w", 0.0))
        mean_r = float(stats.get("mean_political_rate_100w", 0.0))
        ww_r = float(stats.get("word_weighted_political_rate_100w", 0.0))
        hit_share = float(stats.get("comment_hit_share", 0.0))

        is_political_topic = topic in POLITICAL_TOPICS
        high_score = median_r >= threshold or mean_r >= threshold or ww_r >= threshold
        low_score = median_r < threshold and mean_r < threshold

        high_score_non_political = (not is_political_topic) and high_score and topic not in SPECIAL_TOPICS
        low_score_political = is_political_topic and low_score
        informational_seed_low = (
            source in ("metadata_override", "config_topic_list") and low_score_political
        )
        special_topic_high = topic in SPECIAL_TOPICS and high_score

        note_parts: List[str] = []
        if high_score_non_political:
            note_parts.append(
                f"Assigned {topic} via {source} but lexicon rates (med={median_r:.2f}, "
                f"ww={ww_r:.2f}) ≥ threshold {threshold:.2f} — consider it_political."
            )
        elif informational_seed_low:
            note_parts.append(
                f"Seeded {topic} ({source}) with low lexicon (med={median_r:.2f}) — expected manual seed."
            )
        elif low_score_political and not informational_seed_low:
            note_parts.append(
                f"Labeled {topic} but median={median_r:.2f} and mean={mean_r:.2f} below threshold {threshold:.2f}."
            )
        elif special_topic_high:
            note_parts.append(
                f"Special topic {topic} with elevated political lexicon (ww={ww_r:.2f}) — informational."
            )
        else:
            note_parts.append(f"Topic {topic} ({source}) consistent with lexicon scores.")

        family = topic_to_family.get(topic, sub_to_family.get(subreddit, "italian"))
        rows.append(
            {
                "subreddit": subreddit,
                "topic": topic,
                "topic_family": family,
                "assignment_source": source,
                "political_threshold": round(threshold, 4),
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


def enrich_dataframe(
    df: pd.DataFrame,
    subreddit: str,
    meta_row: Dict[str, str],
    topic: str,
    topic_family: str,
    screening_row: Dict[str, Any],
    screening: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: add enrichment columns and thread roll-ups to one month dataframe.

    Parameters:
    - df: cleaned month data.
    - subreddit: subreddit name.
    - meta_row: arm/forum_type/primary_lexicon.
    - topic: assigned topic.
    - topic_family: topic family name.
    - screening_row: pooled screening record.
    - screening: screening config.

    Returns:
    - Enriched dataframe.
    """
    enrich_cols = [
        "political_lexicon_hits",
        "n_words",
        "political_rate_100w",
        "lang_comment",
        "thread_id",
        "arm",
        "forum_type",
        "primary_lexicon",
        "topic",
        "topic_family",
        "volume_band",
        "analysis_tier",
        "exclusion_code",
        "thread_political_rate_100w",
        "thread_is_political",
    ]
    out = df.drop(columns=[c for c in enrich_cols if c in df.columns], errors="ignore").copy()
    lex_lang = meta_row["primary_lexicon"]
    hits: List[int] = []
    n_words_list: List[int] = []
    lang_comments: List[str] = []
    for body in out["body"].astype(str).tolist():
        h, nw = count_political_hits(body, lex_lang, PROJECT_ROOT)
        hits.append(h)
        n_words_list.append(nw)
        lang, _ = langid.classify(body)
        lang_comments.append(lang)
    out["political_lexicon_hits"] = hits
    out["n_words"] = n_words_list
    out["political_rate_100w"] = [
        political_rate_100w(h, nw) for h, nw in zip(hits, n_words_list, strict=True)
    ]
    out["lang_comment"] = lang_comments
    out["thread_id"] = out["link_id"].astype(str)
    out["arm"] = meta_row["arm"]
    out["forum_type"] = meta_row["forum_type"]
    out["primary_lexicon"] = lex_lang
    out["topic"] = topic
    out["topic_family"] = topic_family
    volume_band = str(
        screening_row.get("volume_band", screening_row.get("analysis_tier", "low_volume"))
    )
    out["volume_band"] = volume_band
    out["exclusion_code"] = str(screening_row.get("exclusion_codes", "")) or pd.NA

    tau = float(screening["thread_political_rate_threshold"])
    min_hits = int(screening["thread_political_min_hits"])
    thread_stats = (
        out.groupby("thread_id", as_index=False)
        .agg(
            thread_political_hits=("political_lexicon_hits", "sum"),
            thread_n_words=("n_words", "sum"),
        )
    )
    thread_stats["thread_political_rate_100w"] = thread_stats.apply(
        lambda r: political_rate_100w(int(r["thread_political_hits"]), int(r["thread_n_words"])),
        axis=1,
    )
    thread_stats["thread_is_political"] = (
        (thread_stats["thread_political_rate_100w"] >= tau)
        | (thread_stats["thread_political_hits"] >= min_hits)
    )
    out = out.merge(
        thread_stats[["thread_id", "thread_political_rate_100w", "thread_is_political"]],
        on="thread_id",
        how="left",
    )
    return out


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


def main() -> None:
    """Function summary: enrich all cleaned Parquet shards and write assignment tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    screening = load_screening_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    screening_df = load_screening_pooled(tables_dir)
    screening_by_sub = {str(r["subreddit"]): r for r in screening_df.to_dict(orient="records")}

    if args.prune_family_copies:
        family_dir = interim_dir / "cleaned_monthly_by_family"
        if family_dir.is_dir():
            import shutil

            shutil.rmtree(family_dir)
            print(f"[enrich_cleaned_chunks] removed {family_dir}", flush=True)

    metadata = load_subreddit_metadata(config, project_root=PROJECT_ROOT)
    meta_table = build_subreddit_metadata_table(config, project_root=PROJECT_ROOT)
    sub_to_family = subreddit_family_map(config, include_family_aliases=False)

    subreddit_stats: Dict[str, Dict[str, float]] = {}
    for subreddit in resolve_primary_subreddits(config):
        screen_row = screening_by_sub.get(subreddit, {})
        action = str(screen_row.get("action", ""))
        if action in ("excluded", "exclude_analysis") and not args.include_excluded:
            continue
        subreddit_stats[subreddit] = collect_subreddit_political_stats(
            interim_dir, subreddit, meta_table[subreddit]
        )

    assignments = assign_topics(config, subreddit_stats, screening, metadata)
    topic_to_family = topic_family_map(config, include_family_aliases=False)
    family_month_parts: Dict[tuple[str, str], List[pd.DataFrame]] = defaultdict(list)
    thread_political_by_sub: Dict[str, List[bool]] = defaultdict(list)

    for subreddit in resolve_primary_subreddits(config):
        screen_row = screening_by_sub.get(subreddit, {})
        action = str(screen_row.get("action", ""))
        if action in ("excluded", "exclude_analysis") and not args.include_excluded:
            continue
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.exists():
            continue
        meta_row = meta_table[subreddit]
        topic = assignments.get(subreddit, {}).get(
            "topic", infer_subreddit_topic(config, subreddit, metadata=metadata)
        )
        if isinstance(topic, dict):
            topic = topic.get("topic", "it_general")
        family = topic_to_family.get(str(topic), sub_to_family.get(subreddit, "italian"))
        for shard in sorted(shard_dir.glob("*.parquet")):
            df = pd.read_parquet(shard)
            enriched = enrich_dataframe(
                df=df,
                subreddit=subreddit,
                meta_row=meta_row,
                topic=str(topic),
                topic_family=family,
                screening_row=screen_row,
                screening=screening,
            )
            enriched.to_parquet(shard, index=False, engine="pyarrow", compression="snappy")
            thread_political_by_sub[subreddit].extend(thread_political_share_from_enriched(enriched))
            if args.write_by_family:
                family_month_parts[(family, shard.stem)].append(enriched)

    assignment_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    for subreddit, info in assignments.items():
        topic = str(info["topic"])
        family = topic_to_family.get(topic, sub_to_family.get(subreddit, "italian"))
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
                "assignment_source": info["assignment_source"],
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

    if args.write_by_family:
        for (family, year_month), parts in family_month_parts.items():
            target_dir = interim_dir / "cleaned_monthly_by_family" / family
            target_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(
                target_dir / f"{year_month}.parquet",
                index=False,
                engine="pyarrow",
                compression="snappy",
            )

    print("[enrich_cleaned_chunks] done", flush=True)


if __name__ == "__main__":
    main()
