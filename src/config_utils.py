"""
Script summary:
Shared configuration helpers for the dump-first data pipeline.
This module provides minimal utilities for loading YAML config, converting
UTC ISO timestamps to unix seconds, and parsing optional calendar reference dates for plots.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROFILE_USER_PATTERN = re.compile(r"^u_", re.IGNORECASE)
CREATOR_NAME_SUFFIXES = ("Submissions", "Official")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Function summary: load YAML configuration from disk and return a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


DESCRIPTIVE_FIGURE_VIEWS = (
    "by_family",
    "by_topic",
    "by_topic_italian",
    "country_panel",
    "ideology",
)


def study_id_from_config(config: Dict[str, Any]) -> str:
    """Function summary: resolve study slug for results layout (explicit key or tables_dir leaf).

    Parameters:
    - config: loaded YAML with optional `study_id` and `paths.tables_dir`.

    Returns:
    - Study identifier string (e.g. `italy_polarization`).
    """
    explicit = config.get("study_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    tables_dir = (config.get("paths") or {}).get("tables_dir", "")
    parts = Path(str(tables_dir)).parts
    if parts:
        return parts[-1]
    return "unknown_study"


def _resolve_path_from_config(config: Dict[str, Any], key: str) -> Path:
    """Function summary: resolve a paths.* directory from config as an absolute or repo-relative Path."""
    raw = (config.get("paths") or {}).get(key, "")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parent.parent
    return root / path


def parallel_political_lexicon_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.political_lexicon_parallel CSV for graded salience.

    Parameters:
    - config: loaded study YAML.
    - project_root: optional repo root override.

    Returns:
    - Path to political_lexicon_parallel.csv.
    """
    paths = config.get("paths") or {}
    raw = paths.get("political_lexicon_parallel", "data/raw/political_lexicon_parallel.csv")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = project_root or Path(__file__).resolve().parent.parent
    return root / path


def _parallel_path_from_config(
    config: Dict[str, Any],
    key: str,
    default_rel: str,
    project_root: Optional[Path] = None,
) -> Path:
    """Function summary: resolve a paths.<key> entry under the project root.

    Parameters:
    - config: loaded study YAML.
    - key: paths dict key.
    - default_rel: fallback relative path.
    - project_root: optional repo root override.

    Returns:
    - Resolved Path.
    """
    paths = config.get("paths") or {}
    raw = paths.get(key, default_rel)
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = project_root or Path(__file__).resolve().parent.parent
    return root / path


def polarization_lexicon_parallel_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.polarization_lexicon_parallel CSV."""
    return _parallel_path_from_config(
        config,
        "polarization_lexicon_parallel",
        "data/raw/polarization_lexicon_parallel.csv",
        project_root,
    )


def emotion_cognition_parallel_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.emotion_cognition_parallel CSV."""
    return _parallel_path_from_config(
        config,
        "emotion_cognition_parallel",
        "data/raw/emotion_cognition_parallel.csv",
        project_root,
    )


def aggression_parallel_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.aggression_parallel CSV for semantic-axis insult seeds."""
    return _parallel_path_from_config(
        config,
        "aggression_parallel",
        "data/raw/seeds/aggression_parallel.csv",
        project_root,
    )


def style_phrase_parallel_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.style_phrase_parallel CSV."""
    return _parallel_path_from_config(
        config,
        "style_phrase_parallel",
        "data/raw/style_phrase_parallel.csv",
        project_root,
    )


def italian_lexicon_v4_pairs_path(
    config: Dict[str, Any], project_root: Optional[Path] = None
) -> Path:
    """Function summary: resolve paths.italian_lexicon_v4_pairs CSV (pairs section)."""
    return _parallel_path_from_config(
        config,
        "italian_lexicon_v4_pairs",
        "data/raw/italian_political_lexicon_v4.csv",
        project_root,
    )


def tables_subdir(config: Dict[str, Any], *parts: str) -> Path:
    """Function summary: join paths.tables_dir with optional subfolders for CSV/parquet outputs.

    Parameters:
    - config: loaded YAML.
    - parts: zero or more path segments under the study tables root.

    Returns:
    - Path under results/tables/<study_id>/...
    """
    return _resolve_path_from_config(config, "tables_dir").joinpath(*parts)


def figures_subdir(config: Dict[str, Any], *parts: str) -> Path:
    """Function summary: join paths.figures_dir with optional subfolders for PNG outputs.

    Parameters:
    - config: loaded YAML.
    - parts: zero or more path segments under the study figures root.

    Returns:
    - Path under results/figures/<study_id>/...
    """
    return _resolve_path_from_config(config, "figures_dir").joinpath(*parts)


def logs_subdir(config: Dict[str, Any], *parts: str) -> Path:
    """Function summary: join paths.logs_dir with study-scoped log subfolders.

    Parameters:
    - config: loaded YAML.
    - parts: zero or more path segments (e.g. study_id, filter_dump, runs).

    Returns:
    - Path under results/logs/...
    """
    base = _resolve_path_from_config(config, "logs_dir")
    study = study_id_from_config(config)
    if parts and parts[0] == study:
        return base.joinpath(*parts)
    return base.joinpath(study, *parts)


def filter_dump_logs_dir(config: Dict[str, Any]) -> Path:
    """Function summary: directory for filter_dump state JSON and log files for this study.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Path results/logs/<study_id>/filter_dump/ (created by callers if needed).
    """
    return logs_subdir(config, "filter_dump")


def utc_ts(iso_utc: str) -> int:
    """Function summary: convert an ISO UTC timestamp string to unix epoch seconds."""
    return int(datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).timestamp())


def plot_reference_dates_calendar_utc(config: Dict[str, Any]) -> List[datetime]:
    """Function summary: build naive UTC datetimes for vertical plot markers from optional YAML or defaults.

    Parameters:
    - config: loaded YAML; may contain `plot_reference_dates_utc` as a list of ISO strings (Z or offset).

    Returns:
    - Non-empty list of naive UTC datetimes. If the key is missing, not a list, empty, or parsing yields nothing,
      returns Italy ChatGPT ban onset (`2023-03-31`) and lift (`2023-04-28`) calendar dates for the active study.
    """
    default = [datetime(2023, 3, 31), datetime(2023, 4, 28)]
    raw = config.get("plot_reference_dates_utc")
    if not isinstance(raw, list) or not raw:
        return default
    out: List[datetime] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        out.append(dt)
    return out if out else default


def comment_dump_filenames(start_utc_iso: str, end_utc_exclusive_iso: str) -> List[str]:
    """Function summary: list Reddit monthly comment dump basenames (RC_YYYY-MM.zst) spanning the event window.

    Parameters:
    - start_utc_iso: inclusive window start as ISO-8601 UTC string (e.g. ends with Z).
    - end_utc_exclusive_iso: exclusive window end as ISO-8601 UTC string.

    Returns:
    - Sorted filenames from the first calendar month overlapping start through the last
      calendar month that can contain timestamps strictly before the exclusive end.
    """
    start = datetime.fromisoformat(start_utc_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    end_excl = datetime.fromisoformat(end_utc_exclusive_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    if end_excl <= start:
        return []
    last_moment = end_excl - timedelta(microseconds=1)
    names: List[str] = []
    y, m = start.year, start.month
    y_end, m_end = last_moment.year, last_moment.month
    while (y < y_end) or (y == y_end and m <= m_end):
        names.append(f"RC_{y}-{m:02d}.zst")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return names


def topic_groups(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Function summary: parse topic-group subreddit lists from config and return topic-to-subreddits mapping.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Dictionary mapping topic name -> list of subreddit names.
    """
    raw_topics = config.get("topics", {})
    if not isinstance(raw_topics, dict):
        raise ValueError("Config key `topics` must be a mapping of topic names to settings.")
    groups: Dict[str, List[str]] = {}
    for topic_name, topic_value in raw_topics.items():
        if isinstance(topic_value, dict):
            topic_subs = topic_value.get("subreddits", [])
        elif isinstance(topic_value, list):
            topic_subs = topic_value
        else:
            raise ValueError(f"Config topic `{topic_name}` must be a list or mapping with `subreddits`.")
        if not isinstance(topic_subs, list):
            raise ValueError(f"Config topic `{topic_name}` field `subreddits` must be a list.")
        cleaned = [str(sub).strip() for sub in topic_subs if str(sub).strip()]
        groups[str(topic_name)] = cleaned
    return groups


def metadata_config_path(config: Dict[str, Any], project_root: Optional[Path] = None) -> Path:
    """Function summary: resolve path to optional subreddit metadata YAML from study config.

    Parameters:
    - config: loaded study YAML.
    - project_root: repository root for relative paths; defaults to config file parent parent.

    Returns:
    - Path to metadata YAML file.
    """
    raw = str(config.get("metadata_config_path", "config/italy_polarization_subreddit_metadata.yaml")).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    if project_root is not None:
        return project_root / path
    return path


def load_subreddit_metadata(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Function summary: load subreddit metadata overrides YAML referenced by the study config.

    Parameters:
    - config: loaded study YAML.
    - project_root: repository root for relative metadata path resolution.

    Returns:
    - Metadata dictionary (empty dict if file missing).
    """
    path = metadata_config_path(config, project_root=project_root)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def load_screening_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return screening thresholds with documented defaults for Italy pipeline.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Screening settings dictionary.
    """
    defaults: Dict[str, Any] = {
        "url_only_drop": True,
        "forum_url_only_share_exclude": 0.80,
        "min_kept_per_month_soft": 50,
        "min_kept_window_large_volume": 100,
        "langid_sample_per_month": 500,
        "langid_italian_threshold_pooled": 0.70,
        "langid_min_body_chars": 15,
        "langid_rng_seed": 20260318,
        "thread_political_rate_threshold": 0.45,
        "thread_political_min_hits": 0,
        "thread_political_min_points": 3,
        "forum_political_rate_multiplier_vs_politicaita": 0.25,
        "forum_political_soft_threshold": 0.6,
        "forum_political_pure_threshold": 1.2,
        "forum_political_word_weighted_rate_threshold": 1.2,
    }
    raw = config.get("screening", {})
    if isinstance(raw, dict):
        merged = dict(raw)
        if "min_kept_window_large_volume" not in merged and "min_kept_window_tier_a" in merged:
            merged["min_kept_window_large_volume"] = merged.pop("min_kept_window_tier_a")
        defaults.update(merged)
    return defaults


def forum_political_thresholds(screening: Dict[str, Any]) -> Tuple[float, float]:
    """Function summary: return soft and pure forum word-weighted rate cutoffs for Italian topic assignment.

    Parameters:
    - screening: screening config from load_screening_config().

    Returns:
    - Tuple (soft_threshold for it_political, pure_threshold for it_pure_political).
    """
    legacy = screening.get("forum_political_word_weighted_rate_threshold")
    soft = float(screening.get("forum_political_soft_threshold", legacy if legacy is not None else 0.6))
    pure = float(
        screening.get(
            "forum_political_pure_threshold",
            legacy if legacy is not None else 1.2,
        )
    )
    return soft, pure


EXCLUDED_SCREENING_ACTIONS = frozenset({"excluded", "exclude_analysis"})
ENRICHMENT_MARKER_COLUMNS = frozenset(
    {"primary_lexicon", "n_words", "political_weighted_points"}
)


def load_screening_pooled(tables_dir: Path) -> Any:
    """Function summary: load pooled subreddit screening CSV from Stage-2 screening.

    Parameters:
    - tables_dir: study tables directory (e.g. results/tables/italy_polarization).

    Returns:
    - DataFrame with one row per subreddit from subreddit_screening_pooled.csv.

    Raises:
    - FileNotFoundError: if screening has not been run yet.
    """
    import pandas as pd

    path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run screen_subreddits.py first: missing {path}")
    return pd.read_csv(path)


def screening_by_subreddit(screening_df: Any) -> Dict[str, Dict[str, Any]]:
    """Function summary: index pooled screening rows by subreddit name.

    Parameters:
    - screening_df: output of load_screening_pooled.

    Returns:
    - Mapping subreddit -> screening record dict.
    """
    return {str(row["subreddit"]): row for row in screening_df.to_dict(orient="records")}


def subreddit_screening_action(screening_by_sub: Dict[str, Dict[str, Any]], subreddit: str) -> str:
    """Function summary: return screening action for a subreddit (empty if unknown).

    Parameters:
    - screening_by_sub: indexed screening table.
    - subreddit: subreddit name.

    Returns:
    - Action string (e.g. keep, excluded).
    """
    row = screening_by_sub.get(subreddit, {})
    return str(row.get("action", ""))


def should_skip_screened_subreddit(action: str, include_excluded: bool = False) -> bool:
    """Function summary: whether feature/enrich passes should skip this subreddit.

    Parameters:
    - action: screening action from pooled table.
    - include_excluded: when True, process excluded forums too.

    Returns:
    - True if the subreddit should be skipped.
    """
    return action in EXCLUDED_SCREENING_ACTIONS and not include_excluded


def shard_dir_is_enriched(shard_dir: Path) -> bool:
    """Function summary: check whether cleaned monthly shards have Stage-3 enrichment columns.

    Parameters:
    - shard_dir: directory with monthly Parquet files for one subreddit.

    Returns:
    - True if the first shard contains primary_lexicon and n_words.
    """
    import pyarrow.parquet as pq

    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        return False
    for shard in shards:
        if shard.stat().st_size < 8:
            continue
        try:
            names = set(pq.read_schema(shard).names)
        except Exception:
            continue
        return ENRICHMENT_MARKER_COLUMNS <= names
    return False


def load_political_universe_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return political-universe definition settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Political universe settings (mode, thresholds, tree options).
    """
    defaults: Dict[str, Any] = {
        "mode": "tree",
        "comment_political_min_points": 3,
        "tree_include_parent": True,
        "tree_max_depth": None,
        "political_cos_threshold": 0.55,
    }
    raw = config.get("political_universe", {})
    if isinstance(raw, dict):
        merged = dict(raw)
        if merged.get("tree_max_depth") in ("", "none", "None"):
            merged["tree_max_depth"] = None
        defaults.update(merged)
    return defaults


def load_polarization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return polarization feature settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Polarization settings dictionary.
    """
    defaults: Dict[str, Any] = {
        "ideology_scoring": "dominant_v1",
        "restrict_to_political_comments": True,
        "eps": 1.0e-6,
        "negation_window_tokens": 3,
        "lang_match_filter": False,
        "dip_min_n": 30,
        "er_alpha": 1.6,
    }
    raw = config.get("polarization", {})
    if isinstance(raw, dict):
        defaults.update(raw)
    return defaults


def load_semantic_axis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return semantic-axis feature settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Semantic axis settings (vector paths, seeds_dir, pole_cutoffs, panel_bin_days, language_waves).
    """
    defaults: Dict[str, Any] = {
        "lang_match_filter": False,
        "seeds_dir": "data/raw/seeds",
        "vector_paths": {
            "it": "data/external/embeddings/cc.it.300.bin",
            "en": "data/external/embeddings/cc.en.300.bin",
            "de": "data/external/embeddings/cc.de.300.bin",
        },
        "pole_thresholds": {
            "ideology": 0.25,
            "emotion": 0.25,
            "aggression": 0.25,
        },
        "pole_thresholds_by_lexicon": {
            "it": {"ideology": 0.12, "emotion": 0.12, "aggression": 0.08},
            "en": {"ideology": 0.25, "emotion": 0.25, "aggression": 0.25},
            "de": {"ideology": 0.20, "emotion": 0.20, "aggression": 0.18},
        },
        "pole_percentiles": [10, 90],
        "percentile_calibration": {
            "enabled": True,
            "max_comments_per_lang": 50000,
        },
        "write_vector_cache": True,
        "language_waves": True,
        "language_wave_order": ["it", "en", "de"],
        "vector_cache_exclusive": True,
        "pole_cutoffs": [0.25],
        "panel_bin_days": [1, 3, 7],
    }
    raw = config.get("semantic_axis", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "vector_paths" and isinstance(val, dict):
                defaults["vector_paths"].update(val)
            elif key == "pole_thresholds" and isinstance(val, dict):
                defaults["pole_thresholds"].update(val)
            elif key == "pole_thresholds_by_lexicon" and isinstance(val, dict):
                merged = dict(defaults["pole_thresholds_by_lexicon"])
                for lex, axes in val.items():
                    if isinstance(axes, dict):
                        merged.setdefault(str(lex), {}).update(axes)
                defaults["pole_thresholds_by_lexicon"] = merged
            elif key == "percentile_calibration" and isinstance(val, dict):
                defaults["percentile_calibration"].update(val)
            else:
                defaults[key] = val
    return defaults


def load_circumvention_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return circumvention proxy settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Circumvention paths, treated/control geos, country_panel_geo_map, panel_bin_days.
    """
    defaults: Dict[str, Any] = {
        "raw_dir": "data/raw/circumvention",
        "google_trends_combined": "google_trends_vpn_by_country.csv",
        "google_trends_chatgpt_combined": "google_trends_chatgpt_by_country.csv",
        "tor_relay_combined": "tor_relay_users_by_country.csv",
        "tor_bridge_combined": "tor_bridge_users_by_country.csv",
        "treated_geo": "IT",
        "control_geos": ["DE", "FR", "ES", "GB", "US"],
        "country_panel_geo_map": {
            "Italy_political": "IT",
            "Italy_others": "IT",
            "Germany": "DE",
            "US_political": "US",
            "UK": "GB",
        },
        "panel_bin_days": [1, 3, 7],
    }
    raw = config.get("circumvention", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "country_panel_geo_map" and isinstance(val, dict):
                defaults["country_panel_geo_map"].update(val)
            else:
                defaults[key] = val
    return defaults


def require_dominant_v1_ideology_scoring(config: Dict[str, Any]) -> None:
    """Function summary: assert study config uses mandatory dominant_v1 ideology scoring.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - None; raises AssertionError if ideology_scoring is not dominant_v1.
    """
    pol = load_polarization_config(config)
    scoring = pol.get("ideology_scoring")
    assert scoring == "dominant_v1", (
        f"polarization.ideology_scoring must be 'dominant_v1' (got {scoring!r}). "
        "Re-export lexicons with export_italian_lexicon_v4.py --policy dominant and re-run features."
    )


def load_wordfish_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return Wordfish robustness settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Wordfish settings dictionary.
    """
    defaults: Dict[str, Any] = {
        "enabled": True,
        "min_doc_tokens": 200,
        "min_doc_freq": 2,
        "min_token_len": 3,
        "max_vocab_terms": 5000,
        "top_freq_drop_n": 50,
        "stopwords_dir": "config/lexicons",
        "languages": ["it", "en"],
        "min_subreddits_per_language": 2,
        "time_bins": ["day", "week"],
        "ban_anchor_date": "2023-03-31",
        "weekly_bin_days": 7,
        "train_iters": 5000,
        "learning_rate": 1.0e-5,
        "convergence": {
            "check_final_objective": True,
            "min_objective_improvement": 1.0e-4,
        },
        "anchor_subreddit": {
            "it": "politicaITA",
            "en": "PoliticalDiscussion",
        },
        "sensitivity_profiles": [
            {"min_doc_freq": 2, "top_freq_drop_n": 50},
            {"min_doc_freq": 3, "top_freq_drop_n": 100},
        ],
        "top_axis_words": 25,
        "change_window_days": [7, 3],
        "placebo_launch_date": "2023-03-16",
        "primary_time_bin": "day",
        "week_learning_rate_scale": 0.1,
    }
    raw = config.get("wordfish", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "anchor_subreddit" and isinstance(val, dict):
                defaults["anchor_subreddit"].update(val)
            elif key == "convergence" and isinstance(val, dict):
                defaults["convergence"].update(val)
            elif key == "sensitivity_profiles" and isinstance(val, list):
                defaults["sensitivity_profiles"] = val
            else:
                defaults[key] = val
    return defaults


def load_wordfish_authors_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return author-level Wordfish settings merged with wordfish defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - wordfish_authors settings dict (includes prune/train/stopwords from wordfish).
    """
    base = load_wordfish_config(config)
    defaults: Dict[str, Any] = {
        **base,
        "enabled": True,
        "languages": ["it", "en", "de"],
        "ban_anchor_date": base.get("ban_anchor_date", "2023-03-31"),
        "placebo_launch_date": base.get("placebo_launch_date", "2023-03-16"),
        "rolling_bins_w": 2,
        "time_bins": [
            {"name": "week7", "time_bin": "week", "weekly_bin_days": 7, "min_doc_tokens": 100},
            {"name": "week3", "time_bin": "week", "weekly_bin_days": 3, "min_doc_tokens": 50},
            {"name": "window", "time_bin": "window", "min_doc_tokens": 100},
        ],
        "panel_modes": ["full", "balanced"],
        "drop_cross_language": False,
        "min_authors_per_language": 50,
        "primary_lang_priority": ["it", "de", "en"],
        "filter_comments_to_assigned_lang": True,
        "sign_anchor_mode": "subreddit_modal",
        "anchor_subreddit": {
            "it": "politicaITA",
            "en": "PoliticalDiscussion",
            "de": "de",
        },
        "headline_spec": "week7",
        "headline_mode": "balanced",
        "note": "Robustness; headline balanced+week7 for prompt 04.",
    }
    raw = config.get("wordfish_authors", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "anchor_subreddit" and isinstance(val, dict):
                defaults["anchor_subreddit"] = {**defaults.get("anchor_subreddit", {}), **val}
            elif key == "convergence" and isinstance(val, dict):
                defaults["convergence"] = {**defaults.get("convergence", {}), **val}
            elif key == "time_bins" and isinstance(val, list):
                defaults["time_bins"] = val
            elif key == "panel_modes" and isinstance(val, list):
                defaults["panel_modes"] = val
            elif key == "primary_lang_priority" and isinstance(val, list):
                defaults["primary_lang_priority"] = val
            else:
                defaults[key] = val
    return defaults


def load_wordfish_authors_v2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: author Wordfish v2 settings (alternating MLE, token cap, validation gate).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Merged wordfish_authors_v2 dict built on wordfish_authors defaults.
    """
    defaults = load_wordfish_authors_config(config)
    defaults.update(
        {
            "output_tables_subdir": "wordfish_authors_v2",
            "output_figures_subdir": "wordfish_authors_v2",
            "max_tokens_per_doc": 8000,
            "token_subsample_seed": 42,
            "backend": "python_v2",
            "en_fit_mode": "split_us_uk",
            "validation": {
                "gate_abs_rho_sem_axis": 0.5,
                "min_authors": 100,
            },
            "convergence": {
                **defaults.get("convergence", {}),
                "max_cycles": 40,
                "min_objective_improvement": 1.0e-4,
                "opposite_frac_threshold": 0.05,
            },
            "note": (
                "Author-level v2: political-universe docs, token cap, alternating MLE. "
                "Pre-registered gate on |rho| vs sem_axis; demote theta if gate fails."
            ),
        }
    )
    raw = config.get("wordfish_authors_v2", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "validation" and isinstance(val, dict):
                defaults["validation"] = {**defaults.get("validation", {}), **val}
            elif key == "convergence" and isinstance(val, dict):
                defaults["convergence"] = {**defaults.get("convergence", {}), **val}
            elif key == "anchor_subreddit" and isinstance(val, dict):
                defaults["anchor_subreddit"] = {**defaults.get("anchor_subreddit", {}), **val}
            elif key == "time_bins" and isinstance(val, list):
                defaults["time_bins"] = val
            elif key == "panel_modes" and isinstance(val, list):
                defaults["panel_modes"] = val
            else:
                defaults[key] = val
    return defaults


def load_wordfish_forum_v2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: forum Wordfish v2 (shard topic_family, token cap, alternating MLE).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Merged wordfish_forum_v2 settings on wordfish defaults.
    """
    defaults = load_wordfish_config(config)
    defaults.update(
        {
            "output_tables_subdir": "wordfish_forum_v2",
            "output_figures_subdir": "wordfish_forum_v2",
            "max_tokens_per_doc": 8000,
            "token_subsample_seed": 42,
            "backend": "python_v2",
            "use_shard_topic_family": True,
            "validation": {
                "gate_abs_rho_sem_axis": 0.5,
                "min_subreddits": 5,
                "expect_gate_fail": True,
            },
            "convergence": {
                **defaults.get("convergence", {}),
                "max_cycles": 40,
                "min_objective_improvement": 1.0e-4,
                "opposite_frac_threshold": 0.05,
            },
            "note": (
                "Forum v2: preserve enrich topic_family labels, token cap, alternating MLE. "
                "Theta is NOT validated as ideology (gate expected to fail)."
            ),
        }
    )
    raw = config.get("wordfish_forum_v2", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key == "validation" and isinstance(val, dict):
                defaults["validation"] = {**defaults.get("validation", {}), **val}
            elif key == "convergence" and isinstance(val, dict):
                defaults["convergence"] = {**defaults.get("convergence", {}), **val}
            elif key == "anchor_subreddit" and isinstance(val, dict):
                defaults["anchor_subreddit"] = {**defaults.get("anchor_subreddit", {}), **val}
            else:
                defaults[key] = val
    return defaults


def load_ai_use_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return AI-use feature settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - AI-use settings dictionary.
    """
    defaults: Dict[str, Any] = {
        "lang_match_filter": False,
        "validation_sample_n": 200,
    }
    raw = config.get("ai_use", {})
    if isinstance(raw, dict):
        defaults.update(raw)
    return defaults


def load_comment_style_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return comment-style feature settings with defaults.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Comment-style settings dictionary.
    """
    defaults: Dict[str, Any] = {
        "lang_match_filter": False,
        "enable_phrase_lexicons": True,
    }
    raw = config.get("comment_style", {})
    if isinstance(raw, dict):
        defaults.update(raw)
    return defaults


def user_week_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Function summary: return the user_week config block if present.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Dict (possibly empty).
    """
    raw = config.get("user_week")
    return raw if isinstance(raw, dict) else {}


def user_week_default_features(config: Dict[str, Any]) -> List[str]:
    """Function summary: default feature list for within-user pre/post analysis.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - List of feature column names.
    """
    uw = user_week_section(config)
    feats = uw.get("default_features")
    if isinstance(feats, list) and feats:
        return [str(f) for f in feats]
    return []


def user_week_composites(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Function summary: composite definitions for user-week shift analysis.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - List of dicts with keys ``name`` and ``components`` (list of {feature, sign}).
    """
    uw = user_week_section(config)
    out: List[Dict[str, Any]] = []
    for key in ("polarization_composite", "style_composite", "semantic_composite"):
        block = uw.get(key)
        if not isinstance(block, dict):
            continue
        name = str(block.get("name", "")).strip()
        comps = block.get("components")
        if not name or not isinstance(comps, list):
            continue
        parsed: List[Tuple[str, int]] = []
        for item in comps:
            if not isinstance(item, dict):
                continue
            feat = str(item.get("feature", "")).strip()
            if not feat:
                continue
            sign = int(item.get("sign", 1))
            parsed.append((feat, -1 if sign < 0 else 1))
        if parsed:
            out.append({"name": name, "components": parsed})
    return out


def user_week_drop_ban_week_default(config: Dict[str, Any]) -> bool:
    """Function summary: whether to drop the ISO week containing the ban anchor by default.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - True if drop_ban_week is set in YAML.
    """
    return bool(user_week_section(config).get("drop_ban_week", False))


def user_week_placebo_offset_weeks_default(config: Dict[str, Any]) -> int:
    """Function summary: default placebo offset in weeks from config.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Non-negative integer week offset.
    """
    raw = user_week_section(config).get("placebo_offset_weeks", 8)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 8


def infer_user_week_input_mode(config: Dict[str, Any]) -> str:
    """Function summary: infer user-week panel input layout from study config (archive scripts only).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - ``enriched_shards`` for Italy-style in-place shards, else ``comment_features``.
      Active Italy scripts always use enriched shards; see ``scripts/archive/user_week/``.
    """
    explicit = config.get("user_week", {}).get("input_mode") if isinstance(config.get("user_week"), dict) else None
    if explicit in {"enriched_shards", "comment_features"}:
        return str(explicit)
    project_name = str(config.get("project", {}).get("name", ""))
    if "ItalyPolarization" in project_name or "italy_polarization" in str(config.get("paths", {}).get("interim_dir", "")):
        return "enriched_shards"
    return "comment_features"


def italian_arms_for_langid(config: Dict[str, Any]) -> set[str]:
    """Function summary: return arm labels subject to pooled Italian langid validation.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Set of arm names requiring Italian langid pass.
    """
    return {"discovered_italian", "discovery_seed_italian"}


def infer_subreddit_topic(config: Dict[str, Any], subreddit: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Function summary: infer topic label for a subreddit using arms, metadata lists, and name heuristics.

    Parameters:
    - config: loaded study YAML.
    - subreddit: subreddit name.
    - metadata: optional pre-loaded metadata dict.

    Returns:
    - Topic name string.
    """
    meta = metadata if metadata is not None else load_subreddit_metadata(config)
    overrides = meta.get("topic_overrides", {})
    if isinstance(overrides, dict) and subreddit in overrides:
        return str(overrides[subreddit])

    arm = subreddit_arm_map(config).get(subreddit, "")
    if arm == "control_english_political":
        return "us"
    if arm == "control_europe_political":
        return "uk_political"
    if subreddit == "de":
        return "de"
    if subreddit == "unitedkingdom":
        return "uk"
    if subreddit == "europe":
        return "eu"

    nsfw = set(meta.get("nsfw_subreddits", []) or [])
    memes = set(meta.get("meme_humor_subreddits", []) or [])
    creators = set(meta.get("creator_celebrity_subreddits", []) or [])
    if subreddit in nsfw or subreddit in memes or subreddit in creators:
        return "it_others"
    if any(subreddit.endswith(sfx) for sfx in CREATOR_NAME_SUFFIXES):
        return "it_others"
    if PROFILE_USER_PATTERN.match(subreddit):
        return "it_others"
    return "it_others"


def infer_subreddit_forum_type(config: Dict[str, Any], subreddit: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Function summary: infer forum_type for a subreddit (dedicated_political, general_hub, etc.).

    Parameters:
    - config: loaded study YAML.
    - subreddit: subreddit name.
    - metadata: optional pre-loaded metadata dict.

    Returns:
    - Forum type string.
    """
    meta = metadata if metadata is not None else load_subreddit_metadata(config)
    overrides = meta.get("forum_type_overrides", {})
    if isinstance(overrides, dict) and subreddit in overrides:
        return str(overrides[subreddit])

    topic = infer_subreddit_topic(config, subreddit, metadata=meta)
    if topic in {"us", "uk_political", "it_pure_political"}:
        return "dedicated_political"
    if topic in {"de", "eu", "uk", "it_political", "it_others"}:
        return "general_hub"
    if PROFILE_USER_PATTERN.match(subreddit):
        return "profile_user"
    return "general_hub"


def infer_subreddit_primary_lexicon(
    config: Dict[str, Any], subreddit: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Function summary: infer primary political lexicon language code (it, en, de).

    Parameters:
    - config: loaded study YAML.
    - subreddit: subreddit name.
    - metadata: optional pre-loaded metadata dict.

    Returns:
    - Lexicon language code.
    """
    meta = metadata if metadata is not None else load_subreddit_metadata(config)
    overrides = meta.get("primary_lexicon_overrides", {})
    if isinstance(overrides, dict) and subreddit in overrides:
        return str(overrides[subreddit])

    topic = infer_subreddit_topic(config, subreddit, metadata=meta)
    if topic in {"us", "uk_political", "eu", "uk"}:
        return "en"
    if topic == "de":
        return "de"
    return "it"


def subreddit_forum_type(config: Dict[str, Any], subreddit: str, project_root: Optional[Path] = None) -> str:
    """Function summary: return forum_type for a subreddit using metadata overrides and inference.

    Parameters:
    - config: loaded study YAML.
    - subreddit: subreddit name.
    - project_root: optional repo root for metadata path.

    Returns:
    - Forum type string.
    """
    meta = load_subreddit_metadata(config, project_root=project_root)
    return infer_subreddit_forum_type(config, subreddit, metadata=meta)


def subreddit_primary_lexicon(config: Dict[str, Any], subreddit: str, project_root: Optional[Path] = None) -> str:
    """Function summary: return primary lexicon language code for a subreddit.

    Parameters:
    - config: loaded study YAML.
    - subreddit: subreddit name.
    - project_root: optional repo root for metadata path.

    Returns:
    - Lexicon language code (`it`, `en`, `de`).
    """
    meta = load_subreddit_metadata(config, project_root=project_root)
    return infer_subreddit_primary_lexicon(config, subreddit, metadata=meta)


def build_subreddit_metadata_table(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Function summary: build per-subreddit metadata records for all primary subreddits.

    Parameters:
    - config: loaded study YAML.
    - project_root: optional repo root.

    Returns:
    - Mapping subreddit -> {arm, forum_type, primary_lexicon, topic}.
    """
    meta = load_subreddit_metadata(config, project_root=project_root)
    arms = subreddit_arm_map(config)
    out: Dict[str, Dict[str, str]] = {}
    for subreddit in resolve_primary_subreddits(config):
        out[subreddit] = {
            "arm": arms.get(subreddit, "discovered_italian"),
            "forum_type": infer_subreddit_forum_type(config, subreddit, metadata=meta),
            "primary_lexicon": infer_subreddit_primary_lexicon(config, subreddit, metadata=meta),
            "topic": infer_subreddit_topic(config, subreddit, metadata=meta),
        }
    return out


def subreddit_topic_map(config: Dict[str, Any], include_topic_aliases: bool = True) -> Dict[str, str]:
    """Function summary: build subreddit-to-topic mapping from config topic groups with optional topic aliases.

    Parameters:
    - config: full loaded YAML config dictionary.
    - include_topic_aliases: if true, map each topic name to itself for convenience.

    Returns:
    - Dictionary mapping subreddit (or alias) -> topic.
    """
    mapping: Dict[str, str] = {}
    primary_subreddits = {str(s) for s in config.get("subreddits", {}).get("primary", [])}
    meta = load_subreddit_metadata(config)
    for topic_name, subreddits in topic_groups(config).items():
        for subreddit in subreddits:
            if subreddit not in primary_subreddits:
                continue
            previous = mapping.get(subreddit)
            if previous and previous != topic_name:
                raise ValueError(
                    f"Subreddit `{subreddit}` appears in multiple topics: `{previous}` and `{topic_name}`."
                )
            mapping[subreddit] = topic_name
        if include_topic_aliases:
            mapping[topic_name] = topic_name
    for subreddit in primary_subreddits:
        if subreddit not in mapping:
            mapping[subreddit] = infer_subreddit_topic(config, subreddit, metadata=meta)
    return mapping


def topic_families(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Function summary: parse topic-family topic lists from config and return family-to-topics mapping.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Dictionary mapping family name -> list of topic names.
    """
    raw_families = config.get("topic_families", {})
    if not isinstance(raw_families, dict):
        raise ValueError("Config key `topic_families` must be a mapping of family names to settings.")
    parsed: Dict[str, List[str]] = {}
    for family_name, family_value in raw_families.items():
        if isinstance(family_value, dict):
            family_topics = family_value.get("topics", [])
        elif isinstance(family_value, list):
            family_topics = family_value
        else:
            raise ValueError(
                f"Config topic family `{family_name}` must be a list or mapping with `topics`."
            )
        if not isinstance(family_topics, list):
            raise ValueError(f"Config topic family `{family_name}` field `topics` must be a list.")
        cleaned_topics = [str(topic).strip() for topic in family_topics if str(topic).strip()]
        parsed[str(family_name)] = cleaned_topics
    return parsed


def topic_family_map(config: Dict[str, Any], include_family_aliases: bool = True) -> Dict[str, str]:
    """Function summary: validate family coverage and return topic-to-family mapping with optional family aliases.

    Parameters:
    - config: full loaded YAML config dictionary.
    - include_family_aliases: if true, map each family name to itself for convenience.

    Returns:
    - Dictionary mapping topic (or family alias) -> family.
    """
    configured_topics = set(topic_groups(config).keys())
    families = topic_families(config)
    if not families:
        raise ValueError("Config key `topic_families` is required and must define at least one family.")

    mapping: Dict[str, str] = {}
    unknown_topics: List[str] = []
    for family_name, topics in families.items():
        for topic in topics:
            if topic not in configured_topics:
                unknown_topics.append(topic)
                continue
            previous = mapping.get(topic)
            if previous and previous != family_name:
                raise ValueError(
                    f"Topic `{topic}` appears in multiple families: `{previous}` and `{family_name}`."
                )
            mapping[topic] = family_name
        if include_family_aliases:
            mapping[family_name] = family_name

    if unknown_topics:
        unknown_sorted = ", ".join(sorted(set(unknown_topics)))
        raise ValueError(f"Config `topic_families` contains unknown topics: {unknown_sorted}")

    missing_topics = sorted(configured_topics - set(mapping.keys()))
    if missing_topics:
        raise ValueError(
            "Every configured topic must be assigned to exactly one family. "
            f"Missing topics: {', '.join(missing_topics)}"
        )
    return mapping


def subreddit_family_map(config: Dict[str, Any], include_family_aliases: bool = False) -> Dict[str, str]:
    """Function summary: compose subreddit-to-topic and topic-to-family mappings into subreddit-to-family mapping.

    Parameters:
    - config: full loaded YAML config dictionary.
    - include_family_aliases: if true, map each family name to itself for convenience.

    Returns:
    - Dictionary mapping subreddit (and optional family aliases) -> family.
    """
    sub_to_topic = subreddit_topic_map(config, include_topic_aliases=False)
    topic_to_family = topic_family_map(config, include_family_aliases=include_family_aliases)
    mapping: Dict[str, str] = {}
    for subreddit, topic in sub_to_topic.items():
        family = topic_to_family.get(topic)
        if family is None:
            raise ValueError(f"Topic `{topic}` for subreddit `{subreddit}` is missing from `topic_families`.")
        mapping[subreddit] = family

    primary_subreddits = sorted(str(s) for s in config.get("subreddits", {}).get("primary", []))
    unmapped_subreddits = [sub for sub in primary_subreddits if sub not in mapping]
    if unmapped_subreddits:
        raise ValueError(
            "Every primary subreddit must map to a topic and a topic family. "
            f"Missing subreddits: {', '.join(unmapped_subreddits)}"
        )
    return mapping


def subreddit_control_lists(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Function summary: return named control/discovery subreddit lists from config `subreddits` section.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Mapping of list name (e.g. controls_english_political) -> subreddit names.
    """
    raw = config.get("subreddits", {})
    if not isinstance(raw, dict):
        return {}
    list_keys = (
        "controls_english_political",
        "controls_europe_hub",
        "controls_europe_political",
        "discovery_seeds_italian",
        "discovered_italian",
    )
    out: Dict[str, List[str]] = {}
    for key in list_keys:
        value = raw.get(key, [])
        if isinstance(value, list):
            out[key] = [str(item).strip() for item in value if str(item).strip()]
    return out


def resolve_primary_subreddits(config: Dict[str, Any]) -> List[str]:
    """Function summary: build deduplicated primary subreddit list for filtering and cleaning.

    Uses explicit `subreddits.primary` when non-empty; otherwise unions control lists,
    discovery seeds, and discovered Italian subreddits.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Sorted list of unique subreddit names.
    """
    raw = config.get("subreddits", {})
    if not isinstance(raw, dict):
        raise ValueError("Config key `subreddits` must be a mapping.")
    explicit = raw.get("primary", [])
    if isinstance(explicit, list) and explicit:
        return sorted({str(item).strip() for item in explicit if str(item).strip()})

    lists = subreddit_control_lists(config)
    combined: set[str] = set()
    for key in (
        "controls_english_political",
        "controls_europe_hub",
        "controls_europe_political",
        "discovery_seeds_italian",
        "discovered_italian",
    ):
        combined.update(lists.get(key, []))
    if not combined:
        raise ValueError(
            "No subreddits to extract: set `subreddits.primary` or populate control/discovered lists."
        )
    return sorted(combined)


def subreddit_arm_map(config: Dict[str, Any]) -> Dict[str, str]:
    """Function summary: map subreddit name to comparison arm label for discovery preview tables.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Dictionary subreddit -> arm name (last list wins on duplicate).
    """
    arm_labels = {
        "controls_english_political": "control_english_political",
        "controls_europe_hub": "control_europe_hub",
        "controls_europe_political": "control_europe_political",
        "discovery_seeds_italian": "discovery_seed_italian",
        "discovered_italian": "discovered_italian",
    }
    mapping: Dict[str, str] = {}
    for list_key, arm in arm_labels.items():
        for subreddit in subreddit_control_lists(config).get(list_key, []):
            mapping[subreddit] = arm
    return mapping


def control_subreddits_for_discovery(config: Dict[str, Any]) -> set[str]:
    """Function summary: subreddits fixed by config that skip Italian langid sampling but stay in census.

    Parameters:
    - config: full loaded YAML config dictionary.

    Returns:
    - Set of subreddit names in any controls_* list.
    """
    lists = subreddit_control_lists(config)
    out: set[str] = set()
    for key in ("controls_english_political", "controls_europe_hub", "controls_europe_political"):
        out.update(lists.get(key, []))
    return out
