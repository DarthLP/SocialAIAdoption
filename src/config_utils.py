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
from typing import Any, Dict, List, Optional

import yaml

PROFILE_USER_PATTERN = re.compile(r"^u_", re.IGNORECASE)
CREATOR_NAME_SUFFIXES = ("Submissions", "Official")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Function summary: load YAML configuration from disk and return a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def utc_ts(iso_utc: str) -> int:
    """Function summary: convert an ISO UTC timestamp string to unix epoch seconds."""
    return int(datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).timestamp())


def plot_reference_dates_calendar_utc(config: Dict[str, Any]) -> List[datetime]:
    """Function summary: build naive UTC datetimes for vertical plot markers from optional YAML or defaults.

    Parameters:
    - config: loaded YAML; may contain `plot_reference_dates_utc` as a list of ISO strings (Z or offset).

    Returns:
    - Non-empty list of naive UTC datetimes. If the key is missing, not a list, empty, or parsing yields nothing,
      returns ChatGPT (`2022-11-30`) and GPT-4 (`2023-03-14`) calendar dates for the main launch study.
    """
    default = [datetime(2022, 11, 30), datetime(2023, 3, 14)]
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
        "thread_political_rate_threshold": 0.35,
        "thread_political_min_hits": 2,
        "forum_political_rate_multiplier_vs_politicaita": 0.25,
    }
    raw = config.get("screening", {})
    if isinstance(raw, dict):
        merged = dict(raw)
        if "min_kept_window_large_volume" not in merged and "min_kept_window_tier_a" in merged:
            merged["min_kept_window_large_volume"] = merged.pop("min_kept_window_tier_a")
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
        return "en_us_political"
    if arm == "control_europe_political":
        return "uk_political"
    if subreddit == "de":
        return "de_hub"
    if subreddit == "spain":
        return "es_hub"
    if subreddit == "unitedkingdom":
        return "uk_hub"
    if subreddit == "europe":
        return "europe_hub"

    nsfw = set(meta.get("nsfw_subreddits", []) or [])
    memes = set(meta.get("meme_humor_subreddits", []) or [])
    creators = set(meta.get("creator_celebrity_subreddits", []) or [])
    if subreddit in nsfw:
        return "it_nsfw_sensitivity"
    if subreddit in memes:
        return "it_meme_humor"
    if subreddit in creators or any(subreddit.endswith(sfx) for sfx in CREATOR_NAME_SUFFIXES):
        return "it_creator_celebrity"
    if PROFILE_USER_PATTERN.match(subreddit):
        return "it_general"
    return "it_general"


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
    if topic in {"en_us_political", "uk_political", "it_political"}:
        return "dedicated_political"
    if topic in {"de_hub", "es_hub", "uk_hub", "europe_hub"}:
        return "general_hub"
    if topic == "it_creator_celebrity":
        return "creator_celebrity"
    if topic == "it_nsfw_sensitivity":
        return "nsfw_sensitivity"
    if PROFILE_USER_PATTERN.match(subreddit):
        return "profile_user"
    return "general_hub"


def infer_subreddit_primary_lexicon(
    config: Dict[str, Any], subreddit: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Function summary: infer primary political lexicon language code (it, en, de, es).

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
    if topic in {"en_us_political", "uk_political", "europe_hub", "uk_hub"}:
        return "en"
    if topic == "de_hub":
        return "de"
    if topic == "es_hub":
        return "es"
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
    - Lexicon language code (`it`, `en`, `de`, `es`).
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
