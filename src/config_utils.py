"""
Script summary:
Shared configuration helpers for the dump-first data pipeline.
This module provides minimal utilities for loading YAML config, converting
UTC ISO timestamps to unix seconds, and parsing optional calendar reference dates for plots.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


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
