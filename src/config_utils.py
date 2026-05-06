"""
Script summary:
Shared configuration helpers for the dump-first data pipeline.
This module provides minimal utilities for loading YAML config and converting
UTC ISO timestamps to unix seconds.
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
