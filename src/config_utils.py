"""
Script summary:
Shared configuration helpers for the dump-first data pipeline.
This module provides minimal utilities for loading YAML config and converting
UTC ISO timestamps to unix seconds.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Function summary: load YAML configuration from disk and return a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def utc_ts(iso_utc: str) -> int:
    """Function summary: convert an ISO UTC timestamp string to unix epoch seconds."""
    return int(datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).timestamp())
