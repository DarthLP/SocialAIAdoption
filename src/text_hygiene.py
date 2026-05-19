"""
Script summary:
Shared text-hygiene helpers for Reddit comment cleaning (URL-only detection).

Functionality:
- Detect bare-URL bodies, single-markdown-link bodies, and near-empty text after link stripping.

How to apply/run:
- Imported by `scripts/cleaning/clean_daily_chunks.py` and screening utilities.
"""

from __future__ import annotations

import re

URL_ONLY_PATTERN = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
MARKDOWN_LINK_ONLY_PATTERN = re.compile(
    r"^\s*\[https?://[^\]]+\]\(https?://[^)]+\)\s*$",
    re.IGNORECASE,
)
URL_STRIP_PATTERN = re.compile(r"https?://\S+|\[https?://[^\]]+\]\(https?://[^)]+\)", re.IGNORECASE)


def alphanumeric_content_length(text: str) -> int:
    """Function summary: count alphanumeric characters remaining after URL/markdown removal.

    Parameters:
    - text: raw comment body.

    Returns:
    - Integer count of alphanumeric characters in the residual text.
    """
    residual = URL_STRIP_PATTERN.sub("", text or "")
    return sum(1 for ch in residual if ch.isalnum())


def is_url_only_text(body: str) -> bool:
    """Function summary: return True when the body carries no substantive text beyond links.

    Parameters:
    - body: comment body string.

    Returns:
    - True if the body is URL-only under configured rules.
    """
    normalized = (body or "").strip()
    if not normalized:
        return False
    if URL_ONLY_PATTERN.match(normalized):
        return True
    if MARKDOWN_LINK_ONLY_PATTERN.match(normalized):
        return True
    if alphanumeric_content_length(normalized) < 3:
        return True
    return False
